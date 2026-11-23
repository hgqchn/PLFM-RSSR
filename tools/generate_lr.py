
import argparse
import os
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplesr.utils.degradations import (
    circular_lowpass_kernel,
    random_mixed_kernels,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from simplesr.utils.diffjpeg import DiffJPEG
from simplesr.utils.img_process_utils import filter2D,USMSharp

from simplesr.utils.img_utils import read_img_as_rgb_float, tensor_to_img_array, img_array_to_tensor, write_rgb_float_img





DEFAULT_DEGRADATION_OPT = {

    "scale": 4,



    "use_usm": True,


    "use_jpeg": False,


    "use_jpeg2": False,





    "resize_prob": [0.0, 0.0, 1.0],

    "resize_range": [0.15, 1.5],


    "gaussian_noise_prob": 0.5,

    "noise_range": [1, 30],

    "poisson_scale_range": [0.05, 3],

    "gray_noise_prob": 0.4,


    "jpeg_range": [30, 95],





    "second_blur_prob": 0.8,


    "resize_prob2": [0.0, 0.0, 1.0],

    "resize_range2": [0.3, 1.2],


    "gaussian_noise_prob2": 0.5,

    "noise_range2": [1, 25],

    "poisson_scale_range2": [0.05, 2.5],

    "gray_noise_prob2": 0.4,


    "jpeg_range2": [30, 95],





    "blur_kernel_size": 21,


    "kernel_list": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],

    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],

    "sinc_prob": 0.1,

    "blur_sigma": [0.2, 3],

    "betag_range": [0.5, 4],

    "betap_range": [1, 2],





    "blur_kernel_size2": 21,

    "kernel_list2": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],

    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],

    "sinc_prob2": 0.1,

    "blur_sigma2": [0.2, 1.5],

    "betag_range2": [0.5, 4],

    "betap_range2": [1, 2],


    "final_sinc_prob": 0.8,
}






def generate_blur_kernel(
    opt: Dict,
    stage: int = 1,
    kernel_range: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if kernel_range is None:

        kernel_range = [2 * v + 1 for v in range(3, 11)]

    kernel_size = random.choice(kernel_range)

    if stage == 1:
        sinc_prob = opt["sinc_prob"]
        kernel_list = opt["kernel_list"]
        kernel_prob = opt["kernel_prob"]
        blur_sigma = opt["blur_sigma"]
        betag_range = opt["betag_range"]
        betap_range = opt["betap_range"]
        target_kernel_size = opt["blur_kernel_size"]
    else:
        sinc_prob = opt["sinc_prob2"]
        kernel_list = opt["kernel_list2"]
        kernel_prob = opt["kernel_prob2"]
        blur_sigma = opt["blur_sigma2"]
        betag_range = opt["betag_range2"]
        betap_range = opt["betap_range2"]
        target_kernel_size = opt["blur_kernel_size2"]


    if np.random.uniform() < sinc_prob:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)

        kernel = circular_lowpass_kernel(
            omega_c,
            kernel_size,
            pad_to=False,
        )


    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None,
        )


    pad_size = (target_kernel_size - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    return kernel.astype(np.float32)


def generate_final_sinc_kernel(
    opt: Dict,
    kernel_range: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if kernel_range is None:
        kernel_range = [2 * v + 1 for v in range(3, 11)]

    target_kernel_size = 21

    if np.random.uniform() < opt["final_sinc_prob"]:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)

        sinc_kernel = circular_lowpass_kernel(
            omega_c,
            kernel_size,
            pad_to=target_kernel_size,
        )
    else:

        sinc_kernel = np.zeros((target_kernel_size, target_kernel_size), dtype=np.float32)
        sinc_kernel[target_kernel_size // 2, target_kernel_size // 2] = 1.0

    return sinc_kernel.astype(np.float32)


def kernel_to_tensor(kernel: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(kernel).float().unsqueeze(0).to(device)









class RealESRGANStyleDegrader:

    def __init__(
        self,
        opt: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        self.opt = DEFAULT_DEGRADATION_OPT.copy()
        if opt is not None:
            self.opt.update(opt)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


        if self.opt.get("use_jpeg", True) or self.opt.get("use_jpeg2", True):
            self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        else:
            self.jpeger = None


        self.usm_sharpener = USMSharp().to(self.device)

    @torch.no_grad()
    def degrade_image_array(self, img_rgb: np.ndarray) -> np.ndarray:
        opt = self.opt
        scale = int(opt["scale"])


        gt = img_array_to_tensor(img_rgb, return_batch_tensor=True).to(self.device)
        ori_h, ori_w = gt.shape[2:4]


        if ori_h // scale <= 0 or ori_w // scale <= 0:
            raise ValueError(
                f"输入图像尺寸小于 scale={scale}，"
                f"当前尺寸为 H={ori_h}, W={ori_w}"
            )


        if opt.get("use_usm", True):
            gt_for_degradation = self.usm_sharpener(gt)
        else:
            gt_for_degradation = gt


        kernel1 = kernel_to_tensor(generate_blur_kernel(opt, stage=1), self.device)
        kernel2 = kernel_to_tensor(generate_blur_kernel(opt, stage=2), self.device)
        sinc_kernel = kernel_to_tensor(generate_final_sinc_kernel(opt), self.device)






        out = filter2D(gt_for_degradation, kernel1)


        updown_type = random.choices(["up", "down", "keep"], opt["resize_prob"])[0]
        if updown_type == "up":
            resize_scale = np.random.uniform(1, opt["resize_range"][1])
        elif updown_type == "down":
            resize_scale = np.random.uniform(opt["resize_range"][0], 1)
        else:
            resize_scale = 1.0

        mode = random.choice(["area", "bilinear", "bicubic"])
        if resize_scale != 1.0:
            out = F.interpolate(out, scale_factor=resize_scale, mode=mode)


        if np.random.uniform() < opt["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=opt["noise_range"],
                clip=True,
                rounds=False,
                gray_prob=opt["gray_noise_prob"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt["poisson_scale_range"],
                gray_prob=opt["gray_noise_prob"],
                clip=True,
                rounds=False,
            )


        if opt.get("use_jpeg", True):
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)






        if np.random.uniform() < opt["second_blur_prob"]:
            out = filter2D(out, kernel2)


        updown_type = random.choices(["up", "down", "keep"], opt["resize_prob2"])[0]
        if updown_type == "up":
            resize_scale = np.random.uniform(1, opt["resize_range2"][1])
        elif updown_type == "down":
            resize_scale = np.random.uniform(opt["resize_range2"][0], 1)
        else:
            resize_scale = 1.0

        mode = random.choice(["area", "bilinear", "bicubic"])


        target_h = int(ori_h / scale * resize_scale)
        target_w = int(ori_w / scale * resize_scale)
        target_h = max(1, target_h)
        target_w = max(1, target_w)

        out = F.interpolate(out, size=(target_h, target_w), mode=mode)


        if np.random.uniform() < opt["gaussian_noise_prob2"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=opt["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=opt["gray_noise_prob2"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt["poisson_scale_range2"],
                gray_prob=opt["gray_noise_prob2"],
                clip=True,
                rounds=False,
            )







        final_h = ori_h // scale
        final_w = ori_w // scale
        already_at_final_size = (target_h == final_h and target_w == final_w)

        if not opt.get("use_jpeg2", True):
            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        elif np.random.uniform() < 0.5:

            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)

            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        else:

            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt["jpeg_range2"])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            if not already_at_final_size:
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, size=(final_h, final_w), mode=mode)
            out = filter2D(out, sinc_kernel)


        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0


        lr_rgb = tensor_to_img_array(out)
        return lr_rgb


    def degrade_file(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        img_rgb = read_img_as_rgb_float(input_path)
        lr_rgb = self.degrade_image_array(img_rgb)
        write_rgb_float_img(lr_rgb, output_path)


class RealESRGANStyleDegraderGPU:

    def __init__(self, opt: Dict, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' 但当前环境不可用 CUDA")

        self.opt = DEFAULT_DEGRADATION_OPT.copy()
        self.opt.update(opt)
        self.device = device

        if self.opt.get("use_jpeg", True) or self.opt.get("use_jpeg2", True):
            self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        else:
            self.jpeger = None

        self.usm_sharpener = USMSharp().to(self.device)

    def _make_kernel_batch(self, batch_size: int, stage: int) -> torch.Tensor:
        kernels = [
            generate_blur_kernel(self.opt, stage=stage)
            for _ in range(batch_size)
        ]
        kernel_np = np.stack(kernels, axis=0).astype(np.float32)
        return torch.from_numpy(kernel_np).to(self.device)

    def _make_sinc_kernel_batch(self, batch_size: int) -> torch.Tensor:
        kernels = [
            generate_final_sinc_kernel(self.opt)
            for _ in range(batch_size)
        ]
        kernel_np = np.stack(kernels, axis=0).astype(np.float32)
        return torch.from_numpy(kernel_np).to(self.device)

    @torch.no_grad()
    def degrade_batch_tensor(self, gt: torch.Tensor) -> torch.Tensor:
        if gt.ndim != 4:
            raise ValueError(f"gt 必须是 4 维张量 [B, C, H, W]，当前 shape={tuple(gt.shape)}")

        gt = gt.to(self.device, non_blocking=True).float()
        b, _, ori_h, ori_w = gt.shape
        scale = int(self.opt["scale"])


        if ori_h // scale <= 0 or ori_w // scale <= 0:
            raise ValueError(
                f"输入图像尺寸小于 scale={scale}，当前尺寸为 H={ori_h}, W={ori_w}"
            )

        if self.opt.get("use_usm", True):
            gt_for_degradation = self.usm_sharpener(gt)
        else:
            gt_for_degradation = gt

        kernel1_batch = self._make_kernel_batch(b, stage=1)
        kernel2_batch = self._make_kernel_batch(b, stage=2)
        sinc_kernel_batch = self._make_sinc_kernel_batch(b)


        out_batch = filter2D(gt_for_degradation, kernel1_batch)
        results = []

        for idx in range(b):
            out = out_batch[idx:idx + 1]
            kernel2 = kernel2_batch[idx:idx + 1]
            sinc_kernel = sinc_kernel_batch[idx:idx + 1]


            updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
            if updown_type == "up":
                resize_scale = np.random.uniform(1, self.opt["resize_range"][1])
            elif updown_type == "down":
                resize_scale = np.random.uniform(self.opt["resize_range"][0], 1)
            else:
                resize_scale = 1.0

            mode = random.choice(["area", "bilinear", "bicubic"])
            if resize_scale != 1.0:
                out = F.interpolate(out, scale_factor=resize_scale, mode=mode)


            if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range"],
                    clip=True,
                    rounds=False,
                    gray_prob=self.opt["gray_noise_prob"],
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range"],
                    gray_prob=self.opt["gray_noise_prob"],
                    clip=True,
                    rounds=False,
                )


            if self.opt.get("use_jpeg", True):
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)


            if np.random.uniform() < self.opt["second_blur_prob"]:
                out = filter2D(out, kernel2)


            updown_type = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[0]
            if updown_type == "up":
                resize_scale = np.random.uniform(1, self.opt["resize_range2"][1])
            elif updown_type == "down":
                resize_scale = np.random.uniform(self.opt["resize_range2"][0], 1)
            else:
                resize_scale = 1.0

            mode = random.choice(["area", "bilinear", "bicubic"])
            target_h = max(1, int(ori_h / scale * resize_scale))
            target_w = max(1, int(ori_w / scale * resize_scale))
            out = F.interpolate(out, size=(target_h, target_w), mode=mode)


            if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.opt["noise_range2"],
                    clip=True,
                    rounds=False,
                    gray_prob=self.opt["gray_noise_prob2"],
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt["poisson_scale_range2"],
                    gray_prob=self.opt["gray_noise_prob2"],
                    clip=True,
                    rounds=False,
                )

            final_h = ori_h // scale
            final_w = ori_w // scale
            already_at_final_size = (target_h == final_h and target_w == final_w)

            if not self.opt.get("use_jpeg2", True):
                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)
            elif np.random.uniform() < 0.5:
                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)

                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range2"])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                if not already_at_final_size:
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(out, size=(final_h, final_w), mode=mode)
                out = filter2D(out, sinc_kernel)

            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
            results.append(out)

        return torch.cat(results, dim=0)






def degrade_single_image(
    input_path: str,
    output_path: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    device: Optional[str] = None,
) -> None:
    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegrader(opt=final_opt, device=device)
    degrader.degrade_file(input_path, output_path)






def collect_image_paths(
    input_dir: str,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    recursive: bool = True,
) -> Sequence[Path]:
    input_dir = Path(input_dir)

    if recursive:
        paths = [
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
    else:
        paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

    return sorted(paths)


def degrade_folder(
    input_dir: str,
    output_dir: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    device: Optional[str] = None,
    recursive: bool = True,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegrader(opt=final_opt, device=device)
    image_paths = collect_image_paths(str(input_dir), recursive=recursive)

    print(f"发现 {len(image_paths)} 张图像，开始退化...")

    for input_path in tqdm(image_paths, desc="Degrading images"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        degrader.degrade_file(str(input_path), str(output_path))

    print(f"完成，LR 图像保存到: {output_dir}")


def degrade_folder_gpu(
    input_dir: str,
    output_dir: str,
    scale: int = 4,
    opt: Optional[Dict] = None,
    batch_size: int = 16,
    recursive: bool = True,
    device: str = "cuda",
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device='{device}' 但当前环境不可用 CUDA")

    torch.backends.cudnn.benchmark = True

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_opt = {"scale": scale}
    if opt is not None:
        final_opt.update(opt)

    degrader = RealESRGANStyleDegraderGPU(opt=final_opt, device=device)
    image_paths = collect_image_paths(str(input_dir), recursive=recursive)


    paths_by_size = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            paths_by_size.setdefault(image.size, []).append(image_path)

    batches = [paths[start:start + batch_size] for paths in paths_by_size.values() for start in range(0, len(paths), batch_size)]
    print(f"发现 {len(image_paths)} 张图像，batch_size={batch_size}，使用 {device} 开始退化...")

    with tqdm(total=len(image_paths), desc="Degrading images (GPU batch)", unit="image") as progress:
        for batch_paths in batches:
            imgs_rgb = [read_img_as_rgb_float(str(path)) for path in batch_paths]
            gt_batch = img_array_to_tensor(imgs_rgb, return_batch_tensor=True).to(device, non_blocking=True)
            lq_batch = degrader.degrade_batch_tensor(gt_batch)
            lq_imgs = tensor_to_img_array(lq_batch)

            if isinstance(lq_imgs, np.ndarray):
                lq_imgs = [lq_imgs]
            for input_path, lq_img in zip(batch_paths, lq_imgs, strict=True):
                rel_path = input_path.relative_to(input_dir)
                output_path = output_dir / rel_path
                write_rgb_float_img(lq_img, str(output_path))
            progress.update(len(batch_paths))

    print(f"完成，LR 图像保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate low-resolution inputs with the SimpleSR degradation pipeline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("gpu", "single"), default="single")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()
    if args.scale < 2 or args.batch_size < 1:
        parser.error("--scale must be at least 2 and --batch-size must be positive")
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")

    options = {"scale": args.scale, "device": args.device}
    if args.input.is_file():
        degrade_single_image(str(args.input), str(args.output), **options)
        return
    options["recursive"] = not args.no_recursive
    if args.mode == "gpu":
        degrade_folder_gpu(str(args.input), str(args.output), batch_size=args.batch_size, **options)
    else:
        degrade_folder(str(args.input), str(args.output), **options)


if __name__ == "__main__":
    main()
