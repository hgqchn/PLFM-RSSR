from typing import Dict, List, Tuple, Union, Literal
import cv2
import math
import numpy as np
import os
import torch
import random
from pathlib import Path
from natsort import natsorted

def img_from_bytes(content):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"暂不支持该图像维度: {img.shape}")

    img = img_uint8_to_float32(img)

    return img

def read_img_as_rgb_float(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"暂不支持该图像维度: {img.shape}, path={image_path}")

    if img.dtype == np.uint8:
        img = img_uint8_to_float32(img)
    else:
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        if info is not None:
            img = img.astype(np.float32) / float(info.max)
        else:
            raise TypeError(
                f"Unsupported image dtype: {img.dtype}. "
                "Expected uint8 or integer image. "
                "If your image is float, please normalize it explicitly before degradation."
            )

    return img


def write_rgb_float_img(img_rgb: np.ndarray, save_path: str) -> None:
    img_uint8 = img_float32_to_uint8(img_rgb)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _write_img(img_bgr, str(save_path))


def _write_img(
    img: np.ndarray,
    file_path: str,
    params: Union[None, List[int]] = None,
    auto_mkdir: bool = True
) -> None:
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError("Failed in writing images.")


def img_uint8_to_float32(img: np.ndarray) -> np.ndarray:
    return (img / 255.0).astype(np.float32)


def img_float32_to_uint8(img: np.ndarray) -> np.ndarray:
    return (img.clip(0, 1) * 255.0).round().astype(np.uint8)

def img_array_to_tensor(
    images_rgb: Union[np.ndarray, List[np.ndarray]],
    return_batch_tensor: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(images_rgb, np.ndarray):
        tensor = torch.from_numpy(images_rgb).permute(2, 0, 1).float()
        return tensor.unsqueeze(0) if return_batch_tensor else tensor

    if not images_rgb:
        raise ValueError("images_rgb 不能为空")

    tensors = [torch.from_numpy(img).permute(2, 0, 1).float() for img in images_rgb]
    if not return_batch_tensor:
        return tensors

    shapes = [img.shape for img in images_rgb]
    first_shape = shapes[0]
    if any(shape != first_shape for shape in shapes):
        raise ValueError(f"一个 batch 内所有图像 shape 必须一致，当前 shapes={shapes}")
    return torch.stack(tensors, dim=0)


def tensor_to_img_array(tensor: torch.Tensor) -> Union[np.ndarray, List[np.ndarray]]:
    tensor = tensor.detach().float().cpu().clamp(0, 1)

    if tensor.ndim == 4:
        if tensor.shape[0] == 1:
            return tensor[0].permute(1, 2, 0).numpy()
        return [img.permute(1, 2, 0).numpy() for img in tensor]

    if tensor.ndim == 3:
        return tensor.permute(1, 2, 0).numpy()

    raise ValueError(f"tensor 必须是 3D 或 4D，当前 shape={tuple(tensor.shape)}")


def crop_img_by_border(
    img: np.ndarray,
    crop_border: int
) -> np.ndarray:
    if crop_border == 0:
        return img
    return img[crop_border:-crop_border, crop_border:-crop_border, ...]

def crop_img_by_shape(
    img: np.ndarray,
    crop_shape: tuple[int,int],
    mode: Literal["default", "center"] = "default",
) -> np.ndarray:
    new_h,new_w = crop_shape

    assert img.ndim==2 or img.ndim==3
    height, width = img.shape[0], img.shape[1]

    if new_h>height or new_w>width:
        raise ValueError(" ")

    h_start, w_start = get_crop_start(height, width, new_h, new_w, mode)

    h_end = h_start + new_h
    w_end = w_start + new_w

    img_crop = img[h_start:h_end, w_start:w_end, ...]
    return img_crop

def crop_img_by_scale(
    img: np.ndarray,
    scale: int,
    mode: Literal["default", "center"] = "default",
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"scale 必须大于 0，但得到 {scale}")

    assert img.ndim==2 or img.ndim==3
    height, width = img.shape[0], img.shape[1]
    new_h = height // scale * scale
    new_w = width // scale * scale

    h_start, w_start = get_crop_start(height, width, new_h, new_w, mode)

    h_end = h_start + new_h
    w_end = w_start + new_w

    img_crop = img[h_start:h_end, w_start:w_end, ...]
    return img_crop


def get_crop_start(
    height: int,
    width: int,
    crop_h: int,
    crop_w: int,
    mode: Literal["default", "center"] = "default",
) -> tuple[int, int]:
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"裁剪尺寸必须大于 0，但得到 crop_h={crop_h}, crop_w={crop_w}")

    if height < crop_h or width < crop_w:
        raise ValueError(
            f"图像尺寸小于裁剪尺寸: image=({height}, {width}), crop=({crop_h}, {crop_w})"
        )

    if mode == "center":
        h_start = (height - crop_h) // 2
        w_start = (width - crop_w) // 2
        return h_start, w_start
    return 0, 0



def paired_random_crop_numpy(img_gt, img_lq, gt_patch_size, scale, gt_path=None):
    h_lq, w_lq = img_lq.shape[0:2]
    h_gt, w_gt = img_gt.shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt == gt_patch_size and w_gt == gt_patch_size:
        return img_gt, img_lq

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})."
        )

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        msg = (
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
            f"({lq_patch_size}, {lq_patch_size})."
        )
        if gt_path is not None:
            msg += f" Please remove {gt_path}."
        raise ValueError(msg)

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lq = img_lq[top:top + lq_patch_size, left:left + lq_patch_size, ...]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gt = img_gt[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]

    return img_gt, img_lq


def paired_random_crop_tensor(img_gts, img_lqs, gt_patch_size, scale):
    if not (torch.is_tensor(img_gts) and torch.is_tensor(img_lqs)):
        raise TypeError("paired_random_crop_tensor 仅支持 torch.Tensor 输入")

    h_lq, w_lq = img_lqs.shape[-2:]
    h_gt, w_gt = img_gts.shape[-2:]

    if h_gt == gt_patch_size and w_gt == gt_patch_size:
        return img_gts, img_lqs

    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})."
        )

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_lqs = img_lqs[:, :, top:top + lq_patch_size, left:left + lq_patch_size]
    img_gts = img_gts[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size]
    return img_gts, img_lqs


def img_augment(img, mode=0):

    if mode == 0:
        out = img
    elif mode == 1:
        out = np.rot90(img, k=1)
    elif mode == 2:
        out = np.flipud(np.rot90(img, k=1))
    elif mode == 3:
        out = np.rot90(img, k=2)
    elif mode == 4:
        out = np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        out = np.rot90(img, k=3)
    elif mode == 6:
        out = np.flipud(np.rot90(img, k=3))
    elif mode == 7:
        out = np.flipud(img)
    else:
        raise ValueError(f"mode must be in [0, 7], but got {mode}")


    img = np.ascontiguousarray(out)

    return img

def img_rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def transform_points_by_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:

    points = np.asarray(points, dtype=np.float32)

    original_shape = points.shape

    points = points.reshape(-1, 2)

    ones = np.ones((points.shape[0], 1), dtype=np.float32)

    points_h = np.concatenate([points, ones], axis=1)

    transformed = points_h @ np.asarray(matrix, dtype=np.float32).T

    return transformed.astype(np.float32).reshape(original_shape)


def get_scale_affine_matrix(scale_x: float, scale_y: float) -> np.ndarray:
    return np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)


def get_img_augment_affine_matrix(mode: int, img_shape) -> np.ndarray:
    height, width = img_shape[:2]
    if mode == 0:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    if mode == 1:
        return np.array([[0, 1, 0], [-1, 0, width - 1]], dtype=np.float32)
    if mode == 2:
        return np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    if mode == 3:
        return np.array([[-1, 0, width - 1], [0, -1, height - 1]], dtype=np.float32)
    if mode == 4:
        return np.array([[-1, 0, width - 1], [0, 1, 0]], dtype=np.float32)
    if mode == 5:
        return np.array([[0, -1, height - 1], [1, 0, 0]], dtype=np.float32)
    if mode == 6:
        return np.array([[0, -1, height - 1], [-1, 0, width - 1]], dtype=np.float32)
    if mode == 7:
        return np.array([[1, 0, 0], [0, -1, height - 1]], dtype=np.float32)
    raise ValueError(f"mode must be in [0, 7], but got {mode}")


def get_img_rotate_affine_matrix(img_shape, angle, center=None, scale=1.0) -> np.ndarray:
    height, width = img_shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    return cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)


def transform_polygons_by_affine(polygons: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return transform_points_by_affine(polygons, matrix)


image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def is_image_file(filename: str) -> bool:

    return filename.lower().endswith(image_extensions)


def get_image_paths(imagedir: str | os.PathLike[str]) -> list[str]:
    folder = Path(imagedir)
    img_list = natsorted(
        [
            str(path)
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
    )
    return img_list


if __name__ == "__main__":
    pass
