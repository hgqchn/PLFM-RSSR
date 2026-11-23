import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from plfm_rssr.codec import VQF4
from plfm_rssr.network import build_model
from plfm_rssr.runtime import image_seed, load_weights, select_device, super_resolve


def main():
    parser = argparse.ArgumentParser(description="Run 4x PLFM-RSSR inference on RGB images.")
    parser.add_argument("--input", type=Path, required=True, help="An image or a directory of images.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG or directory.")
    parser.add_argument("--weights", type=Path, required=True, help="Three-way PLFM-RSSR CVN checkpoint.")
    parser.add_argument("--vq-model", type=Path, help="Local VQ-f4 folder or its parent.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--steps", type=int, default=10, help="Number of Euler updates.")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for per-image noise.")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive.")
    if not args.input.exists():
        parser.error(f"Input not found: {args.input}")
    is_folder = args.input.is_dir()
    suffixes = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    inputs = [args.input]
    if is_folder:
        inputs = sorted(path for path in args.input.iterdir()
                        if path.is_file() and path.suffix.lower() in suffixes)

    if not is_folder and args.output.suffix and not args.output.is_dir():
        if args.output.suffix.lower() != ".png":
            parser.error("A single output file must have the .png extension.")
        outputs = [args.output]
    else:
        outputs = [args.output / f"{path.stem}.png" for path in inputs]

    output_paths = {path.resolve() for path in outputs}
    input_paths = {path.resolve() for path in inputs}
    if len(output_paths) != len(outputs):
        parser.error("Two input files would have the same output name.")
    if output_paths & input_paths:
        parser.error("Input files cannot also be output files.")

    device = select_device(args.device)
    model = build_model().eval()
    load_weights(model, args.weights)
    model.to(device)
    codec = VQF4(args.vq_model).to(device)
    for source, target in zip(inputs, outputs, strict=True):
        with Image.open(source) as image:
            array = np.array(image.convert("RGB"))
        lr = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255)
        sr = super_resolve(model, codec, lr, args.steps, image_seed(source.name, args.seed))
        pixels = sr[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target, format="PNG")
        print(target)


if __name__ == "__main__":
    main()
