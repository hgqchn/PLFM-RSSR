
import argparse
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dota.dota_yolo_dataset_utils import (
    drop_incomplete_patches,
    make_lowres_dataset,
    split_dataset,
    write_yaml,
)


def parse_numbers(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析数字列表：{text}") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"数字必须为正数：{text}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Split YOLO-OBB DOTA images and generate LR patches.')
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "trainval", "test"), default="trainval")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--rates", type=parse_numbers, default=(1.0,))
    parser.add_argument("--lowres-scales", type=parse_numbers, default=(2.0, 4.0))
    parser.add_argument("--drop-incomplete", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--degrade-mode", choices=("gpu", "single"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data_root, save_dir = Path(args.data_root), Path(args.save_dir)
    if not data_root.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{data_root}")
    if save_dir.exists():
        raise FileExistsError(f"输出目录已存在：{save_dir}")
    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于 0")

    split_dataset(data_root, save_dir, args.split, args.crop_size, args.gap, args.rates)
    if args.drop_incomplete:
        drop_incomplete_patches(save_dir, args.split, args.crop_size, args.rates)
    write_yaml(save_dir)
    for scale in tqdm(args.lowres_scales, desc="Generating low-resolution datasets", unit="scale"):
        if not scale.is_integer() or scale < 2:
            raise ValueError("--lowres-scales 只支持不小于 2 的整数，例如 2,4")
        print(f"low-resolution dataset: {make_lowres_dataset(save_dir, int(scale), args.device, args.degrade_mode, args.batch_size)}")


if __name__ == "__main__":
    main()
