
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dota.dota_split_custom_utils import split_dota_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Split DOTA images and labels into patches.')
    parser.add_argument("--src-img-dir", required=True)
    parser.add_argument("--src-label-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--keep-policy", choices=("inside", "center"), default="inside")
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--image-format", choices=("png", "jpg", "jpeg"), default="png")
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()
    split_dota_dataset(
        args.src_img_dir,
        args.src_label_dir,
        args.out_dir,
        crop_size=args.crop_size,
        stride=args.stride,
        keep_policy=args.keep_policy,
        drop_empty=args.drop_empty,
        image_format=args.image_format,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
