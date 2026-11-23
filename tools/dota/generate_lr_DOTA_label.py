
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dota.label_transform_utils import scale_dota_label_folder


def main() -> None:
    parser = argparse.ArgumentParser(description='Scale DOTA label files to the low-resolution coordinate system.')
    parser.add_argument("--src-label-dir", required=True)
    parser.add_argument("--dst-label-dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-box-size", type=float)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()
    if args.scale < 1:
        raise ValueError("--scale 必须大于 0")
    scale_dota_label_folder(
        args.src_label_dir,
        args.dst_label_dir,
        scale=args.scale,
        recursive=not args.no_recursive,
        min_box_size=args.min_box_size,
    )


if __name__ == "__main__":
    main()
