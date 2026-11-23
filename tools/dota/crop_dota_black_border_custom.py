
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dota.dota_black_border_custom_utils import remove_black_border_for_dota_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Remove black borders from DOTA images and adjust labels.')
    parser.add_argument("--src-img-dir", required=True)
    parser.add_argument("--src-label-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--black-threshold", type=int, default=5)
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--align-to", type=int)
    parser.add_argument("--keep-policy", choices=("inside", "center", "intersect", "all"), default="inside")
    parser.add_argument("--clip-coords", action="store_true")
    args = parser.parse_args()
    remove_black_border_for_dota_dataset(
        args.src_img_dir,
        args.src_label_dir,
        args.out_dir,
        black_threshold=args.black_threshold,
        margin=args.margin,
        align_to=args.align_to,
        keep_policy=args.keep_policy,
        clip_coords=args.clip_coords,
    )


if __name__ == "__main__":
    main()
