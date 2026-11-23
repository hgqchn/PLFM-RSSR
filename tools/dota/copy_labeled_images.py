
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dota.dota_pick import copy_labeled_images


def main() -> None:
    parser = argparse.ArgumentParser(description='Copy DOTA image pairs with nonempty labels.')
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    kept, removed = copy_labeled_images(args.source_dir, args.output_dir)
    print(f"完成：保留 {kept} 张有标签图像，跳过 {removed} 张无标签或空标签图像")


if __name__ == "__main__":
    main()
