
import shutil
import sys
from pathlib import Path

from PIL import Image
from ultralytics.data.split_dota import split_images_and_labels
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_lr import degrade_folder, degrade_folder_gpu


NAMES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
]


def write_yaml(dataset_dir: Path) -> None:
    if not (dataset_dir / "images" / "train").is_dir() or not (dataset_dir / "images" / "val").is_dir():
        return

    lines = [f"path: {dataset_dir.as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(NAMES)]
    (dataset_dir / f"{dataset_dir.name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_dataset(data_root: Path, save_dir: Path, split: str, crop_size: int, gap: int, rates: tuple[float, ...]) -> None:
    crop_sizes = tuple(int(crop_size / rate) for rate in rates)
    gaps = tuple(int(gap / rate) for rate in rates)
    splits = ("train", "val") if split == "trainval" else (split,)
    for split_name in splits:
        print(f"开始切分 {split_name}：crop_sizes={crop_sizes}, gaps={gaps}")
        split_images_and_labels(str(data_root), str(save_dir), split=split_name, crop_sizes=crop_sizes, gaps=gaps)
        print(f"完成切分 {split_name}")


def drop_incomplete_patches(save_dir: Path, split: str, crop_size: int, rates: tuple[float, ...]) -> None:
    valid_sizes = {int(crop_size / rate) for rate in rates}
    splits = ("train", "val") if split == "trainval" else (split,)
    for split_name in splits:
        image_dir = save_dir / "images" / split_name
        label_dir = save_dir / "labels" / split_name
        removed = 0
        image_paths = [path for path in image_dir.iterdir() if path.is_file()]
        for image_path in tqdm(image_paths, desc=f"Filtering {split_name}", unit="image"):
            with Image.open(image_path) as image:
                keep = image.width == image.height and image.width in valid_sizes
            if not keep:
                image_path.unlink()
                (label_dir / f"{image_path.stem}.txt").unlink(missing_ok=True)
                removed += 1
        print(f"{split_name}: 保留 {len(image_paths) - removed} 张完整切块，删除 {removed} 张边缘切块")


def make_lowres_dataset(src_dir: Path, scale: int, device: str, degrade_mode: str, batch_size: int) -> Path:
    dst_dir = src_dir.parent / f"{src_dir.name}_lr_x{scale}"
    if dst_dir.exists():
        raise FileExistsError(f"低分辨率输出目录已存在，请先更换 --save-dir 或手动删除：{dst_dir}")

    for image_dir in sorted((src_dir / "images").iterdir()):
        if not image_dir.is_dir():
            continue
        output_dir = dst_dir / "images" / image_dir.name
        if degrade_mode == "gpu":
            degrade_folder_gpu(
                str(image_dir), str(output_dir), scale=scale, batch_size=batch_size, device=device, recursive=False
            )
        else:
            degrade_folder(str(image_dir), str(output_dir), scale=scale, device=device, recursive=False)
        label_dir = src_dir / "labels" / image_dir.name
        if label_dir.is_dir():
            shutil.copytree(label_dir, dst_dir / "labels" / image_dir.name)

    write_yaml(dst_dir)
    return dst_dir
