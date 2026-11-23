
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def copy_labeled_images(source_dir: Path, output_dir: Path) -> tuple[int, int]:
    image_root, label_root = source_dir / "images", source_dir / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise NotADirectoryError(f"数据集必须包含 images/ 和 labels/：{source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")

    kept = removed = 0
    image_paths = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    progress = tqdm(image_paths, desc="复制有标签样本", unit="image")
    for image_path in progress:
        relative_path = image_path.relative_to(image_root)
        label_path = label_root / relative_path.with_suffix(".txt")
        if not label_path.is_file() or not label_path.read_text(encoding="utf-8").strip():
            removed += 1
            progress.set_postfix(kept=kept, skipped=removed)
            continue
        target_image = output_dir / "images" / relative_path
        target_label = output_dir / "labels" / relative_path.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        kept += 1
        progress.set_postfix(kept=kept, skipped=removed)
    return kept, removed


def collect_dota_pairs(
    image_dir: str | Path,
    label_dir: str | Path,
    image_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    require_label: bool = True,
) -> list[tuple[Path, Path]]:
    image_dir = Path(image_dir).expanduser().resolve()
    label_dir = Path(label_dir).expanduser().resolve()

    if not image_dir.is_dir():
        raise NotADirectoryError(f"图像目录不存在或不是目录: {image_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(f"标签目录不存在或不是目录: {label_dir}")

    suffixes = {s.lower() for s in image_suffixes}
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)
            if not require_label:
                pairs.append((image_path, label_path))

    if require_label and missing_labels:
        print(f"[Warning] 有 {len(missing_labels)} 张图像缺少标签，已跳过。")

    if len(pairs) == 0:
        raise RuntimeError(
            f"没有找到有效的 DOTA 图像-标签配对: image_dir={image_dir}, label_dir={label_dir}"
        )

    return pairs


def random_select_dota_samples(
    image_dir: str | Path,
    label_dir: str | Path,
    manifest_path: str | Path,
    num_samples: int,
    seed: int = 0,
    image_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    require_label: bool = True,
) -> list[tuple[Path, Path]]:
    if num_samples <= 0:
        raise ValueError(f"num_samples 必须大于 0，但得到: {num_samples}")

    pairs = collect_dota_pairs(
        image_dir=image_dir,
        label_dir=label_dir,
        image_suffixes=image_suffixes,
        require_label=require_label,
    )

    if num_samples > len(pairs):
        raise ValueError(
            f"num_samples={num_samples} 大于可用样本数 {len(pairs)}"
        )

    rng = random.Random(seed)

    selected_pairs = rng.sample(pairs, num_samples)

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_path", "name"])

        for image_path, label_path in selected_pairs:
            writer.writerow([
                str(image_path),
                str(label_path),
                image_path.stem,
            ])

    print(f"[Info] 已随机挑选 {len(selected_pairs)} 个样本")
    print(f"[Info] manifest 保存到: {manifest_path}")

    return selected_pairs


def read_dota_manifest(
    manifest_path: str | Path,
) -> list[tuple[Path, Path, str]]:
    manifest_path = Path(manifest_path).expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 文件不存在: {manifest_path}")

    samples: list[tuple[Path, Path, str]] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_fields = {"image_path", "label_path", "name"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                f"manifest 文件字段错误，需要包含: {required_fields}，"
                f"当前字段: {reader.fieldnames}"
            )

        for row in reader:
            image_path = Path(row["image_path"]).expanduser().resolve()
            label_path = Path(row["label_path"]).expanduser().resolve()
            name = row["name"]
            samples.append((image_path, label_path, name))

    return samples


def copy_dota_samples_from_manifest(
    manifest_path: str | Path,
    output_image_dir: str | Path,
    output_label_dir: str | Path,
    overwrite: bool = False,
) -> None:
    output_image_dir = Path(output_image_dir).expanduser().resolve()
    output_label_dir = Path(output_label_dir).expanduser().resolve()

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    samples = read_dota_manifest(manifest_path)

    for image_path, label_path, name in samples:
        if not image_path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        dst_image_path = output_image_dir / image_path.name
        dst_label_path = output_label_dir / label_path.name

        if not overwrite:
            if dst_image_path.exists():
                raise FileExistsError(f"目标图像已存在: {dst_image_path}")
            if dst_label_path.exists():
                raise FileExistsError(f"目标标签已存在: {dst_label_path}")

        shutil.copy2(image_path, dst_image_path)
        shutil.copy2(label_path, dst_label_path)

    print(f"[Info] 已复制 {len(samples)} 个样本")
    print(f"[Info] 图像保存到: {output_image_dir}")
    print(f"[Info] 标签保存到: {output_label_dir}")


def make_dota_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    output_root: str | Path,
    num_samples: int,
    seed: int = 0,
    manifest_name: str = "selected_samples.csv",
    overwrite: bool = False,
) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    output_image_dir = output_root / "images"
    output_label_dir = output_root / "labelTxt"
    manifest_path = output_root / manifest_name

    random_select_dota_samples(
        image_dir=image_dir,
        label_dir=label_dir,
        manifest_path=manifest_path,
        num_samples=num_samples,
        seed=seed,
    )

    copy_dota_samples_from_manifest(
        manifest_path=manifest_path,
        output_image_dir=output_image_dir,
        output_label_dir=output_label_dir,
        overwrite=overwrite,
    )

    return manifest_path


