
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np



LabelKeepPolicy = Literal["inside", "center", "intersect", "all"]


def find_non_black_bbox(
    img: np.ndarray,
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
) -> tuple[int, int, int, int]:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:

        gray = img.mean(axis=2)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    h, w = gray.shape[:2]

    valid_mask = gray > black_threshold
    ys, xs = np.where(valid_mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No non-black region found in image.")

    x_min = max(int(xs.min()) - margin, 0)
    x_max = min(int(xs.max()) + 1 + margin, w)
    y_min = max(int(ys.min()) - margin, 0)
    y_max = min(int(ys.max()) + 1 + margin, h)

    if align_to is not None:
        if align_to <= 0:
            raise ValueError(f"align_to must be positive, but got {align_to}")

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        new_w = crop_w // align_to * align_to
        new_h = crop_h // align_to * align_to

        if new_w <= 0 or new_h <= 0:
            raise ValueError(
                f"Aligned crop size is invalid: ({new_h}, {new_w}). "
                f"Original crop size: ({crop_h}, {crop_w}), align_to={align_to}"
            )


        x_max = x_min + new_w
        y_max = y_min + new_h

    return x_min, y_min, x_max, y_max


def crop_image_by_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max, ...]


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _format_coord(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _polygon_keep_by_policy(
    points: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    policy: LabelKeepPolicy,
) -> bool:
    if policy == "all":
        return True

    x_min, y_min, x_max, y_max = crop_bbox

    xs = points[:, 0]
    ys = points[:, 1]

    if policy == "inside":
        return bool(
            np.all(xs >= x_min)
            and np.all(xs < x_max)
            and np.all(ys >= y_min)
            and np.all(ys < y_max)
        )

    if policy == "center":
        cx = float(xs.mean())
        cy = float(ys.mean())
        return bool(x_min <= cx < x_max and y_min <= cy < y_max)

    if policy == "intersect":
        poly_x_min = float(xs.min())
        poly_x_max = float(xs.max())
        poly_y_min = float(ys.min())
        poly_y_max = float(ys.max())

        no_intersection = (
            poly_x_max < x_min
            or poly_x_min >= x_max
            or poly_y_max < y_min
            or poly_y_min >= y_max
        )
        return not no_intersection

    raise ValueError(f"Unsupported label keep policy: {policy}")


def shift_dota_label_by_crop(
    src_label_path: str | os.PathLike[str],
    dst_label_path: str | os.PathLike[str],
    crop_bbox: tuple[int, int, int, int],
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
    encoding: str = "utf-8",
) -> int:
    src_label_path = Path(src_label_path)
    dst_label_path = Path(dst_label_path)
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, y_min, x_max, y_max = crop_bbox
    new_w = x_max - x_min
    new_h = y_max - y_min

    if not src_label_path.exists():

        dst_label_path.write_text("", encoding=encoding)
        return 0

    kept_lines: list[str] = []
    kept_count = 0

    with src_label_path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()





        if len(parts) < 9 or not all(_is_number(v) for v in parts[:8]):
            kept_lines.append(line)
            continue

        coords = np.array([float(v) for v in parts[:8]], dtype=np.float32)
        points = coords.reshape(4, 2)

        if not _polygon_keep_by_policy(points, crop_bbox, keep_policy):
            continue


        points[:, 0] -= x_min
        points[:, 1] -= y_min

        if clip_coords:
            points[:, 0] = np.clip(points[:, 0], 0, new_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, new_h - 1)

        new_coords = points.reshape(-1).tolist()
        new_coord_strs = [_format_coord(v) for v in new_coords]


        new_line = " ".join(new_coord_strs + parts[8:])
        kept_lines.append(new_line)
        kept_count += 1

    with dst_label_path.open("w", encoding=encoding) as f:
        for line in kept_lines:
            f.write(line + "\n")

    return kept_count


def crop_dota_image_and_label_remove_black_border(
    src_img_path: str | os.PathLike[str],
    src_label_path: str | os.PathLike[str],
    dst_img_path: str | os.PathLike[str],
    dst_label_path: str | os.PathLike[str],
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
) -> tuple[tuple[int, int, int, int], int]:
    src_img_path = Path(src_img_path)
    dst_img_path = Path(dst_img_path)
    dst_label_path = Path(dst_label_path)

    img = cv2.imread(str(src_img_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Failed to read image: {src_img_path}")

    bbox = find_non_black_bbox(
        img,
        black_threshold=black_threshold,
        margin=margin,
        align_to=align_to,
    )

    cropped_img = crop_image_by_bbox(img, bbox)

    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_img_path), cropped_img)

    if not ok:
        raise IOError(f"Failed to write image: {dst_img_path}")

    kept_count = shift_dota_label_by_crop(
        src_label_path=src_label_path,
        dst_label_path=dst_label_path,
        crop_bbox=bbox,
        keep_policy=keep_policy,
        clip_coords=clip_coords,
    )

    return bbox, kept_count

def remove_black_border_for_dota_dataset(
    src_img_dir: str | os.PathLike[str],
    src_label_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    label_name: str = "labelTxt",
    black_threshold: int = 5,
    margin: int = 0,
    align_to: int | None = None,
    keep_policy: LabelKeepPolicy = "inside",
    clip_coords: bool = False,
) -> None:
    src_img_dir = Path(src_img_dir).expanduser().resolve()
    src_label_dir = Path(src_label_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()

    out_img_dir = out_dir / "images"
    out_label_dir = out_dir / label_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {s.lower() for s in image_suffixes}

    image_paths = sorted(
        p for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )

    print(f"Found {len(image_paths)} images.")
    print(f"Output image dir: {out_img_dir}")
    print(f"Output label dir: {out_label_dir}")

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = src_label_dir / f"{img_path.stem}.txt"

        dst_img_path = out_img_dir / img_path.name
        dst_label_path = out_label_dir / f"{img_path.stem}.txt"

        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")

        bbox, kept_count = crop_dota_image_and_label_remove_black_border(
            src_img_path=img_path,
            src_label_path=label_path,
            dst_img_path=dst_img_path,
            dst_label_path=dst_label_path,
            black_threshold=black_threshold,
            margin=margin,
            align_to=align_to,
            keep_policy=keep_policy,
            clip_coords=clip_coords,
        )

        print(f"  bbox={bbox}, kept objects={kept_count}")

    print("Done.")

