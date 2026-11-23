
from pathlib import Path
import numpy as np
from typing import List, Dict
from PIL import Image

IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
    "airport",
    "helipad",
]

ULTRALYTICS_DOTA_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
    "container-crane",
    "airport",
    "helipad",
]

def parse_dota_label(label_path):
    label_path = Path(label_path)
    objects = []

    if not label_path.exists():
        raise FileNotFoundError(f"Warning: label file not found: {label_path}")


    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("imagesource") or line.startswith("gsd"):
                continue

            parts = line.split()

            if len(parts) < 9:
                print(f"Skip invalid line: {line}")
                continue

            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                print(f"Skip invalid coordinates: {line}")
                continue

            polygon = np.array(coords, dtype=np.float32).reshape(4, 2)

            class_name = parts[8]

            if len(parts) >= 10:
                difficult = parts[9]
            else:
                difficult = "0"

            objects.append(
                {
                    "polygon": polygon,
                    "class_name": class_name,
                    "difficult": difficult,
                }
            )

    return objects

def collect_images(img_dir: Path) -> List[Path]:
    image_paths = []

    for suffix in IMAGE_SUFFIXES:
        image_paths.extend(img_dir.glob(f"*{suffix}"))
        image_paths.extend(img_dir.glob(f"*{suffix.upper()}"))

    image_paths = sorted(set(image_paths))
    return image_paths

def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-4:
        return str(int(round(x)))
    return f"{x:.2f}"


def write_dota_label(label_path: Path|str, objects: List[Dict]) -> None:
    if isinstance(label_path, str):
        label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as f:
        f.write("imagesource:GoogleEarth\n")
        f.write("gsd:None\n")

        for obj in objects:
            polygon = obj["polygon"].reshape(-1)
            class_name = obj["class_name"]
            difficult = obj["difficult"]

            coord_str = " ".join(format_number(float(v)) for v in polygon)
            line = f"{coord_str} {class_name} {difficult}\n"
            f.write(line)


def parse_yolo_obb_label(
    label_path: Path | str,
    image_size: tuple[int, int],
    class_names: List[str] = DOTA_CLASSES,
    difficult: str = "0",
) -> List[Dict]:
    label_path = Path(label_path)
    width, height = image_size
    objects = []

    if not label_path.exists():
        return objects

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) not in (9, 10):
                print(f"Skip invalid YOLO-OBB line: {line.strip()}")
                continue

            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(class_names):
                raise ValueError(f"class_id={class_id} 超出 class_names 范围: {label_path}")

            coords = np.asarray([float(v) for v in parts[1:9]], dtype=np.float32).reshape(4, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            confidence = float(parts[9]) if len(parts) == 10 else None

            obj = {
                "polygon": coords,
                "class_name": class_names[class_id],
                "difficult": difficult,
            }
            if confidence is not None:
                obj["confidence"] = confidence
            objects.append(obj)

    return objects


def yolo_obb_label_to_dota_label(
    src_label_path: Path | str,
    image_path: Path | str,
    dst_label_path: Path | str,
    class_names: List[str] = DOTA_CLASSES,
    difficult: str = "0",
) -> None:
    with Image.open(image_path) as image:
        objects = parse_yolo_obb_label(
            src_label_path,
            image_size=image.size,
            class_names=class_names,
            difficult=difficult,
        )
    write_dota_label(dst_label_path, objects)


def yolo_obb_dir_to_dota_label_dir(
    image_dir: Path | str,
    yolo_label_dir: Path | str,
    dota_label_dir: Path | str,
    class_names: List[str] = DOTA_CLASSES,
    difficult: str = "0",
) -> None:
    image_dir = Path(image_dir)
    yolo_label_dir = Path(yolo_label_dir)
    dota_label_dir = Path(dota_label_dir)
    dota_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in collect_images(image_dir):
        src_label_path = yolo_label_dir / f"{image_path.stem}.txt"
        dst_label_path = dota_label_dir / f"{image_path.stem}.txt"
        yolo_obb_label_to_dota_label(
            src_label_path=src_label_path,
            image_path=image_path,
            dst_label_path=dst_label_path,
            class_names=class_names,
            difficult=difficult,
        )


def dota_objects_to_polygons(objects: List[Dict]) -> np.ndarray:
    return np.asarray([obj["polygon"] for obj in objects], dtype=np.float32).reshape(-1, 4, 2)


def update_dota_objects_polygons(objects: List[Dict], polygons: np.ndarray) -> List[Dict]:
    polygons = np.asarray(polygons, dtype=np.float32).reshape(-1, 4, 2)
    new_objects = []
    for obj, polygon in zip(objects, polygons):
        new_obj = dict(obj)
        new_obj["polygon"] = polygon
        new_objects.append(new_obj)
    return new_objects


def filter_small_dota_objects(objects: List[Dict], min_box_size=None) -> List[Dict]:
    if min_box_size is None:
        return objects

    filtered = []
    for obj in objects:
        polygon = np.asarray(obj["polygon"], dtype=np.float32).reshape(4, 2)
        box_w = polygon[:, 0].max() - polygon[:, 0].min()
        box_h = polygon[:, 1].max() - polygon[:, 1].min()
        if box_w >= min_box_size and box_h >= min_box_size:
            filtered.append(obj)
    return filtered
