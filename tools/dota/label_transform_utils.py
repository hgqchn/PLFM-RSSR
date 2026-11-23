
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from tools.dota.utils import (
    dota_objects_to_polygons,
    filter_small_dota_objects,
    parse_dota_label,
    update_dota_objects_polygons,
    write_dota_label,
)
from simplesr.utils.img_utils import (
    get_img_augment_affine_matrix,
    get_img_rotate_affine_matrix,
    get_scale_affine_matrix,
    transform_polygons_by_affine,
)


def transform_dota_polygon_by_affine(polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    return transform_polygons_by_affine(polygon, matrix)


def transform_dota_labels_by_affine(labels: List[Dict], matrix: np.ndarray) -> List[Dict]:
    polygons = dota_objects_to_polygons(labels)
    polygons = transform_polygons_by_affine(polygons, matrix)
    return update_dota_objects_polygons(labels, polygons)


def img_augment_DOTA_label(labels: List[Dict], mode=0, img_shape=None) -> List[Dict]:
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    return transform_dota_labels_by_affine(labels, matrix)


def img_rotate_DOTA_label(labels: List[Dict], img_shape, angle, center=None, scale=1.0) -> List[Dict]:
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    return transform_dota_labels_by_affine(labels, matrix)


def transform_dota_objects(objects, matrix, min_box_size=None):
    objects = transform_dota_labels_by_affine(objects, matrix)
    return filter_small_dota_objects(objects, min_box_size=min_box_size)


def scale_dota_objects(objects, scale_x, scale_y, min_box_size=None):
    matrix = get_scale_affine_matrix(scale_x, scale_y)
    return transform_dota_objects(objects, matrix, min_box_size=min_box_size)


def augment_dota_objects(objects, img_shape, mode):
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    return transform_dota_objects(objects, matrix)


def rotate_dota_objects(objects, img_shape, angle, center=None, scale=1.0):
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    return transform_dota_objects(objects, matrix)


def transform_dota_label_file(src_label_path, dst_label_path, matrix, min_box_size=None):
    src_label_path = Path(src_label_path)
    dst_label_path = Path(dst_label_path)

    if not src_label_path.exists():
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.write_text("", encoding="utf-8")
        return

    objects = parse_dota_label(src_label_path)
    transformed_objects = transform_dota_objects(objects, matrix, min_box_size=min_box_size)
    write_dota_label(dst_label_path, transformed_objects)


def scale_dota_label_file(src_label_path, dst_label_path, scale_x, scale_y, min_box_size=None):
    matrix = get_scale_affine_matrix(scale_x, scale_y)
    transform_dota_label_file(
        src_label_path,
        dst_label_path,
        matrix,
        min_box_size=min_box_size,
    )


def augment_dota_label_file(src_label_path, dst_label_path, img_shape, mode):
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)


def rotate_dota_label_file(src_label_path, dst_label_path, img_shape, angle, center=None, scale=1.0):
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)


def scale_dota_label_folder(
    src_label_dir: str | Path,
    dst_label_dir: str | Path,
    scale: int = 4,
    recursive: bool = True,
    min_box_size: float | None = None,
) -> None:
    src_label_dir, dst_label_dir = Path(src_label_dir), Path(dst_label_dir)
    label_paths = sorted(src_label_dir.rglob("*.txt") if recursive else src_label_dir.glob("*.txt"))
    for src_label_path in tqdm(label_paths, desc="Scaling DOTA labels"):
        scale_dota_label_file(
            src_label_path,
            dst_label_dir / src_label_path.relative_to(src_label_dir),
            scale_x=1.0 / scale,
            scale_y=1.0 / scale,
            min_box_size=min_box_size,
        )
