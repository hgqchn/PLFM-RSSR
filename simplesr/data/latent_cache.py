
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


SCHEMA = 1
PER_IMAGE_SCHEMA = 2
PROTOCOL = ("rgb_float32_0_1;augment_image;nearest_x4;image*2-1;posterior_mode;fp32;"
            "encoder_cudnn_tf32_false;encoder_matmul_tf32_false")


def data_transform_signature():
    from simplesr.utils.img_utils import img_augment, img_array_to_tensor, img_from_bytes
    source = "\n".join(inspect.getsource(function) for function in
                       (img_augment, img_array_to_tensor, img_from_bytes))
    return hashlib.sha256(source.encode()).hexdigest()


LEGACY_TRANSFORM_SHA256 = "34effa07faa3617667fe6846b4ddc949f225c6374c6fddd10765c7ebd373950f"
LEGACY_VAE_SOURCE_SHA256 = "bcfeb6b0113fec2e8753266603b9ca9d3f16f9aa597963ded861137b5883f47d"
RELEASE_TRANSFORM_SHA256 = "697cd77f1ad15091b96af28e94b3797a1a1c26783a473095d0ec0ff88a2f1cc0"
RELEASE_VAE_SOURCE_SHA256 = "411998ee5524647c0a76e77af921ff8d478a05f69b10ba95e05f17fa01570d53"


def data_transform_matches(recorded):
    current = data_transform_signature()
    return (recorded == current or
            (recorded == LEGACY_TRANSFORM_SHA256 and current == RELEASE_TRANSFORM_SHA256))


def encoder_matches(recorded, current):
    if recorded == current:
        return True
    if (current.get("data_transform_sha256") != RELEASE_TRANSFORM_SHA256
            or current.get("encode_source_sha256") != RELEASE_VAE_SOURCE_SHA256):
        return False
    compatible_legacy = dict(current, data_transform_sha256=LEGACY_TRANSFORM_SHA256,
                             encode_source_sha256=LEGACY_VAE_SOURCE_SHA256)
    return recorded == compatible_legacy


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_vae_signature(model, vae_name):
    import diffusers
    import torch
    from plfm_rssr import diffusers_vae
    from plfm_rssr.diffusers_vae_registry import get_vae_spec

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("A latent cache requires an eval-mode, frozen VAE")
    spec = dict(get_vae_spec(vae_name))
    spec.pop("local_path", None)
    if spec.get("sample_posterior", False):
        raise ValueError("Cannot cache a sampled VAE posterior without changing training")
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("Latent caches require FP32 frozen VAE weights")
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())

    config = {key: value for key, value in dict(model.config).items() if not key.startswith("_")}
    source = Path(diffusers_vae.__file__).read_text(encoding="utf-8")
    return dict(vae=vae_name, weights_sha256=digest.hexdigest(), config_sha256=json_digest(config),
                spec=spec, protocol=PROTOCOL, model_class=type(model).__qualname__,
                encoder_precision=dict(dtype="float32", cudnn_allow_tf32=False,
                                       matmul_allow_tf32=False),
                data_transform_sha256=data_transform_signature(),
                encode_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                torch=str(torch.__version__), diffusers=diffusers.__version__)


def source_records(dataset):
    records = []
    for pair in dataset.paths:
        record = {}
        for key in ("gt", "lq"):
            path = Path(pair[f"{key}_path"])
            root = Path(getattr(dataset, f"dataroot_{key}"))
            record[key] = path.relative_to(root).as_posix()
            record[f"{key}_sha256"] = file_digest(path)
        records.append(record)
    return records


def load_cache_manifest(root):
    root = Path(root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    if (manifest.get("schema") not in (SCHEMA, PER_IMAGE_SCHEMA) or manifest.get("status") != "complete"
            or manifest.get("signature") != json_digest(unsigned)
            or manifest.get("dtype") != "float32" or manifest.get("shape") != [3, 64, 64]
            or manifest.get("augmentation_modes") not in ([0], list(range(8)))
            or manifest.get("encoder", {}).get("protocol") != PROTOCOL
            or not data_transform_matches(manifest.get("encoder", {}).get("data_transform_sha256"))
            or manifest.get("scale") != 4 or manifest.get("gt_size") != 256):
        raise ValueError(f"Invalid or incomplete latent cache manifest: {path}")
    count, modes = len(manifest["samples"]), len(manifest["augmentation_modes"])
    if not count or len(manifest["row_sha256"]) != count * modes:
        raise ValueError("Latent cache sample/checksum count differs")
    if manifest["schema"] == SCHEMA:
        artifact = manifest["artifact"]
        if artifact.get("file") != "latents.npy":
            raise ValueError("Unexpected latent cache artifact")
        cache_path = root / "latents.npy"
        if cache_path.stat().st_size != artifact["bytes"]:
            raise ValueError("Latent cache artifact is truncated or changed")
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (count, modes, 2, 3, 64, 64) or values.dtype != np.float32:
            raise ValueError("Latent cache array shape/dtype differs")
    else:
        artifacts = manifest.get("artifacts", [])
        if manifest.get("storage") != "per_image_npy" or len(artifacts) != count:
            raise ValueError("Invalid per-image latent cache layout")
        seen = set()
        for sample, pair in zip(manifest["samples"], artifacts):
            for key, source, folder in (("hr", "gt", "hr_latent"),
                                        ("lr_nearest", "lq", "lr_nearest_latent")):
                artifact = pair[key]
                relative = artifact["file"]
                if relative != f"{folder}/{sample[source]}.npy" or relative in seen:
                    raise ValueError("Latent cache filename differs from its source image")
                seen.add(relative)
                cache_path = cache_artifact_path(root, relative)
                if not cache_path.is_file() or cache_path.stat().st_size != artifact["bytes"]:
                    raise ValueError(f"Latent cache artifact is missing or changed: {relative}")
    return manifest


def cache_artifact_path(root, relative):
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError(f"Invalid latent cache artifact path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Latent cache artifact escapes its root: {relative}")
    return path


def read_cached_pair(root, manifest, row, mode_index, *, values=None):
    count, modes = len(manifest["samples"]), len(manifest["augmentation_modes"])
    if not 0 <= row < count or not 0 <= mode_index < modes:
        raise IndexError("Latent cache sample or augmentation index is out of range")
    if manifest["schema"] == SCHEMA:
        if values is None:
            values = np.load(Path(root) / "latents.npy", mmap_mode="r", allow_pickle=False)
        latent = np.array(values[row, mode_index], copy=True)
    elif manifest["schema"] == PER_IMAGE_SCHEMA:
        pair = manifest["artifacts"][row]
        parts = []
        for key in ("hr", "lr_nearest"):
            path = cache_artifact_path(root, pair[key]["file"])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (modes, 3, 64, 64) or array.dtype != np.float32:
                raise ValueError(f"Latent cache array shape/dtype differs: {path}")
            parts.append(np.array(array[mode_index], copy=True))
        latent = np.stack(parts)
    else:
        raise ValueError("Unsupported latent cache schema")
    checksum = manifest["row_sha256"][row * modes + mode_index]
    if (latent.shape != (2, 3, 64, 64) or latent.dtype != np.float32
            or not np.isfinite(latent).all()
            or hashlib.sha256(latent.tobytes()).hexdigest() != checksum):
        raise ValueError(f"Latent cache row corrupted: {row}, mode {mode_index}")
    return latent


def validate_dataset_options(dataset):
    if (dataset.scale != 4 or dataset.phase not in ("train", "val", "test")
            or dataset.color is not None or dataset.mean is not None or dataset.std is not None
            or dataset.io_backend_opt.get("type", "disk") != "disk"
            or (dataset.phase == "train" and dataset.gt_size != 256)):
        raise ValueError("Cache supports RGB disk pairs, x4, full 256px patches without normalization")


def validate_full_pair(shapes):
    if tuple(shapes[0]) != (256, 256, 3) or tuple(shapes[1]) != (64, 64, 3):
        raise ValueError("Cache requires full HR256/LR64 pairs; random-crop caches are unsupported")
