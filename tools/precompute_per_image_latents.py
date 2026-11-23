
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from simplesr.data.latent_cache import (
    PER_IMAGE_SCHEMA, SCHEMA, cache_artifact_path, encoder_matches, file_digest,
    frozen_vae_signature, json_digest, load_cache_manifest, read_cached_pair,
    source_records,
)
from tools.precompute_frozen_latents import (
    CacheBuildImages, encode_pair, write_json,
)
from plfm_rssr.diffusers_vae import build_vae


def verify_files(root, manifest):
    for pair in manifest["artifacts"]:
        for artifact in pair.values():
            if file_digest(cache_artifact_path(root, artifact["file"])) != artifact["sha256"]:
                raise ValueError(f"Per-image cache file changed: {artifact['file']}")


def write_pair(root, record, latents):
    if (latents.dtype != np.float32 or latents.shape[1:] != (2, 3, 64, 64)
            or not np.isfinite(latents).all()):
        raise ValueError("Expected finite FP32 [modes,2,3,64,64] latents")
    artifacts = {}
    for index, key, source, folder in ((0, "hr", "gt", "hr_latent"),
                                       (1, "lr_nearest", "lq", "lr_nearest_latent")):
        relative = f"{folder}/{record[source]}.npy"
        path = cache_artifact_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to replace an existing latent: {path}")
        temporary = path.with_suffix(path.suffix + ".partial")
        expected = np.ascontiguousarray(latents[:, index])
        with temporary.open("xb") as stream:
            np.save(stream, expected, allow_pickle=False)
        saved = np.load(temporary, allow_pickle=False)
        if not np.array_equal(saved, expected) or saved.dtype != np.float32:
            raise ValueError(f"Saved latent differs: {temporary}")
        temporary.replace(path)
        artifacts[key] = dict(file=relative, bytes=path.stat().st_size, sha256=file_digest(path))
    checksums = [hashlib.sha256(row.tobytes()).hexdigest() for row in latents]
    return artifacts, checksums


def verify_online(root, manifest, dataset, model, vae_name):
    largest, checks = 0.0, 0
    device = next(model.parameters()).device
    for row in sorted({0, len(dataset.paths) // 2, len(dataset.paths) - 1}):
        for mode_index in range(len(dataset.modes)):
            sample = dataset[row * len(dataset.modes) + mode_index]
            online = encode_pair(model, vae_name, sample["gt"].unsqueeze(0).to(device),
                                 sample["lq"].unsqueeze(0).to(device))[0]
            saved = read_cached_pair(root, manifest, row, mode_index)
            largest = max(largest, float(np.abs(saved - online).max()))
            if not np.allclose(saved, online, rtol=1e-4, atol=1e-4):
                raise ValueError(f"Online/per-image mismatch: pair {row}, mode {mode_index}; "
                                 f"max absolute difference {np.abs(saved - online).max():.8g}")
            checks += 1
    return dict(checked_pairs_and_modes=checks, max_absolute_difference=largest,
                rtol=1e-4, atol=1e-4, encoding_batch_size=1)


def encoded_pairs(dataset, model, vae_name, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    device = next(model.parameters()).device
    pending, completed = [], 0
    for batch in loader:
        indices = batch["index"].tolist()
        if indices != list(range(completed, completed + len(indices))):
            raise ValueError("Precompute image order changed")
        encoded = encode_pair(model, vae_name, batch["gt"].to(device, non_blocking=True),
                              batch["lq"].to(device, non_blocking=True))
        for row in encoded:
            pending.append(row)
            completed += 1
            if len(pending) == len(dataset.modes):
                yield np.stack(pending)
                pending = []
    if pending or completed != len(dataset):
        raise ValueError("Incomplete image augmentation group")


def converted_pairs(source_root, source):
    values = np.load(source_root / "latents.npy", mmap_mode="r", allow_pickle=False)
    for row in range(len(source["samples"])):
        yield np.stack([read_cached_pair(source_root, source, row, mode, values=values)
                        for mode in range(len(source["augmentation_modes"]))])


def verify_first_batch(dataset, model, vae_name, batch_size):
    samples = [dataset[index] for index in range(min(batch_size, len(dataset)))]
    device = next(model.parameters()).device
    batched = encode_pair(model, vae_name, torch.stack([s["gt"] for s in samples]).to(device),
                          torch.stack([s["lq"] for s in samples]).to(device))
    for index in sorted({0, len(samples) // 2, len(samples) - 1}):
        sample = samples[index]
        single = encode_pair(model, vae_name, sample["gt"].unsqueeze(0).to(device),
                             sample["lq"].unsqueeze(0).to(device))[0]
        if not np.allclose(batched[index], single, rtol=1e-4, atol=1e-4):
            raise ValueError("Encoding batch differs from single-image reference before cache creation; "
                             f"max absolute difference {np.abs(batched[index] - single).max():.8g}")


def build_split(kwargs, root, vae_name, *, model=None, identity=None,
                source_root=None, batch_size=16, workers=2):
    root = Path(root)
    dataset = CacheBuildImages(kwargs)
    print(f"Checking {len(dataset.paths)} source image pairs for {root.name}", flush=True)
    sources = source_records(dataset)
    if not sources:
        raise ValueError("Cannot create an empty latent cache")
    source = None
    if source_root is not None:
        source_root = Path(source_root)
        source = load_cache_manifest(source_root)
        if (source["schema"] != SCHEMA or source["vae"] != vae_name
                or source["samples"] != sources or source["augmentation_modes"] != dataset.modes):
            raise ValueError("Source legacy cache does not match the configured images/protocol")
        if file_digest(source_root / "latents.npy") != source["artifact"]["sha256"]:
            raise ValueError("Source legacy cache checksum changed")
        identity = source["encoder"]
    elif model is None or identity is None:
        raise ValueError("Fresh encoding requires the frozen VAE and its identity")
    if (root / "manifest.json").exists():
        existing = load_cache_manifest(root)
        if (existing["schema"] != PER_IMAGE_SCHEMA or existing["samples"] != sources
                or existing["augmentation_modes"] != dataset.modes or not encoder_matches(existing["encoder"], identity)
                or existing["vae"] != vae_name):
            raise ValueError("Existing per-image cache belongs to a different source/protocol")
        verify_files(root, existing)
        print(f"Reusing verified complete split: {root}", flush=True)
        return existing
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Incomplete output exists; inspect it before choosing a fresh root: {root}")
    if source is None:
        verify_first_batch(dataset, model, vae_name, batch_size)
    root.mkdir(parents=True, exist_ok=True)
    required = len(sources) * len(dataset.modes) * 2 * 3 * 64 * 64 * 4
    if shutil.disk_usage(root).free < required + 512 * 1024**2:
        raise OSError("Insufficient disk space for the complete per-image cache")
    (root / "build.lock").mkdir()
    started = time.perf_counter()
    artifacts, checksums = [], []
    iterator = (converted_pairs(source_root, source) if source is not None else
                encoded_pairs(dataset, model, vae_name, batch_size, workers))
    for row, latents in enumerate(iterator):
        pair, hashes = write_pair(root, sources[row], latents)
        artifacts.append(pair)
        checksums.extend(hashes)
        completed = row + 1
        if completed % 128 == 0 or completed == len(sources):
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress = dict(status="building", completed_pairs=completed, total_pairs=len(sources),
                            seconds=elapsed, pairs_per_second=completed / elapsed)
            write_json(root / "progress.json", progress)
            print(f"{root.name}: {completed}/{len(sources)} pairs, "
                  f"{completed / elapsed:.2f} pairs/s", flush=True)
    if len(artifacts) != len(sources) or source_records(dataset) != sources:
        raise ValueError("Source image set changed or output count differs")
    manifest = dict(schema=PER_IMAGE_SCHEMA, status="complete", storage="per_image_npy",
                    vae=vae_name, encoder=identity, dtype="float32", shape=[3, 64, 64],
                    scale=4, gt_size=256, augmentation_modes=dataset.modes, samples=sources,
                    artifacts=artifacts, row_sha256=checksums)
    if source is not None:
        if checksums != source["row_sha256"]:
            raise ValueError("Repacked cache differs from the accepted source cache")
        manifest["encoding_origin"] = dict(method="lossless_repack", source_signature=source["signature"],
                                           source_artifact_sha256=source["artifact"]["sha256"],
                                           all_pairs_bitwise_equal=True)
        manifest["online_parity"] = source["online_parity"]
    else:
        manifest["encoding_origin"] = dict(method="encode_images", build_batch_size=batch_size,
                                           input_layout="contiguous_nchw", cudnn_benchmark=False,
                                           cudnn_deterministic=True)
        manifest["online_parity"] = verify_online(root, manifest, dataset, model, vae_name)
    manifest.update(build_seconds=time.perf_counter() - started,
                    created_at=datetime.now(timezone.utc).isoformat(), python=sys.executable)
    manifest["signature"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    load_cache_manifest(root)
    write_json(root / "progress.json", dict(status="complete", completed_pairs=len(sources),
                                            file_count=len(sources) * 2, signature=manifest["signature"]))
    print(f"Complete {root}: {len(sources) * 2} files; {manifest['encoding_origin']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Precompute or convert a per-image VQ-f4 latent cache.')
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vq-model", type=Path, help="Local VQ-f4 model folder")
    parser.add_argument("--source-cache-root", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Positive batch size and nonnegative workers required")
    print(f"Python: {sys.executable}", flush=True)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    vae_name = config["vae"]
    if vae_name != "ldm-vq-f4-sr":
        parser.error("This protocol supports the frozen ldm-vq-f4-sr VAE")
    model, identity = None, None
    if args.source_cache_root is None:
        if not torch.cuda.is_available():
            parser.error("Fresh encoding requires an explicitly selected idle GPU")
        torch.backends.cudnn.benchmark = True
        model = build_vae(vae_name, source=args.vq_model or config.get("vq_model")).to("cuda").eval().requires_grad_(False)
        identity = frozen_vae_signature(model, vae_name)
    for phase in args.splits:
        build_split(config["datasets"][phase]["kwargs"], args.output_root / phase, vae_name,
                    model=model, identity=identity, batch_size=args.batch_size, workers=args.workers,
                    source_root=args.source_cache_root / phase if args.source_cache_root else None)


if __name__ == "__main__":
    main()
