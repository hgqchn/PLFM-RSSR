
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from simplesr.data.dataset import PairedImageDataset
from simplesr.utils.img_utils import img_augment
from plfm_rssr.diffusers_vae import build_vae, encode_vae
from simplesr.data.latent_cache import (SCHEMA, encoder_matches, file_digest, frozen_vae_signature, json_digest,
                                       load_cache_manifest, source_records,
                                       validate_dataset_options, validate_full_pair)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class CacheBuildImages(PairedImageDataset):
    def __init__(self, kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("latent_cache_root", None)
        kwargs.pop("vae", None)
        super().__init__(**kwargs)
        validate_dataset_options(self)
        self.modes = list(range(8)) if self.phase == "train" and self.use_augment else [0]

        self.phase, self.use_augment = "val", False

    def __len__(self):
        return len(self.paths) * len(self.modes)

    def _finish_sample(self, sample, index, augmentation_mode, original_shapes):
        validate_full_pair(original_shapes)
        return sample

    def __getitem__(self, index):
        pair_index, mode_index = divmod(index, len(self.modes))
        sample = super().__getitem__(pair_index)
        for key in ("gt", "lq"):
            image = sample[key].permute(1, 2, 0).numpy()
            sample[key] = torch.from_numpy(img_augment(image, self.modes[mode_index])).permute(2, 0, 1)
        return {"gt": sample["gt"], "lq": sample["lq"], "index": index}


@torch.no_grad()
def encode_pair(model, vae_name, gt, lq):


    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        gt, lq = gt.contiguous(), lq.contiguous()
        nearest = F.interpolate(lq, scale_factor=4, mode="nearest")
        hr_latent = encode_vae(model, gt.mul(2).sub(1), vae_name, sample_posterior=False)
        lr_latent = encode_vae(model, nearest.mul(2).sub(1), vae_name, sample_posterior=False)
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
    if hr_latent.dtype != torch.float32 or lr_latent.dtype != torch.float32:
        raise ValueError("Precompute must retain FP32 encoding precision")
    if hr_latent.shape[1:] != (3, 64, 64) or hr_latent.shape != lr_latent.shape:
        raise ValueError("Expected the configured VQ-f4 3x64x64 latent")
    return torch.stack((hr_latent, lr_latent), dim=1).cpu().numpy()


def verify_encoding(root, dataset, model, vae_name):
    values = np.load(root / "latents.npy", mmap_mode="r", allow_pickle=False)
    device = next(model.parameters()).device
    count = len(dataset.paths)
    largest = 0.0
    checks = 0
    for pair_index in sorted({0, count // 2, count - 1}):
        for mode_index in range(len(dataset.modes)):
            sample = dataset[pair_index * len(dataset.modes) + mode_index]
            actual = encode_pair(model, vae_name, sample["gt"].unsqueeze(0).to(device),
                                 sample["lq"].unsqueeze(0).to(device))[0]
            expected = np.array(values[pair_index, mode_index], copy=True)
            largest = max(largest, float(np.abs(expected - actual).max()))
            if not np.allclose(expected, actual, rtol=1e-4, atol=1e-4):
                raise ValueError(f"Online/cache encoding mismatch at pair {pair_index}, mode {mode_index}")
            checks += 1
    return dict(checked_pairs_and_modes=checks, max_absolute_difference=largest,
                rtol=1e-4, atol=1e-4, encoding_batch_size=1)


def build_split(kwargs, root, model, vae_name, identity, batch_size, workers):
    dataset = CacheBuildImages(kwargs)
    print(f"Hashing {len(dataset.paths)} image pairs for {root.name}", flush=True)
    sources = source_records(dataset)
    if (root / "manifest.json").exists():
        existing = load_cache_manifest(root)
        if (existing["samples"] != sources or not encoder_matches(existing["encoder"], identity)
                or existing["augmentation_modes"] != dataset.modes):
            raise ValueError(f"Existing cache belongs to a different source/protocol: {root}")
        if file_digest(root / "latents.npy") != existing["artifact"]["sha256"]:
            raise ValueError("Existing cache artifact hash changed")
        print(f"Reusing verified complete cache: {root}", flush=True)
        return existing
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Partial cache exists; inspect it before building into a fresh root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    (root / "build.lock").mkdir()
    started = time.monotonic()
    count, modes = len(dataset.paths), len(dataset.modes)
    if not count:
        raise ValueError("Cannot create an empty latent cache")
    shape = (count, modes, 2, 3, 64, 64)
    temporary = root / "latents.npy.partial"
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    flat = values.reshape(-1, 2, 3, 64, 64)
    checksums = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    device = next(model.parameters()).device
    completed = 0
    for batch in loader:
        indices = batch["index"].tolist()
        if indices != list(range(completed, completed + len(indices))):
            raise ValueError("Precompute data order changed")
        encoded = encode_pair(model, vae_name, batch["gt"].to(device, non_blocking=True),
                              batch["lq"].to(device, non_blocking=True))
        if not np.isfinite(encoded).all():
            raise FloatingPointError("Frozen VAE returned non-finite latents")
        flat[completed:completed + len(indices)] = encoded
        checksums.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in encoded)
        completed += len(indices)
        if completed % (batch_size * 100) == 0 or completed == len(dataset):
            elapsed = time.monotonic() - started
            progress = dict(status="building", completed=completed, total=len(dataset),
                            seconds=elapsed, rows_per_second=completed / elapsed,
                            updated_at=datetime.now(timezone.utc).isoformat())
            write_json(root / "progress.json", progress)
            print(f"{root.name}: {completed}/{len(dataset)} rows, {completed / elapsed:.2f} rows/s", flush=True)
    values.flush()
    del flat, values
    temporary.replace(root / "latents.npy")
    parity = verify_encoding(root, dataset, model, vae_name)
    if source_records(dataset) != sources:
        raise ValueError("Source images changed while precomputing")
    artifact = root / "latents.npy"
    manifest = dict(schema=SCHEMA, status="complete", vae=vae_name, encoder=identity,
                    dtype="float32", shape=[3, 64, 64], scale=4, gt_size=256,
                    augmentation_modes=dataset.modes, samples=sources, row_sha256=checksums,
                    artifact=dict(file=artifact.name, bytes=artifact.stat().st_size,
                                  sha256=file_digest(artifact)), online_parity=parity,
                    build_batch_size=batch_size, build_seconds=time.monotonic() - started,
                    created_at=datetime.now(timezone.utc).isoformat())
    manifest["signature"] = json_digest(manifest)
    write_json(root / "manifest.json", manifest)
    load_cache_manifest(root)
    write_json(root / "progress.json", dict(status="complete", completed=completed,
                                            signature=manifest["signature"]))
    print(f"Complete {root}: {completed} rows; parity {parity}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Precompute augmentation-matched VQ-f4 latents into an aggregate cache.')
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vq-model", type=Path, help="Local VQ-f4 model folder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Positive batch size and nonnegative workers required")
    print(f"Python: {sys.executable}", flush=True)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    vae_name = config["vae"]
    if vae_name != "ldm-vq-f4-sr":
        parser.error("This cache protocol currently supports the frozen ldm-vq-f4-sr VAE")
    if not torch.cuda.is_available():
        parser.error("Use a verified idle GPU through CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.benchmark = True
    model = build_vae(vae_name, source=args.vq_model or config.get("vq_model")).to("cuda").eval().requires_grad_(False)
    identity = frozen_vae_signature(model, vae_name)
    for phase in ("train", "val"):
        kwargs = config["datasets"][phase]["kwargs"]
        build_split(kwargs, args.output_root / phase, model, vae_name, identity,
                    args.batch_size, args.workers)


if __name__ == "__main__":
    main()
