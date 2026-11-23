import hashlib

import torch
from torch.nn import functional as F


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_weights(module, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(f"Expected a model state dictionary in {path}")
    module.load_state_dict(state, strict=True)


def image_seed(name, seed):
    key = name.encode("utf-8") if isinstance(name, str) else name
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, "little") + seed) % (2**63 - 1)


def _seeded_noise(like, seed):
    generator = torch.Generator(device=like.device).manual_seed(seed)
    return torch.randn(like.shape, generator=generator, device=like.device, dtype=like.dtype)


@torch.inference_mode()
def super_resolve(model, codec, lr, steps=10, seed=12345, *, encoded_lr=None):
    if steps < 1:
        raise ValueError("The Euler step count must be positive.")
    height, width = lr.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    if pad_h or pad_w:
        if encoded_lr is not None:
            raise ValueError("Cached LR latents require image dimensions divisible by 8.")
        pad_mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        lr = F.pad(lr, (0, pad_w, 0, pad_h), mode=pad_mode)

    anchor, features = model.anchor(lr)
    condition = model.velocity.prepare_condition(features)
    if encoded_lr is None:
        encoded_lr = codec.encode(F.interpolate(lr, scale_factor=4, mode="nearest"))
    if encoded_lr.shape != anchor.shape:
        raise ValueError("Encoded LR and LPP latents must have the same shape.")

    if isinstance(seed, int):
        noise = _seeded_noise(anchor, seed)
    else:
        noise = torch.cat([
            _seeded_noise(image.unsqueeze(0), image_seed_value)
            for image, image_seed_value in zip(anchor, seed, strict=True)
        ])
    latent = anchor + noise
    times = torch.linspace(1, 0, steps + 1, device=anchor.device, dtype=anchor.dtype)
    for time, next_time in zip(times[:-1], times[1:], strict=True):
        labels = time.float().expand(latent.shape[0]) * 1000.0
        velocity = model.velocity(latent, labels, lq=encoded_lr, cond_features=condition)
        latent = latent + (next_time - time) * velocity

    return codec.decode(latent)[..., :4 * height, :4 * width].clamp(0, 1)
