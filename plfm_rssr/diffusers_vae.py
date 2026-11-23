import torch

from plfm_rssr.codec import VQF4

from .diffusers_vae_registry import get_vae_spec


def build_vae(vae_name, source=None):
    get_vae_spec(vae_name)
    return VQF4(source).model


@torch.no_grad()
def encode_vae(model, image, vae_name, *, sample_posterior=False):
    get_vae_spec(vae_name)
    if sample_posterior:
        raise ValueError("The frozen VQ-f4 encoder is deterministic.")
    return model.encode(image).latents


def decode_vae(model, latent, vae_name):
    get_vae_spec(vae_name)
    return model.decode(latent).sample
