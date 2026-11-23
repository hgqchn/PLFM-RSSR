_SPEC = {
    "family": "vq",
    "model_id": "CompVis/ldm-super-resolution-4x-openimages",
    "subfolder": "vqvae",
    "variant": "fp16",
    "use_safetensors": True,
    "downsample_factor": 4,
    "scale_factor": 1.0,
    "latent_channels": 3,
}


def get_vae_spec(vae_name):
    if vae_name != "ldm-vq-f4-sr":
        raise KeyError(f"Unknown VAE: {vae_name}")
    return dict(_SPEC)
