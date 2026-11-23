from pathlib import Path

import torch
from diffusers import VQModel
from torch import nn


class VQF4(nn.Module):
    def __init__(self, source=None):
        super().__init__()
        options = dict(
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        if source is None:
            self.model = VQModel.from_pretrained(
                "CompVis/ldm-super-resolution-4x-openimages",
                subfolder="vqvae",
                revision="0b55ddf931a8e3a1b426b3a50ddcf325ff84f668",
                **options,
            )
        else:
            path = Path(source).expanduser()
            if (path / "vqvae").is_dir():
                path = path / "vqvae"
            for filename in ("config.json", "diffusion_pytorch_model.fp16.safetensors"):
                if not (path / filename).is_file():
                    raise FileNotFoundError(path / filename)
            self.model = VQModel.from_pretrained(path, local_files_only=True, **options)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, image):
        return self.model.encode(image.mul(2).sub(1)).latents

    def decode(self, latent):
        return self.model.decode(latent).sample.add(1).div(2)