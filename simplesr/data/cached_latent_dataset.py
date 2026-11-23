
from pathlib import Path

import numpy as np
import torch

from simplesr.data.dataset import PairedImageDataset
from .latent_cache import (SCHEMA, load_cache_manifest, read_cached_pair, source_records,
                           validate_dataset_options, validate_full_pair)


class CachedLatentPairedImageDataset(PairedImageDataset):
    def __init__(self, latent_cache_root, vae="ldm-vq-f4-sr", **kwargs):
        super().__init__(**kwargs)
        self.latent_cache_root = Path(latent_cache_root)
        self.vae = vae
        validate_dataset_options(self)
        self.cache_manifest = load_cache_manifest(self.latent_cache_root)
        modes = list(range(8)) if self.phase == "train" and self.use_augment else [0]
        if (self.cache_manifest["vae"] != self.vae
                or self.cache_manifest["augmentation_modes"] != modes):
            raise ValueError("Latent cache VAE or augmentation protocol differs from the dataset")

        records = source_records(self)
        indexed = {(item["gt"], item["lq"]): (index, item)
                   for index, item in enumerate(self.cache_manifest["samples"])}
        if len(indexed) != len(records):
            raise ValueError("Latent cache image set differs")
        self._cache_indices = []
        for record in records:
            found = indexed.get((record["gt"], record["lq"]))
            if found is None or found[1] != record:
                raise ValueError(f"Latent cache source image changed: {record['gt']}")
            self._cache_indices.append(found[0])
        self._latent_values = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_latent_values"] = None
        return state

    def _finish_sample(self, sample, index, augmentation_mode, original_shapes):
        validate_full_pair(original_shapes)
        if self.cache_manifest["schema"] == SCHEMA and self._latent_values is None:
            self._latent_values = np.load(self.latent_cache_root / "latents.npy",
                                          mmap_mode="r", allow_pickle=False)
        row = self._cache_indices[index]
        mode_index = self.cache_manifest["augmentation_modes"].index(augmentation_mode)
        latent = read_cached_pair(self.latent_cache_root, self.cache_manifest, row,
                                  mode_index, values=self._latent_values)
        sample.update(z_hr=torch.from_numpy(latent[0]), z_lr_nearest=torch.from_numpy(latent[1]),
                      latent_cache_signature=self.cache_manifest["signature"])
        return sample
