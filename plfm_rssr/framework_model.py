from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn import functional as F

from simplesr.data.latent_cache import encoder_matches, frozen_vae_signature, load_cache_manifest
from simplesr.models.base_sr_model import SRModel

from .codec import VQF4
from .runtime import image_seed, super_resolve


class _VQSRModel(SRModel):
    def __init__(self, opt):
        if opt["scale"] != 4 or opt["vae"] != "ldm-vq-f4-sr":
            raise ValueError("PLFM-RSSR uses the frozen ldm-vq-f4-sr autoencoder at x4.")
        super().__init__(opt)
        self.codec = VQF4(opt.get("vq_model")).to(self.device)

    def _init_losses(self):
        pass

    def _validation_network(self):
        network = self.net_g if self.net_g_ema is None else self.get_ema_module(self.net_g_ema)
        return self.get_bare_model(network)

    def _train_step(self, loss, logs, current_iter):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PLFM-RSSR training loss.")
        def check_gradients(*_):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), float("inf"), error_if_nonfinite=True)
        hook = self.optimizer_g.register_step_pre_hook(check_gradients)
        try:
            self.backward_and_step(loss, current_iter)
        finally:
            hook.remove()
        self.log_dict = self.reduce_loss_dict(logs, current_iter)


class LPPModel(_VQSRModel):
    def optimize_parameters(self, current_iter):
        target = self.codec.encode(self.gt)
        predicted, _ = self.net_g(self.lq)
        self.output = self.codec.decode(predicted)
        latent_loss = F.mse_loss(predicted, target)
        image_loss = F.l1_loss(self.output, self.gt)
        settings = self.opt["train"]
        loss = settings["lambda_latent"] * latent_loss + settings["lambda_image"] * image_loss
        self._train_step(
            loss, OrderedDict(l_latent=latent_loss, l_image=image_loss, l_total=loss), current_iter,
        )

    @torch.no_grad()
    def inference(self):
        network = self._validation_network()
        was_training = network.training
        network.eval()
        try:
            predicted, _ = network(self.lq)
            self.output = self.codec.decode(predicted).clamp_(0, 1)
        finally:
            network.train(was_training)


class PLFMModel(_VQSRModel):
    def __init__(self, opt):
        if opt.get("distributed"):
            raise ValueError("Use a single device for PLFM-RSSR velocity training and evaluation.")
        settings = opt["flow_matching"]
        self.seed = int(settings["validation_seed"])
        self.steps = int(settings["inference"]["sample_steps"])
        if self.steps < 1 or settings["inference"]["ode_method"] != "euler":
            raise ValueError("PLFM-RSSR uses at least one Euler update.")
        if not opt["is_train"] and not opt["path"].get("pretrain_network_g"):
            raise ValueError("Provide the full three-way checkpoint in path.pretrain_network_g.")
        super().__init__(opt)
        self.cache_signatures = self._validate_caches()

    def init_training_settings(self):
        paths = self.opt["path"]
        if self.load_path is None and not paths.get("resume_ckpt"):
            lpp_weights = paths.get("lpp_weights")
            if not lpp_weights:
                raise ValueError("Provide the LPP checkpoint in path.lpp_weights before training the CVN.")
            self.load_network(self.get_bare_model(self.net_g).anchor, lpp_weights, strict=True)
        super().init_training_settings()

    def _validate_caches(self):
        roots = [dataset.get("kwargs", {}).get("latent_cache_root") for dataset in self.opt["datasets"].values()]
        if not any(roots):
            return set()
        if not all(roots):
            raise ValueError("Configure a latent cache for every dataset or use online encoding for all.")
        encoder = frozen_vae_signature(self.codec.model, self.opt["vae"])
        signatures = set()
        for root in set(roots):
            manifest = load_cache_manifest(root)
            if manifest["vae"] != self.opt["vae"] or not encoder_matches(manifest["encoder"], encoder):
                raise ValueError(f"Latent cache encoder differs from the loaded frozen VQ-f4: {root}")
            if manifest["shape"] != [3, 64, 64] or manifest["dtype"] != "float32":
                raise ValueError(f"Expected FP32 3x64x64 latents in {root}")
            signatures.add(manifest["signature"])
        return signatures

    def feed_data(self, data):
        self.cached_hr = self.cached_lr = None
        super().feed_data(data)
        self.lq_paths = data.get("lq_path")
        if isinstance(self.lq_paths, str):
            self.lq_paths = [self.lq_paths]
        fields = {"z_hr", "z_lr_nearest", "latent_cache_signature"}
        if not self.cache_signatures:
            if fields.intersection(data):
                raise ValueError("Received cached latents without a configured, verified cache.")
            return
        if not fields.issubset(data):
            raise ValueError("A cached dataset must return its latent pair and cache signature.")
        signatures = data["latent_cache_signature"]
        if isinstance(signatures, str):
            signatures = [signatures]
        batch_size = self.lq.shape[0]
        if (not isinstance(signatures, (list, tuple)) or len(signatures) != batch_size
                or any(signature not in self.cache_signatures for signature in signatures)):
            raise ValueError("Batch latent cache signature is not verified.")
        expected_shape = (batch_size, 3, 64, 64)
        if tuple(self.lq.shape) != expected_shape or (
            self.gt is not None and tuple(self.gt.shape) != (batch_size, 3, 256, 256)
        ):
            raise ValueError("Cached VQ-f4 requires aligned 64x64 LR and 256x256 HR RGB patches.")
        cached = []
        for key in ("z_hr", "z_lr_nearest"):
            value = data[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != expected_shape or value.requires_grad
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid cached FP32 latent batch: {key}")
            cached.append(value.to(self.device, non_blocking=True))
        self.cached_hr, self.cached_lr = cached

    def optimize_parameters(self, current_iter):
        with torch.no_grad():
            target = self.cached_hr
            encoded_lr = self.cached_lr
            if target is None:
                target = self.codec.encode(self.gt)
                encoded_lr = self.codec.encode(F.interpolate(self.lq, scale_factor=4, mode="nearest"))
            anchor, features = self.get_bare_model(self.net_g).anchor(self.lq)
        time = torch.rand(self.lq.shape[0], device=self.device)
        t = time[:, None, None, None]
        noise = torch.randn_like(target)
        delta = anchor - target
        state = target + t * delta + t * noise
        predicted = self.net_g(state, 1000.0 * time, lq=encoded_lr, cond_features=features)
        endpoint = state - t * predicted
        velocity_loss = (predicted.float() - (delta + noise).float()).square().mean((1, 2, 3)).mean()
        endpoint_loss = F.mse_loss(endpoint.float(), target.float())
        settings = self.opt["train"]
        weighted_endpoint = settings["auxiliary_losses"]["endpoint_latent_mse"] * endpoint_loss
        loss = settings["fm_weight"] * velocity_loss + weighted_endpoint
        self._train_step(
            loss, OrderedDict(loss=velocity_loss, endpoint_latent_mse=weighted_endpoint, loss_total=loss),
            current_iter,
        )

    def inference(self):
        network = self._validation_network()
        was_training = network.training
        network.eval()
        try:
            if self.lq_paths is None:
                keys = [image.detach().cpu().contiguous().numpy().tobytes() for image in self.lq]
            else:
                keys = [Path(path).name for path in self.lq_paths]
            seeds = [image_seed(key, self.seed) for key in keys]
            self.output = super_resolve(
                network, self.codec, self.lq, self.steps, seeds, encoded_lr=self.cached_lr,
            )
        finally:
            network.train(was_training)
