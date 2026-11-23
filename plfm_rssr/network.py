import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .haar import FourScaleHaarConditioner
from .swin import RSTBLatentFrontend, SwinTransformerBlock, window_partition, window_reverse


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _batch_scalar(value, x, name):
    value = torch.as_tensor(value, device=x.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(x.shape[0])
    if value.shape != (x.shape[0],) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape [B].")
    return value


def _position_embedding(height, width, channels, device, dtype):
    row = timestep_embedding(torch.arange(height, device=device), channels // 2)
    col = timestep_embedding(torch.arange(width, device=device), channels // 2)
    return torch.cat((row[:, None].expand(-1, width, -1),
                      col[None].expand(height, -1, -1)), -1).to(dtype)[None]


class TimeConditionedSwinBlock(SwinTransformerBlock):

    def __init__(self, channels, emb_channels, input_resolution, num_heads,
                 window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__(
            dim=channels, input_resolution=tuple(input_resolution), num_heads=num_heads,
            window_size=window_size, shift_size=shift_size, mlp_ratio=mlp_ratio,
            qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
            norm_layer=partial(nn.LayerNorm, elementwise_affine=False, eps=1e-6),
        )
        self.emb_channels = emb_channels
        self.modulation = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(self.emb_channels, 6 * self.dim)))
        self.register_buffer("attn_mask", self.attn_mask, persistent=False)

    def _window_attention(self, x):
        batch, height, width, channels = x.shape
        size = self.window_size
        pad_h, pad_w = (-height) % size, (-width) % size
        padded_height, padded_width = height + pad_h, width + pad_w
        shift = self.shift_size if min(height, width) > size else 0
        padded = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        if shift:
            padded = torch.roll(padded, (-shift, -shift), (1, 2))
        windows = window_partition(padded, size).view(-1, size * size, channels)
        mask = None
        if shift:
            mask = (self.attn_mask if (padded_height, padded_width) == self.input_resolution
                    else self.calculate_mask((padded_height, padded_width)))
            mask = mask.to(device=x.device, dtype=x.dtype)
            mask = mask.masked_fill(mask != 0, float("-inf"))
        if pad_h or pad_w:
            valid = torch.zeros((1, padded_height, padded_width, 1), dtype=torch.bool, device=x.device)
            valid[:, :height, :width] = True
            if shift:
                valid = torch.roll(valid, (-shift, -shift), (1, 2))
            valid = window_partition(valid, size).view(-1, size * size)
            if mask is None:
                mask = x.new_zeros((valid.shape[0], size * size, size * size))
            mask = mask.masked_fill(~valid[:, None, :], float("-inf"))
            empty_rows = torch.isneginf(mask).all(dim=-1)
            mask = mask.masked_fill(empty_rows[..., None] & torch.eye(
                size * size, dtype=torch.bool, device=x.device)[None], 0)
        attended = self.attn(windows, mask=mask)
        merged = window_reverse(attended.view(-1, size, size, channels),
                                size, padded_height, padded_width)
        if shift:
            merged = torch.roll(merged, (shift, shift), (1, 2))
        return merged[:, :height, :width].reshape(batch, height, width, channels)

    def forward(self, x, emb):
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError("Swin velocity block requires BHWC with matching channels")
        if emb.shape != (x.shape[0], self.emb_channels):
            raise ValueError("Time embedding must have shape [B, emb_channels]")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(emb).to(x.dtype)[:, None, None].chunk(6, dim=-1))
        query = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.drop_path(self._window_attention(query))
        return x + gate_m * self.drop_path(self.mlp(self.norm2(x) * (1 + scale_m) + shift_m))


class TimeConditionedSwinStage(nn.Module):

    def __init__(self, channels, emb_channels, input_resolution, depth, num_heads,
                 window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimeConditionedSwinBlock(
                channels, emb_channels, input_resolution, num_heads, window_size,
                window_size // 2 if index % 2 else 0, mlp_ratio)
            for index in range(depth)
        ])

    def forward(self, x, emb):
        x = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x, emb)
        return x.permute(0, 3, 1, 2).contiguous()


class TimeConditionedRSTB(nn.Module):
    def __init__(self, channels, resolution):
        super().__init__()
        self.add_module("residual_group", None)
        self.conv = nn.Conv2d(160, 160, 3, padding=1)
        self.input_projection = nn.Conv2d(channels, 160, 1)
        self.output_projection = nn.Conv2d(160, channels, 1)
        self.residual_group = TimeConditionedSwinStage(
            160, 640, (resolution, resolution), 4, 5, window_size=8, mlp_ratio=4.0,
        )

    def forward(self, x, emb):
        updated = self.input_projection(x)
        updated = self.conv(self.residual_group(updated, emb))
        return x + self.output_projection(updated)


class RSTBStage(nn.Module):
    def __init__(self, channels, resolution):
        super().__init__()
        self.groups = nn.ModuleList([TimeConditionedRSTB(channels, resolution)])

    def forward(self, x, emb):
        return self.groups[0](x, emb)


class RSTBRawEnhancedSkipVelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = (160, 320, 320, 640)
        self.time_embed = nn.Sequential(nn.Linear(160, 640), nn.SiLU(), nn.Linear(640, 640))
        self.stem = nn.Conv2d(6, self.channels[0], 3, padding=1)
        self.encoder = nn.ModuleList([
            RSTBStage(width, 64 // (2 ** level))
            for level, width in enumerate(self.channels)
        ])
        self.down = nn.ModuleList(
            nn.Conv2d(current, next_width, 3, stride=2, padding=1)
            for current, next_width in zip(self.channels, self.channels[1:])
        )
        self.middle = RSTBStage(self.channels[-1], 8)
        self.up = nn.ModuleList(
            nn.Conv2d(next_width, current, 3, padding=1)
            for current, next_width in zip(self.channels, self.channels[1:])
        )
        self.skip_projection = nn.ModuleList(
            nn.Conv2d(2 * width, width, 1) for width in self.channels
        )
        self.decoder = nn.ModuleList([
            RSTBStage(width, 64 // (2 ** level))
            for level, width in enumerate(self.channels)
        ])
        self.conditioner = FourScaleHaarConditioner(128, self.channels, 4, 8)
        self.final_norm = nn.LayerNorm(self.channels[0], elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(640, 2 * self.channels[0])),
        )
        self.out = zero_module(nn.Conv2d(self.channels[0], 3, 3, padding=1))
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for level, width in enumerate(self.channels):
                original = self.skip_projection[level]
                projection = nn.Conv2d(3 * width, width, 1)
                decoder_weight, skip_weight = original.weight.split(width, dim=1)
                half_skip = 0.5 * skip_weight
                projection.weight.copy_(torch.cat((decoder_weight, half_skip, half_skip), dim=1))
                projection.bias.copy_(original.bias)
                self.skip_projection[level] = projection

    def prepare_condition(self, features, cache=None):
        if isinstance(features, dict):
            if set(features) != {0, 1, 2, 3}:
                raise ValueError("Prepared features must contain all four levels")
            return features
        if cache is None:
            cache = not self.training and not torch.is_grad_enabled()
        return self.conditioner.prepare(features, cache=cache)

    def forward(self, x, timesteps, lq, cond_features):
        if (x.ndim != 4 or x.shape[1] != 3
                or min(x.shape[-2:]) < 8 or any(size % 8 for size in x.shape[-2:])):
            raise ValueError("Expected BCHW latents with three channels and sides divisible by eight")
        if lq.ndim != 4 or lq.shape[:2] != x.shape[:2]:
            raise ValueError("The LR latent must match the evolving latent batch and channels")
        condition = self.prepare_condition(cond_features)
        labels = _batch_scalar(timesteps, x, "timesteps")
        emb = self.time_embed(timestep_embedding(labels, 160).to(self.time_embed[0].weight.dtype))
        low_res = F.interpolate(lq, x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.stem(torch.cat((x, low_res), dim=1))
        h = h + _position_embedding(*h.shape[-2:], h.shape[1], h.device, h.dtype).permute(0, 3, 1, 2)
        skips = []
        for level, stage in enumerate(self.encoder):
            h = stage(h, emb)
            skips.append(h)
            if level < 3:
                h = self.down[level](h)
        h = self.middle(h, emb)
        for level in (3, 2, 1, 0):
            raw_skip = skips[level]
            if level < 3:
                h = self.up[level](F.interpolate(h, size=raw_skip.shape[-2:], mode="nearest"))
            enhanced_skip = self.conditioner(raw_skip, condition, level)
            h = self.skip_projection[level](torch.cat((h, raw_skip, enhanced_skip), dim=1))
            h = self.decoder[level](h, emb)
        shift, scale = self.final_modulation(emb).to(h.dtype)[:, None, None].chunk(2, dim=-1)
        h = self.final_norm(h.permute(0, 2, 3, 1)) * (1 + scale) + shift
        return self.out(h.permute(0, 3, 1, 2).contiguous()).to(x.dtype)


class PLFMRSSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = RSTBRawEnhancedSkipVelocityNet()
        self.anchor = build_frontend().requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, x, timesteps, lq, cond_features):
        return self.velocity(x, timesteps, lq=lq, cond_features=cond_features)


def build_frontend():
    return RSTBLatentFrontend()


def build_model():
    return PLFMRSSR()
