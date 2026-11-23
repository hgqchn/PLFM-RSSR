import ptwt
import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x, window_size):
    b, h, w, c = x.shape
    if h % window_size or w % window_size:
        raise ValueError("window_partition requires padded spatial dimensions")
    return (x.reshape(b, h // window_size, window_size, w // window_size,
                      window_size, c).permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window_size * window_size, c))


def window_reverse(windows, window_size, height, width):
    return (windows.reshape(-1, height // window_size, width // window_size,
                            window_size, window_size, windows.shape[-1])
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, height, width, windows.shape[-1]))


def haar_decompose(x):
    if x.ndim != 4 or min(x.shape[-2:]) < 1:
        raise ValueError("Haar input must be a nonempty BCHW tensor")
    with torch.autocast(device_type=x.device.type, enabled=False):
        working = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        ll, (ch, cv, cd) = ptwt.wavedec2(working, "haar", level=1, mode="constant")
    return tuple(band.to(dtype=x.dtype) for band in (ll, -cv, -ch, cd))


class WindowCrossAttention(nn.Module):

    def __init__(self, query_channels, condition_channels, num_heads=4, window_size=8):
        super().__init__()
        self.query_channels = query_channels
        self.condition_channels = condition_channels
        self.num_heads = num_heads
        self.window_size = window_size
        if min(query_channels, condition_channels, num_heads, window_size) < 1:
            raise ValueError("Channels, heads, and window size must be positive")
        if self.condition_channels % self.num_heads:
            raise ValueError("Condition channels must be divisible by attention heads")
        self.query_norm = nn.LayerNorm(self.query_channels)
        self.condition_norm = nn.LayerNorm(self.condition_channels)
        self.query = nn.Linear(self.query_channels, self.condition_channels)
        self.key = nn.Linear(self.condition_channels, self.condition_channels)
        self.value = nn.Linear(self.condition_channels, self.condition_channels)
        self.output = nn.Linear(self.condition_channels, self.query_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.relative_position_bias = nn.Parameter(torch.zeros(
            (2 * self.window_size - 1) ** 2, self.num_heads))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij"))
        coords = coords.flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative += self.window_size - 1
        index = relative[0] * (2 * self.window_size - 1) + relative[1]
        self.register_buffer("relative_position_index", index, persistent=False)

    def _windows(self, x):
        b, _, h, w = x.shape
        pad_h, pad_w = (-h) % self.window_size, (-w) % self.window_size
        padded = F.pad(x, (0, pad_w, 0, pad_h))
        windows = window_partition(padded.permute(0, 2, 3, 1), self.window_size)
        windows = windows.reshape(-1, self.window_size ** 2, self.num_heads,
                                  self.condition_channels // self.num_heads).transpose(1, 2)
        valid = torch.ones((b, 1, h, w), dtype=torch.bool, device=x.device)
        valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)
        valid = window_partition(valid.permute(0, 2, 3, 1), self.window_size).squeeze(-1)
        return windows, valid

    def _project_kv(self, condition):
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("Condition must be BCHW with the configured channel count")
        if min(condition.shape[-2:]) < 1:
            raise ValueError("Condition spatial dimensions must be nonempty")
        normalized = self.condition_norm(condition.permute(0, 2, 3, 1))
        key, valid = self._windows(self.key(normalized).permute(0, 3, 1, 2))
        value, _ = self._windows(self.value(normalized).permute(0, 3, 1, 2))
        shape = (condition.shape[0], *condition.shape[-2:])
        return key, value, valid, shape, id(self)

    def prepare_cache(self, condition):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("K/V caching requires eval mode and no_grad/inference_mode")
        return self._project_kv(condition)

    def forward(self, x, condition):
        if x.ndim != 4 or x.shape[1] != self.query_channels:
            raise ValueError("Query must be BCHW with the configured channel count")
        if isinstance(condition, torch.Tensor):
            key, value, valid, shape, owner = self._project_kv(condition)
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Cached K/V may only be used during no-grad evaluation")
            key, value, valid, shape, owner = condition
        if owner != id(self):
            raise ValueError("K/V cache belongs to another module (possibly non-EMA weights)")
        if shape != (x.shape[0], *x.shape[-2:]):
            raise ValueError("Query and condition batch/spatial dimensions must match")
        if key.device != x.device:
            raise ValueError("Query and condition must be on the same device")
        query = self.query(self.query_norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        query, _ = self._windows(query)
        tokens = self.window_size ** 2
        bias = self.relative_position_bias[self.relative_position_index.flatten()]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        mask = bias.to(query.dtype).masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(
            query, key.to(query.dtype), value.to(query.dtype), attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(-1, tokens, self.condition_channels)
        h, w = x.shape[-2:]
        merged = window_reverse(attended, self.window_size,
                                h + (-h) % self.window_size, w + (-w) % self.window_size)
        residual = self.output(merged[:, :h, :w]).permute(0, 3, 1, 2)
        return x + residual


class HaarDownsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if any(not isinstance(c, int) or isinstance(c, bool) or c <= 0
               for c in (self.in_channels, self.out_channels)):
            raise ValueError("Haar downsampling requires positive integer channels")
        self.projection = nn.Conv2d(4 * self.in_channels, self.out_channels, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Haar input must be BCHW with configured channels")
        return self.projection(torch.cat(haar_decompose(x), dim=1))


class FourScaleHaarConditioner(nn.Module):

    def __init__(self, feature_channels=128, stage_channels=(128, 256, 512, 1024),
                 num_heads=4, window_size=8):
        super().__init__()
        self.feature_channels = feature_channels
        self.stage_channels = dict(enumerate(stage_channels))
        self.num_heads = num_heads
        self.window_size = window_size
        if len(self.stage_channels) != 4:
            raise ValueError("The Haar conditioner requires exactly four stage widths")
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in
               (self.feature_channels, self.num_heads, self.window_size,
                *self.stage_channels.values())):
            raise ValueError("Feature/stage channels, heads and window size must be positive integers")
        if any(c % self.num_heads for c in self.stage_channels.values()):
            raise ValueError("Every condition width must be divisible by num_heads")
        self.input_projection = nn.Conv2d(self.feature_channels, self.stage_channels[0], 1)
        self.downsamples = nn.ModuleList([
            HaarDownsample(self.stage_channels[level - 1], self.stage_channels[level])
            for level in (1, 2, 3)
        ])
        self.attention = nn.ModuleDict({
            str(level): WindowCrossAttention(
                channels, channels, self.num_heads, self.window_size)
            for level, channels in self.stage_channels.items()
        })

    def prepare(self, features, cache=False):
        if features.ndim != 4 or features.shape[1] != self.feature_channels:
            raise ValueError("Features must be BCHW with configured feature_channels")
        if min(features.shape[-2:]) < 1:
            raise ValueError("Features must have nonempty spatial dimensions")
        if cache and (self.training or torch.is_grad_enabled()):
            raise RuntimeError("K/V caching requires eval mode and no_grad/inference_mode")
        current = self.input_projection(features)
        conditions = {}
        for level in range(4):
            if level:
                current = self.downsamples[level - 1](current)
            conditions[level] = (self.attention[str(level)].prepare_cache(current)
                                 if cache else current)
        return conditions

    def forward(self, x, prepared, level):
        if level not in self.stage_channels or level not in prepared:
            raise ValueError("A prepared condition is required for level 0, 1, 2 or 3")
        return self.attention[str(level)](x, prepared[level])
