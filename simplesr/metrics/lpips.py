from functools import lru_cache
import warnings

import lpips
import numpy as np
import torch

from simplesr.metrics.metric_util import reorder_image


@lru_cache(maxsize=None)
def _get_lpips_model(net, device):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=".*parameter 'pretrained' is deprecated.*", category=UserWarning)
        warnings.filterwarnings('ignore', message='Arguments other than a weight enum.*', category=UserWarning)
        return lpips.LPIPS(net=net, verbose=False).eval().to(device)


def _to_tensor(img, input_order, crop_border, device):
    img = reorder_image(img, input_order)
    if img.shape[2] != 3:
        raise ValueError(f'LPIPS expects an RGB image, but got {img.shape[2]} channel(s)')
    if crop_border < 0 or crop_border * 2 >= min(img.shape[:2]):
        raise ValueError(f'Invalid crop_border {crop_border} for image shape {img.shape}')
    if crop_border:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    img = img.astype(np.float32, copy=False)
    if img.max() > 1:
        img = img / 255.
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0).to(device) * 2 - 1


@torch.no_grad()
def calculate_lpips(img, img2, crop_border=0, input_order='HWC', net='alex', device=None, **kwargs):
    if img.shape != img2.shape:
        raise ValueError(f'Image shapes are different: {img.shape}, {img2.shape}')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = _get_lpips_model(net, device)
    return model(_to_tensor(img, input_order, crop_border, device),
                 _to_tensor(img2, input_order, crop_border, device)).item()
