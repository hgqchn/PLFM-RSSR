from copy import deepcopy

from .dists import calculate_dists
from .lpips import calculate_lpips
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_psnr_pt, calculate_ssim, calculate_ssim_pt

__all__ = ['calculate_metric', 'calculate_dists', 'calculate_lpips',
           'calculate_psnr', 'calculate_psnr_pt', 'calculate_ssim', 'calculate_ssim_pt', 'calculate_niqe']


METRIC_FUNCS = {
    'calculate_dists': calculate_dists,
    'calculate_lpips': calculate_lpips,
    'calculate_psnr': calculate_psnr,
    'calculate_psnr_pt': calculate_psnr_pt,
    'calculate_ssim': calculate_ssim,
    'calculate_ssim_pt': calculate_ssim_pt,
    'calculate_niqe': calculate_niqe,
    'dists': calculate_dists,
    'DISTS': calculate_dists,
    'lpips': calculate_lpips,
    'LPIPS': calculate_lpips,
    'psnr': calculate_psnr,
    'PSNR': calculate_psnr,
    'ssim': calculate_ssim,
    'SSIM': calculate_ssim,
    'niqe': calculate_niqe,
    'NIQE': calculate_niqe,
}


def calculate_metric(data, opt):
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    if metric_type not in METRIC_FUNCS:
        raise KeyError(f'Unsupported metric type: {metric_type}. Supported metrics: {tuple(METRIC_FUNCS)}')
    return METRIC_FUNCS[metric_type](**data, **opt)
