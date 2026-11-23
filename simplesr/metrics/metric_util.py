import numpy as np

from simplesr.utils.color_utils import rgb2ycbcr


def reorder_image(img, input_order='HWC'):

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def img_to_255(img):
    img = img.astype(np.float32, copy=False)
    if img.size > 0 and img.max() <= 1.0:
        img = img * 255.0
    return img
