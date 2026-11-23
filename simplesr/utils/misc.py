import numpy as np
import os
import random
import torch
from os import path as osp

import datetime as dt

def get_filename(path,with_ext=False):
    basename = os.path.basename(path)
    name,ext=os.path.splitext(basename)
    if with_ext:
        return name,ext
    else:
        return name


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def get_parameter_nums(model, show=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    if show:
        print(f"\n模型: {model.__class__.__name__}")
        print(f"总参数量:      {total_params:,}")
        print(f"可训练参数量:  {trainable_params:,}")
        print(f"不可训练参数量: {non_trainable:,}\n")
    return total_params, trainable_params, non_trainable

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_current_time():
    format="%Y-%m-%d_%H-%M-%S"

    return dt.datetime.now().strftime(format)



def mkdir_and_rename(path):
    if osp.exists(path):
        new_name = path + '_archived_' + get_current_time()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def scandir(dir_path, suffix=None, recursive=False):

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                return_path = entry.path

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)



