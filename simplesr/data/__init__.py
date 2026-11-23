import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from simplesr.data.prefetch_dataloader import PrefetchDataLoader
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config
from simplesr.utils.distributed_utils import get_rank,get_world_size


__all__ = ['build_dataset', 'build_dataloader']


def build_dataset(dataset_opt):
    logger = get_root_logger()
    dataset_path = dataset_opt.get('path', dataset_opt.get('target', '-'))

    try:
        dataset = instantiate_from_config(dataset_opt)
    except Exception:
        logger.exception(f'Failed to build dataset from {dataset_path}.')
        raise

    dataset_name = dataset.name
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_name} is built.')
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    phase = dataset_opt['phase']
    rank=get_rank()
    if phase == 'train':
        if dist:
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True

        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:
        batch_size = dataset_opt.get('batch_size', 1)
        num_workers = dataset_opt.get('num_worker',0)
        dataloader_args = dict(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,)
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', True)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:


        return torch.utils.data.DataLoader(**dataloader_args)




def worker_init_fn(worker_id, num_workers, rank, seed):

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
