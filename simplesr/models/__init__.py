from copy import deepcopy
from os import path as osp

from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config

def build_model(opt):
    model_opt = opt['model']
    name = model_opt.get('name', '-')
    model_path = model_opt.get('path', model_opt.get('model_path', '-'))
    logger = get_root_logger()

    try:
        model = instantiate_from_config(model_opt, extra_kwargs={'opt': opt})
    except Exception:
        logger.exception(f'Failed to create model [{name}] from {model_path}.')
        raise

    logger.info(f'Model [{model.__class__.__name__} - {name}] is created.')
    return model
