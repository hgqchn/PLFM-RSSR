from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config



def build_loss(loss_opt):
    logger = get_root_logger()
    loss_path = loss_opt.get('path', loss_opt.get('target', '-'))

    try:
        loss = instantiate_from_config(loss_opt)
    except Exception:
        logger.exception(f'Failed to build loss from {loss_path}.')
        raise

    logger.info(f'Loss [{loss.__class__.__name__}].')
    return loss
