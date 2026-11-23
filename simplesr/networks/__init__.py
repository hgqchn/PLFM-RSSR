
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.object_utils import instantiate_from_config



def build_network(network_opt):
    logger = get_root_logger()
    network_name = network_opt.get('name', '')
    network_path = network_opt.get('path', network_opt.get('target', '-'))

    try:
        net = instantiate_from_config(network_opt)
    except Exception:
        logger.exception(
            f'Failed to build network [{network_name}] from {network_path}.')
        raise

    logger.info(f'Network [{net.__class__.__name__}] - {network_name} is built.')
    return net
