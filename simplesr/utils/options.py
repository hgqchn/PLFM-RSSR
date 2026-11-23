import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
from omegaconf import OmegaConf,DictConfig

from simplesr.utils.misc import set_random_seed,get_current_time
from simplesr.utils.distributed_utils import init_distributed_mode,master_only,is_main_process



def load_yaml(yaml_file,return_dict=True):
    opt = OmegaConf.load(yaml_file)
    return opt if not return_dict else OmegaConf.to_container(opt,resolve=True, throw_on_missing=True)

@master_only
def save_yaml(opt: dict, file_path: str) -> None:

    cfg = OmegaConf.create(opt)

    OmegaConf.save(config=cfg, f=file_path)


def opt_dict_to_str(opt):
    opt=OmegaConf.create(opt)
    opt_str=OmegaConf.to_yaml(opt)
    return opt_str

def overwrite_options(opt,dotlist):
    opt=load_yaml(opt,return_dict=False)
    OmegaConf.set_struct(opt, False)

    opt_cli=OmegaConf.from_dotlist(dotlist)
    opt_new=OmegaConf.merge(opt,opt_cli)
    opt_dict=OmegaConf.to_container(opt_new,resolve=True)
    return opt_dict

def parse_args(cmd_args=None):



    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--debug', action='store_true')


    parser.add_argument("--add-time", action=argparse.BooleanOptionalAction, default=True)


    args, unknown = parser.parse_known_args(cmd_args)

    opt=overwrite_options(args.opt,unknown)


    if args.debug and not opt['exp_name'].startswith('debug_'):
        opt['exp_name'] = 'debug_' + opt['exp_name']

    return opt,args


def apply_debug_options(opt):
    if 'val' in opt:
        opt['val']['val_freq'] = 5
        opt['val']['save_img'] = False
        opt['val']['max_samples'] = 8
    opt['log_settings']['print_freq'] = 5
    opt['log_settings']['save_checkpoint_freq'] = 5
    opt['train']['total_iter'] = 10


def parse_options(opt,is_train=True):







    init_distributed_mode(opt)


    seed = opt.get('manual_seed')
    if seed is None:



        seed = random.randint(1, 2**32 - 1) if opt['rank'] == 0 else 0
        if opt['distributed']:
            seed_device = (
                torch.device('cuda', opt['local_rank'])
                if torch.cuda.is_available()
                else torch.device('cpu')
            )
            seed_tensor = torch.tensor(seed, dtype=torch.int64, device=seed_device)

            torch.distributed.broadcast(seed_tensor, src=0)
            seed = int(seed_tensor.item())
        opt['manual_seed'] = seed

    set_random_seed(seed + opt['rank'])


    opt['is_train'] = is_train


    for phase, dataset in opt['datasets'].items():

        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])


    output_dir = opt['output_dir']
    current_time=get_current_time()

    path_opt = opt.get('path') or {}
    opt['path'] = dict(path_opt)

    resume_ckpt=opt["path"].get("resume_ckpt")

    if is_train:

        if resume_ckpt:

            experiments_root = infer_experiment_root_from_checkpoint(resume_ckpt)
            opt['is_resume_train']=True
        else:

            experiments_root = opt['path'].get('experiments_root')

            if experiments_root is None:
                experiments_root = osp.join(output_dir, opt['exp_name'])

            experiments_root = osp.join(experiments_root, f'train_{current_time}')
            opt['path']['experiments_root'] = experiments_root


        opt['path']['experiments_root'] = experiments_root
        opt['path']['weights'] = osp.join(experiments_root, 'weights')
        opt['path']['checkpoints'] = osp.join(experiments_root, 'checkpoints')
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')


        if opt['exp_name'].startswith('debug_'):
            apply_debug_options(opt)




        requested_accumulation_steps = int(opt['train'].get('accumulation_steps', 1))
        if requested_accumulation_steps < 1:
            raise ValueError('train.accumulation_steps must be greater than or equal to 1.')
        if opt.get('val') is not None and opt['val'].get('val_freq') is not None:
            if int(opt['val']['val_freq']) < 1:
                raise ValueError('val.val_freq must be greater than or equal to 1.')
        if int(opt['log_settings']['save_checkpoint_freq']) < 1:
            raise ValueError('log_settings.save_checkpoint_freq must be greater than or equal to 1.')
        if int(opt['log_settings']['print_freq']) < 1:
            raise ValueError('log_settings.print_freq must be greater than or equal to 1.')
    else:
        results_root = opt['path'].get('results_root')

        if results_root is None:
            results_root = osp.join(output_dir, opt['exp_name'])

        results_root = osp.join(results_root, f'test_{current_time}')

        opt['path']['results_root'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt

def infer_experiment_root_from_checkpoint(checkpoint):

    ckpt_path = osp.abspath(osp.expanduser(checkpoint))
    ckpt_dir = osp.dirname(ckpt_path)
    ckpt_dir_name = osp.basename(ckpt_dir)

    if ckpt_dir_name == "checkpoints":
        return osp.dirname(ckpt_dir)

    raise ValueError(
        "Cannot infer experiment root from checkpoint path. "
        "Expected checkpoint path like: <exp_root>/checkpoints/checkpoint_xxx.pth"
    )

@master_only
def make_exp_dirs(opt):
    path_opt=opt['path'].copy()
    for key, path in path_opt.items():
            os.makedirs(path, exist_ok=True)
