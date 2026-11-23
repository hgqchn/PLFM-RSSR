import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler

from timm.utils import ModelEmaV2 as ModelEma

from simplesr.models.lr_scheduler import CosineAnnealingHoldLR
from simplesr.utils.log_utils import get_root_logger
from simplesr.utils.distributed_utils import master_only,get_local_rank
from simplesr.utils.misc import count_parameters


class BaseModel():

    def __init__(self, opt):
        self.opt = opt
        self.local_rank = get_local_rank()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.is_train = opt['is_train']
        requested_accumulation_steps = int(opt.get('train', {}).get('accumulation_steps', 1))
        if requested_accumulation_steps < 1:
            raise ValueError('train.accumulation_steps must be greater than or equal to 1.')
        self.accumulation_steps = requested_accumulation_steps


        self.total_micro_iters = int(opt.get('train', {}).get('total_iter', 0)) * self.accumulation_steps
        self.schedulers = []
        self.optimizers = []
        self.log_dict = OrderedDict()
        self._accumulated_log_dict = OrderedDict()
        self.best_metric_results = {}

    def feed_data(self, data):
        pass

    def optimize_parameters(self,current_iter):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        pass

    def validation(
            self, dataloader, current_iter, wandb_logger, save_img=False,
            dataset_name=None, max_samples=None, save_reference_img=False):
        if self.opt['distributed']:
            return self.dist_validation(
                dataloader, current_iter, wandb_logger, save_img, dataset_name, max_samples,
                save_reference_img)
        else:
            return self.nondist_validation(
                dataloader, current_iter, wandb_logger, save_img, dataset_name, max_samples,
                save_reference_img)

    def _initialize_best_metric_results(self, dataset_name):
        if dataset_name in self.best_metric_results:
            return


        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def init_ema(self,network,decay=0.999):
        net=self.get_bare_model(network)
        net_ema=ModelEma(
            net,
            decay=decay,
            device=self.device
        )

        for p in net_ema.module.parameters():
            p.requires_grad_(False)

        net_ema.module.eval()

        return net_ema

    @torch.no_grad()
    def update_ema(
        self,
        network: nn.Module | None = None,
        net_ema: ModelEma | None = None,
    ) -> None:

        net = self.get_bare_model(network)
        net_ema.update(net)

    def get_ema_module(self, net_ema):
        return net_ema.module

    def get_current_log(self):
        return self.log_dict

    def is_optimizer_step(self, current_iter):
        return current_iter % self.accumulation_steps == 0 or current_iter == self.total_micro_iters

    def get_optimizer_iter(self, current_iter):
        return current_iter // self.accumulation_steps

    def backward_and_step(self, loss, current_iter):
        if (current_iter - 1) % self.accumulation_steps == 0:
            self.optimizer_g.zero_grad()


        (loss / self.accumulation_steps).backward()
        if not self.is_optimizer_step(current_iter):
            return False

        self.optimizer_g.step()
        if self.ema_decay > 0:
            self.update_ema(self.net_g, self.net_g_ema)
        return True

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt['distributed']:
            if self.device.type == 'cuda':
                net = DistributedDataParallel(
                    net,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False)
            else:
                net = DistributedDataParallel(net, find_unused_parameters=False)

        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        train_opt = self.opt['train']
        scheduler_opt = train_opt.get('scheduler')
        if not scheduler_opt:
            return
        scheduler_type = scheduler_opt.pop('type')
        if scheduler_type == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingHoldLR':
            for optimizer in self.optimizers:
                self.schedulers.append(CosineAnnealingHoldLR(optimizer, **scheduler_opt))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        if isinstance(net, DistributedDataParallel):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        if isinstance(net, DistributedDataParallel):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = count_parameters(net)

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')


    def _set_lr(self, lr_groups_l):
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        init_lr_groups_l = []
        for optimizer in self.optimizers:


            init_lr_groups_l.append([
                v.setdefault('initial_lr', v['lr']) for v in optimizer.param_groups
            ])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):


        if (current_iter - 1) % self.accumulation_steps != 0:
            return
        update_iter = (current_iter - 1) // self.accumulation_steps + 1
        if update_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

        if update_iter <= warmup_iter:

            init_lr_g_l = self._get_init_lr()

            warm_up_lr_l = []

            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * update_iter for v in init_lr_g])

            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter,):
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['weights'], save_filename)

        net = self.get_bare_model(net)
        state_dict = net.state_dict()

        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')


        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_sd = self._load_network_state(load_path)

        logger.info(f'Loading {net.__class__.__name__} model from {load_path}')





        self._print_different_keys_loading(net, load_sd, strict)
        net.load_state_dict(load_sd, strict=strict)

    @staticmethod
    def _load_network_state(load_path):
        if str(load_path).lower().endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(load_path, device='cpu')
        return torch.load(load_path, map_location='cpu', weights_only=False)

    @master_only
    def save_checkpoint(self, epoch, current_iter, file_iter=None):
        if current_iter != -1:
            net_g_ema = getattr(self, 'net_g_ema', None)
            state = {
                'epoch': epoch,
                'iter': current_iter,

                'accumulation_steps': self.accumulation_steps,
                'net_g': self.get_bare_model(self.net_g).state_dict(),
                'net_g_ema': net_g_ema.module.state_dict() if net_g_ema else None,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{file_iter if file_iter is not None else current_iter}.state'
            save_path = os.path.join(self.opt['path']['checkpoints'], save_filename)

            torch.save(state, save_path)
            return save_path
        else:
            return None


    def resume_checkpoint(self, resume_state):




        saved_accumulation_steps = resume_state.get('accumulation_steps')
        if (saved_accumulation_steps is not None
                and int(saved_accumulation_steps) != self.accumulation_steps):
            raise ValueError(
                'Checkpoint accumulation_steps '
                f'({saved_accumulation_steps}) does not match current training config '
                f'({self.accumulation_steps}).'
            )

        if 'net_g' in resume_state:
            self.get_bare_model(self.net_g).load_state_dict(resume_state['net_g'], strict=True)

        net_g_ema = getattr(self, 'net_g_ema', None)
        if net_g_ema is not None and resume_state.get('net_g_ema') is not None:
            net_g_ema.module.load_state_dict(resume_state['net_g_ema'], strict=True)

        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict, current_iter):
        with torch.no_grad():
            if self.opt['distributed']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    value = torch.as_tensor(
                        value, device=self.device, dtype=torch.float32)
                    if value.numel() != 1:
                        raise ValueError(
                            f'Loss "{name}" must be scalar, but got shape '
                            f'{tuple(value.shape)}.')
                    losses.append(value.reshape(()))
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        raise ValueError(
                            f'Loss "{name}" must be scalar, but got shape '
                            f'{tuple(value.shape)}.')
                    value = value.item()
                log_dict[name] = float(value)



            if (current_iter - 1) % self.accumulation_steps == 0:
                self._accumulated_log_dict.clear()
            for name, value in log_dict.items():
                self._accumulated_log_dict[name] = (
                    self._accumulated_log_dict.get(name, 0.0)
                    + value / self.accumulation_steps)

            return OrderedDict(self._accumulated_log_dict)
