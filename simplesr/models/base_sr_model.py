import sys
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from simplesr.networks import build_network
from simplesr.losses import build_loss
from simplesr.metrics import calculate_metric
from simplesr.utils.log_utils import AverageMeter, CSVLogger, get_root_logger
from simplesr.utils.img_utils import tensor_to_img_array, write_rgb_float_img
from .base_model import BaseModel


class SRModel(BaseModel):

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.load_path = None
        self.net_g_ema = None
        self.ema_decay = 0
        self.cri_pix = None
        self.cri_perceptual = None
        self.optimizer_g = None
        self.lq = None
        self.gt = None
        self.output = None
        self.metric_results = {}


        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)


        load_path = self.opt['path'].get('pretrain_network_g', None)
        self.load_path = load_path
        if load_path is not None:
            self.load_network(self.net_g, load_path, strict=True)

        if self.is_train:
            self.init_training_settings()


    def init_training_settings(self):
        self.net_g.train()

        self._init_ema()

        self._init_losses()


        self.setup_optimizers()
        self.setup_schedulers()

    def _init_ema(self):
        self.ema_decay = self.opt['train'].get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'使用 EMA，decay={self.ema_decay}')
            self.net_g_ema = self.init_ema(self.net_g, decay=self.ema_decay)


            if self.load_path is not None:
                self.load_network(self.net_g_ema.module, self.load_path, strict=True)

    def _init_losses(self):
        train_opt = self.opt["train"]

        if train_opt.get("pixel_opt") is not None:
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)

        if train_opt.get("perceptual_opt") is not None:
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("At least one loss should be defined.")


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'[setup_optimizers]Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = None
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style



        self.backward_and_step(l_total, current_iter)

        self.log_dict = self.reduce_loss_dict(loss_dict, current_iter)

    def inference(self):
        if self.net_g_ema is not None:
            net_g_ema = self.get_ema_module(self.net_g_ema)
            net_g_ema.eval()
            with torch.no_grad():
                self.output = net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def inference_selfensemble(self):




        def _transform(v, op):
            if op == 'v':
                return torch.flip(v, dims=[3])
            elif op == 'h':
                return torch.flip(v, dims=[2])
            elif op == 't':
                return v.transpose(2, 3)
            else:
                raise ValueError(f'Unsupported self-ensemble transform: {op}')














        lq_list = [self.lq]

        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])


        if self.net_g_ema is not None:
            net_g_ema = self.get_ema_module(self.net_g_ema)
            net_g_ema.eval()
            with torch.no_grad():
                out_list = [net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()


        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.stack(out_list, dim=0)

        self.output = output.mean(dim=0)

    def dist_validation(
            self, dataloader, current_iter, wandb_logger, save_img, dataset_name=None,
            max_samples=None, save_reference_img=False):

        if self.opt['rank'] == 0:
            return self.nondist_validation(
                dataloader, current_iter, wandb_logger, save_img, dataset_name, max_samples,
                save_reference_img)
        return None

    def nondist_validation(
            self, dataloader, current_iter, wandb_logger, save_img, dataset_name=None,
            max_samples=None, save_reference_img=False):
        dataset_name = dataset_name or dataloader.dataset.name
        assert self.opt['val'].get('metrics') is not None
        with_metrics=True
        use_pbar = self.opt['log_settings'].get('val_pbar', False)
        if max_samples is not None:
            if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
                raise ValueError('val.max_samples must be a positive integer or null.')

        per_image_logger = None
        if self.opt['val'].get('save_per_image_metrics', False):
            root_key = 'experiments_root' if self.opt['is_train'] else 'results_root'
            per_image_logger = CSVLogger(osp.join(
                self.opt['path'][root_key], 'per_image_metrics', dataset_name,
                f'iter_{current_iter}.csv'), flush_freq=1)
            if per_image_logger.csv_file.exists():
                raise FileExistsError(f'Per-image metrics already exist: {per_image_logger.csv_file}')

        if with_metrics:

            self._initialize_best_metric_results(dataset_name)
            metric_meters = {metric: AverageMeter() for metric in self.opt['val']['metrics'].keys()}

        num_samples = 0
        pbar_total = min(len(dataloader.dataset), max_samples) if max_samples is not None else len(dataloader.dataset)
        pbar = tqdm(total=pbar_total, unit='image',
                    desc=f'Validate {dataset_name} @ iter {current_iter}', file=sys.stdout) if use_pbar else None

        for idx, val_data in enumerate(dataloader):

            self.feed_data(val_data)
            self.inference()



            visuals = self.get_current_visuals()
            sr_img = tensor_to_img_array(visuals['result'])
            sr_imgs = sr_img if isinstance(sr_img, list) else [sr_img]

            gt_imgs = None
            if 'gt' in visuals:
                gt_img = tensor_to_img_array(visuals['gt'])
                gt_imgs = gt_img if isinstance(gt_img, list) else [gt_img]

            lq_imgs = None
            if save_reference_img and 'lq' in visuals:
                lq_img = tensor_to_img_array(visuals['lq'])
                lq_imgs = lq_img if isinstance(lq_img, list) else [lq_img]

            lq_paths = val_data.get('lq_path', None)
            if lq_paths is None:
                lq_paths = [f'{idx:08d}_{i:04d}' for i in range(len(sr_imgs))]
            elif isinstance(lq_paths, str):
                lq_paths = [lq_paths]


            for sample_idx, sr_img in enumerate(sr_imgs):
                if max_samples is not None and num_samples >= max_samples:
                    break
                img_name = osp.splitext(osp.basename(lq_paths[sample_idx]))[0]

                if save_img:
                    if self.opt['is_train']:
                        iter_dir = osp.join(
                            self.opt['path']['visualization'], dataset_name, f'iter_{current_iter}')
                        save_img_path = osp.join(iter_dir, 'sr', f'{img_name}.png')
                        if save_reference_img:
                            if lq_imgs is not None:
                                write_rgb_float_img(
                                    lq_imgs[sample_idx], osp.join(iter_dir, 'lq', f'{img_name}.png'))
                            if gt_imgs is not None:
                                write_rgb_float_img(
                                    gt_imgs[sample_idx], osp.join(iter_dir, 'gt', f'{img_name}.png'))
                    else:
                        suffix = self.opt['val'].get('suffix')
                        if suffix:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{suffix}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["exp_name"]}.png')
                    write_rgb_float_img(sr_img, save_img_path)

                if with_metrics:
                    metric_data = {'img': sr_img}
                    if gt_imgs is not None:
                        metric_data['img2'] = gt_imgs[sample_idx]
                    sample_metrics = {}
                    for name, opt_ in self.opt['val']['metrics'].items():
                        value = calculate_metric(metric_data, opt_)
                        metric_meters[name].update(value)
                        sample_metrics[name] = value
                    if per_image_logger is not None:
                        per_image_logger.write({
                            'image': img_name, 'lq_path': str(lq_paths[sample_idx]), **sample_metrics})

                num_samples += 1
                if pbar is not None:
                    pbar.update(1)

            self.gt = None

            if max_samples is not None and num_samples >= max_samples:
                break






        if pbar is not None:
            pbar.close()

        if with_metrics:
            if num_samples == 0:
                raise ValueError(f'Validation dataloader for {dataset_name} is empty.')
            self.metric_results = {metric: meter.avg for metric, meter in metric_meters.items()}
            for metric in self.metric_results.keys():

                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, wandb_logger)
            return {'dataset': dataset_name, **self.metric_results}
        return {'dataset': dataset_name}

    def _log_validation_metric_values(self, current_iter, dataset_name, wandb_logger):
        log_str = f'Validation {dataset_name} @ iter {current_iter}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t# {metric} | Current: {value:.4f}'
            log_str += (f' | Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} '
                        f'@ iter {self.best_metric_results[dataset_name][metric]["iter"]}')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if wandb_logger:
            for metric, value in self.metric_results.items():
                wandb_logger.log({f'metrics/{dataset_name}/{metric}': value}, step=current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, display_iter=None):
        file_iter = current_iter if display_iter is None else display_iter
        self.save_network(self.net_g, 'net_g', file_iter)
        net_g_ema = getattr(self, 'net_g_ema', None)
        if net_g_ema is not None:
            ema_module = self.get_ema_module(net_g_ema)
            self.save_network(ema_module, 'net_g_ema', file_iter)
        return self.save_checkpoint(epoch, current_iter, file_iter=file_iter)
