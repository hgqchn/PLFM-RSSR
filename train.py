import datetime
import logging
import math
import time
import torch
import os
import sys
from os import path as osp
from tqdm import tqdm
from simplesr.data import build_dataloader, build_dataset
from simplesr.data.data_sampler import EnlargedSampler
from simplesr.models import build_model

from simplesr.utils.options import parse_args,parse_options,opt_dict_to_str,save_yaml
from simplesr.utils.log_utils import AvgTimer, TrainMessageLogger, init_root_logger, CSVLogger, init_wandb_logger
from simplesr.utils.distributed_utils import is_main_process, master_only
from simplesr.utils.misc import get_current_time


def validation_kind(micro_iter, accumulation_steps, quick_frequency, full_updates, total_updates, debug=False):
    if micro_iter % accumulation_steps:
        return None
    update = micro_iter // accumulation_steps
    if not debug and update in full_updates and update < total_updates:
        return 'full'
    if quick_frequency is not None and update % quick_frequency == 0:
        return 'quick'
    return None

@master_only
def make_dirs(opt):
    path_opt=opt['path']
    os.makedirs(path_opt['experiments_root'], exist_ok=True)
    os.makedirs(path_opt['weights'], exist_ok=True)
    os.makedirs(path_opt['checkpoints'], exist_ok=True)
    os.makedirs(path_opt['visualization'], exist_ok=True)


def create_train_val_dataloader(opt, logger):

    train_loader, val_loaders = None, []
    train_sampler = None
    total_epochs = 0
    total_iters = 0
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)

            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)

            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['world_size'],
                dist=opt['distributed'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            batch_size_per_gpu = dataset_opt['batch_size_per_gpu']
            samples_per_rank = len(train_sampler)

            num_iter_per_epoch = len(train_loader)
            if num_iter_per_epoch == 0:
                raise ValueError(
                    f'Train dataloader is empty: samples_per_rank={samples_per_rank}, '
                    f'batch_size_per_gpu={batch_size_per_gpu}.')
            accumulation_steps = int(opt['train'].get('accumulation_steps', 1))


            total_iters = int(opt['train']['total_iter']) * accumulation_steps
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['world_size'], dist=opt['distributed'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {val_set.name}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

def load_checkpoint(cpt_path):
    return torch.load(cpt_path, map_location='cpu', weights_only=False)


def train_pipeline(cmd_args=None):
    opt, _ = parse_args(cmd_args)

    opt = parse_options(opt, is_train=True)

    exp_name = opt['exp_name']
    debug = exp_name.startswith('debug_')
    if debug:
        print("=" * 30 + "调试模式" + "=" * 30)


    torch.backends.cudnn.benchmark = True


    current_time=get_current_time()
    resume_train = opt.get('is_resume_train', False)

    experiments_root=opt['path']['experiments_root']
    if not resume_train and not debug:
        make_dirs(opt)
        save_yaml(opt,osp.join(experiments_root,'config.yaml'))


    opt['path']['log_file'] = osp.join(experiments_root, f'log_{current_time}.log')
    log_file=opt['path']['log_file'] if not debug else None


    logger = init_root_logger(logger_name='basic_logger', log_level=logging.INFO, log_file_path=log_file)
    logger.info("experiment opt:\n"+opt_dict_to_str(opt))


    csv_flush_freq = opt['log_settings'].get('csv_flush_freq', 10)
    train_csv_logger = None if debug else CSVLogger(osp.join(experiments_root, 'train_log.csv'), csv_flush_freq)
    val_csv_logger = None if debug else CSVLogger(osp.join(experiments_root, 'val_log.csv'), 1)



    wandb_logger=None
    wandb_opt=opt["log_settings"].get("wandb")
    train_dataset_opt = opt['datasets']['train']
    train_kwargs = train_dataset_opt.get('kwargs', {})
    val_dataset_names = [
        dataset_opt.get('kwargs', {}).get('name', phase)
        for phase, dataset_opt in opt['datasets'].items()
        if phase.split('_')[0] == 'val'
    ]
    network_g_opt = opt.get('network_g', {})
    network_g_kwargs = network_g_opt.get('kwargs', {})
    train_opt = opt['train']
    optim_g_opt = train_opt.get('optim_g', {})
    scheduler_opt = train_opt.get('scheduler', {})
    pixel_opt = train_opt.get('pixel_opt', {})
    perceptual_opt = train_opt.get('perceptual_opt', {})
    val_opt = opt.get('val', {})

    exp_opt = {
        "exp_name": exp_name,
        "resume_train": resume_train,
        "scale": opt.get('scale'),
        "manual_seed": opt.get('manual_seed'),
        "distributed": opt.get('distributed'),
        "train_dataset": train_kwargs.get('name'),
        "val_datasets": val_dataset_names,
        "batch_size_per_gpu": train_dataset_opt.get('batch_size_per_gpu'),
        "num_worker_per_gpu": train_dataset_opt.get('num_worker_per_gpu'),
        "model": opt.get('model', {}).get('name'),
        "model_path": opt.get('model', {}).get('path'),
        "network_g": network_g_opt.get('name'),
        "network_g_path": network_g_opt.get('path'),
        "network_g_kwargs": network_g_kwargs,
        "total_iter": train_opt['total_iter'],
        "warmup_iter": train_opt.get('warmup_iter', -1),
        "ema_decay": train_opt.get('ema_decay', 0),
        "optimizer": optim_g_opt.get('type'),
        "initial_lr": optim_g_opt.get('lr'),
        "pixel_loss": pixel_opt.get('path'),
        "perceptual_loss": perceptual_opt.get('path'),
        "experiments_root": experiments_root,
        "vae": opt.get("vae"),
        "flow_matching": opt.get("flow_matching"),
        "network_anchor": opt.get("network_anchor"),
        "accumulation_steps": train_opt.get("accumulation_steps", 1),
        "val_freq": val_opt.get("val_freq"),
        "provenance": opt.get("provenance"),
    }
    if wandb_opt and not debug and wandb_opt.get('mode') != 'disabled':
        wandb_logger = init_wandb_logger(
            exp_name=exp_name,
            wandb_opt=wandb_opt,
            exp_opt=exp_opt,
            save_dir=experiments_root,
        )


    model=build_model(opt)


    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    epoch_multiplier = int(getattr(model, 'training_epoch_multiplier', 1))
    if epoch_multiplier < 1:
        raise ValueError('training_epoch_multiplier must be positive.')
    total_epochs *= epoch_multiplier



    train_opt = opt['datasets']['train']
    num_train_images = len(train_loader.dataset)
    dataset_enlarge_ratio = train_opt.get('dataset_enlarge_ratio', 1)
    enlarged_size = num_train_images * dataset_enlarge_ratio
    batch_size_per_gpu = train_opt['batch_size_per_gpu']
    world_size = opt['world_size']
    global_batch_size = batch_size_per_gpu * world_size
    accumulation_steps = model.accumulation_steps
    effective_batch_size = global_batch_size * accumulation_steps
    samples_per_rank = len(train_sampler)
    num_iter_per_epoch = len(train_loader)
    total_optim_steps = int(opt['train']['total_iter'])
    logger.info('Training statistics:'
                f'\n\tVisible GPU count: {torch.cuda.device_count()}'
                f'\n\tDistributed training: {opt["distributed"]}'
                f'\n\tWorld size (training processes): {world_size}'
                f'\n\tCurrent rank / local rank: {opt["rank"]} / {opt["local_rank"]}'
                f'\n\tNumber of train images: {num_train_images}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tEnlarged sample count: {num_train_images} * {dataset_enlarge_ratio} = {enlarged_size}'
                f'\n\tSamples per rank: ceil({enlarged_size} / {world_size}) = {samples_per_rank}'
                f'\n\tBatch size per GPU/process: {batch_size_per_gpu}'
                f'\n\tGlobal batch size: {batch_size_per_gpu} * {world_size} = {global_batch_size}'
                f'\n\tGradient accumulation steps: {accumulation_steps}'
                f'\n\tEffective batch size: {global_batch_size} * {accumulation_steps} = {effective_batch_size}'
                f'\n\tIterations per epoch: floor({samples_per_rank} / {batch_size_per_gpu}) '
                f'= len(train_loader) = {num_iter_per_epoch}'
                f'\n\tConfigured optimizer steps: {total_optim_steps}'
                f'\n\tActual micro-batch iterations: {total_optim_steps} * {accumulation_steps} = {total_iters}'
                f'\n\tRequired epochs: ceil({total_iters} / {num_iter_per_epoch}) = {total_epochs}')


    print_freq_optim_steps = int(opt['log_settings']['print_freq'])
    save_freq_optim_steps = int(opt['log_settings']['save_checkpoint_freq'])
    val_freq_optim_steps = opt.get('val', {}).get('val_freq')
    full_validation_updates = opt.get('val', {}).get('full_iters', [])
    if not debug and any(isinstance(step, bool) or not isinstance(step, int)
                         or step <= 0 or step > total_optim_steps or step % save_freq_optim_steps
                         for step in full_validation_updates):
        raise ValueError('val.full_iters must be optimizer-update checkpoint boundaries within total_iter')
    full_validation_updates = set(full_validation_updates) if not debug else set()


    actual_print_freq = print_freq_optim_steps * accumulation_steps
    actual_save_freq = save_freq_optim_steps * accumulation_steps


    start_epoch = 1
    current_iter = 0

    if resume_train:
        resume_ckpt = opt['path']['resume_ckpt']
        logger.info(f'Resuming training from checkpoint: {resume_ckpt}')
        resume_state = load_checkpoint(resume_ckpt)
        model.resume_checkpoint(resume_state)


        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']



    current_optimizer_iter = model.get_optimizer_iter(current_iter)

    msg_logger = TrainMessageLogger(opt, current_optimizer_iter, wandb_logger=wandb_logger)



    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_optimizer_iter}')

    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.perf_counter()

    msg_logger.reset_start_time()
    use_train_pbar = opt['log_settings'].get('train_pbar', False) and is_main_process()
    save_validation_references = True

    for epoch in range(start_epoch,total_epochs+1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_batches = (
            model.training_batches(train_loader, current_iter)
            if hasattr(model, 'training_batches') else train_loader
        )
        train_pbar = tqdm(
            epoch_batches,
            total=min(len(train_loader), total_iters - current_iter),
            desc=f'Epoch {epoch}/{total_epochs}',
            unit='batch',
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
            disable=not use_train_pbar)
        for train_data in train_pbar:
            data_timer.record()




            current_iter += 1
            if current_iter > total_iters:
                break


            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            model.feed_data(train_data)
            model.optimize_parameters(current_iter)


            current_optimizer_iter = model.get_optimizer_iter(current_iter)
            iter_timer.record()
            if use_train_pbar:
                train_pbar.set_postfix({
                    'iter': f'{current_optimizer_iter}/{total_optim_steps}',
                    'lr': f'{model.get_current_learning_rate()[0]:.3e}',
                    'time': f'{iter_timer.get_current_time():.3f}',
                    'data': f'{data_timer.get_current_time():.3f}',
                })

            if current_iter % actual_print_freq == 0:
                log_vars = {'epoch': epoch, 'iter': current_optimizer_iter}


                log_vars.update({
                    'time': iter_timer.get_avg_time() * accumulation_steps,
                    'data_time': data_timer.get_avg_time() * accumulation_steps,
                })

                results = {'lrs': model.get_current_learning_rate()}
                results.update(model.get_current_log())
                log_vars['results'] = results
                msg_logger(log_vars)

                train_csv_row = {
                    'epoch': epoch,
                    'iter': current_optimizer_iter,
                    'lr': model.get_current_learning_rate()[0],
                }
                train_csv_row.update(model.get_current_log())
                if train_csv_logger is not None:
                    train_csv_logger.write(train_csv_row)


            if current_iter % actual_save_freq == 0:


                logger.info('Saving checkpoint.')
                if not debug:
                    saved_path=model.save(epoch, current_iter, display_iter=current_optimizer_iter)
                    logger.info('Saving checkpoint to {}'.format(saved_path))
                else:
                    logger.info("Debug mode")


            validation = validation_kind(current_iter, accumulation_steps, val_freq_optim_steps,
                                         full_validation_updates, total_optim_steps, debug)
            if validation is not None:
                full_validation = validation == 'full'
                logger.info(f'[epoch:{epoch:3d}, iter:{current_optimizer_iter:8,d}] Starting {validation} validation.')

                for val_loader in val_loaders:
                    val_results = model.validation(
                        val_loader,
                        current_optimizer_iter,
                        wandb_logger,
                        True if full_validation else opt['val']['save_img'],
                        dataset_name=f'{val_loader.dataset.name}_{validation}',
                        max_samples=None if full_validation else opt['val'].get('max_samples'),
                        save_reference_img=save_validation_references)
                    if val_csv_logger is not None and val_results is not None:
                        val_csv_logger.write({'iter': current_optimizer_iter, **val_results})
                save_validation_references = False

            data_timer.start()
            iter_timer.start()




    consumed_time = str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')


    if not debug and opt.get('val') is not None and val_loaders:
        logger.info(
            f'[iter:{current_optimizer_iter:8,d}] Starting final full validation.'
        )
        for val_loader in val_loaders:
            val_results = model.validation(
                val_loader,
                current_optimizer_iter,
                wandb_logger,
                True,
                dataset_name=f'{val_loader.dataset.name}_full',
                max_samples=None)
            if val_csv_logger is not None and val_results is not None:
                val_csv_logger.write({'iter': current_optimizer_iter, **val_results})






    if train_csv_logger is not None:
        train_csv_logger.flush()
    if val_csv_logger is not None:
        val_csv_logger.flush()

    if wandb_logger is not None:
        wandb_logger.finish()

if __name__ == '__main__':
    train_pipeline()
