import datetime as dt
import logging
import time
import os
import sys
import csv
from typing import Any
from pathlib import Path

from .distributed_utils import get_rank, master_only,is_main_process

initialized_logger = {}
_root_logger = None

class AvgTimer():

    def __init__(self, window=200):
        self.window = window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.perf_counter()

    def record(self):
        self.count += 1
        self.toc = time.perf_counter()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time

        self.avg_time = self.total_time / self.count


        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.perf_counter()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time




class TrainMessageLogger():

    def __init__(self, opt, start_iter=1, logger=None, wandb_logger=None):
        self.exp_name = opt['exp_name']



        self.max_iters = opt['train']['total_iter']
        self.wandb_logger = wandb_logger
        self.start_time = time.perf_counter()
        self.start_iter=start_iter
        self.logger=logger if logger else get_root_logger()

    def reset_start_time(self):
        self.start_time = time.perf_counter()

    @master_only
    def __call__(self, log_dict):

        epoch = log_dict.get('epoch')
        current_iter = log_dict.get('iter')

        message = (f'[epoch:{epoch:3d}, iter:{current_iter:8,d}]')


        if 'time' in log_dict.keys():
            iter_time = log_dict.get('time')
            data_time = log_dict.get('data_time')

            total_time = time.perf_counter() - self.start_time



            completed_iters = max(1, current_iter - self.start_iter)
            time_sec_avg = total_time / completed_iters

            eta_sec = time_sec_avg * max(0, self.max_iters - current_iter)
            eta_str = str(dt.timedelta(seconds=int(eta_sec)))
            elapsed_str = str(dt.timedelta(seconds=int(total_time)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f}s ({data_time:.3f}s), elapsed: {elapsed_str}] '


        results=log_dict.get('results')

        if self.wandb_logger is not None:
            self.wandb_logger.log(results,step=current_iter)

        results_str=format_kwargs(**results)
        message+=results_str


        self.logger.info(message)


@master_only
def init_wandb_logger(
    exp_name,
    wandb_opt,
    exp_opt,
    save_dir,
):
    import wandb
    logger = get_root_logger()


    name=exp_name
    wandb_opt = wandb_opt

    project = wandb_opt.get("project")
    if project is None:
        raise ValueError("['wandb']['project'] must be specified.")

    mode = wandb_opt.get("mode", "online")
    tags = wandb_opt.get("tags", None)



    resume_id = wandb_opt.get("resume_id", None)

    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb_opt["id"] = wandb_id

    run=wandb.init(id=wandb_id,
               resume=resume,
               name=name,
               config=exp_opt,
               project=project,
               tags=tags,
               mode=mode,
               dir=save_dir,
               )

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')
    return run

def get_logger(logger_name='basic_logger', log_level=logging.INFO, log_file_path=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger


    format_str = "[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False

    rank=get_rank()
    if rank != 0:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)

    if log_file_path is not None:
        logger.setLevel(log_level)

        file_handler = logging.FileHandler(log_file_path, 'a',encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def init_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file_path=None):
    global _root_logger
    if _root_logger is not None:
        raise RuntimeError('Root logger has already been initialized.')
    if logger_name in initialized_logger:
        raise RuntimeError(f'Logger {logger_name!r} was initialized before the root logger.')
    _root_logger = get_logger(logger_name, log_level, log_file_path)
    return _root_logger


def get_root_logger():
    if _root_logger is None:
        raise RuntimeError('Root logger is not initialized. Call init_root_logger() first.')
    return _root_logger

def format_kwargs(**kwargs):
    str_parts=[]
    str_space=" "*2

    for key, value in kwargs.items():
        if hasattr(value, "item"):
                value = value.item()

        val_str = str(value)
        str_parts.append(f"{key}:{val_str}")
    return str_space.join(str_parts)

def _to_scalar(self,value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class CSVLogger:




    def __init__(self,csv_file,flush_freq):

        self.csv_file = Path(csv_file)
        self.buffer: list[dict[str, Any]] = []
        self.flush_freq = flush_freq
        self.fieldnames = None

    @master_only
    def write(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())


        if list(row.keys()) != self.fieldnames:
            raise ValueError(
                "CSV row keys do not match fieldnames.\n"
                f"Expected: {self.fieldnames}\n"
                f"Got: {list(row.keys())}"
            )

        self.buffer.append(row)

        if len(self.buffer) >= self.flush_freq:
            self.flush()

    @master_only
    def flush(self) -> None:
        if not self.buffer:
            return

        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_file.exists()

        with self.csv_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.buffer)
        self.buffer.clear()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




























