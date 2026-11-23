import functools
import os
import torch
import torch.distributed as dist
from typing import Any



def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(cfg):


    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:


        cfg["rank"] = int(os.environ["RANK"])
        cfg["world_size"] = int(os.environ["WORLD_SIZE"])
        cfg["local_rank"] = int(os.environ.get("LOCAL_RANK", 0))
        cfg["gpu"] = cfg["local_rank"]
    else:
        print("未使用分布式模式")
        cfg["rank"] = 0
        cfg["world_size"] = 1
        cfg["local_rank"] = 0
        cfg["gpu"] = 0

    cfg["distributed"] = cfg["world_size"] > 1
    if not cfg["distributed"]:
        print('分布式未初始化，仅使用单进程')
        return


    torch.cuda.set_device(cfg["local_rank"])



    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"

    print(f"| 初始化分布式 (rank {cfg['rank']}, local_rank {cfg['local_rank']}): env://", flush=True)



    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=cfg["world_size"],
        rank=cfg["rank"]
    )


    dist.barrier()


    setup_for_distributed(cfg["rank"] == 0)



    print("分布式训练初始化成功！")
    print(f"总进程数: {cfg['world_size']}, 全局 rank: {cfg['rank']}, 本地 rank: {cfg['local_rank']}")

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def reduce_dict(
    input_dict: dict[str, torch.Tensor | float | int],
    average: bool = True,
    to_float: bool = True,
) -> dict[str, torch.Tensor | float]:
    world_size = get_world_size()

    if world_size < 2:
        if to_float:
            return {
                k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                for k, v in input_dict.items()
            }
        return {
            k: v if torch.is_tensor(v) else torch.tensor(v)
            for k, v in input_dict.items()
        }

    if len(input_dict) == 0:
        return {}

    with torch.no_grad():
        names = sorted(input_dict.keys())

        values = []
        device = None


        for name in names:
            value = input_dict[name]
            if torch.is_tensor(value):
                device = value.device
                break


        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        for name in names:
            value = input_dict[name]

            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                value = value.detach()
                if value.ndim != 0:
                    value = value.mean()
                value = value.to(device=device, dtype=torch.float32)

            values.append(value)

        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)

        if average:
            values /= world_size

        if to_float:
            reduced_dict = {
                name: float(value.detach().cpu().item())
                for name, value in zip(names, values)
            }
        else:
            reduced_dict = {
                name: value
                for name, value in zip(names, values)
            }

    return reduced_dict


def all_gather_tensor(data: torch.Tensor) -> list[torch.Tensor]:
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    data_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(data_list, data)
    return data_list


def all_gather_object(data: Any) -> list[Any]:
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, data)
    return object_list

def save_weight_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper