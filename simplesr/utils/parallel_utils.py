from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any, List, Optional, Sequence, Dict, Tuple, Union

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os



@dataclass
class ParallelConfig:
    mode: str = "process"
    max_workers: Optional[int] = None
    show_progress: bool = True
    desc: str = "Working"
    total: Optional[int] = None
    chunksize: int = 1



def _get_executor(mode: str):
    mode = mode.lower()
    if mode in ("thread", "threads", "t"):
        return ThreadPoolExecutor
    if mode in ("process", "proc", "p"):
        return ProcessPoolExecutor
    raise ValueError(f"mode must be 'thread' or 'process', got: {mode}")


def _default_workers(mode: str) -> int:
    cpu = os.cpu_count() or 1

    return cpu * 4 if mode.lower().startswith("thread") else cpu


def _infer_total(it: Any, fallback: Optional[int] = None) -> Optional[int]:
    try:
        return len(it)
    except Exception:
        return fallback


def run_map(
    fn: Callable[..., Any],
    *iterables: Iterable[Any],
    cfg: ParallelConfig = ParallelConfig(),
) -> List[Any]:
    if cfg.max_workers is None:
        cfg.max_workers = _default_workers(cfg.mode)

    total = cfg.total
    if total is None and iterables:
        total = _infer_total(iterables[0], None)

    Executor = _get_executor(cfg.mode)

    with Executor(max_workers=cfg.max_workers) as ex:

        try:
            it = ex.map(fn, *iterables, chunksize=cfg.chunksize)
        except TypeError:
            it = ex.map(fn, *iterables)

        if cfg.show_progress:
            return list(tqdm(it, total=total, desc=cfg.desc))
        return list(it)


def run_submit(
    fn: Callable[..., Any],
    tasks: Sequence[Union[Tuple[Any, ...], Dict[str, Any]]],
    cfg: ParallelConfig = ParallelConfig(),
) -> List[Any]:
    if cfg.max_workers is None:
        cfg.max_workers = _default_workers(cfg.mode)

    total = cfg.total
    if total is None:
        total = len(tasks)

    Executor = _get_executor(cfg.mode)

    with Executor(max_workers=cfg.max_workers) as ex:
        futures = []
        for i, t in enumerate(tasks):
            if isinstance(t, dict):
                fut = ex.submit(fn, **t)
            elif isinstance(t, tuple):
                fut = ex.submit(fn, *t)
            else:

                raise TypeError(
                    f"tasks[{i}] is of type {type(t).__name__}, but expected 'tuple' or 'dict'.\n"
                    f" - If you meant to pass a single argument (even if it's a list), wrap it in a tuple: `(arg,)`.\n"
                    f" - If you meant to pass multiple positional arguments, use a tuple: `(arg1, arg2)`.\n"
                    f" - If you accidentally passed the raw data list instead of a list of task tuples, "
                    f"try `tasks=[(x,) for x in data_list]`."
                )
            futures.append(fut)

        if not cfg.show_progress:

            results = [f.result() for f in futures]

        else:

            for _ in tqdm(as_completed(futures), total=total, desc=cfg.desc):
                pass



            results = [f.result() for f in futures]

        return results


if __name__ == '__main__':

    mode = "thread"
    cfg=ParallelConfig(mode=mode, max_workers=os.cpu_count(), show_progress=True, desc="")



    pass
