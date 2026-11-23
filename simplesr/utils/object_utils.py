import importlib
from functools import partial
import os
from typing import Any, Mapping, Sequence

def get_object_path(obj):
    return obj.__module__ + "." + obj.__class__.__name__


def get_object_from_path(obj_path):
    module_path, object_name = obj_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"{e} 无法导入模块 {module_path}")
    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(f"{e} 模块 {module_path} 中不存在 {object_name} 的对象")
    return obj


def instantiate_object(class_path, *args, **kwargs):
    cls = get_object_from_path(class_path)
    if not isinstance(cls, type):
        raise TypeError(f"{class_path} 不是一个类，而是 {type(cls)}")
    try:
        instance = cls(*args, **kwargs)
        return instance
    except Exception as e:
        raise TypeError(f"实例化类 {class_path} 失败: {e}")


def partial_function(func, *args, **kwargs):
    if isinstance(func, str):
        func = get_object_from_path(func)
    if not callable(func):
        raise TypeError(f"{func}不是可调用对象")
    try:
        return partial(func, *args, **kwargs)
    except Exception as e:
        raise TypeError(f"partial 失败: {e}")


def instantiate_model_from_cfg(model_config: Mapping[str, Any]):
    if "model_path" in model_config:
        model_path = model_config["model_path"]
    elif "path" in model_config:
        model_path = model_config["path"]
    else:
        raise KeyError("配置文件中缺少 model_path 或 path")

    kwargs = model_config.get("model_kwargs") or {}
    return instantiate_object(model_path, **kwargs)

def get_by_dotted_key(config: Mapping[str, Any], key: str) -> Any:
    current: Any = config

    for part in key.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(f"无法继续索引 '{part}'，当前对象不是字典: {current}")

        if part not in current:
            raise KeyError(f"配置中不存在 key: {key}")

        current = current[part]

    return current


def instantiate_from_config(
    config: Mapping[str, Any],
    *,
    path_keys: Sequence[str] = (
        "target",
        "path",
        "object_path",
        "class_path",
        "model_path",
    ),
    kwargs_keys: Sequence[str] = (
        "params",
        "kwargs",
        "model_kwargs",
        "init_args",
    ),
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if not isinstance(config, Mapping):
        raise TypeError(f"config 必须是字典类型，但得到: {type(config)}")

    object_path = None
    for path_key in path_keys:
        if path_key in config:
            object_path = config[path_key]
            break

    if object_path is None:
        raise KeyError(
            f"配置中缺少对象路径字段，支持的字段包括: {tuple(path_keys)}"
        )

    kwargs: dict[str, Any] = {}
    for kwargs_key in kwargs_keys:
        if kwargs_key in config:
            raw_kwargs = config[kwargs_key]
            if raw_kwargs is None:
                raw_kwargs = {}

            if not isinstance(raw_kwargs, Mapping):
                raise TypeError(
                    f"'{kwargs_key}' 必须是字典类型，但得到: {type(raw_kwargs)}"
                )

            kwargs.update(raw_kwargs)
            break

    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    return instantiate_object(object_path, **kwargs)



