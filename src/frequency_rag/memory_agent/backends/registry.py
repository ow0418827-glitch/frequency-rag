"""后端名称到实现的注册表；使用时才导入所选后端。"""
from __future__ import annotations

from importlib import import_module
from typing import Callable


BACKENDS = {
    "murag": ("frequency_rag.memory_agent.backends.murag", "MuRAGMemory"),
    "ngmemory": ("frequency_rag.memory_agent.backends.ngmemory", "NGMemory"),
    "augustus": ("frequency_rag.memory_agent.backends.augustus", "AugustusMemory"),
    "universalrag": ("frequency_rag.memory_agent.backends.universalrag", "UniversalRAGMemory"),
    "mem0": ("frequency_rag.memory_agent.backends.mem0", "Mem0Memory"),
}
_custom_factories: dict[str, Callable] = {}


def available_backends():
    return tuple(sorted(set(BACKENDS) | set(_custom_factories)))


def register_backend(name: str, factory: Callable):
    """扩展工厂签名为 (config, encoder, llm)，返回统一记忆接口的实例。"""
    if not isinstance(name, str) or not name.strip() or not callable(factory):
        raise ValueError("后端名称与工厂必须有效。")
    if name in available_backends():
        raise ValueError(f"记忆后端已注册：{name}")
    _custom_factories[name] = factory


def make_memory(config, encoder, llm):
    name = config["backend"]
    if name in _custom_factories:
        return _custom_factories[name](config, encoder, llm)
    if name not in BACKENDS:
        raise ValueError(f"未知记忆后端：{name}；可选值：{', '.join(available_backends())}")
    module, class_name = BACKENDS[name]
    cls = getattr(import_module(module), class_name)
    options = {"candidates": config.get("candidates", 10), "top_k": config.get("top_k", 3)}
    if name == "mem0":
        if "mem0" not in config:
            raise ValueError("事实记忆后端需要单独的 mem0 配置。")
        return cls(config["mem0"], llm, **options)
    return cls(encoder, llm, **options)
