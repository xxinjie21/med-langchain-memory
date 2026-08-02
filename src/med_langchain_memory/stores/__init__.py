"""存储适配层：统一的医疗会话历史抽象、工厂注册器与各存储引擎实现。

导入本包即完成内置存储适配器的注册（当前为 ``memory``），
上层可直接通过 ``StoreFactory.create("memory", ...)`` 取用。
"""

from __future__ import annotations

from .base import (
    MED_ROLE_KEY,
    MedChatMessageHistory,
    from_langchain_message,
    to_langchain_message,
)
from .factory import StoreConfig, StoreFactory
from .memory_store import InMemoryMedHistory

__all__ = [
    "MED_ROLE_KEY",
    "InMemoryMedHistory",
    "MedChatMessageHistory",
    "StoreConfig",
    "StoreFactory",
    "from_langchain_message",
    "to_langchain_message",
]
