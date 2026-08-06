"""存储适配层：统一的医疗会话历史抽象、工厂注册器与各存储引擎实现。

导入本包即完成内置存储适配器的注册：``memory`` 与 ``file`` 始终可用；
``redis`` 依赖可选包 ``redis``，未安装时静默跳过注册，其余后端不受影响。
上层可直接通过 ``StoreFactory.create("memory", ...)`` 取用。
"""

from __future__ import annotations

import contextlib

from .base import (
    MED_ROLE_KEY,
    MedChatMessageHistory,
    from_langchain_message,
    to_langchain_message,
)
from .factory import StoreConfig, StoreFactory
from .file_lock import FileLock
from .file_store import FileFormat, FileMedHistory
from .memory_store import InMemoryMedHistory

with contextlib.suppress(ImportError):  # redis 为可选依赖，缺失时不注册该后端
    from .redis_store import RedisMedHistory

__all__ = [
    "MED_ROLE_KEY",
    "FileFormat",
    "FileLock",
    "FileMedHistory",
    "InMemoryMedHistory",
    "MedChatMessageHistory",
    "RedisMedHistory",
    "StoreConfig",
    "StoreFactory",
    "from_langchain_message",
    "to_langchain_message",
]
