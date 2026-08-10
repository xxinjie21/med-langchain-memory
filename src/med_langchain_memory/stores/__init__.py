"""存储适配层：统一的医疗会话历史抽象、工厂注册器与各存储引擎实现。

导入本包即完成内置存储适配器的注册：``memory`` 与 ``file`` 始终可用；
``redis`` 依赖可选包 ``redis``、MySQL 分表路由依赖可选包 ``SQLAlchemy``、
``elasticsearch`` 归档依赖可选包 ``elasticsearch``，缺失时静默跳过，其余后端不受影响。
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

with contextlib.suppress(ImportError):  # redis 为可选依赖，缺失时不注册 redis 后端
    from .redis_cluster_store import RedisClusterMedHistory, build_cluster_client
    from .redis_store import ExpiryCallback, RedisMedHistory

with contextlib.suppress(ImportError):  # SQLAlchemy 为可选依赖，缺失时不导出分表路由
    from .mysql_shard_router import DEFAULT_ROUTER, ShardRouter

with contextlib.suppress(ImportError):  # elasticsearch 为可选依赖，缺失时不注册归档后端
    from .es_store import (
        ARCHIVE_INDEX_PREFIX,
        ARCHIVE_TEMPLATE_NAME,
        EsArchiveMedHistory,
        build_index_template,
        monthly_index,
        search_archive,
    )

__all__ = [
    "ARCHIVE_INDEX_PREFIX",
    "ARCHIVE_TEMPLATE_NAME",
    "DEFAULT_ROUTER",
    "MED_ROLE_KEY",
    "EsArchiveMedHistory",
    "ExpiryCallback",
    "FileFormat",
    "FileLock",
    "FileMedHistory",
    "InMemoryMedHistory",
    "MedChatMessageHistory",
    "RedisClusterMedHistory",
    "RedisMedHistory",
    "ShardRouter",
    "StoreConfig",
    "StoreFactory",
    "build_cluster_client",
    "build_index_template",
    "from_langchain_message",
    "monthly_index",
    "search_archive",
    "to_langchain_message",
]
