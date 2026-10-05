"""存储适配层：统一的医疗会话历史抽象、工厂注册器、历史解析器与各存储引擎实现。

导入本包即完成内置存储适配器的注册：``memory`` 与 ``file`` 始终可用；
``redis`` / ``redis-cluster`` 依赖可选包 ``redis``、``mysql`` 分表后端依赖可选包
``SQLAlchemy``、``elasticsearch`` 归档依赖可选包 ``elasticsearch``，
缺失时静默跳过，其余后端不受影响。
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
from .history_resolver import HistoryResolver, StoreFactoryHistoryResolver
from .memory_store import InMemoryMedHistory
from .message_repository import InMemoryMessageRepository, MessageRepository
from .session_repository import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    InMemorySessionRepository,
    SessionRepository,
    SessionScope,
    validate_pagination,
)

with contextlib.suppress(ImportError):  # redis 为可选依赖，缺失时不注册 redis 后端
    from .redis_cluster_store import RedisClusterMedHistory, build_cluster_client
    from .redis_store import ExpiryCallback, RedisMedHistory

with contextlib.suppress(ImportError):  # SQLAlchemy 为可选依赖，缺失时不注册 mysql 后端
    from .mysql_shard_router import DEFAULT_ROUTER, ShardRouter
    from .mysql_store import DEFAULT_MYSQL_URL, MySQLMedHistory

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
    "DEFAULT_MYSQL_URL",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_ROUTER",
    "MAX_PAGE_SIZE",
    "MED_ROLE_KEY",
    "EsArchiveMedHistory",
    "ExpiryCallback",
    "FileFormat",
    "FileLock",
    "FileMedHistory",
    "HistoryResolver",
    "InMemoryMedHistory",
    "InMemoryMessageRepository",
    "InMemorySessionRepository",
    "MedChatMessageHistory",
    "MessageRepository",
    "MySQLMedHistory",
    "RedisClusterMedHistory",
    "RedisMedHistory",
    "SessionRepository",
    "SessionScope",
    "ShardRouter",
    "StoreConfig",
    "StoreFactory",
    "StoreFactoryHistoryResolver",
    "build_cluster_client",
    "build_index_template",
    "from_langchain_message",
    "monthly_index",
    "search_archive",
    "to_langchain_message",
    "validate_pagination",
]
