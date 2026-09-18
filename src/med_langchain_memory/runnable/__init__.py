"""医疗增强 Runnable 层（LCEL 上下文工程）。

当前提供基于存储工厂注入的会话历史增强 Runnable，以及多租户/科室命名空间隔离
所需的身份上下文、命名空间值对象与越权守卫；后续迭代将叠加时序/Token 预算裁剪、
LLM 摘要压缩、并发会话锁与读写降级等能力。
"""

from __future__ import annotations

from .med_history_runnable import MedRunnableWithMessageHistory
from .tenant import (
    NAMESPACE_SEPARATOR,
    STORAGE_KEY_PREFIX,
    SessionNamespace,
    TenantContext,
    TenantGuard,
    build_namespace_key,
    parse_namespace_key,
)

__all__ = [
    "NAMESPACE_SEPARATOR",
    "STORAGE_KEY_PREFIX",
    "MedRunnableWithMessageHistory",
    "SessionNamespace",
    "TenantContext",
    "TenantGuard",
    "build_namespace_key",
    "parse_namespace_key",
]
