"""会话生命周期层：跨存储迁移、TTL 归档、快照与合规保留。

本模块当前实现跨存储迁移器 :class:`Migrator`，后续迭代将补充
TTL 归档调度、会话快照与软删除保留期等能力。本层不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from .migrator import (
    MigrationCursor,
    MigrationResult,
    Migrator,
    migrate_session,
)

__all__ = [
    "MigrationCursor",
    "MigrationResult",
    "Migrator",
    "migrate_session",
]
