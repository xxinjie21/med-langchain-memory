"""会话生命周期层：跨存储迁移、TTL 归档、快照、合规保留。

本模块实现跨存储迁移器 :class:`Migrator`、TTL 自动归档调度器
:class:`TtlArchiver`、会话快照备份/恢复 :class:`SessionSnapshotter` 与
合规保留期调度器 :class:`RetentionManager`（软删除标记 + 到期物理清理）。
本层不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from .migrator import (
    MigrationCursor,
    MigrationResult,
    Migrator,
    migrate_session,
)
from .retention import (
    PurgeResult,
    RetentionManager,
    RetentionPolicy,
    RetentionReport,
    SessionRetentionResult,
    SoftDeleteResult,
    purge_expired_session,
    soft_delete_session,
)
from .snapshot import (
    DEFAULT_SCHEMA_VERSION,
    SessionSnapshotPackage,
    SessionSnapshotter,
    SnapshotSummary,
    restore_session,
    snapshot_session,
)
from .ttl_archiver import (
    ArchiveReport,
    ArchiveResult,
    TtlArchiver,
    archive_expired_session,
)

__all__ = [
    "MigrationCursor",
    "MigrationResult",
    "Migrator",
    "migrate_session",
    "PurgeResult",
    "RetentionManager",
    "RetentionPolicy",
    "RetentionReport",
    "SessionRetentionResult",
    "SoftDeleteResult",
    "purge_expired_session",
    "soft_delete_session",
    "DEFAULT_SCHEMA_VERSION",
    "SessionSnapshotPackage",
    "SessionSnapshotter",
    "SnapshotSummary",
    "restore_session",
    "snapshot_session",
    "ArchiveReport",
    "ArchiveResult",
    "TtlArchiver",
    "archive_expired_session",
]
