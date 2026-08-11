"""跨存储迁移工具：将单个会话从源存储批量迁移到目标存储。

提供可断点续传的游标（:class:`MigrationCursor`）与迁移后完整性校验
（:meth:`Migrator.migrate` 默认校验源/目标消息集合一致）。

迁移过程仅依赖 ``MedChatMessageHistory`` 公共接口，不触碰任何底层存储实现，
因此任意两个已注册后端之间均可互迁（内存→Redis、文件→ES 等）。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import (
    StorageError,
    TenantIsolationError,
    ValidationError,
)
from med_langchain_memory.stores.base import MedChatMessageHistory


@dataclass
class MigrationCursor:
    """迁移断点续传游标。

    记录已处理（尝试写入）的源消息条数作为续传偏移量，可序列化为 JSON
    持久化到磁盘，使迁移在进程崩溃或目标写入失败后能从断点恢复。

    Attributes:
        session_key: 被迁移会话的统一存储键。
        migrated: 已处理的源消息条数（续传偏移量）。
        total: 源会话消息总数，首次迁移时解析。
        finished: 迁移是否已完成。
    """

    session_key: str
    migrated: int = 0
    total: int = 0
    finished: bool = False

    def advance(self, count: int) -> None:
        """标记 ``count`` 条源消息已成功处理，推进游标。"""
        self.migrated += count

    def to_dict(self) -> dict[str, Any]:
        """序列化为可落盘的字典。"""
        return {
            "session_key": self.session_key,
            "migrated": self.migrated,
            "total": self.total,
            "finished": self.finished,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationCursor:
        """从字典还原游标；缺失字段按默认值补齐，保证向前兼容。"""
        return cls(
            session_key=str(data["session_key"]),
            migrated=int(data.get("migrated", 0)),
            total=int(data.get("total", 0)),
            finished=bool(data.get("finished", False)),
        )

    def save(self, path: str | Path) -> None:
        """将游标持久化到 JSON 文件（先写临时文件再 rename，保证原子性）。"""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> MigrationCursor:
        """从 JSON 文件加载游标；文件不存在时抛出 :class:`FileNotFoundError`。"""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        """判断游标文件是否存在。"""
        return Path(path).is_file()


@dataclass
class MigrationResult:
    """单次迁移执行结果摘要。

    Attributes:
        session_key: 被迁移会话的统一存储键。
        source_total: 源会话消息总数。
        target_total: 迁移后目标会话消息总数。
        newly_written: 本次实际新写入目标的消息条数。
        skipped: 因目标已存在而跳过的消息条数（幂等续传）。
        verified: 完整性校验结果；``None`` 表示未启用校验。
        finished: 迁移是否全部完成。
        errors: 校验或执行中记录的问题描述。
    """

    session_key: str
    source_total: int = 0
    target_total: int = 0
    newly_written: int = 0
    skipped: int = 0
    verified: bool | None = None
    finished: bool = False
    errors: list[str] = field(default_factory=list)


class Migrator:
    """将单个会话从源存储批量迁移到目标存储，支持断点续传与完整性校验。

    源与目标必须属于同一会话命名空间（``storage_key`` 一致），否则拒绝迁移。
    续传机制：每次成功写入一个批次后推进 :class:`MigrationCursor` 并持久化；
    迁移前先扫描目标已存在消息的 ``message_id``，已存在者直接跳过，从而保证
    失败后重跑不会产生重复数据。
    """

    def __init__(
        self,
        source: MedChatMessageHistory,
        target: MedChatMessageHistory,
        *,
        batch_size: int = 100,
        verify: bool = True,
        cursor_path: str | Path | None = None,
    ) -> None:
        """初始化迁移器。

        Args:
            source: 源存储（提供全量消息）。
            target: 目标存储（接收迁移消息）。
            batch_size: 单批次写入消息条数，必须为正整数。
            verify: 迁移完成后是否做源/目标一致性校验。
            cursor_path: 游标持久化文件路径；提供则启用断点续传。

        Raises:
            ValidationError: ``batch_size`` 非正数时。
            TenantIsolationError: 源与目标不属于同一会话命名空间时。
        """
        if batch_size <= 0:
            raise ValidationError("batch_size must be a positive integer")
        if source.storage_key != target.storage_key:
            raise TenantIsolationError(
                f"cannot migrate across namespaces: {source.storage_key} -> {target.storage_key}"
            )
        self.source = source
        self.target = target
        self.batch_size = batch_size
        self.verify = verify
        self.cursor_path = Path(cursor_path) if cursor_path is not None else None

    def _load_cursor(self) -> MigrationCursor:
        """加载既有游标（若存在且未损坏），否则新建指向当前会话的游标。"""
        if self.cursor_path is not None and MigrationCursor.exists(self.cursor_path):
            try:
                return MigrationCursor.load(self.cursor_path)
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                # 游标损坏则从头开始，不阻断迁移
                pass
        return MigrationCursor(session_key=self.source.storage_key)

    def _save_cursor(self, cursor: MigrationCursor) -> None:
        """持久化游标（仅当配置了 ``cursor_path`` 时）。"""
        if self.cursor_path is not None:
            cursor.save(self.cursor_path)

    def migrate(self) -> MigrationResult:
        """执行迁移：分批写入 + 推进游标 + 可选校验。

        失败（目标写入抛错）时游标停留在出错批次之前，调用方可修复后
        再次调用 :meth:`migrate` 从断点续传。

        Returns:
            迁移结果摘要 :class:`MigrationResult`。

        Raises:
            StorageError: 目标存储写入失败且不可恢复时透传为存储错误。
        """
        cursor = self._load_cursor()
        source_msgs = self.source.get_med_messages()
        cursor.total = len(source_msgs)

        result = MigrationResult(session_key=self.source.storage_key)
        result.source_total = len(source_msgs)

        # 已存在于目标的消息直接跳过，保证幂等续传
        target_ids = {m.message_id for m in self.target.get_med_messages()}

        pending = source_msgs[cursor.migrated :]
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            to_write = [m for m in batch if m.message_id not in target_ids]
            result.skipped += len(batch) - len(to_write)
            if to_write:
                try:
                    self.target.add_med_messages(to_write)
                except Exception as exc:  # 写入失败：游标不推进，便于续传
                    result.errors.append(f"batch write failed: {exc}")
                    self._save_cursor(cursor)
                    raise StorageError(
                        f"migration aborted at batch {i // self.batch_size}: {exc}"
                    ) from exc
                result.newly_written += len(to_write)
                target_ids.update(m.message_id for m in to_write)
            cursor.advance(len(batch))
            self._save_cursor(cursor)

        cursor.finished = cursor.migrated >= cursor.total
        self._save_cursor(cursor)

        result.target_total = len(self.target.get_med_messages())
        result.finished = cursor.finished
        if self.verify:
            ok, errs = self._verify(source_msgs)
            result.verified = ok
            result.errors.extend(errs)
        return result

    def _verify(self, source_msgs: list[MedMessage]) -> tuple[bool, list[str]]:
        """校验目标与源消息集合（含关键字段）一致。

        比较消息 ``message_id`` 集合、总条数，并逐条核对 ``content`` /
        ``role`` / ``created_at`` / 命名空间等关键字段。

        Returns:
            ``(是否通过, 问题描述列表)``。
        """
        errors: list[str] = []
        target_msgs = self.target.get_med_messages()
        if len(target_msgs) != len(source_msgs):
            errors.append(f"count mismatch: source={len(source_msgs)} target={len(target_msgs)}")
            return False, errors
        src_by_id = {m.message_id: m for m in source_msgs}
        tgt_by_id = {m.message_id: m for m in target_msgs}
        if set(src_by_id) != set(tgt_by_id):
            errors.append("message id set mismatch between source and target")
            return False, errors
        for mid, src in src_by_id.items():
            tgt = tgt_by_id[mid]
            if (
                src.content != tgt.content
                or src.role != tgt.role
                or src.created_at != tgt.created_at
                or src.storage_key != tgt.storage_key
            ):
                errors.append(f"content mismatch for {mid}")
                return False, errors
        return True, errors


def migrate_session(
    source: MedChatMessageHistory,
    target: MedChatMessageHistory,
    *,
    batch_size: int = 100,
    verify: bool = True,
    cursor_path: str | Path | None = None,
) -> MigrationResult:
    """便捷函数：将单个会话从 ``source`` 迁移到 ``target``。

    Args:
        source: 源存储。
        target: 目标存储。
        batch_size: 单批次写入消息条数。
        verify: 是否做迁移后一致性校验。
        cursor_path: 游标持久化路径（启用断点续传）。

    Returns:
        迁移结果摘要。
    """
    return Migrator(
        source,
        target,
        batch_size=batch_size,
        verify=verify,
        cursor_path=cursor_path,
    ).migrate()
