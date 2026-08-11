"""跨存储迁移器 :class:`Migrator` 单元测试。

测试直接针对 ``MedChatMessageHistory`` 抽象接口：用隔离的 :class:`FakeHistory`
作为源/目标存储（各自持有独立数据，不按 storage_key 共享），覆盖批量迁移、
断点续传游标、幂等重跑与迁移后完整性校验等场景。无需真实中间件即可跑通。
"""

from __future__ import annotations

import pytest

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.exceptions import (
    StorageError,
    TenantIsolationError,
    ValidationError,
)
from med_langchain_memory.lifecycle import (
    MigrationCursor,
    MigrationResult,
    Migrator,
    migrate_session,
)
from med_langchain_memory.stores.base import MedChatMessageHistory

NAMESPACE = {
    "session_id": "s-1",
    "tenant_id": "hosp-a",
    "dept_id": "cardio",
    "patient_id": "p-1",
}


class FakeHistory(MedChatMessageHistory):
    """隔离的内存存储替身：每条实例持有独立数据，不按 storage_key 共享。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data: list[MedMessage] = []

    def _append(self, messages: list[MedMessage]) -> None:
        self._data.extend(messages)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        if limit is None:
            return list(self._data)
        return list(self._data[-limit:])

    def clear(self) -> None:
        self._data.clear()


class FlakyTarget(FakeHistory):
    """前 ``fail_times`` 次写入抛错，用于模拟目标写入中途失败。"""

    def __init__(self, *args, fail_times: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fail_remaining = fail_times

    def add_med_messages(self, messages) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise StorageError("simulated target failure")
        return super().add_med_messages(messages)


def make_messages(n: int, **overrides) -> list[MedMessage]:
    """构造 ``n`` 条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    role_cycle = [
        MessageRole.PATIENT,
        MessageRole.DOCTOR,
        MessageRole.ASSISTANT,
        MessageRole.SYSTEM,
    ]
    out: list[MedMessage] = []
    for i in range(n):
        out.append(
            MedMessage(
                session_id=NAMESPACE["session_id"],
                tenant_id=NAMESPACE["tenant_id"],
                dept_id=NAMESPACE["dept_id"],
                patient_id=NAMESPACE["patient_id"],
                role=role_cycle[i % len(role_cycle)],
                content=f"msg-{i}",
                **overrides,
            )
        )
    return out


def seed_source(n: int) -> FakeHistory:
    """构造含 ``n`` 条消息的源存储。"""
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(n))
    return src


# --------------------------------------------------------------------------- #
# 构造与校验
# --------------------------------------------------------------------------- #
def test_init_ok() -> None:
    m = Migrator(seed_source(3), FakeHistory(**NAMESPACE))
    assert m.batch_size == 100
    assert m.verify is True
    assert m.cursor_path is None


def test_init_batch_size_zero_raises() -> None:
    with pytest.raises(ValidationError):
        Migrator(seed_source(1), FakeHistory(**NAMESPACE), batch_size=0)


def test_init_batch_size_negative_raises() -> None:
    with pytest.raises(ValidationError):
        Migrator(seed_source(1), FakeHistory(**NAMESPACE), batch_size=-5)


def test_init_cross_namespace_raises() -> None:
    src = seed_source(1)
    other = FakeHistory(
        session_id="s-2",
        tenant_id="hosp-a",
        dept_id="cardio",
        patient_id="p-1",
    )
    with pytest.raises(TenantIsolationError):
        Migrator(src, other)


# --------------------------------------------------------------------------- #
# 迁移核心
# --------------------------------------------------------------------------- #
def test_migrate_copies_all_messages() -> None:
    src = seed_source(10)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt).migrate()
    assert len(tgt.get_med_messages()) == 10
    assert result.source_total == 10
    assert result.target_total == 10
    assert result.newly_written == 10
    assert result.verified is True


def test_migrate_preserves_fields() -> None:
    src = seed_source(5)
    original = src.get_med_messages()[2]
    tgt = FakeHistory(**NAMESPACE)
    Migrator(src, tgt).migrate()
    copied = {m.message_id: m for m in tgt.get_med_messages()}[original.message_id]
    assert copied.content == original.content
    assert copied.role == original.role
    assert copied.created_at == original.created_at
    assert copied.masked == original.masked
    assert copied.storage_key == original.storage_key


def test_migrate_empty_source() -> None:
    src = seed_source(0)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt).migrate()
    assert result.source_total == 0
    assert result.target_total == 0
    assert result.newly_written == 0
    assert result.finished is True
    assert result.verified is True


def test_migrate_idempotent_rerun() -> None:
    src = seed_source(8)
    tgt = FakeHistory(**NAMESPACE)
    m = Migrator(src, tgt)
    first = m.migrate()
    second = m.migrate()
    assert first.newly_written == 8
    assert second.newly_written == 0
    assert second.skipped == 8
    assert second.verified is True
    assert len(tgt.get_med_messages()) == 8


# --------------------------------------------------------------------------- #
# 批处理
# --------------------------------------------------------------------------- #
def test_batch_size_respected() -> None:
    src = seed_source(250)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt, batch_size=100).migrate()
    assert result.newly_written == 250
    assert result.target_total == 250


def test_batch_size_one() -> None:
    src = seed_source(5)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt, batch_size=1).migrate()
    assert result.newly_written == 5
    assert len(tgt.get_med_messages()) == 5


# --------------------------------------------------------------------------- #
# 游标
# --------------------------------------------------------------------------- #
def test_cursor_advance() -> None:
    c = MigrationCursor(session_key="k")
    c.advance(3)
    c.advance(2)
    assert c.migrated == 5


def test_cursor_roundtrip_dict() -> None:
    c = MigrationCursor(session_key="k", migrated=3, total=10, finished=False)
    c2 = MigrationCursor.from_dict(c.to_dict())
    assert c2.session_key == c.session_key
    assert c2.migrated == c.migrated
    assert c2.total == c.total
    assert c2.finished == c.finished


def test_cursor_from_dict_missing_fields_default() -> None:
    c = MigrationCursor.from_dict({"session_key": "k"})
    assert c.migrated == 0
    assert c.total == 0
    assert c.finished is False


def test_cursor_save_load_file(tmp_path) -> None:
    p = tmp_path / "cur.json"
    MigrationCursor(session_key="k", migrated=4, total=10, finished=False).save(p)
    assert p.is_file()
    loaded = MigrationCursor.load(p)
    assert loaded.migrated == 4
    assert loaded.total == 10


def test_cursor_load_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        MigrationCursor.load("/nonexistent/cursor.json")


def test_cursor_exists() -> None:
    assert MigrationCursor.exists("/nonexistent/cursor.json") is False


# --------------------------------------------------------------------------- #
# 断点续传
# --------------------------------------------------------------------------- #
def test_resume_after_failure() -> None:
    src = seed_source(10)
    tgt = FlakyTarget(**NAMESPACE, fail_times=1)
    m = Migrator(src, tgt, batch_size=3)
    with pytest.raises(StorageError):
        m.migrate()
    assert len(tgt.get_med_messages()) == 0
    result = m.migrate()
    assert result.newly_written == 10
    assert result.verified is True
    assert len(tgt.get_med_messages()) == 10


def test_resume_with_cursor_file(tmp_path) -> None:
    p = tmp_path / "cur.json"
    src = seed_source(12)
    tgt = FlakyTarget(**NAMESPACE, fail_times=1)
    m = Migrator(src, tgt, batch_size=5, cursor_path=p)
    with pytest.raises(StorageError):
        m.migrate()
    assert p.is_file()
    result = Migrator(src, tgt, batch_size=5, cursor_path=p).migrate()
    assert result.newly_written == 12
    assert result.finished is True
    assert MigrationCursor.load(p).finished is True


def test_resume_skips_already_written() -> None:
    src = seed_source(10)
    first3 = src.get_med_messages()[:3]
    tgt = FakeHistory(**NAMESPACE)
    tgt.add_med_messages(first3)
    result = Migrator(src, tgt).migrate()
    assert result.newly_written == 7
    assert result.skipped == 3
    assert result.verified is True
    assert len(tgt.get_med_messages()) == 10


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def test_verify_disabled() -> None:
    src = seed_source(6)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt, verify=False).migrate()
    assert result.verified is None
    assert result.newly_written == 6


def test_verify_detects_extra_in_target() -> None:
    src = seed_source(4)
    tgt = FakeHistory(**NAMESPACE)
    tgt.add_med_messages(
        [
            MedMessage(
                session_id=NAMESPACE["session_id"],
                tenant_id=NAMESPACE["tenant_id"],
                dept_id=NAMESPACE["dept_id"],
                patient_id=NAMESPACE["patient_id"],
                role=MessageRole.SYSTEM,
                content="extra",
            )
        ]
    )
    result = Migrator(src, tgt).migrate()
    assert result.verified is False
    assert any("count mismatch" in e for e in result.errors)


def test_verify_detects_content_mismatch() -> None:
    src = seed_source(3)
    original = src.get_med_messages()[0]
    tgt = FakeHistory(**NAMESPACE)
    spoof = original.model_copy(update={"content": "tampered"})
    tgt.add_med_messages([spoof])
    result = Migrator(src, tgt).migrate()
    assert result.verified is False
    assert any("content mismatch" in e for e in result.errors)


# --------------------------------------------------------------------------- #
# 便捷函数与健壮性
# --------------------------------------------------------------------------- #
def test_migrate_session_helper() -> None:
    src = seed_source(7)
    tgt = FakeHistory(**NAMESPACE)
    result = migrate_session(src, tgt)
    assert isinstance(result, MigrationResult)
    assert result.newly_written == 7
    assert result.verified is True


def test_cursor_corrupt_file_falls_back(tmp_path) -> None:
    p = tmp_path / "cur.json"
    p.write_text("{not valid json", encoding="utf-8")
    src = seed_source(5)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt, cursor_path=p).migrate()
    assert result.newly_written == 5
    assert result.verified is True


def test_migrate_large_scale() -> None:
    src = seed_source(1000)
    tgt = FakeHistory(**NAMESPACE)
    result = Migrator(src, tgt, batch_size=250).migrate()
    assert result.newly_written == 1000
    assert result.verified is True
    assert len(tgt.get_med_messages()) == 1000


def test_result_dataclass_defaults() -> None:
    r = MigrationResult(session_key="k")
    assert r.source_total == 0
    assert r.newly_written == 0
    assert r.verified is None
    assert r.errors == []
