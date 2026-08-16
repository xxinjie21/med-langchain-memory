"""``domain/audit.py`` 的单元测试：审计事件模型与落盘接口。

覆盖：事件模型构造/校验、序列化往返、工厂函数、抽象接口约束、
内存与文件两类落盘实现（正向 + 边界/异常）。
"""

from __future__ import annotations

import threading
import uuid

import pytest
from pydantic import ValidationError

from med_langchain_memory.domain import (
    AuditAction,
    AuditEvent,
    AuditSink,
    AuditStatus,
    FileAuditSink,
    InMemoryAuditSink,
    make_audit_event,
)
from med_langchain_memory.exceptions import AuditSinkError, MedMemoryError


# --------------------------------------------------------------------------- #
# AuditEvent 模型
# --------------------------------------------------------------------------- #
class TestAuditEventModel:
    def test_construct_full_fields(self) -> None:
        """正向：完整字段构造，默认值正确。"""
        event = AuditEvent(
            action=AuditAction.WRITE,
            actor="doctor:1024",
            session_id="s-1",
            tenant_id="t-1",
            dept_id="d-1",
            target="message:m-1",
            metadata={"k": "v"},
        )
        assert event.action is AuditAction.WRITE
        assert event.actor == "doctor:1024"
        assert event.status is AuditStatus.SUCCESS  # 默认成功
        assert event.session_id == "s-1"
        assert event.occurred_at > 0
        # event_id 为合法 UUIDv7 字符串
        assert uuid.UUID(event.event_id)

    def test_construct_invalid_action_rejected(self) -> None:
        """边界：非法 action 字符串触发校验错误。"""
        with pytest.raises(ValidationError):
            AuditEvent(action="not-an-action", actor="system")

    def test_construct_invalid_session_id_rejected(self) -> None:
        """边界：含非法字符的 session_id 触发校验错误。"""
        with pytest.raises(ValidationError):
            AuditEvent(action=AuditAction.READ, actor="x", session_id="bad:id")

    def test_construct_empty_actor_rejected(self) -> None:
        """边界：空 actor 触发校验错误。"""
        with pytest.raises(ValidationError):
            AuditEvent(action=AuditAction.READ, actor="")

    def test_frozen_immutable(self) -> None:
        """边界：冻结模型不可原地修改。"""
        event = AuditEvent(action=AuditAction.READ, actor="system")
        with pytest.raises(ValidationError):  # 冻结模型不可修改
            event.actor = "other"

    def test_serialization_roundtrip_equal(self) -> None:
        """正向：JSON 序列化往返后与原事件等价。"""
        event = AuditEvent(
            action=AuditAction.DELETE,
            actor="patient:9",
            status=AuditStatus.FAILURE,
            session_id="s-2",
            error="boom",
            metadata={"trace": "abc"},
        )
        restored = AuditEvent.model_validate_json(event.model_dump_json())
        assert restored == event
        assert restored.status is AuditStatus.FAILURE
        assert restored.error == "boom"


# --------------------------------------------------------------------------- #
# 工厂函数
# --------------------------------------------------------------------------- #
class TestMakeAuditEvent:
    def test_factory_accepts_strings(self) -> None:
        """正向：action/status 接受字符串，字段被正确解析。"""
        event = make_audit_event("migrate", "service:migrator", session_id="s-3")
        assert event.action is AuditAction.MIGRATE
        assert event.status is AuditStatus.SUCCESS
        assert event.actor == "service:migrator"
        assert event.session_id == "s-3"

    def test_factory_empty_actor_raises(self) -> None:
        """边界：空 actor 抛 ValueError。"""
        with pytest.raises(ValueError):
            make_audit_event(AuditAction.READ, "")

    def test_factory_metadata_default_empty(self) -> None:
        """正向：未传 metadata 时默认为空字典而非 None。"""
        event = make_audit_event("read", "system")
        assert event.metadata == {}


# --------------------------------------------------------------------------- #
# 抽象接口
# --------------------------------------------------------------------------- #
class TestAuditSinkABC:
    def test_cannot_instantiate_abstract(self) -> None:
        """边界：抽象基类不可直接实例化。"""
        with pytest.raises(TypeError):
            AuditSink()  # type: ignore[abstract]

    def test_is_med_memory_error_hierarchy(self) -> None:
        """正向：审计异常继承自 MedMemoryError（统一捕获）。"""
        assert issubclass(AuditSinkError, MedMemoryError)


# --------------------------------------------------------------------------- #
# 内存落盘
# --------------------------------------------------------------------------- #
class TestInMemoryAuditSink:
    def test_record_and_len(self) -> None:
        """正向：记录后长度与内容正确。"""
        sink = InMemoryAuditSink()
        sink.record(make_audit_event("read", "system"))
        sink.record(make_audit_event("write", "doctor:1", session_id="s-1"))
        assert len(sink) == 2
        assert sink.events[0].action is AuditAction.READ

    def test_events_returns_copy(self) -> None:
        """正向：events 返回快照副本，外部修改不污染内部。"""
        sink = InMemoryAuditSink()
        sink.record(make_audit_event("read", "system"))
        snapshot = sink.events
        snapshot.clear()
        assert len(sink) == 1  # 内部未被改动

    def test_record_many_default(self) -> None:
        """正向：抽象基类默认批量实现逐条写入。"""
        sink = InMemoryAuditSink()
        events = [
            make_audit_event("read", "a"),
            make_audit_event("write", "b"),
        ]
        sink.record_many(events)
        assert len(sink) == 2

    def test_clear(self) -> None:
        """边界：clear 后长度归零。"""
        sink = InMemoryAuditSink()
        sink.record(make_audit_event("read", "system"))
        sink.clear()
        assert len(sink) == 0

    def test_concurrent_record_thread_safe(self) -> None:
        """边界：多线程并发写入不丢事件。"""
        sink = InMemoryAuditSink()

        def worker() -> None:
            for _ in range(100):
                sink.record(make_audit_event("read", "t"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(sink) == 800


# --------------------------------------------------------------------------- #
# 文件落盘（JSONL）
# --------------------------------------------------------------------------- #
class TestFileAuditSink:
    def test_record_then_read_roundtrip(self, tmp_path) -> None:
        """正向：写入后读取得到等价事件列表（按落盘顺序）。"""
        log = tmp_path / "audit.log"
        sink = FileAuditSink(log)
        e1 = make_audit_event("write", "doctor:1", session_id="s-1")
        e2 = make_audit_event("delete", "patient:2", status="failure", error="x")
        sink.record(e1)
        sink.record(e2)

        loaded = FileAuditSink(log).read()
        assert loaded == [e1, e2]
        assert loaded[1].status is AuditStatus.FAILURE

    def test_append_across_instances(self, tmp_path) -> None:
        """正向：不同实例写入同一文件，内容追加不覆盖。"""
        log = tmp_path / "audit.log"
        FileAuditSink(log).record(make_audit_event("read", "a"))
        FileAuditSink(log).record(make_audit_event("read", "b"))
        assert len(FileAuditSink(log).read()) == 2

    def test_read_missing_file_returns_empty(self, tmp_path) -> None:
        """边界：文件不存在时 read 返回空列表。"""
        sink = FileAuditSink(tmp_path / "missing" / "audit.log")
        assert sink.read() == []

    def test_record_to_directory_raises(self, tmp_path) -> None:
        """边界：路径为目录时写入触发 AuditSinkError。"""
        directory = tmp_path / "subdir"
        directory.mkdir()
        sink = FileAuditSink(directory)
        with pytest.raises(AuditSinkError):
            sink.record(make_audit_event("read", "system"))

    def test_parent_is_file_raises(self, tmp_path) -> None:
        """边界：父路径为文件时构造触发 AuditSinkError。"""
        not_a_dir = tmp_path / "afile"
        not_a_dir.write_text("x")
        with pytest.raises(AuditSinkError):
            FileAuditSink(not_a_dir / "log")
