"""TTL 自动归档调度器 :mod:`ttl_archiver` 单元测试。

覆盖单会话归档（过期判定 / 状态流转 / 消息写入 / 删除热数据 / 幂等跳过）、
调度器批量扫描（混合结果 / 错误容忍 / 0 候选 / 构造校验）以及一处受
``ImportError`` 保护的 Elasticsearch 归档层集成用例（真实 ``EsArchiveMedHistory``
配合内存 ``FakeElasticsearch`` 客户端）。

仅依赖 ``MedChatMessageHistory`` 公共接口，无需任何真实中间件即可跑通；
热存储用依赖独立的 ``FakeTtlHistory``，归档层用实例隔离的 ``FakeArchiveHistory``。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest
from conftest import NAMESPACE, make_messages

from med_langchain_memory.domain.session import SessionStatus
from med_langchain_memory.exceptions import StorageError
from med_langchain_memory.lifecycle import (
    ArchiveResult,
    TtlArchiver,
    archive_expired_session,
)
from med_langchain_memory.stores.base import MedChatMessageHistory


class FakeTtlHistory(MedChatMessageHistory):
    """支持原生 TTL 的内存热存储替身（实例间数据隔离）。"""

    #: 本替身支持原生 TTL，``is_expired`` 走基类 ``updated_at + ttl`` 逻辑。
    supports_ttl: ClassVar[bool] = True

    def __init__(self, *args, ttl_seconds: int | None = None, **kwargs) -> None:
        super().__init__(*args, ttl_seconds=ttl_seconds, **kwargs)
        self._data: list = []

    def _apply_ttl(self, ttl_seconds: int) -> None:
        """原生 TTL 下发钩子（本替身为空操作，过期判定由基类负责）。"""

    def _append(self, messages: list) -> None:
        self._data.extend(messages)

    def _read(self, limit: int | None = None) -> list:
        if limit is None:
            return list(self._data)
        return list(self._data[-limit:])

    def clear(self) -> None:
        self._data.clear()


class FakeArchiveHistory(MedChatMessageHistory):
    """实例隔离的内存归档层替身，仅实现 ``add_med_messages`` / 读取 / 清理。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data: list = []

    def _append(self, messages: list) -> None:
        self._data.extend(messages)

    def _read(self, limit: int | None = None) -> list:
        if limit is None:
            return list(self._data)
        return list(self._data[-limit:])

    def clear(self) -> None:
        self._data.clear()


def _make_hot(ttl_seconds: int = 60, *, n_msgs: int = 3) -> FakeTtlHistory:
    """构造带消息的过期热存储句柄（具体是否过期由调用方传入 ``now_ms`` 决定）。"""
    hot = FakeTtlHistory(
        session_id=NAMESPACE["session_id"],
        tenant_id=NAMESPACE["tenant_id"],
        dept_id=NAMESPACE["dept_id"],
        patient_id=NAMESPACE["patient_id"],
        ttl_seconds=ttl_seconds,
    )
    hot.add_med_messages(make_messages(n_msgs))
    return hot


def _make_archive() -> FakeArchiveHistory:
    return FakeArchiveHistory(
        session_id=NAMESPACE["session_id"],
        tenant_id=NAMESPACE["tenant_id"],
        dept_id=NAMESPACE["dept_id"],
        patient_id=NAMESPACE["patient_id"],
    )


def _expiry_now(hot: FakeTtlHistory) -> int:
    """返回足够靠后的时间戳，使 ``hot`` 必然被判过期。"""
    return hot.session_meta.updated_at + hot.ttl_seconds * 1000 + 1


# ====================================================================== #
# 单会话归档 archive_expired_session
# ====================================================================== #
def test_archive_expired_when_expired():
    hot, archive = _make_hot(), _make_archive()
    result = archive_expired_session(hot, archive, now_ms=_expiry_now(hot))
    assert result.archived is True
    assert result.message_count == 3
    assert hot.session_meta.status == SessionStatus.ARCHIVED


def test_archive_expired_not_expired_skips():
    hot, archive = _make_hot(), _make_archive()
    now = hot.session_meta.updated_at  # 距过期时间尚远
    result = archive_expired_session(hot, archive, now_ms=now)
    assert result.archived is False
    assert result.reason == "not_expired"
    assert hot.session_meta.status == SessionStatus.ACTIVE
    assert archive.get_med_messages() == []


def test_archive_expired_already_archived_skips():
    hot, archive = _make_hot(), _make_archive()
    # 预先流转到 ARCHIVED（如历史已归档但未清热数据）
    hot.archive()
    result = archive_expired_session(hot, archive, now_ms=_expiry_now(hot))
    assert result.archived is False
    assert result.reason == f"status:{SessionStatus.ARCHIVED.value}"
    assert archive.get_med_messages() == []


def test_archive_expired_writes_all_messages():
    hot, archive = _make_hot(n_msgs=5), _make_archive()
    archive_expired_session(hot, archive, now_ms=_expiry_now(hot))
    # 归档() 不清除热数据；以 hot 全量消息 id 为基准校验归档层写入完整
    src_ids = {m.message_id for m in hot.get_med_messages()}
    assert {m.message_id for m in archive.get_med_messages()} == src_ids
    assert len(archive.get_med_messages()) == 5


def test_archive_expired_empty_messages():
    hot = FakeTtlHistory(
        session_id=NAMESPACE["session_id"],
        tenant_id=NAMESPACE["tenant_id"],
        dept_id=NAMESPACE["dept_id"],
        patient_id=NAMESPACE["patient_id"],
        ttl_seconds=60,
    )
    archive = _make_archive()
    result = archive_expired_session(hot, archive, now_ms=_expiry_now(hot))
    assert result.archived is True
    assert result.message_count == 0
    assert archive.get_med_messages() == []
    assert hot.session_meta.status == SessionStatus.ARCHIVED


def test_archive_expired_delete_after_archive():
    hot, archive = _make_hot(), _make_archive()
    archive_expired_session(hot, archive, delete_after_archive=True, now_ms=_expiry_now(hot))
    assert archive.get_med_messages()  # 归档层保留
    assert hot.get_med_messages() == []  # 热存储已清空
    assert hot.session_meta.status == SessionStatus.ARCHIVED


def test_archive_expired_archive_write_failure_keeps_status():
    hot, archive = _make_hot(), _make_archive()

    def _boom(messages):  # type: ignore[no-untyped-def]
        raise StorageError("archive bulk failed")

    archive.add_med_messages = _boom  # type: ignore[method-assign]
    with pytest.raises(StorageError):
        archive_expired_session(hot, archive, now_ms=_expiry_now(hot))
    # 写入失败：状态不得流转，便于安全重跑
    assert hot.session_meta.status == SessionStatus.ACTIVE


# ====================================================================== #
# 调度器 TtlArchiver.run
# ====================================================================== #
def test_archiver_construct_requires_callables():
    with pytest.raises(ValueError):
        TtlArchiver(
            list_sessions="not-callable",  # type: ignore[arg-type]
            make_hot=lambda k: _make_hot(),
            make_archive=lambda m: _make_archive(),
        )


def test_archiver_run_mixed_report():
    expired = _make_hot(ttl_seconds=60)
    fresh = _make_hot(ttl_seconds=10_000)  # 永不过期（now_ms 在附近）
    keys = {"med:chat:hosp-a:cardio:exp": expired, "med:chat:hosp-a:cardio:fresh": fresh}

    def list_sessions():
        return list(keys)

    def make_hot(key):
        return keys[key]

    def make_archive(meta):
        return FakeArchiveHistory(
            session_id=meta.session_id,
            tenant_id=meta.tenant_id,
            dept_id=meta.dept_id,
            patient_id=meta.patient_id,
        )

    archiver = TtlArchiver(
        list_sessions=list_sessions, make_hot=make_hot, make_archive=make_archive
    )
    report = archiver.run(now_ms=_expiry_now(expired))
    assert report.scanned == 2
    assert report.archived == 1
    assert report.skipped == 1
    assert report.failed == 0
    assert report.message_total == 3
    assert expired.session_meta.status == SessionStatus.ARCHIVED


def test_archiver_run_error_tolerance():
    good = _make_hot()
    keys = {"good": good, "bad": None}

    def list_sessions():
        return list(keys)

    def make_hot(key):
        if key == "bad":
            raise RuntimeError("cannot build hot handle")
        return keys[key]

    def make_archive(meta):
        return _make_archive()

    archiver = TtlArchiver(
        list_sessions=list_sessions, make_hot=make_hot, make_archive=make_archive
    )
    report = archiver.run(now_ms=_expiry_now(good))
    assert report.scanned == 2
    assert report.archived == 1
    assert report.failed == 1
    assert "bad: cannot build hot handle" in report.errors


def test_archiver_run_none_expired_all_skipped():
    fresh = _make_hot(ttl_seconds=10_000)

    def list_sessions():
        return ["only"]

    def make_hot(key):
        return fresh

    def make_archive(meta):
        return _make_archive()

    archiver = TtlArchiver(
        list_sessions=list_sessions, make_hot=make_hot, make_archive=make_archive
    )
    report = archiver.run(now_ms=fresh.session_meta.updated_at)
    assert report.scanned == 1
    assert report.archived == 0
    assert report.skipped == 1
    assert report.failed == 0


def test_archiver_make_archive_receives_meta():
    hot = _make_hot()
    seen = {}

    def make_archive(meta):
        seen["meta"] = meta
        return _make_archive()

    archiver = TtlArchiver(
        list_sessions=lambda: ["k"],
        make_hot=lambda k: hot,
        make_archive=make_archive,
    )
    archiver.run(now_ms=_expiry_now(hot))
    assert seen["meta"].storage_key == hot.storage_key


def test_archiver_archive_session_delegates():
    hot, archive = _make_hot(), _make_archive()
    archiver = TtlArchiver(
        list_sessions=lambda: [],
        make_hot=lambda k: hot,
        make_archive=lambda m: archive,
    )
    result = archiver.archive_session(hot, archive, now_ms=_expiry_now(hot))
    assert isinstance(result, ArchiveResult)
    assert result.archived is True
    assert hot.session_meta.status == SessionStatus.ARCHIVED


# ====================================================================== #
# Elasticsearch 归档层集成（受 ImportError 保护）
# ====================================================================== #
def _es_integration():
    """返回 ``(client, EsArchiveMedHistory 构造器)``；不可用时抛 ``ImportError``。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_stores"))
    from fake_es import FakeElasticsearch  # type: ignore

    from med_langchain_memory.stores.es_store import EsArchiveMedHistory

    client = FakeElasticsearch()

    def make_archive(meta):
        return EsArchiveMedHistory(
            session_id=meta.session_id,
            tenant_id=meta.tenant_id,
            dept_id=meta.dept_id,
            patient_id=meta.patient_id,
            client=client,
        )

    return client, make_archive


def test_es_integration_run_archives_to_es():
    try:
        client, make_archive = _es_integration()
    except ImportError:
        pytest.skip("es_store / fake_es not available")

    hot = _make_hot(n_msgs=4)
    archiver = TtlArchiver(
        list_sessions=lambda: [hot.storage_key],
        make_hot=lambda k: hot,
        make_archive=make_archive,
    )
    report = archiver.run(now_ms=_expiry_now(hot))
    assert report.archived == 1
    assert report.message_total == 4
    sources = client.all_sources()
    assert len(sources) == 4
    assert {s["message_id"] for s in sources} == {m.message_id for m in hot.get_med_messages()}


def test_es_integration_delete_after_archive():
    try:
        client, make_archive = _es_integration()
    except ImportError:
        pytest.skip("es_store / fake_es not available")

    hot = _make_hot(n_msgs=2)
    archiver = TtlArchiver(
        list_sessions=lambda: [hot.storage_key],
        make_hot=lambda k: hot,
        make_archive=make_archive,
        delete_after_archive=True,
    )
    archiver.run(now_ms=_expiry_now(hot))
    assert len(client.all_sources()) == 2
    assert hot.get_med_messages() == []
