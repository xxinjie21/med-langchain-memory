"""合规保留期 :mod:`med_langchain_memory.lifecycle.retention` 单元测试。

测试直接针对 ``MedChatMessageHistory`` 抽象接口：用隔离的 :class:`FakeHistory`
作为存储替身（各自持有独立数据，不按 storage_key 共享），覆盖保留期策略判定、
软删除标记、到期物理清理与调度器两阶段执行。无需真实中间件即可跑通。
"""

from __future__ import annotations

import pytest

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.domain.message import now_millis
from med_langchain_memory.domain.session import SessionStatus
from med_langchain_memory.exceptions import StateTransitionError
from med_langchain_memory.lifecycle import (
    PurgeResult,
    RetentionManager,
    RetentionPolicy,
    RetentionReport,
    SoftDeleteResult,
    purge_expired_session,
    soft_delete_session,
)
from med_langchain_memory.stores.base import MedChatMessageHistory

MS_DAY = 86_400_000

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


def _rewind(history: MedChatMessageHistory, days: int) -> None:
    """将会话元数据的 created_at/updated_at 回拨 ``days`` 天，模拟历史时间戳。"""
    old = now_millis() - days * MS_DAY - 1000
    history._meta = history._meta.model_copy(update={"created_at": old, "updated_at": old})


def make_archived(days_old: int) -> FakeHistory:
    """构造一个 ``days_old`` 天前归档的会话。"""
    h = FakeHistory(**NAMESPACE)
    h.add_med_messages(make_messages(3))
    h.archive()
    _rewind(h, days_old)
    return h


def make_deleted(days_old: int) -> FakeHistory:
    """构造一个 ``days_old`` 天前软删除的会话。"""
    h = FakeHistory(**NAMESPACE)
    h.add_med_messages(make_messages(3))
    h.archive()
    h.delete()
    _rewind(h, days_old)
    return h


class TestRetentionPolicy:
    def test_default_grace_is_zero(self) -> None:
        assert RetentionPolicy(retention_days=5).grace_days == 0

    def test_reject_non_positive_retention(self) -> None:
        with pytest.raises(ValueError):
            RetentionPolicy(retention_days=0)
        with pytest.raises(ValueError):
            RetentionPolicy(retention_days=-1)

    def test_reject_negative_grace(self) -> None:
        with pytest.raises(ValueError):
            RetentionPolicy(retention_days=1, grace_days=-1)

    def test_is_due_for_soft_delete_positive(self) -> None:
        policy = RetentionPolicy(retention_days=365)
        meta = make_archived(400).session_meta
        assert policy.is_due_for_soft_delete(meta, now_millis()) is True

    def test_is_due_for_soft_delete_boundary(self) -> None:
        policy = RetentionPolicy(retention_days=10)
        meta = make_archived(0).session_meta
        due_now = meta.updated_at + 10 * MS_DAY
        assert policy.is_due_for_soft_delete(meta, due_now) is True
        assert policy.is_due_for_soft_delete(meta, due_now - 1) is False

    def test_is_due_for_soft_delete_ignores_non_archived(self) -> None:
        policy = RetentionPolicy(retention_days=365)
        meta = make_deleted(400).session_meta
        assert policy.is_due_for_soft_delete(meta, now_millis()) is False

    def test_is_due_for_purge_positive(self) -> None:
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        meta = make_deleted(10).session_meta
        assert policy.is_due_for_purge(meta, now_millis()) is True

    def test_is_due_for_purge_boundary(self) -> None:
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        meta = make_deleted(0).session_meta
        due_now = meta.updated_at + 7 * MS_DAY
        assert policy.is_due_for_purge(meta, due_now) is True
        assert policy.is_due_for_purge(meta, due_now - 1) is False

    def test_is_due_for_purge_ignores_non_deleted(self) -> None:
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        meta = FakeHistory(**NAMESPACE).session_meta
        assert policy.is_due_for_purge(meta, now_millis()) is False


class TestSoftDeleteSession:
    def test_soft_delete_archived(self) -> None:
        h = make_archived(400)
        res = soft_delete_session(h)
        assert isinstance(res, SoftDeleteResult)
        assert res.deleted is True
        assert h.session_meta.status is SessionStatus.DELETED
        assert h.get_med_messages()  # 宽限期内数据保留

    def test_soft_delete_active_rejected(self) -> None:
        h = FakeHistory(**NAMESPACE)
        h.add_med_messages(make_messages(2))
        res = soft_delete_session(h)
        assert res.deleted is False
        assert res.reason == "not_archived"
        assert h.session_meta.status is SessionStatus.ACTIVE

    def test_soft_delete_already_deleted(self) -> None:
        h = make_deleted(10)
        res = soft_delete_session(h)
        assert res.deleted is False
        assert res.reason == "not_archived"


class TestPurgeExpiredSession:
    def test_purge_deleted_due(self) -> None:
        h = make_deleted(10)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        res = purge_expired_session(h, policy)
        assert isinstance(res, PurgeResult)
        assert res.purged is True
        assert res.purged_count == 3
        assert h.get_med_messages() == []

    def test_purge_not_deleted(self) -> None:
        h = make_archived(400)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        res = purge_expired_session(h, policy)
        assert res.purged is False
        assert res.reason == "not_deleted"

    def test_purge_within_grace(self) -> None:
        h = make_deleted(3)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        res = purge_expired_session(h, policy)
        assert res.purged is False
        assert res.reason == "not_due"

    def test_purge_already_purged(self) -> None:
        h = make_deleted(10)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        h.clear()
        res = purge_expired_session(h, policy)
        assert res.purged is False
        assert res.reason == "already_purged"

    def test_purge_is_idempotent(self) -> None:
        h = make_deleted(10)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        assert purge_expired_session(h, policy).purged is True
        second = purge_expired_session(h, policy)
        assert second.purged is False
        assert second.reason == "already_purged"

    def test_purge_with_explicit_now_boundary(self) -> None:
        h = make_deleted(0)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        meta = h.session_meta
        due_now = meta.updated_at + 7 * MS_DAY
        assert purge_expired_session(h, policy, now_ms=due_now).purged is True
        again = purge_expired_session(h, policy, now_ms=due_now - 1)
        assert again.purged is False
        assert again.reason == "not_due"


class TestRetentionManagerConstruction:
    def test_requires_policy_instance(self) -> None:
        with pytest.raises(ValueError):
            RetentionManager(
                policy=None,  # type: ignore[arg-type]
                list_sessions=lambda: [],
                make_history=lambda k: None,  # type: ignore[arg-type]
            )

    def test_requires_callable_list_sessions(self) -> None:
        with pytest.raises(ValueError):
            RetentionManager(
                policy=RetentionPolicy(retention_days=365),
                list_sessions="not-callable",  # type: ignore[arg-type]
                make_history=lambda k: None,  # type: ignore[arg-type]
            )

    def test_requires_callable_make_history(self) -> None:
        with pytest.raises(ValueError):
            RetentionManager(
                policy=RetentionPolicy(retention_days=365),
                list_sessions=lambda: [],
                make_history="not-callable",  # type: ignore[arg-type]
            )


class TestRetentionManagerRun:
    def _build_sessions(self) -> dict[str, FakeHistory]:
        return {
            "k-arch-due": make_archived(400),
            "k-arch-nd": make_archived(100),
            "k-del-due": make_deleted(10),
            "k-del-nd": make_deleted(3),
            "k-active": (lambda h: (h.add_med_messages(make_messages(1)), h)[1])(
                FakeHistory(**NAMESPACE)
            ),
        }

    def test_run_two_phase_mixed(self) -> None:
        sessions = self._build_sessions()
        purged: list[str] = []
        manager = RetentionManager(
            policy=RetentionPolicy(retention_days=365, grace_days=7),
            list_sessions=lambda: list(sessions.keys()),
            make_history=lambda k: sessions[k],
            after_purge=lambda k: purged.append(k),
        )
        report = manager.run()
        assert isinstance(report, RetentionReport)
        assert report.scanned == 5
        assert report.soft_deleted == 1
        assert report.purged == 1
        assert report.skipped == 3
        assert report.failed == 0
        assert purged == ["k-del-due"]
        # 状态流转确实发生
        assert sessions["k-arch-due"].session_meta.status is SessionStatus.DELETED
        assert sessions["k-del-due"].get_med_messages() == []

    def test_run_tolerates_errors(self) -> None:
        sessions = self._build_sessions()

        def make_history(key: str) -> FakeHistory:
            if key == "k-boom":
                raise RuntimeError("boom")
            return sessions[key]

        manager = RetentionManager(
            policy=RetentionPolicy(retention_days=365, grace_days=7),
            list_sessions=lambda: ["k-arch-due", "k-boom", "k-del-due"],
            make_history=make_history,
        )
        report = manager.run()
        assert report.scanned == 3
        assert report.failed == 1
        assert report.errors == ["k-boom: boom"]
        assert report.purged == 1

    def test_run_respects_explicit_now(self) -> None:
        sessions = {"only": make_deleted(0)}
        manager = RetentionManager(
            policy=RetentionPolicy(retention_days=365, grace_days=7),
            list_sessions=lambda: list(sessions.keys()),
            make_history=lambda k: sessions[k],
        )
        # 未到宽限期 -> 跳过
        before = manager.run(now_ms=sessions["only"].session_meta.updated_at + 1)
        assert before.purged == 0
        assert before.skipped == 1
        # 超过宽限期 -> 清理
        after = manager.run(now_ms=sessions["only"].session_meta.updated_at + 8 * MS_DAY)
        assert after.purged == 1


class TestRetentionManagerMethods:
    def test_soft_delete_method(self) -> None:
        h = make_archived(400)
        manager = RetentionManager(
            policy=RetentionPolicy(retention_days=365, grace_days=7),
            list_sessions=lambda: [],
            make_history=lambda k: h,
        )
        res = manager.soft_delete(h)
        assert res.deleted is True
        assert h.session_meta.status is SessionStatus.DELETED

    def test_purge_method(self) -> None:
        h = make_archived(400)
        policy = RetentionPolicy(retention_days=365, grace_days=7)
        manager = RetentionManager(
            policy=policy,
            list_sessions=lambda: [],
            make_history=lambda k: h,
        )
        manager.soft_delete(h)
        res = manager.purge(h, now_ms=h.session_meta.updated_at + 8 * MS_DAY)
        assert res.purged is True
        assert res.purged_count == 3


class TestBaseDeleteHook:
    def test_delete_transitions_archived_to_deleted(self) -> None:
        h = FakeHistory(**NAMESPACE)
        h.add_med_messages(make_messages(2))
        h.archive()
        meta, messages = h.delete()
        assert meta.status is SessionStatus.DELETED
        assert h.session_meta.status is SessionStatus.DELETED
        assert h.is_deleted is True
        assert len(messages) == 2
        assert h.get_med_messages()  # 软删除保留数据

    def test_delete_rejects_non_archived(self) -> None:
        h = FakeHistory(**NAMESPACE)
        h.add_med_messages(make_messages(1))
        with pytest.raises(StateTransitionError):
            h.delete()

    def test_is_deleted_false_when_active(self) -> None:
        h = FakeHistory(**NAMESPACE)
        assert h.is_deleted is False
