"""SessionMeta / SessionStatus 状态机单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from med_langchain_memory.domain import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import StateTransitionError


def make_session(**overrides: object) -> SessionMeta:
    """构造一个合法会话元数据，允许字段覆盖。"""
    kwargs: dict = {
        "session_id": "s-001",
        "tenant_id": "hospital_a",
        "dept_id": "cardiology",
        "patient_id": "p-1024",
    }
    kwargs.update(overrides)
    return SessionMeta(**kwargs)


class TestSessionMetaDefaults:
    def test_create_with_defaults(self) -> None:
        meta = make_session()
        assert meta.status is SessionStatus.ACTIVE
        assert meta.message_count == 0
        assert meta.created_at > 0
        assert meta.updated_at == meta.created_at

    def test_storage_key_format(self) -> None:
        meta = make_session()
        assert meta.storage_key == "med:chat:hospital_a:cardiology:s-001"

    def test_updated_at_earlier_than_created_at_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_session(created_at=2000, updated_at=1000)

    @pytest.mark.parametrize("bad_id", ["", "a:b", "含中文"])
    def test_illegal_session_id_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            make_session(session_id=bad_id)


class TestSessionTransitions:
    def test_active_to_closed_to_reopened(self) -> None:
        meta = make_session()
        closed = meta.transition_to(SessionStatus.CLOSED)
        assert closed.status is SessionStatus.CLOSED
        reopened = closed.transition_to(SessionStatus.ACTIVE)
        assert reopened.status is SessionStatus.ACTIVE

    def test_full_archive_lifecycle(self) -> None:
        meta = make_session()
        archived = meta.transition_to(SessionStatus.ARCHIVED)
        deleted = archived.transition_to(SessionStatus.DELETED)
        assert deleted.status is SessionStatus.DELETED

    def test_transition_returns_new_instance(self) -> None:
        meta = make_session()
        closed = meta.transition_to(SessionStatus.CLOSED)
        assert meta.status is SessionStatus.ACTIVE  # 原实例不变
        assert closed is not meta

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (SessionStatus.ACTIVE, SessionStatus.DELETED),
            (SessionStatus.CLOSED, SessionStatus.DELETED),
            (SessionStatus.DELETED, SessionStatus.ACTIVE),
            (SessionStatus.ARCHIVED, SessionStatus.ACTIVE),
        ],
    )
    def test_illegal_transition_raises(self, source: SessionStatus, target: SessionStatus) -> None:
        meta = make_session(status=source)
        assert not meta.can_transition_to(target)
        with pytest.raises(StateTransitionError):
            meta.transition_to(target)

    def test_can_transition_to_positive(self) -> None:
        meta = make_session()
        assert meta.can_transition_to(SessionStatus.CLOSED)
        assert meta.can_transition_to(SessionStatus.ARCHIVED)


class TestSessionTouch:
    def test_touch_increments_count_and_refreshes_timestamp(self) -> None:
        meta = make_session()
        touched = meta.touch(added_messages=3)
        assert touched.message_count == 3
        assert touched.updated_at >= meta.updated_at
        assert meta.message_count == 0  # 原实例不变

    def test_touch_default_adds_one(self) -> None:
        assert make_session().touch().message_count == 1

    def test_touch_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="added_messages"):
            make_session().touch(added_messages=-1)
