"""MedMessage / MessageRole / new_message_id 单元测试。"""

from __future__ import annotations

import time
import uuid

import pytest
from pydantic import ValidationError

from med_langchain_memory.domain import MedMessage, MessageRole, new_message_id


def make_message(**overrides: object) -> MedMessage:
    """构造一条合法消息，允许字段覆盖。"""
    kwargs: dict = {
        "session_id": "s-001",
        "tenant_id": "hospital_a",
        "dept_id": "cardiology",
        "patient_id": "p-1024",
        "role": MessageRole.PATIENT,
        "content": "最近胸闷气短",
    }
    kwargs.update(overrides)
    return MedMessage(**kwargs)


class TestNewMessageId:
    def test_is_valid_uuid_version_7(self) -> None:
        mid = new_message_id()
        parsed = uuid.UUID(mid)
        assert parsed.version == 7
        assert parsed.variant == uuid.RFC_4122

    def test_ids_are_time_ordered(self) -> None:
        first = new_message_id()
        time.sleep(0.002)  # 跨毫秒边界
        second = new_message_id()
        assert first < second

    def test_ids_are_unique(self) -> None:
        ids = {new_message_id() for _ in range(1000)}
        assert len(ids) == 1000


class TestMedMessage:
    def test_create_with_defaults(self) -> None:
        msg = make_message()
        assert uuid.UUID(msg.message_id).version == 7
        assert msg.token_count == 0
        assert msg.masked is False
        assert msg.created_at > 0
        assert msg.metadata == {}

    def test_role_accepts_string_value(self) -> None:
        msg = make_message(role="doctor")
        assert msg.role is MessageRole.DOCTOR

    def test_storage_key_format(self) -> None:
        msg = make_message()
        assert msg.storage_key == "med:chat:hospital_a:cardiology:s-001"

    def test_is_frozen(self) -> None:
        msg = make_message()
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_message(content="")

    @pytest.mark.parametrize("bad_id", ["", "a:b", "a{b}", "中文id", "x" * 65])
    def test_illegal_tenant_id_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            make_message(tenant_id=bad_id)

    def test_negative_token_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_message(token_count=-1)

    def test_malformed_message_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_message(message_id="not-a-uuid")

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            make_message(diagnosis="unexpected")

    def test_unknown_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_message(role="nurse")
