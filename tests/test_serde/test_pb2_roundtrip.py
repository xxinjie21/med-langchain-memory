"""生成代码 ``med_session_pb2`` 的编解码与领域模型对齐测试。"""

from __future__ import annotations

import pytest
from google.protobuf.message import DecodeError

from med_langchain_memory.domain.message import MessageRole
from med_langchain_memory.domain.session import SessionStatus
from med_langchain_memory.serde import med_session_pb2 as pb


def _sample_message() -> pb.MedMessage:
    msg = pb.MedMessage(
        message_id="0198f6f0-1234-7abc-8def-0123456789ab",
        session_id="sess-001",
        tenant_id="hospital-a",
        dept_id="cardiology",
        patient_id="p-1001",
        role=pb.MESSAGE_ROLE_PATIENT,
        content="最近胸口偶尔发闷",
        token_count=12,
        masked=False,
        created_at=1_753_700_000_000,
    )
    msg.metadata["channel"] = "app"
    return msg


class TestMedMessageRoundTrip:
    def test_serialize_and_parse_round_trip(self) -> None:
        original = _sample_message()
        data = original.SerializeToString()
        restored = pb.MedMessage()
        restored.ParseFromString(data)
        assert restored == original
        assert restored.content == "最近胸口偶尔发闷"
        assert restored.metadata["channel"] == "app"

    def test_default_message_has_zero_values(self) -> None:
        """边界：空消息所有字段为 proto3 零值。"""
        empty = pb.MedMessage()
        assert empty.message_id == ""
        assert empty.role == pb.MESSAGE_ROLE_UNSPECIFIED
        assert empty.token_count == 0
        assert empty.masked is False
        assert len(empty.metadata) == 0

    def test_parse_garbage_bytes_raises(self) -> None:
        """边界：非法字节流解析必须抛 DecodeError。"""
        with pytest.raises(DecodeError):
            pb.MedMessage().ParseFromString(b"\xff\xff\xff\xff")


class TestSessionSnapshotRoundTrip:
    def test_snapshot_round_trip(self) -> None:
        snapshot = pb.SessionSnapshot(
            meta=pb.SessionMeta(
                session_id="sess-001",
                tenant_id="hospital-a",
                dept_id="cardiology",
                patient_id="p-1001",
                status=pb.SESSION_STATUS_ACTIVE,
                message_count=1,
                created_at=1_753_700_000_000,
                updated_at=1_753_700_000_500,
            ),
            messages=[_sample_message()],
            snapshot_at=1_753_700_001_000,
            schema_version="1",
        )
        restored = pb.SessionSnapshot()
        restored.ParseFromString(snapshot.SerializeToString())
        assert restored == snapshot
        assert restored.meta.status == pb.SESSION_STATUS_ACTIVE
        assert len(restored.messages) == 1

    def test_empty_batch_serializes(self) -> None:
        """边界：空消息批次可正常编解码。"""
        batch = pb.MedMessageBatch(session_id="sess-empty")
        restored = pb.MedMessageBatch()
        restored.ParseFromString(batch.SerializeToString())
        assert restored.session_id == "sess-empty"
        assert len(restored.messages) == 0


class TestEnumAlignmentWithDomain:
    """protobuf 枚举必须覆盖领域层枚举的全部取值。"""

    @pytest.mark.parametrize("role", list(MessageRole))
    def test_message_role_alignment(self, role: MessageRole) -> None:
        assert pb.MessageRole.Value(f"MESSAGE_ROLE_{role.name}") > 0

    @pytest.mark.parametrize("status", list(SessionStatus))
    def test_session_status_alignment(self, status: SessionStatus) -> None:
        assert pb.SessionStatus.Value(f"SESSION_STATUS_{status.name}") > 0

    def test_unknown_enum_name_raises(self) -> None:
        """边界：不存在的枚举名必须抛 ValueError。"""
        with pytest.raises(ValueError):
            pb.MessageRole.Value("MESSAGE_ROLE_NURSE")
