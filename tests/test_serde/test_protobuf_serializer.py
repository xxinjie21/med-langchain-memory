"""ProtobufSerializer 单元测试。

覆盖单条消息、批量消息、会话元数据、完整快照四类转换，
边界用例含非法字节、未映射枚举、空批处理等；所有用例不依赖任何外部中间件。
"""

from __future__ import annotations

import uuid

import pytest

from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.serde import ProtobufSerializer, SerializationError, Serializer
from med_langchain_memory.serde import med_session_pb2 as pb


def _make_message(
    role: MessageRole = MessageRole.PATIENT, content: str = "主诉：咳嗽三天"
) -> MedMessage:
    return MedMessage(
        session_id="s-001",
        tenant_id="t-001",
        dept_id="d-001",
        patient_id="p-001",
        role=role,
        content=content,
        token_count=6,
        metadata={"source": "web"},
    )


def _make_session(status: SessionStatus = SessionStatus.ACTIVE) -> SessionMeta:
    return SessionMeta(
        session_id="s-001",
        tenant_id="t-001",
        dept_id="d-001",
        patient_id="p-001",
        status=status,
        message_count=2,
        metadata={"tag": "flu"},
    )


# --------------------------------------------------------------------------- #
# 抽象契约
# --------------------------------------------------------------------------- #
def test_serializer_is_abstract() -> None:
    """Serializer 不可被直接实例化。"""
    with pytest.raises(TypeError):
        Serializer()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# 单条消息
# --------------------------------------------------------------------------- #
def test_serialize_message_roundtrip() -> None:
    """单条消息 model -> bytes -> model 字段完整保留。"""
    ser = ProtobufSerializer()
    msg = _make_message(role=MessageRole.DOCTOR, content="建议拍胸片")
    restored = ser.deserialize_message(ser.serialize_message(msg))
    assert restored == msg
    assert restored.role is MessageRole.DOCTOR
    assert restored.metadata == {"source": "web"}


def test_deserialize_message_invalid_bytes_raises() -> None:
    """损坏字节流应抛出 SerializationError。"""
    ser = ProtobufSerializer()
    with pytest.raises(SerializationError):
        ser.deserialize_message(b"this is not a valid protobuf message")


def test_deserialize_message_unmapped_role_raises() -> None:
    """proto 层 UNSPECIFIED(0) 角色在 domain 无对应值，应判为损坏。"""
    ser = ProtobufSerializer()
    raw = pb.MedMessage()
    raw.message_id = str(uuid.uuid4())
    raw.session_id = "s-001"
    raw.tenant_id = "t-001"
    raw.dept_id = "d-001"
    raw.patient_id = "p-001"
    raw.role = pb.MESSAGE_ROLE_UNSPECIFIED
    raw.content = "x"
    with pytest.raises(SerializationError):
        ser.deserialize_message(raw.SerializeToString())


# --------------------------------------------------------------------------- #
# 批量消息
# --------------------------------------------------------------------------- #
def test_serialize_messages_roundtrip() -> None:
    """批量消息顺序与数量在编解码后保持一致。"""
    ser = ProtobufSerializer()
    messages = [
        _make_message(role=MessageRole.PATIENT, content="主诉：发热"),
        _make_message(role=MessageRole.DOCTOR, content="体温多少"),
        _make_message(role=MessageRole.ASSISTANT, content="已记录"),
    ]
    restored = ser.deserialize_messages(ser.serialize_messages(messages))
    assert restored == messages
    assert [m.role for m in restored] == [
        MessageRole.PATIENT,
        MessageRole.DOCTOR,
        MessageRole.ASSISTANT,
    ]


def test_serialize_messages_empty_roundtrip() -> None:
    """空消息列表应编解码为空列表（边界用例）。"""
    ser = ProtobufSerializer()
    assert ser.deserialize_messages(ser.serialize_messages([])) == []


def test_serialize_messages_invalid_bytes_raises() -> None:
    """批量消息的损坏字节流应抛出 SerializationError。"""
    ser = ProtobufSerializer()
    with pytest.raises(SerializationError):
        ser.deserialize_messages(b"\x00\x01not-batch")


# --------------------------------------------------------------------------- #
# 会话元数据
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", list(SessionStatus))
def test_serialize_session_status_preserves_all(status: SessionStatus) -> None:
    """各生命周期状态均可正确往返（含终态 DELETED）。"""
    ser = ProtobufSerializer()
    session = _make_session(status=status)
    restored = ser.deserialize_session(ser.serialize_session(session))
    assert restored == session
    assert restored.status is status


def test_serialize_session_metadata_preserved() -> None:
    """会话元数据扩展字段在编解码后保留。"""
    ser = ProtobufSerializer()
    session = _make_session().model_copy(update={"metadata": {"k1": "v1", "k2": "v2"}})
    restored = ser.deserialize_session(ser.serialize_session(session))
    assert restored.metadata == {"k1": "v1", "k2": "v2"}


def test_deserialize_session_unmapped_status_raises() -> None:
    """proto 层 UNSPECIFIED(0) 状态在 domain 无对应值，应判为损坏。"""
    ser = ProtobufSerializer()
    raw = pb.SessionMeta()
    raw.session_id = "s-001"
    raw.tenant_id = "t-001"
    raw.dept_id = "d-001"
    raw.patient_id = "p-001"
    raw.status = pb.SESSION_STATUS_UNSPECIFIED
    with pytest.raises(SerializationError):
        ser.deserialize_session(raw.SerializeToString())


# --------------------------------------------------------------------------- #
# 完整快照
# --------------------------------------------------------------------------- #
def test_serialize_snapshot_roundtrip() -> None:
    """快照 meta + messages + schema_version 三者均完整往返。"""
    ser = ProtobufSerializer()
    session = _make_session(status=SessionStatus.CLOSED)
    messages = [
        _make_message(role=MessageRole.PATIENT, content="复查"),
        _make_message(role=MessageRole.DOCTOR, content="恢复良好"),
    ]
    meta, restored_msgs, version = ser.deserialize_snapshot(
        ser.serialize_snapshot(session, messages, schema_version="2")
    )
    assert meta == session
    assert restored_msgs == messages
    assert version == "2"


def test_serialize_snapshot_default_version() -> None:
    """未指定 schema_version 时默认写入 "1"。"""
    ser = ProtobufSerializer()
    _, _, version = ser.deserialize_snapshot(
        ser.serialize_snapshot(_make_session(), [_make_message()])
    )
    assert version == "1"


def test_serialize_snapshot_empty_messages() -> None:
    """快照可携带零条消息（边界用例）。"""
    ser = ProtobufSerializer()
    session = _make_session()
    meta, restored_msgs, _ = ser.deserialize_snapshot(ser.serialize_snapshot(session, []))
    assert meta == session
    assert restored_msgs == []
