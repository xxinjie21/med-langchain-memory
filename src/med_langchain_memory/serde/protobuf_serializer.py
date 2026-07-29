"""基于 Protobuf 的序列化器实现。

``MedMessage`` / ``SessionMeta`` 与 ``protos/med_session.proto`` 中的消息一一对应，
枚举通过显式映射表与 domain 枚举对齐（proto3 的 ``*_UNSPECIFIED=0`` 在 domain 层不存在，
故未在映射表中，反序列化遇到未映射值即视为损坏）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.serde import med_session_pb2 as pb
from med_langchain_memory.serde.base import SerializationError, Serializer

if TYPE_CHECKING:
    pass

# domain 枚举 -> protobuf 枚举成员（proto 枚举成员即 int 子类实例）
_ROLE_TO_PB: dict[MessageRole, pb.MessageRole] = {
    MessageRole.PATIENT: pb.MESSAGE_ROLE_PATIENT,
    MessageRole.DOCTOR: pb.MESSAGE_ROLE_DOCTOR,
    MessageRole.ASSISTANT: pb.MESSAGE_ROLE_ASSISTANT,
    MessageRole.SYSTEM: pb.MESSAGE_ROLE_SYSTEM,
}
_PB_TO_ROLE: dict[int, MessageRole] = {int(v): k for k, v in _ROLE_TO_PB.items()}

_STATUS_TO_PB: dict[SessionStatus, pb.SessionStatus] = {
    SessionStatus.ACTIVE: pb.SESSION_STATUS_ACTIVE,
    SessionStatus.CLOSED: pb.SESSION_STATUS_CLOSED,
    SessionStatus.ARCHIVED: pb.SESSION_STATUS_ARCHIVED,
    SessionStatus.DELETED: pb.SESSION_STATUS_DELETED,
}
_PB_TO_STATUS: dict[int, SessionStatus] = {int(v): k for k, v in _STATUS_TO_PB.items()}


class ProtobufSerializer(Serializer):
    """将领域模型编码 / 解码为 protobuf 二进制。"""

    # ------------------------------------------------------------------ #
    # 单条消息
    # ------------------------------------------------------------------ #
    def serialize_message(self, message: MedMessage) -> bytes:
        return bytes(self._to_pb_message(message).SerializeToString())

    def deserialize_message(self, data: bytes) -> MedMessage:
        try:
            proto = pb.MedMessage()
            proto.ParseFromString(data)
        except Exception as exc:  # protobuf 解码异常类型不统一，统一包装
            raise SerializationError(f"invalid MedMessage bytes: {exc}") from exc
        return self._from_pb_message(proto)

    # ------------------------------------------------------------------ #
    # 批量消息
    # ------------------------------------------------------------------ #
    def serialize_messages(self, messages: list[MedMessage]) -> bytes:
        batch = pb.MedMessageBatch()
        batch.messages.extend(self._to_pb_message(m) for m in messages)
        return bytes(batch.SerializeToString())

    def deserialize_messages(self, data: bytes) -> list[MedMessage]:
        try:
            batch = pb.MedMessageBatch()
            batch.ParseFromString(data)
        except Exception as exc:
            raise SerializationError(f"invalid MedMessageBatch bytes: {exc}") from exc
        return [self._from_pb_message(m) for m in batch.messages]

    # ------------------------------------------------------------------ #
    # 会话元数据
    # ------------------------------------------------------------------ #
    def serialize_session(self, session: SessionMeta) -> bytes:
        return bytes(self._to_pb_session(session).SerializeToString())

    def deserialize_session(self, data: bytes) -> SessionMeta:
        try:
            proto = pb.SessionMeta()
            proto.ParseFromString(data)
        except Exception as exc:
            raise SerializationError(f"invalid SessionMeta bytes: {exc}") from exc
        return self._from_pb_session(proto)

    # ------------------------------------------------------------------ #
    # 完整快照
    # ------------------------------------------------------------------ #
    def serialize_snapshot(
        self,
        session: SessionMeta,
        messages: list[MedMessage],
        schema_version: str = "1",
    ) -> bytes:
        snapshot = pb.SessionSnapshot()
        snapshot.meta.CopyFrom(self._to_pb_session(session))
        snapshot.messages.extend(self._to_pb_message(m) for m in messages)
        snapshot.schema_version = schema_version
        return bytes(snapshot.SerializeToString())

    def deserialize_snapshot(self, data: bytes) -> tuple[SessionMeta, list[MedMessage], str]:
        try:
            snapshot = pb.SessionSnapshot()
            snapshot.ParseFromString(data)
        except Exception as exc:
            raise SerializationError(f"invalid SessionSnapshot bytes: {exc}") from exc
        meta = self._from_pb_session(snapshot.meta)
        messages = [self._from_pb_message(m) for m in snapshot.messages]
        return meta, messages, snapshot.schema_version

    # ------------------------------------------------------------------ #
    # 内部转换
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_pb_message(message: MedMessage) -> pb.MedMessage:
        proto = pb.MedMessage()
        proto.message_id = message.message_id
        proto.session_id = message.session_id
        proto.tenant_id = message.tenant_id
        proto.dept_id = message.dept_id
        proto.patient_id = message.patient_id
        proto.role = _ROLE_TO_PB[message.role]
        proto.content = message.content
        proto.token_count = message.token_count
        proto.masked = message.masked
        proto.created_at = message.created_at
        proto.metadata.update(message.metadata)
        return proto

    @staticmethod
    def _from_pb_message(proto: pb.MedMessage) -> MedMessage:
        role = _PB_TO_ROLE.get(int(proto.role))
        if role is None:
            raise SerializationError(f"unmapped MessageRole value: {proto.role}")
        return MedMessage(
            message_id=proto.message_id,
            session_id=proto.session_id,
            tenant_id=proto.tenant_id,
            dept_id=proto.dept_id,
            patient_id=proto.patient_id,
            role=role,
            content=proto.content,
            token_count=proto.token_count,
            masked=proto.masked,
            created_at=proto.created_at,
            metadata=dict(proto.metadata),
        )

    @staticmethod
    def _to_pb_session(session: SessionMeta) -> pb.SessionMeta:
        proto = pb.SessionMeta()
        proto.session_id = session.session_id
        proto.tenant_id = session.tenant_id
        proto.dept_id = session.dept_id
        proto.patient_id = session.patient_id
        proto.status = _STATUS_TO_PB[session.status]
        proto.message_count = session.message_count
        proto.created_at = session.created_at
        proto.updated_at = session.updated_at
        proto.metadata.update(session.metadata)
        return proto

    @staticmethod
    def _from_pb_session(proto: pb.SessionMeta) -> SessionMeta:
        status = _PB_TO_STATUS.get(int(proto.status))
        if status is None:
            raise SerializationError(f"unmapped SessionStatus value: {proto.status}")
        return SessionMeta(
            session_id=proto.session_id,
            tenant_id=proto.tenant_id,
            dept_id=proto.dept_id,
            patient_id=proto.patient_id,
            status=status,
            message_count=proto.message_count,
            created_at=proto.created_at,
            updated_at=proto.updated_at,
            metadata=dict(proto.metadata),
        )
