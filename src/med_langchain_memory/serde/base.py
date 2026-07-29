"""序列化层抽象接口与异常定义。

本模块只定义与具体编码格式无关的契约，不绑定任何第三方序列化库。
所有跨语言落库数据最终由 :class:`ProtobufSerializer` 产出 protobuf 二进制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from med_langchain_memory.exceptions import MedMemoryError

if TYPE_CHECKING:
    from med_langchain_memory.domain.message import MedMessage
    from med_langchain_memory.domain.session import SessionMeta


class SerializationError(MedMemoryError):
    """字节流反序列化失败（数据损坏 / 协议不匹配 / 字段越界）。"""


class Serializer(ABC):
    """模型对象与字节流互相转换的抽象契约。

    子类须同时实现单条消息、批量消息、会话元数据与完整快照四类转换，
    保证存储层（memory/file/redis/mysql/es）与迁移/快照工具可用同一套编解码。
    """

    # --- 单条消息 ---------------------------------------------------------
    @abstractmethod
    def serialize_message(self, message: MedMessage) -> bytes:
        """将单条 :class:`MedMessage` 编码为字节流。"""

    @abstractmethod
    def deserialize_message(self, data: bytes) -> MedMessage:
        """将字节流解码为 :class:`MedMessage`。

        Raises:
            SerializationError: 字节流损坏或字段无法映射时。
        """

    # --- 批量消息 ---------------------------------------------------------
    @abstractmethod
    def serialize_messages(self, messages: list[MedMessage]) -> bytes:
        """将有序消息列表编码为单个字节流（protobuf ``MedMessageBatch``）。"""

    @abstractmethod
    def deserialize_messages(self, data: bytes) -> list[MedMessage]:
        """将批量字节流解码为有序 :class:`MedMessage` 列表。

        Raises:
            SerializationError: 字节流损坏或字段无法映射时。
        """

    # --- 会话元数据 -------------------------------------------------------
    @abstractmethod
    def serialize_session(self, session: SessionMeta) -> bytes:
        """将 :class:`SessionMeta` 编码为字节流。"""

    @abstractmethod
    def deserialize_session(self, data: bytes) -> SessionMeta:
        """将字节流解码为 :class:`SessionMeta`。

        Raises:
            SerializationError: 字节流损坏或状态无法映射时。
        """

    # --- 完整会话快照 -----------------------------------------------------
    @abstractmethod
    def serialize_snapshot(
        self,
        session: SessionMeta,
        messages: list[MedMessage],
        schema_version: str = "1",
    ) -> bytes:
        """将会话元数据 + 全部消息编码为快照字节流（protobuf ``SessionSnapshot``）。"""

    @abstractmethod
    def deserialize_snapshot(self, data: bytes) -> tuple[SessionMeta, list[MedMessage], str]:
        """将快照字节流解码为 ``(meta, messages, schema_version)``。

        Raises:
            SerializationError: 字节流损坏或字段无法映射时。
        """
