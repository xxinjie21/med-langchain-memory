"""会话快照备份与恢复。

将单个会话（元数据 + 全量消息）导出为带 SHA-256 校验和的 protobuf 文件包，
并从文件包恢复。恢复前校验目标与会话命名空间一致，杜绝跨租户越权写入。

文件包为自描述二进制容器，布局如下：

    magic(8B) | schema_version_len(1B) | schema_version(utf-8) |
    payload_len(4B, little-endian) | payload(protobuf SessionSnapshot) |
    sha256(32B)

``sha256`` 覆盖从 magic 到 payload 末尾的全部字节，导入时用于完整性校验。

本模块不触碰任何底层存储实现，仅依赖 ``MedChatMessageHistory`` 公共接口与
``ProtobufSerializer``，因此任意已注册后端均可导出/恢复。不含任何文本解析逻辑。
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from med_langchain_memory.exceptions import IntegrityError, TenantIsolationError
from med_langchain_memory.serde import SerializationError
from med_langchain_memory.serde.protobuf_serializer import ProtobufSerializer
from med_langchain_memory.stores.base import MedChatMessageHistory

#: 文件包魔数，用于导入时快速识别文件类型与版本。
_MAGIC = b"MEDSNAP1"
#: 当前 schema 版本，随序列化协议演进而递增。
DEFAULT_SCHEMA_VERSION = "1"
#: 固定头部最小长度：magic(8) + version_len(1) + payload_len(4) + sha256(32)。
_MIN_PACKAGE_LEN = len(_MAGIC) + 1 + 4 + 32


@dataclass(frozen=True)
class SessionSnapshotPackage:
    """带 SHA-256 校验和的会话快照文件包（不可变值对象）。"""

    schema_version: str
    payload: bytes

    def to_bytes(self) -> bytes:
        """编码为自描述二进制文件包（尾部附 32 字节校验和）。"""
        sv = self.schema_version.encode("utf-8")
        head = _MAGIC + bytes([len(sv)]) + sv + struct.pack("<I", len(self.payload))
        body = head + self.payload
        return body + hashlib.sha256(body).digest()

    @classmethod
    def from_bytes(cls, data: bytes) -> SessionSnapshotPackage:
        """从二进制文件包解析并校验完整性。

        Raises:
            SerializationError: 文件太短、魔数不符或版本长度越界时。
            IntegrityError: 校验和与内容不匹配（文件损坏或被篡改）时。
        """
        if len(data) < _MIN_PACKAGE_LEN:
            raise SerializationError("snapshot package too short")
        if data[: len(_MAGIC)] != _MAGIC:
            raise SerializationError("bad snapshot magic; not a session snapshot file")
        sv_len = data[len(_MAGIC)]
        pos = len(_MAGIC) + 1
        if pos + sv_len + 4 + 32 > len(data):
            raise SerializationError("snapshot package header truncated")
        schema_version = data[pos : pos + sv_len].decode("utf-8")
        pos += sv_len
        (payload_len,) = struct.unpack("<I", data[pos : pos + 4])
        pos += 4
        payload_end = pos + payload_len
        if payload_end + 32 != len(data):
            raise SerializationError("snapshot payload length mismatch")
        payload = data[pos:payload_end]
        expected = data[payload_end : payload_end + 32]
        actual = hashlib.sha256(data[:payload_end]).digest()
        # 32 字节校验和用等值比较即可（非密钥比较场景，无需抵御时序侧信道）。
        if actual != expected:
            raise IntegrityError("snapshot checksum mismatch; file may be corrupted")
        return cls(schema_version=schema_version, payload=payload)

    def save(self, path: str | Path) -> None:
        """将文件包原子写入磁盘（先临时文件再 rename）。"""
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(self.to_bytes())
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> SessionSnapshotPackage:
        """从磁盘读取并校验文件包。"""
        return cls.from_bytes(Path(path).read_bytes())


@dataclass
class SnapshotSummary:
    """单次导出/恢复操作的摘要结果。

    Attributes:
        session_key: 会话统一存储键。
        message_count: 本次包含的消息条数。
        schema_version: 文件包 schema 版本。
        path: 文件包落盘路径（导出时有效，恢复时为源路径）。
        verified: 校验和是否通过（恢复时为 ``True``，导出时为 ``None``）。
    """

    session_key: str
    message_count: int
    schema_version: str
    path: str
    verified: bool | None = None


class SessionSnapshotter:
    """将会话导出为带校验和文件包，或从文件包恢复到目标存储。"""

    def __init__(self, *, serializer: ProtobufSerializer | None = None) -> None:
        """初始化快照器。

        Args:
            serializer: 可选的序列化器实例；缺省使用 ``ProtobufSerializer``。
        """
        self._ser = serializer or ProtobufSerializer()

    def export_session(
        self,
        history: MedChatMessageHistory,
        path: str | Path,
        *,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
    ) -> SnapshotSummary:
        """将 ``history`` 的元数据和全量消息导出到文件包。

        Args:
            history: 源会话历史（任意已注册后端）。
            path: 文件包输出路径。
            schema_version: 文件包 schema 版本标记。

        Returns:
            导出摘要；``verified`` 恒为 ``None``。
        """
        meta = history.session_meta
        messages = history.get_med_messages()
        # 导出前以实际消息数校正计数，保证快照内自洽。
        meta = meta.model_copy(update={"message_count": len(messages)})
        payload = self._ser.serialize_snapshot(meta, messages, schema_version)
        SessionSnapshotPackage(schema_version=schema_version, payload=payload).save(path)
        return SnapshotSummary(
            session_key=meta.storage_key,
            message_count=len(messages),
            schema_version=schema_version,
            path=str(path),
            verified=None,
        )

    def import_session(
        self,
        path: str | Path,
        target: MedChatMessageHistory,
    ) -> SnapshotSummary:
        """从文件包恢复会话到 ``target``。

        导入前校验文件包与会话命名空间一致，禁止跨租户越权写入；消息通过
        ``add_med_messages`` 追加，幂等可重跑（已存在消息由目标按 ID 去重）。

        Args:
            path: 文件包路径。
            target: 恢复目标会话历史（必须与文件包属于同一命名空间）。

        Returns:
            恢复摘要；``verified`` 为 ``True``（校验和已通过）。

        Raises:
            SerializationError: 文件包损坏或 protobuf 解析失败时。
            IntegrityError: 文件包校验和不匹配时。
            TenantIsolationError: 文件包命名空间与目标不一致时。
        """
        package = SessionSnapshotPackage.load(path)
        try:
            meta, messages, _sv = self._ser.deserialize_snapshot(package.payload)
        except SerializationError as exc:
            raise SerializationError(str(exc)) from exc
        if meta.storage_key != target.storage_key:
            raise TenantIsolationError(
                f"cannot restore {meta.storage_key} into {target.storage_key}"
            )
        # 跳过目标已存在的消息，保证导入幂等（append-only 后端重跑不重复）。
        existing = {m.message_id for m in target.get_med_messages()}
        to_write = [m for m in messages if m.message_id not in existing]
        if to_write:
            target.add_med_messages(to_write)
        return SnapshotSummary(
            session_key=meta.storage_key,
            message_count=len(messages),
            schema_version=package.schema_version,
            path=str(path),
            verified=True,
        )


def snapshot_session(
    history: MedChatMessageHistory,
    path: str | Path,
    *,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> SnapshotSummary:
    """便捷函数：将会话导出为带校验和文件包。"""
    return SessionSnapshotter().export_session(history, path, schema_version=schema_version)


def restore_session(path: str | Path, target: MedChatMessageHistory) -> SnapshotSummary:
    """便捷函数：从文件包恢复会话到目标存储。"""
    return SessionSnapshotter().import_session(path, target)
