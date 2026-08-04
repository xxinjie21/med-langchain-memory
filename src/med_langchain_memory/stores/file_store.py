"""本地文件存储适配器 :class:`FileMedHistory`。

定位：单机部署、边缘节点与离线导出场景下的零中间件持久化后端，
数据以**只追加日志**形式落盘，天然保留写入时序，便于人工排查与冷备。

支持两种落盘格式：

* ``jsonl``：每条消息一行 UTF-8 JSON，可读性优先，方便 ``grep`` 与人工核对；
* ``binary``：``4 字节大端长度前缀 + protobuf 消息体`` 的帧式二进制，
  与跨语言存储规范完全一致，体积小、解析快，作为正式落库格式。

并发安全由 :class:`FileLock` 保证：同一会话文件的读、追加、删除全部在
跨进程独占锁内完成，多进程 worker 同时写不会产生交错记录。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import json
import os
import struct
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.serde.base import SerializationError
from med_langchain_memory.serde.protobuf_serializer import ProtobufSerializer

from .base import MedChatMessageHistory
from .factory import StoreFactory
from .file_lock import FileLock

#: 二进制模式下每条消息的长度前缀（4 字节无符号大端整数）。
_LENGTH_PREFIX = struct.Struct(">I")

#: 单条消息体积上限（字节），防止损坏的长度字段触发巨量内存分配。
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: 禁止出现在存储路径中的目录名，避免 ID 越权穿越到上级目录。
_UNSAFE_PATH_PARTS = frozenset({".", ".."})


class FileFormat(StrEnum):
    """文件存储的落盘格式。"""

    JSONL = "jsonl"
    BINARY = "binary"


_SUFFIXES: dict[FileFormat, str] = {
    FileFormat.JSONL: ".jsonl",
    FileFormat.BINARY: ".pb",
}


@StoreFactory.register("file")
class FileMedHistory(MedChatMessageHistory):
    """本地文件会话历史，注册名 ``file``。

    落盘路径为 ``{base_dir}/{tenant_id}/{dept_id}/{session_id}{后缀}``，
    锁文件为同名 ``.lock`` 文件。

    Example:
        >>> import tempfile
        >>> history = FileMedHistory(
        ...     session_id="s-1",
        ...     tenant_id="hosp-a",
        ...     dept_id="cardio",
        ...     patient_id="p-1",
        ...     base_dir=tempfile.mkdtemp(),
        ... )
        >>> history.get_med_messages()
        []
    """

    #: 文件系统无原生过期能力，TTL 由上层 lifecycle 调度器负责。
    supports_ttl: ClassVar[bool] = False

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        base_dir: str | Path = ".med_sessions",
        file_format: FileFormat | str = FileFormat.JSONL,
        lock_timeout: float = 10.0,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化文件会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            base_dir: 数据根目录，按租户/科室分层建子目录。
            file_format: 落盘格式，``jsonl`` 或 ``binary``。
            lock_timeout: 获取文件锁的最长等待秒数。
            ttl_seconds: 必须为 ``None``，本后端不支持原生 TTL。

        Raises:
            ValidationError: ID 不合法、路径分量不安全或格式名未知时。
            StorageError: 传入了 ``ttl_seconds``（本后端不支持原生 TTL）时。
        """
        super().__init__(session_id, tenant_id, dept_id, patient_id, ttl_seconds=ttl_seconds)
        try:
            self._format = FileFormat(file_format)
        except ValueError as exc:
            raise ValidationError(
                f"unknown file format: {file_format!r}; "
                f"expected one of {[f.value for f in FileFormat]}"
            ) from exc
        for part in (self.tenant_id, self.dept_id, self.session_id):
            if part in _UNSAFE_PATH_PARTS:
                raise ValidationError(f"unsafe path component in session namespace: {part!r}")
        self._base_dir = Path(base_dir)
        filename = f"{self.session_id}{_SUFFIXES[self._format]}"
        self._path = self._base_dir / self.tenant_id / self.dept_id / filename
        self._lock = FileLock(Path(f"{self._path}.lock"), timeout=lock_timeout)
        self._serializer = ProtobufSerializer()

    # ------------------------------------------------------------------ #
    # 存储原语
    # ------------------------------------------------------------------ #
    def _append(self, messages: list[MedMessage]) -> None:
        """在文件锁保护下将消息编码为记录追加写入日志尾部并落盘。"""
        payload = b"".join(self._encode(message) for message in messages)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "ab") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """读取整个日志并按 ``created_at`` 稳定排序；文件不存在时返回空列表。

        Raises:
            StorageError: 日志记录损坏（非法 JSON 行、截断的二进制帧等）时。
        """
        with self._lock:
            try:
                raw = self._path.read_bytes()
            except FileNotFoundError:
                return []
        messages = self._decode_all(raw)
        messages.sort(key=lambda message: message.created_at)
        return messages if limit is None else messages[-limit:]

    def clear(self) -> None:
        """删除本会话的日志文件（文件不存在时为空操作）。"""
        with self._lock:
            self._path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 文件后端专有能力
    # ------------------------------------------------------------------ #
    @property
    def path(self) -> Path:
        """本会话日志文件的完整路径。"""
        return self._path

    @property
    def lock_path(self) -> Path:
        """本会话锁文件的完整路径。"""
        return self._lock.path

    @property
    def file_format(self) -> FileFormat:
        """当前落盘格式。"""
        return self._format

    @property
    def exists(self) -> bool:
        """日志文件是否已创建。"""
        return self._path.is_file()

    @property
    def size(self) -> int:
        """当前会话已存储的消息条数。"""
        return len(self._read())

    # ------------------------------------------------------------------ #
    # 编解码
    # ------------------------------------------------------------------ #
    def _encode(self, message: MedMessage) -> bytes:
        """将单条消息编码为一条日志记录。"""
        if self._format is FileFormat.JSONL:
            line = json.dumps(message.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            return f"{line}\n".encode()
        blob = self._serializer.serialize_message(message)
        return _LENGTH_PREFIX.pack(len(blob)) + blob

    def _decode_all(self, raw: bytes) -> list[MedMessage]:
        """将整个日志字节流解码为消息列表。"""
        if self._format is FileFormat.BINARY:
            return self._decode_frames(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StorageError(f"{self._path} is not valid utf-8 jsonl: {exc}") from exc
        return [
            self._decode_line(line, number)
            for number, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]

    def _decode_line(self, line: str, number: int) -> MedMessage:
        """解码一行 JSONL 记录。"""
        try:
            payload = json.loads(line)
            return MedMessage(**payload)
        except (json.JSONDecodeError, TypeError, PydanticValidationError) as exc:
            raise StorageError(
                f"corrupted jsonl record at line {number} of {self._path}: {exc}"
            ) from exc

    def _decode_frames(self, raw: bytes) -> list[MedMessage]:
        """解码长度前缀帧式二进制日志。"""
        messages: list[MedMessage] = []
        offset = 0
        total = len(raw)
        while offset < total:
            if total - offset < _LENGTH_PREFIX.size:
                raise StorageError(
                    f"truncated frame length prefix at offset {offset} of {self._path}"
                )
            (size,) = _LENGTH_PREFIX.unpack_from(raw, offset)
            offset += _LENGTH_PREFIX.size
            if size == 0 or size > MAX_FRAME_BYTES or total - offset < size:
                raise StorageError(
                    f"corrupted frame of declared size {size} at offset {offset} of {self._path}"
                )
            try:
                messages.append(self._serializer.deserialize_message(raw[offset : offset + size]))
            except SerializationError as exc:
                raise StorageError(
                    f"corrupted protobuf frame at offset {offset} of {self._path}: {exc}"
                ) from exc
            offset += size
        return messages
