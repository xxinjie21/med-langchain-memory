"""审计事件模型与落盘接口。

定义医疗会话存储各操作（读 / 写 / 删 / 迁移 / 归档 / 快照 / 保留）的审计事件
``AuditEvent``，以及可插拔的落盘接口 ``AuditSink``（提供内存与文件 JSONL
两种实现）。本模块只记录结构化操作元数据，不做任何文本内容理解。

典型用法::

    sink = FileAuditSink("audit.log")
    sink.record(make_audit_event(AuditAction.WRITE, actor="doctor:1024",
                                 session_id="s-1", tenant_id="t-1"))
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from med_langchain_memory.exceptions import AuditSinkError

from .message import IdStr, new_message_id, now_millis


class AuditAction(StrEnum):
    """被审计的操作类型。"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MIGRATE = "migrate"
    ARCHIVE = "archive"
    SNAPSHOT = "snapshot"
    RETAIN = "retain"


class AuditStatus(StrEnum):
    """操作结果状态。"""

    SUCCESS = "success"
    FAILURE = "failure"


class AuditEvent(BaseModel):
    """一次受审计操作的不可变记录。

    字段对齐领域模型规范：业务 ID 复用 ``IdStr`` 约束（允许为空以覆盖系统级
    操作）；``event_id`` 为 UUIDv7 时序标识，``occurred_at`` 为 epoch 毫秒。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=new_message_id)
    action: AuditAction
    actor: str = Field(min_length=1)
    status: AuditStatus = AuditStatus.SUCCESS
    session_id: IdStr | None = None
    tenant_id: IdStr | None = None
    dept_id: IdStr | None = None
    target: str | None = None
    error: str | None = None
    occurred_at: int = Field(default_factory=now_millis, gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)


def make_audit_event(
    action: AuditAction | str,
    actor: str,
    *,
    status: AuditStatus | str = AuditStatus.SUCCESS,
    session_id: IdStr | None = None,
    tenant_id: IdStr | None = None,
    dept_id: IdStr | None = None,
    target: str | None = None,
    error: str | None = None,
    metadata: dict[str, str] | None = None,
) -> AuditEvent:
    """构造审计事件的便捷工厂，``action`` / ``status`` 接受字符串形式。

    Args:
        action: 操作类型，可为 :class:`AuditAction` 或其字符串值。
        actor: 操作主体标识（如 ``"doctor:1024"``、``"system"``）。
        status: 操作结果，默认 :attr:`AuditStatus.SUCCESS`。
        session_id: 关联会话 ID，系统级操作可为 ``None``。
        tenant_id: 关联租户 ID。
        dept_id: 关联科室 ID。
        target: 操作目标（如 ``"message:<id>"``）。
        error: 失败时的错误描述。
        metadata: 附加标签。

    Returns:
        新构造的 :class:`AuditEvent` 实例。

    Raises:
        ValueError: 当 ``actor`` 为空时。
    """
    if not actor:
        raise ValueError("actor must be a non-empty string")
    return AuditEvent(
        action=AuditAction(action),
        actor=actor,
        status=AuditStatus(status),
        session_id=session_id,
        tenant_id=tenant_id,
        dept_id=dept_id,
        target=target,
        error=error,
        metadata=metadata or {},
    )


class AuditSink(ABC):
    """审计事件落盘接口。

    任意后端（内存、文件、消息队列、数据库）只需实现 :meth:`record` 即可接入
    统一的审计链路；:meth:`record_many` 提供默认批量实现。
    """

    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        """持久化单条审计事件。"""

    def record_many(self, events: list[AuditEvent]) -> None:
        """批量持久化审计事件，默认逐条写入。"""
        for event in events:
            self.record(event)


class InMemoryAuditSink(AuditSink):
    """进程内内存审计落盘（测试与嵌入式场景）。"""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        """追加一条审计事件到内存缓冲区。"""
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[AuditEvent]:
        """返回已记录事件的快照副本（防止外部修改内部状态）。"""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """清空已记录事件。"""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """返回已记录事件数量。"""
        with self._lock:
            return len(self._events)


class FileAuditSink(AuditSink):
    """JSONL 追加写审计落盘（每行一条 ``AuditEvent`` 的 JSON）。

    适合本地合规留痕：文件以 UTF-8 追加模式打开，每条事件独立成行、互不依赖，
    可随时被外部工具按行解析。写入受线程锁保护。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - 目录创建失败分支
            raise AuditSinkError(f"cannot prepare audit log dir: {exc}") from exc

    @property
    def path(self) -> Path:
        """返回审计日志文件路径。"""
        return self._path

    def record(self, event: AuditEvent) -> None:
        """将事件序列化为 JSON 并追加写入日志文件。

        Raises:
            AuditSinkError: 当文件 IO 失败时（如路径为目录）。
        """
        line = event.model_dump_json() + "\n"
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                raise AuditSinkError(f"failed to write audit log: {exc}") from exc

    def read(self) -> list[AuditEvent]:
        """读取并解析全部已落盘审计事件（按落盘顺序）。

        文件不存在时返回空列表。
        """
        if not self._path.exists():
            return []
        events: list[AuditEvent] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    events.append(AuditEvent.model_validate_json(raw))
        return events
