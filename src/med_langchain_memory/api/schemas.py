"""会话与消息接口的请求/响应 DTO（Pydantic v2）。

DTO 与领域模型分离：领域模型带状态机与存储键等行为，DTO 只做线格式契约。
所有 DTO 均为 ``extra="forbid"``（拒绝未声明字段，避免前端拼写错误被静默吞掉），
响应 DTO 额外 ``frozen=True``（构造后不可变）。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator

from med_langchain_memory.domain.message import IdStr, MedMessage, MessageRole
from med_langchain_memory.domain.session import SessionMeta, SessionStatus

#: 单次追加消息的最大条数（防止一个请求写入过多消息）。
MAX_APPEND_BATCH = 100


class SessionCreateRequest(BaseModel):
    """创建会话请求体。

    Attributes:
        session_id: 会话 ID；``None`` 表示由服务端生成 UUID。
        patient_id: 患者 ID。
        metadata: 扩展标签（如 ``{"visit_type": "first"}``）。
    """

    model_config = ConfigDict(extra="forbid")

    session_id: IdStr | None = None
    patient_id: IdStr
    metadata: dict[str, str] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """会话详情响应体。

    Attributes:
        session_id: 会话 ID。
        tenant_id: 租户 ID。
        dept_id: 科室 ID。
        patient_id: 患者 ID。
        status: 会话状态。
        message_count: 已累积消息条数。
        created_at: 创建时间（epoch 毫秒）。
        updated_at: 最近更新时间（epoch 毫秒）。
        metadata: 扩展标签。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    tenant_id: str
    dept_id: str
    patient_id: str
    status: SessionStatus
    message_count: int
    created_at: int
    updated_at: int
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_meta(cls, meta: SessionMeta) -> SessionResponse:
        """由领域模型构造响应体。

        Args:
            meta: 会话元数据。

        Returns:
            对应的 :class:`SessionResponse`。
        """
        return cls(
            session_id=meta.session_id,
            tenant_id=meta.tenant_id,
            dept_id=meta.dept_id,
            patient_id=meta.patient_id,
            status=meta.status,
            message_count=meta.message_count,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
            metadata=dict(meta.metadata),
        )


class SessionListResponse(BaseModel):
    """会话分页列表响应体。

    Attributes:
        total: 命中总数（与分页无关）。
        limit: 本次请求的单页条数。
        offset: 本次请求的起始偏移量。
        items: 当前页会话列表。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    items: list[SessionResponse] = Field(default_factory=list)


class MessageCreate(BaseModel):
    """单条待追加消息。

    命名空间（``tenant_id`` / ``dept_id``）与 ``patient_id`` 由服务端从会话元数据补齐，
    调用方**不得**自行声明，避免越权写入其他租户。

    Attributes:
        role: 消息发送方角色。
        content: 消息正文（按原文落库，脱敏在读取时按开关执行）。
        token_count: 正文 token 数，调用方已知时可直接给出，缺省为 0。
        metadata: 扩展标签。
        message_id: 消息 ID；``None`` 表示由服务端生成 UUIDv7。
        created_at: 创建时间（epoch 毫秒）；``None`` 表示取服务端当前时间。
    """

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)
    message_id: str | None = None
    created_at: int | None = Field(default=None, gt=0)

    @field_validator("message_id")
    @classmethod
    def _validate_message_id(cls, v: str | None) -> str | None:
        """``message_id`` 若显式给出则必须为合法 UUID。"""
        if v is None:
            return None
        uuid.UUID(v)
        return v


class MessageAppendRequest(BaseModel):
    """批量追加消息请求体。

    Attributes:
        messages: 待追加消息，单次 1..``MAX_APPEND_BATCH`` 条。
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[MessageCreate] = Field(min_length=1, max_length=MAX_APPEND_BATCH)


class MessageResponse(BaseModel):
    """单条消息响应体。

    Attributes:
        message_id: 消息 ID。
        session_id: 归属会话 ID。
        tenant_id: 租户 ID。
        dept_id: 科室 ID。
        patient_id: 患者 ID。
        role: 消息发送方角色。
        content: 消息正文（可能已脱敏）。
        token_count: 正文 token 数。
        masked: 该条内容是否已过脱敏引擎。
        created_at: 创建时间（epoch 毫秒）。
        metadata: 扩展标签。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str
    session_id: str
    tenant_id: str
    dept_id: str
    patient_id: str
    role: MessageRole
    content: str
    token_count: int
    masked: bool
    created_at: int
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_med(cls, message: MedMessage) -> MessageResponse:
        """由领域模型构造响应体。

        Args:
            message: 领域消息（脱敏与否由调用方在传入前决定）。

        Returns:
            对应的 :class:`MessageResponse`。
        """
        return cls(
            message_id=message.message_id,
            session_id=message.session_id,
            tenant_id=message.tenant_id,
            dept_id=message.dept_id,
            patient_id=message.patient_id,
            role=message.role,
            content=message.content,
            token_count=message.token_count,
            masked=message.masked,
            created_at=message.created_at,
            metadata=dict(message.metadata),
        )


class MessageAppendResponse(BaseModel):
    """批量追加消息响应体。

    Attributes:
        session_id: 会话 ID。
        appended: 本次成功写入的消息条数。
        message_count: 写入后会话累积的消息总数。
        items: 本次写入的消息（按时序升序）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    appended: int = Field(ge=0)
    message_count: int = Field(ge=0)
    items: list[MessageResponse] = Field(default_factory=list)


class MessageListResponse(BaseModel):
    """消息游标分页响应体。

    Attributes:
        session_id: 会话 ID。
        limit: 本次请求的单页条数。
        masked: 本次响应内容是否已脱敏。
        next_cursor: 下一页游标；``None`` 表示已到末页。
        items: 当前页消息（按时序升序）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    limit: int = Field(ge=1)
    masked: bool
    next_cursor: str | None = None
    items: list[MessageResponse] = Field(default_factory=list)
