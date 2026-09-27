"""会话接口的请求/响应 DTO（Pydantic v2）。

DTO 与领域模型分离：领域模型带状态机与存储键等行为，DTO 只做线格式契约。
所有 DTO 均为 ``extra="forbid"``（拒绝未声明字段，避免前端拼写错误被静默吞掉），
响应 DTO 额外 ``frozen=True``（构造后不可变）。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.domain.session import SessionMeta, SessionStatus


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
