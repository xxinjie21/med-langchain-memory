"""多租户 / 科室命名空间隔离。

医院场景下同一进程会同时服务多家机构（tenant）与多个科室（dept），任何一次会话
访问都必须先落到 ``tenant_id:dept_id:session_id`` 三段式命名空间，再校验调用方
身份是否有权访问该命名空间。本模块提供三个最小原语：

* :class:`TenantContext` —— 调用方身份（租户 + 科室作用域 + 操作者），不可变；
* :class:`SessionNamespace` —— 会话命名空间值对象，负责三段式键与存储键互转；
* :class:`TenantGuard` —— 守卫，绑定身份后校验命名空间归属，越权即拒绝。

设计取舍：隔离判定完全基于**结构化 ID 比较**（纯字符串等值），不做任何文本内容
解析；与存储层 ``MedChatMessageHistory.assert_tenant`` 语义保持一致，本模块负责
「取用前的身份校验」，存储层负责「落库后的归属校验」，两层互补。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.exceptions import TenantIsolationError, ValidationError
from med_langchain_memory.stores.base import MedChatMessageHistory

#: 命名空间键分隔符。
NAMESPACE_SEPARATOR = ":"

#: 统一存储键前缀（与 ``SessionMeta.storage_key`` 对齐）。
STORAGE_KEY_PREFIX = "med:chat"

#: 三段式命名空间键的字段数。
_NAMESPACE_PARTS = 3

#: 允许为空的科室 ID：空串表示该上下文覆盖租户内全部科室（租户级视角）。
OptionalDeptId = Annotated[str, StringConstraints(max_length=64, pattern=r"^[A-Za-z0-9_.-]*$")]

#: 操作者标识（医生工号 / 服务账号），仅用于审计留痕，不参与权限判定。
ActorId = Annotated[str, StringConstraints(max_length=64)]


def build_namespace_key(tenant_id: str, dept_id: str, session_id: str) -> str:
    """构造三段式命名空间键 ``tenant_id:dept_id:session_id``。

    Args:
        tenant_id: 医院/机构租户 ID。
        dept_id: 科室 ID。
        session_id: 会话 ID。

    Returns:
        以 ``:`` 连接的三段式键。

    Raises:
        ValidationError: 任一段为空字符串时。
    """
    for name, value in (("tenant_id", tenant_id), ("dept_id", dept_id), ("session_id", session_id)):
        if not value:
            raise ValidationError(f"{name} must not be empty in a namespace key")
    return NAMESPACE_SEPARATOR.join((tenant_id, dept_id, session_id))


def parse_namespace_key(key: str) -> tuple[str, str, str]:
    """解析三段式命名空间键为 ``(tenant_id, dept_id, session_id)``。

    Args:
        key: 形如 ``h-a:cardiology:s-1`` 的命名空间键。

    Returns:
        按 ``(tenant_id, dept_id, session_id)`` 顺序排列的三元组。

    Raises:
        ValidationError: 段数不等于 3 或任一段为空时。
    """
    parts = key.split(NAMESPACE_SEPARATOR)
    if len(parts) != _NAMESPACE_PARTS:
        raise ValidationError(
            f"invalid namespace key: {key!r}; expect tenant{NAMESPACE_SEPARATOR}"
            f"dept{NAMESPACE_SEPARATOR}session"
        )
    tenant_id, dept_id, session_id = parts
    for name, value in (("tenant_id", tenant_id), ("dept_id", dept_id), ("session_id", session_id)):
        if not value:
            raise ValidationError(f"{name} must not be empty in namespace key {key!r}")
    return tenant_id, dept_id, session_id


class SessionNamespace(BaseModel):
    """会话命名空间值对象（不可变）。

    与存储层键规范一致：``storage_key`` 为 ``med:chat:{tenant}:{dept}:{session}``，
    而 :attr:`key` 去掉前缀，只保留 ``tenant:dept:session`` 三段，便于跨层传递与日志脱敏。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: IdStr
    tenant_id: IdStr
    dept_id: IdStr
    patient_id: OptionalDeptId = ""

    @property
    def key(self) -> str:
        """返回三段式命名空间键 ``tenant_id:dept_id:session_id``。"""
        return build_namespace_key(self.tenant_id, self.dept_id, self.session_id)

    @property
    def storage_key(self) -> str:
        """返回统一存储键 ``med:chat:{tenant_id}:{dept_id}:{session_id}``。"""
        return f"{STORAGE_KEY_PREFIX}:{self.key}"

    def to_config(self) -> dict[str, Any]:
        """转换为 :class:`MedRunnableWithMessageHistory` 可直接使用的调用配置。"""
        return {
            "configurable": {
                "session_id": self.session_id,
                "tenant_id": self.tenant_id,
                "dept_id": self.dept_id,
                "patient_id": self.patient_id,
            }
        }

    @classmethod
    def parse(cls, key: str, patient_id: str = "") -> SessionNamespace:
        """从三段式命名空间键解析出命名空间对象。

        Args:
            key: 形如 ``h-a:cardiology:s-1`` 的键。
            patient_id: 归属患者 ID，缺省为空。

        Returns:
            解析得到的命名空间。

        Raises:
            ValidationError: 键格式非法或字段不符合 ID 规范时。
        """
        tenant_id, dept_id, session_id = parse_namespace_key(key)
        return cls(
            session_id=session_id,
            tenant_id=tenant_id,
            dept_id=dept_id,
            patient_id=patient_id,
        )

    @classmethod
    def from_history(cls, history: MedChatMessageHistory) -> SessionNamespace:
        """从既有的会话历史句柄提取命名空间。

        Args:
            history: 已创建的会话历史实例。

        Returns:
            与该句柄一致的命名空间。
        """
        return cls(
            session_id=history.session_id,
            tenant_id=history.tenant_id,
            dept_id=history.dept_id,
            patient_id=history.patient_id,
        )


class TenantContext(BaseModel):
    """调用方租户身份上下文（不可变）。

    Attributes:
        tenant_id: 调用方所属租户，必须非空。
        dept_id: 调用方科室作用域；**空串表示租户内不限科室**（如院级管理员、
            跨科室会诊服务），非空时只能访问本科室会话。
        actor_id: 操作者标识（医生工号 / 服务账号），仅审计留痕，不参与判定。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: IdStr
    dept_id: OptionalDeptId = ""
    actor_id: ActorId = ""

    @property
    def is_tenant_wide(self) -> bool:
        """是否为租户级视角（未限定科室）。"""
        return self.dept_id == ""

    def allows(self, tenant_id: str, dept_id: str) -> bool:
        """判断本身份是否允许访问给定的租户/科室命名空间。

        Args:
            tenant_id: 目标命名空间租户 ID。
            dept_id: 目标命名空间科室 ID。

        Returns:
            允许访问返回 ``True``，否则 ``False``（不抛异常，便于过滤场景）。
        """
        if tenant_id != self.tenant_id:
            return False
        return self.is_tenant_wide or dept_id == self.dept_id

    def assert_access(self, tenant_id: str, dept_id: str) -> None:
        """校验访问归属，越权时拒绝。

        Args:
            tenant_id: 目标命名空间租户 ID。
            dept_id: 目标命名空间科室 ID。

        Raises:
            ValidationError: 租户或科室为空时（命名空间不完整，无从判定）。
            TenantIsolationError: 跨租户或跨科室越权访问时。
        """
        if not tenant_id:
            raise ValidationError("tenant_id must not be empty when checking tenant access")
        if not dept_id:
            raise ValidationError("dept_id must not be empty when checking tenant access")
        if not self.allows(tenant_id, dept_id):
            raise TenantIsolationError(
                f"access denied: context {self.tenant_id}:{self.dept_id or '*'} "
                f"cannot access {tenant_id}:{dept_id}"
            )

    def namespace(
        self, session_id: str, dept_id: str | None = None, patient_id: str = ""
    ) -> SessionNamespace:
        """按本身份构造会话命名空间。

        Args:
            session_id: 会话 ID。
            dept_id: 目标科室；``None`` 时取本上下文科室。
            patient_id: 归属患者 ID，缺省为空。

        Returns:
            构造出的命名空间。

        Raises:
            ValidationError: 本科室身份未显式给出 ``dept_id`` 之外的情况下，
                上下文为租户级却未指定目标科室时。
        """
        resolved_dept = self.dept_id if dept_id is None else dept_id
        if not resolved_dept:
            raise ValidationError("dept_id is required for a tenant-wide context")
        return SessionNamespace(
            session_id=session_id,
            tenant_id=self.tenant_id,
            dept_id=resolved_dept,
            patient_id=patient_id,
        )


class TenantGuard:
    """命名空间守卫：绑定调用方身份，取用会话前做越权拦截。"""

    def __init__(self, context: TenantContext) -> None:
        """初始化守卫。

        Args:
            context: 调用方租户身份上下文。
        """
        self._context = context

    @property
    def context(self) -> TenantContext:
        """守卫绑定的租户身份上下文。"""
        return self._context

    def assert_access(self, tenant_id: str, dept_id: str) -> None:
        """校验给定命名空间归属。

        Raises:
            ValidationError: 命名空间不完整时。
            TenantIsolationError: 越权访问时。
        """
        self._context.assert_access(tenant_id, dept_id)

    def guard_history(self, history: MedChatMessageHistory) -> MedChatMessageHistory:
        """校验已创建的会话历史句柄归属，通过后原样返回。

        Args:
            history: 待校验的会话历史实例。

        Returns:
            校验通过的同一句柄（便于链式书写）。

        Raises:
            TenantIsolationError: 句柄所属命名空间不在本身份作用域内时。
        """
        self.assert_access(history.tenant_id, history.dept_id)
        return history

    def bind(
        self, get_session_history: Callable[..., MedChatMessageHistory]
    ) -> Callable[..., MedChatMessageHistory]:
        """包装 ``get_session_history``：先鉴权再创建句柄。

        被包装的可调用对象需遵循本项目约定签名
        ``(session_id, tenant_id="", dept_id="", patient_id="")``；
        命名空间字段留空时回退到守卫绑定的上下文。

        Args:
            get_session_history: 原始的会话历史工厂闭包。

        Returns:
            带越权拦截的等价闭包。
        """
        context = self._context

        def guarded(
            session_id: str,
            tenant_id: str = "",
            dept_id: str = "",
            patient_id: str = "",
        ) -> MedChatMessageHistory:
            resolved_tenant = tenant_id or context.tenant_id
            resolved_dept = dept_id or context.dept_id
            context.assert_access(resolved_tenant, resolved_dept)
            return get_session_history(
                session_id,
                tenant_id=resolved_tenant,
                dept_id=resolved_dept,
                patient_id=patient_id,
            )

        return guarded
