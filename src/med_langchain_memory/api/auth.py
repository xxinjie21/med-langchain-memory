"""API Key 鉴权与科室 scope 授权。

医疗场景下「谁能读哪个租户哪个科室的会话」必须显式约束，本模块提供最小可用的
静态密钥方案（无 JWT、无 OAuth、无用户体系）：

* :class:`ApiKeyRecord` —— 一把 API Key 的授权档案：所属租户、允许访问的科室集合、
  启用开关；``dept_ids`` 含通配符 ``"*"`` 表示允许该租户下全部科室；
* :class:`AuthPrincipal` —— 认证成功后得到的调用主体（只含授权必需字段），
  由 :class:`ApiKeyAuthenticator` 从档案投影而来，避免把启用开关等管理字段外泄到路由层；
* :class:`ApiKeyAuthenticator` —— 进程内密钥注册表，**只保存密钥的 SHA-256 摘要**，
  认证时对请求头值取摘要后查表，明文密钥不驻留内存；
* :func:`authorize_scope` —— 把「主体 + 请求声明的租户/科室」校验为 :class:`SessionScope`，
  租户不符或科室不在白名单时抛 :class:`~med_langchain_memory.exceptions.AuthorizationError`。

密钥比较使用摘要查表（``dict`` 命中）而非逐字符比较，天然规避时序侧信道；
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.exceptions import (
    AuthenticationError,
    AuthorizationError,
    IntegrityError,
    ValidationError,
)
from med_langchain_memory.stores.session_repository import SessionScope

#: 默认的 API Key 请求头名。
DEFAULT_API_KEY_HEADER: Final = "X-API-Key"

#: 科室通配符：表示该密钥可访问所属租户下的全部科室。
WILDCARD_DEPT: Final = "*"

#: 合法科室标识：通配符 ``*`` 或与 :data:`~med_langchain_memory.domain.message.IdStr` 同规格的 ID。
_DEPT_ID_PATTERN: Final = re.compile(r"^(?:\*|[A-Za-z0-9_.-]{1,64})$")


def hash_api_key(raw_key: str) -> str:
    """计算 API Key 的 SHA-256 摘要（小写十六进制）。

    Args:
        raw_key: 明文 API Key。

    Returns:
        64 字符的小写十六进制摘要；同一输入恒得同一结果，不同输入碰撞概率可忽略。
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def dept_allowed(dept_ids: frozenset[str], dept_id: str) -> bool:
    """判断科室白名单是否覆盖目标科室。

    匹配为**精确匹配**（``cardio`` 不覆盖 ``cardio-oncology``），仅通配符 ``"*"`` 例外。

    Args:
        dept_ids: 允许访问的科室集合。
        dept_id: 目标科室 ID。

    Returns:
        命中白名单或白名单含通配符时为 ``True``。
    """
    return WILDCARD_DEPT in dept_ids or dept_id in dept_ids


class ApiKeyRecord(BaseModel):
    """一把 API Key 的授权档案。

    Attributes:
        key_id: 密钥标识（用于审计与错误信息，**不是**密钥本身）。
        tenant_id: 该密钥绑定的医院/机构租户 ID。
        dept_ids: 允许访问的科室集合，非空；含 ``"*"`` 表示该租户下全部科室。
        label: 人类可读备注（如「门诊医生站」），可选。
        enabled: 是否启用；停用后认证一律失败。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: IdStr
    tenant_id: IdStr
    dept_ids: frozenset[str]
    label: str | None = None
    enabled: bool = True

    @field_validator("dept_ids")
    @classmethod
    def _validate_dept_ids(cls, value: frozenset[str]) -> frozenset[str]:
        """校验科室集合非空且每个元素为通配符或合法科室 ID。

        Args:
            value: 待校验的科室集合。

        Returns:
            原样返回的科室集合。

        Raises:
            ValueError: 集合为空或含非法科室标识时（由 pydantic 包装为校验错误）。
        """
        if not value:
            raise ValueError("dept_ids must not be empty")
        for dept_id in value:
            if not _DEPT_ID_PATTERN.match(dept_id):
                raise ValueError(f"invalid dept id: {dept_id!r}")
        return value

    def allows_dept(self, dept_id: str) -> bool:
        """判断本档案是否允许访问目标科室。

        Args:
            dept_id: 目标科室 ID。

        Returns:
            允许时为 ``True``。
        """
        return dept_allowed(self.dept_ids, dept_id)


class AuthPrincipal(BaseModel):
    """认证通过后的调用主体（授权所需字段的最小投影）。

    Attributes:
        key_id: 命中的密钥标识。
        tenant_id: 主体绑定的租户 ID。
        dept_ids: 主体可访问的科室集合（可能含通配符 ``"*"``）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: str = Field(min_length=1)
    tenant_id: IdStr
    dept_ids: frozenset[str]

    def allows_dept(self, dept_id: str) -> bool:
        """判断主体是否允许访问目标科室。

        Args:
            dept_id: 目标科室 ID。

        Returns:
            允许时为 ``True``。
        """
        return dept_allowed(self.dept_ids, dept_id)


class ApiKeyAuthenticator:
    """进程内 API Key 注册表与认证器。

    只保存密钥摘要，支持自定义请求头名；线程安全性由「注册阶段写、运行阶段只读」
    的用法保证（注册表在应用启动前构造完毕，运行期仅做查表）。

    Example:
        >>> record = ApiKeyRecord(key_id="k-1", tenant_id="hosp-a", dept_ids=frozenset({"cardio"}))
        >>> auth = ApiKeyAuthenticator()
        >>> _ = auth.add_key("secret-key", record)
        >>> auth.authenticate("secret-key").tenant_id
        'hosp-a'
    """

    def __init__(self, *, header_name: str = DEFAULT_API_KEY_HEADER) -> None:
        """初始化空注册表。

        Args:
            header_name: 承载 API Key 的 HTTP 头名。

        Raises:
            ValidationError: ``header_name`` 为空串时。
        """
        if not header_name:
            raise ValidationError("api key header name must not be empty")
        self.header_name = header_name
        self._records: dict[str, ApiKeyRecord] = {}
        self._key_ids: set[str] = set()

    def add_key(self, raw_key: str, record: ApiKeyRecord) -> ApiKeyRecord:
        """注册一把 API Key。

        Args:
            raw_key: 明文 API Key。
            record: 该密钥的授权档案。

        Returns:
            已注册的授权档案。

        Raises:
            ValidationError: ``raw_key`` 为空串时。
            IntegrityError: 密钥标识或密钥本身已注册时。
        """
        if not raw_key:
            raise ValidationError("api key must not be empty")
        digest = hash_api_key(raw_key)
        if record.key_id in self._key_ids:
            raise IntegrityError(f"duplicate api key id: {record.key_id}")
        if digest in self._records:
            raise IntegrityError("api key already registered")
        self._records[digest] = record
        self._key_ids.add(record.key_id)
        return record

    @classmethod
    def from_records(
        cls,
        records: Mapping[str, ApiKeyRecord],
        *,
        header_name: str = DEFAULT_API_KEY_HEADER,
    ) -> ApiKeyAuthenticator:
        """由「明文密钥 → 授权档案」映射批量构造认证器。

        Args:
            records: 明文 API Key 到授权档案的映射。
            header_name: 承载 API Key 的 HTTP 头名。

        Returns:
            已装载全部密钥的认证器。
        """
        authenticator = cls(header_name=header_name)
        for raw_key, record in records.items():
            authenticator.add_key(raw_key, record)
        return authenticator

    def authenticate(self, raw_key: str | None) -> AuthPrincipal:
        """校验请求携带的 API Key 并返回调用主体。

        Args:
            raw_key: 请求头中的明文密钥；``None`` / 空串 / 纯空白均视为未携带。

        Returns:
            认证通过的主体 :class:`AuthPrincipal`。

        Raises:
            AuthenticationError: 密钥缺失、未注册或已停用时（对应 HTTP 401）。
        """
        candidate = (raw_key or "").strip()
        if not candidate:
            raise AuthenticationError("missing api key")
        record = self._records.get(hash_api_key(candidate))
        if record is None:
            raise AuthenticationError("invalid api key")
        if not record.enabled:
            raise AuthenticationError(f"api key disabled: {record.key_id}")
        return AuthPrincipal(
            key_id=record.key_id,
            tenant_id=record.tenant_id,
            dept_ids=record.dept_ids,
        )


def authorize_scope(principal: AuthPrincipal, *, tenant_id: str, dept_id: str) -> SessionScope:
    """校验主体对「租户 + 科室」的访问权限并构造命名空间坐标。

    Args:
        principal: 已认证的调用主体。
        tenant_id: 请求声明的租户 ID。
        dept_id: 请求声明的科室 ID。

    Returns:
        校验通过的 :class:`SessionScope`。

    Raises:
        AuthorizationError: 租户与主体绑定租户不一致，或科室不在主体白名单内时（对应 HTTP 403）。
    """
    if principal.tenant_id != tenant_id:
        raise AuthorizationError(
            f"api key {principal.key_id} is not authorized for tenant {tenant_id}"
        )
    if not principal.allows_dept(dept_id):
        raise AuthorizationError(f"api key {principal.key_id} is not authorized for dept {dept_id}")
    return SessionScope(tenant_id=tenant_id, dept_id=dept_id)
