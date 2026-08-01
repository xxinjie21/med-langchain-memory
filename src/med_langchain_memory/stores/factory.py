"""存储适配器工厂与注册器。

各存储实现通过装饰器把自己登记到全局注册表::

    @StoreFactory.register("memory")
    class InMemoryMedHistory(MedChatMessageHistory):
        ...

上层只需给出后端名与会话命名空间即可拿到实例，无需 import 具体实现类::

    history = StoreFactory.create("memory", session_id="s-1", tenant_id="h-a", ...)

亦支持配置驱动实例化（:class:`StoreConfig` 可由 YAML/JSON/环境变量填充）::

    history = StoreFactory.create_from_config(
        StoreConfig(backend="redis", ttl_seconds=3600, options={"url": "redis://..."}),
        session_id="s-1", tenant_id="h-a", dept_id="cardiology", patient_id="p-1",
    )
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from med_langchain_memory.exceptions import (
    StorageError,
    StoreNotFoundError,
    StoreRegistrationError,
)

from .base import MedChatMessageHistory

#: 合法后端名：小写字母、数字、下划线与短横线。
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

StoreT = TypeVar("StoreT", bound=MedChatMessageHistory)


def _normalize(name: str) -> str:
    """归一化后端名：去空白并转小写。"""
    if not isinstance(name, str):
        raise StoreRegistrationError(f"backend name must be a string, got {type(name).__name__}")
    return name.strip().lower()


class StoreConfig(BaseModel):
    """存储后端实例化配置。

    Attributes:
        backend: 已注册的后端名（自动去空白并转小写）。
        ttl_seconds: 会话级 TTL（秒），``None`` 表示永不过期。
        options: 传给具体存储实现构造函数的额外关键字参数。
    """

    model_config = ConfigDict(extra="forbid")

    backend: str
    ttl_seconds: int | None = Field(default=None, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, value: str) -> str:
        """校验并归一化后端名。"""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("backend must not be empty")
        return normalized


class StoreFactory:
    """存储适配器注册表与实例化入口（全部为类方法，进程内单例语义）。"""

    _registry: ClassVar[dict[str, type[MedChatMessageHistory]]] = {}

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #
    @classmethod
    def register(
        cls, name: str, *, override: bool = False
    ) -> Callable[[type[StoreT]], type[StoreT]]:
        """返回把存储实现类登记到注册表的类装饰器。

        Args:
            name: 后端名，仅允许小写字母/数字/下划线/短横线。
            override: 是否允许覆盖同名已注册实现。

        Returns:
            原样返回被装饰类的装饰器函数。

        Raises:
            StoreRegistrationError: 名称非法、重复注册（``override=False``）、
                被装饰对象不是 :class:`MedChatMessageHistory` 的具体子类时。
        """
        key = _normalize(name)
        if not _NAME_PATTERN.match(key):
            raise StoreRegistrationError(f"invalid backend name: {name!r}")
        if key in cls._registry and not override:
            raise StoreRegistrationError(
                f"backend {key!r} already registered by {cls._registry[key].__name__}"
            )

        def decorator(target: type[StoreT]) -> type[StoreT]:
            if not (inspect.isclass(target) and issubclass(target, MedChatMessageHistory)):
                raise StoreRegistrationError(
                    f"{target!r} is not a subclass of MedChatMessageHistory"
                )
            if inspect.isabstract(target):
                raise StoreRegistrationError(f"cannot register abstract store {target.__name__}")
            cls._registry[key] = target
            return target

        return decorator

    @classmethod
    def unregister(cls, name: str) -> None:
        """注销已注册的后端。

        Raises:
            StoreNotFoundError: 该后端未注册时。
        """
        key = _normalize(name)
        if key not in cls._registry:
            raise StoreNotFoundError(f"unknown store backend: {name!r}")
        del cls._registry[key]

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """判断后端名是否已注册。"""
        return _normalize(name) in cls._registry

    @classmethod
    def available(cls) -> list[str]:
        """返回全部已注册后端名（字典序）。"""
        return sorted(cls._registry)

    @classmethod
    def get(cls, name: str) -> type[MedChatMessageHistory]:
        """按名称取回存储实现类。

        Raises:
            StoreNotFoundError: 该后端未注册时（错误信息附带可用后端列表）。
        """
        key = _normalize(name)
        try:
            return cls._registry[key]
        except KeyError as exc:
            raise StoreNotFoundError(
                f"unknown store backend: {name!r}; available: {cls.available()}"
            ) from exc

    # ------------------------------------------------------------------ #
    # 实例化
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        backend: str,
        *,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        ttl_seconds: int | None = None,
        **options: Any,
    ) -> MedChatMessageHistory:
        """实例化指定后端的会话历史。

        Args:
            backend: 已注册的后端名。
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            ttl_seconds: 会话级 TTL（秒），``None`` 表示永不过期。
            **options: 透传给具体实现构造函数的额外参数（如连接串、编码模式）。

        Returns:
            对应后端的 :class:`MedChatMessageHistory` 实例。

        Raises:
            StoreNotFoundError: 后端未注册时。
            StorageError: 实现类不接受给定参数时（构造签名不匹配）。
        """
        store_cls = cls.get(backend)
        try:
            return store_cls(
                session_id=session_id,
                tenant_id=tenant_id,
                dept_id=dept_id,
                patient_id=patient_id,
                ttl_seconds=ttl_seconds,
                **options,
            )
        except TypeError as exc:
            raise StorageError(
                f"cannot build {store_cls.__name__} for backend {backend!r}: {exc}"
            ) from exc

    @classmethod
    def create_from_config(
        cls,
        config: StoreConfig,
        *,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
    ) -> MedChatMessageHistory:
        """按 :class:`StoreConfig` 配置实例化会话历史。

        Args:
            config: 后端名 + TTL + 额外参数的配置对象。
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。

        Returns:
            对应后端的 :class:`MedChatMessageHistory` 实例。

        Raises:
            StoreNotFoundError: 配置中的后端未注册时。
            StorageError: 实现类不接受配置给出的参数时。
        """
        return cls.create(
            config.backend,
            session_id=session_id,
            tenant_id=tenant_id,
            dept_id=dept_id,
            patient_id=patient_id,
            ttl_seconds=config.ttl_seconds,
            **config.options,
        )
