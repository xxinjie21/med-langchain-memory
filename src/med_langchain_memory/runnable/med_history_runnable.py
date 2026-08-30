"""医疗增强版 ``RunnableWithMessageHistory`` 骨架。

把 LangChain 的 :class:`RunnableWithMessageHistory` 与本项目存储工厂
:class:`~med_langchain_memory.stores.factory.StoreFactory` 对接：调用方只需声明
后端名与会话命名空间，即可按需获得 ``MedChatMessageHistory`` 实例，无需直接
实例化具体存储类。

会话命名空间由必填的 ``session_id`` 与可配置的 ``tenant_id`` / ``dept_id`` /
``patient_id`` 三段式组成（与存储层键规范一致，``patient_id`` 缺省时回退默认值）。
后续迭代（D24 多租户隔离、D25+ 上下文裁剪/摘要压缩）将在此基础上叠加增强能力，
本迭代只完成「继承封装 + 注入存储工厂」的最小可用骨架。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import (
    GetSessionHistoryCallable,
    RunnableWithMessageHistory,
)

from med_langchain_memory.stores import MedChatMessageHistory, StoreFactory

#: 除 ``session_id`` 外参与会话命名空间的配置字段（与存储键规范一致）。
_NAMESPACE_FIELDS: tuple[str, ...] = ("tenant_id", "dept_id", "patient_id")


class MedRunnableWithMessageHistory(RunnableWithMessageHistory):
    """基于存储工厂注入的会话历史增强 Runnable。

    相比原生 :class:`RunnableWithMessageHistory` 需手工传入 ``get_session_history``
    闭包，本类改为接收**已注册的后端名 + 命名空间**，在内部组装出闭包并调用父类。
    任意已注册的 ``StoreFactory`` 后端（当前含 ``memory`` / ``file`` / ``redis`` /
    ``redis_cluster`` / ``elasticsearch``，缺失的可选依赖后端不会注册）均可热插拔。

    Args:
        runnable: 被包裹的下游 Runnable（通常是 LLM 链或提示词组合）。
        backend: 已注册的存储后端名。
        store_factory: 提供 ``create(...)`` 方法的工厂类/实例，默认全局 ``StoreFactory``。
        ttl_seconds: 会话级 TTL（秒），``None`` 表示永不过期（仅原生支持 TTL 的后端生效）。
        store_options: 透传给具体存储实现构造函数的额外参数（如连接串、编码模式）。
        input_messages_key: 包裹 Runnable 输入中承载用户消息的键。
        output_messages_key: 包裹 Runnable 输出中承载消息的键。
        history_messages_key: 注入历史消息的占位符键。
        default_namespace: 命名空间默认值，当调用方在 ``config`` 中留空时取用。

    Raises:
        StoreNotFoundError: 通过本 Runnable 取用时 ``backend`` 未注册（在 ``invoke`` 时触发）。
    """

    def __init__(
        self,
        runnable: Any,
        *,
        backend: str,
        store_factory: type[StoreFactory] = StoreFactory,
        ttl_seconds: int | None = None,
        store_options: Mapping[str, Any] | None = None,
        input_messages_key: str | None = None,
        output_messages_key: str | None = None,
        history_messages_key: str | None = None,
        default_namespace: Mapping[str, str] | None = None,
    ) -> None:
        self._backend = backend
        self._store_factory = store_factory
        self._ttl_seconds = ttl_seconds
        self._store_options = dict(store_options or {})
        self._default_namespace = dict(default_namespace or {})

        get_history = self._make_get_session_history()
        factory_config = self._build_history_factory_config()

        super().__init__(
            runnable,
            get_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            history_messages_key=history_messages_key,
            history_factory_config=factory_config,
        )

    # ------------------------------------------------------------------ #
    # 内部构造助手
    # ------------------------------------------------------------------ #
    def _make_get_session_history(self) -> GetSessionHistoryCallable:
        """构造 ``get_session_history`` 闭包。

        Returns:
            签名与 ``history_factory_config`` 字段一一对应的可调用对象，
            调用时按命名空间实例化 ``MedChatMessageHistory``；命名空间字段留空时
            回退到 :attr:`_default_namespace` 中登记的默认值。
        """
        backend = self._backend
        factory = self._store_factory
        ttl = self._ttl_seconds
        options = self._store_options
        defaults = self._default_namespace

        def get_session_history(
            session_id: str,
            tenant_id: str = "",
            dept_id: str = "",
            patient_id: str = "",
        ) -> MedChatMessageHistory:
            resolved_tenant = tenant_id or defaults.get("tenant_id", "")
            resolved_dept = dept_id or defaults.get("dept_id", "")
            resolved_patient = patient_id or defaults.get("patient_id", "")
            return factory.create(
                backend,
                session_id=session_id,
                tenant_id=resolved_tenant,
                dept_id=resolved_dept,
                patient_id=resolved_patient,
                ttl_seconds=ttl,
                **options,
            )

        return get_session_history

    @staticmethod
    def _build_history_factory_config() -> list[ConfigurableFieldSpec]:
        """构造 ``history_factory_config``：会话 ID + 三段式命名空间字段。"""
        specs: list[ConfigurableFieldSpec] = [
            ConfigurableFieldSpec(id="session_id", annotation=str, is_shared=False)
        ]
        for field in _NAMESPACE_FIELDS:
            specs.append(ConfigurableFieldSpec(id=field, annotation=str, is_shared=False))
        return specs

    # ------------------------------------------------------------------ #
    # 调用辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_config(
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
    ) -> dict[str, Any]:
        """构造调用本 Runnable 所需的 ``config`` 字典。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。

        Returns:
            ``{"configurable": {...}}`` 形式的调用配置。
        """
        return {
            "configurable": {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "dept_id": dept_id,
                "patient_id": patient_id,
            }
        }
