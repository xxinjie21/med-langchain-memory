"""医疗增强版 ``RunnableWithMessageHistory`` 骨架。

把 LangChain 的 :class:`RunnableWithMessageHistory` 与本项目存储工厂
:class:`~med_langchain_memory.stores.factory.StoreFactory` 对接：调用方只需声明
后端名与会话命名空间，即可按需获得 ``MedChatMessageHistory`` 实例，无需直接
实例化具体存储类。

会话命名空间由必填的 ``session_id`` 与可配置的 ``tenant_id`` / ``dept_id`` /
``patient_id`` 三段式组成（与存储层键规范一致，``patient_id`` 缺省时回退默认值）。
D24 起支持注入 :class:`~.tenant.TenantContext`：若给定调用方身份，取用历史前会先做
``tenant_id:dept_id`` 归属校验，跨租户 / 跨科室访问直接拒绝，不再落到存储层。
D25 起可选注入 :class:`~.trimmer.ContextWindowPolicy`，通过 :meth:`trim_context`
对取用的历史做时序滑动窗口裁剪；D26 起可再叠加 :class:`~.token_budget.TokenBudgetPolicy`，
由 :meth:`build_context` 按「时序窗口 → Token 预算」两级流水线产出最终上下文，
并返回带告警的结构化报告；D27 起可再注入 :class:`~.summarizer.SummaryPolicy` 与
摘要链，流水线升级为「时序窗口 → LLM 摘要压缩 → Token 预算」三级：中间段旧消息被
折叠成一条带区间标记的 system 摘要消息后再做预算兜底裁剪。D28 起可注入
:class:`~.lock.LockPolicy`（可选 Redis 客户端），:meth:`invoke` 会在「读历史 →
调模型 → 写历史」整段临界区外包一层会话锁，Redis 客户端缺失时自动降级为进程内
本地锁；:meth:`session_lock` 供调用方跨多次调用自行圈定临界区。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import (
    GetSessionHistoryCallable,
    RunnableWithMessageHistory,
)

from med_langchain_memory.exceptions import LockError
from med_langchain_memory.stores import MedChatMessageHistory, StoreFactory

from .lock import LockPolicy, SessionLock, SessionLockManager
from .summarizer import (
    SummaryChain,
    SummaryCompressor,
    SummaryPolicy,
    SummaryReport,
    SummaryResult,
)
from .tenant import TenantContext
from .token_budget import (
    TokenBudgetPolicy,
    TokenBudgetTrimmer,
    TokenCounter,
    TokenTrimReport,
    TokenTrimResult,
    resolve_token_counter,
)
from .trimmer import ContextWindowPolicy, TimeWindowTrimmer

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
        tenant_context: 调用方租户身份；非空时对本 Runnable 取用的每个会话做归属校验。
        trim_policy: 时序裁剪策略；非空时可用 :meth:`trim_context` 裁剪历史上下文。
        token_budget: Token 预算策略；非空时 :meth:`build_context` 会在时序裁剪之后
            追加一层预算裁剪。
        token_counter: token 计数器实例或名称（``"heuristic"`` / ``"tiktoken"`` /
            tiktoken 编码表名）；``None`` 表示使用零依赖的启发式计数器。
        summarizer: 摘要压缩策略；非空时 :meth:`summarize_context` 可把中间段旧消息
            折叠成一条 system 摘要消息，:meth:`build_context` 会在时序裁剪与预算裁剪
            之间插入该层。
        summary_chain: 摘要链（满足 ``invoke({"messages": [...]})`` 契约，如
            ``build_summary_prompt() | llm | StrOutputParser()``）；``None`` 时使用
            零依赖的确定性摘要链。
        lock_policy: 并发会话锁策略；非空时 :meth:`invoke` 会在临界区外自动加锁。
        lock_client: Redis 客户端；``None`` 时锁降级为进程内本地锁。
        lock_manager: 现成的锁管理器；给定后优先于 ``lock_policy`` / ``lock_client``。

    Raises:
        StoreNotFoundError: 通过本 Runnable 取用时 ``backend`` 未注册（在 ``invoke`` 时触发）。
        TenantIsolationError: 注入了 ``tenant_context`` 而调用方请求了越权命名空间时。
        ValueError: ``token_counter`` 为未知名称时。
        LockAcquisitionError: 启用会话锁且等待超时仍未获取到锁时。
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
        tenant_context: TenantContext | None = None,
        trim_policy: ContextWindowPolicy | None = None,
        token_budget: TokenBudgetPolicy | None = None,
        token_counter: TokenCounter | str | None = None,
        summarizer: SummaryPolicy | None = None,
        summary_chain: SummaryChain | None = None,
        lock_policy: LockPolicy | None = None,
        lock_client: Any | None = None,
        lock_manager: SessionLockManager | None = None,
    ) -> None:
        options = dict(store_options or {})
        defaults = dict(default_namespace or {})
        counter = resolve_token_counter(token_counter)
        resolved_manager = lock_manager
        if resolved_manager is None and lock_policy is not None:
            resolved_manager = SessionLockManager(client=lock_client, policy=lock_policy)

        get_history = self._make_get_session_history(
            backend=backend,
            factory=store_factory,
            ttl=ttl_seconds,
            options=options,
            defaults=defaults,
            context=tenant_context,
        )
        factory_config = self._build_history_factory_config()

        super().__init__(
            runnable,
            get_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            history_messages_key=history_messages_key,
            history_factory_config=factory_config,
        )

        # 注意：RunnableWithMessageHistory 是 pydantic 模型，其 ``__init__`` 会整体
        # 替换实例 ``__dict__``，故私有状态必须在 super().__init__() **之后**赋值。
        self._backend = backend
        self._store_factory = store_factory
        self._ttl_seconds = ttl_seconds
        self._store_options = options
        self._default_namespace = defaults
        self._tenant_context = tenant_context
        self._trim_policy = trim_policy
        self._token_budget = token_budget
        self._token_counter = counter
        self._summarizer = summarizer
        self._summary_chain = summary_chain
        self._lock_manager = resolved_manager

    # ------------------------------------------------------------------ #
    # 内部构造助手
    # ------------------------------------------------------------------ #
    def _make_get_session_history(
        self,
        *,
        backend: str,
        factory: type[StoreFactory],
        ttl: int | None,
        options: Mapping[str, Any],
        defaults: Mapping[str, str],
        context: TenantContext | None,
    ) -> GetSessionHistoryCallable:
        """构造 ``get_session_history`` 闭包。

        Args:
            backend: 已注册的后端名。
            factory: 存储工厂（类或兼容 ``create`` 签名的对象）。
            ttl: 会话级 TTL（秒）。
            options: 透传给存储实现的额外参数。
            defaults: 命名空间默认值。
            context: 调用方租户身份；``None`` 表示不做归属校验。

        Returns:
            签名与 ``history_factory_config`` 字段一一对应的可调用对象，
            调用时按命名空间实例化 ``MedChatMessageHistory``；命名空间字段留空时
            依次回退到 ``defaults`` 与 ``context``；注入了租户身份时，解析完命名空间
            后先做归属校验再创建句柄。
        """

        def get_session_history(
            session_id: str,
            tenant_id: str = "",
            dept_id: str = "",
            patient_id: str = "",
        ) -> MedChatMessageHistory:
            resolved_tenant = tenant_id or defaults.get("tenant_id", "")
            resolved_dept = dept_id or defaults.get("dept_id", "")
            if context is not None:
                resolved_tenant = resolved_tenant or context.tenant_id
                resolved_dept = resolved_dept or context.dept_id
                context.assert_access(resolved_tenant, resolved_dept)
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
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def tenant_context(self) -> TenantContext | None:
        """注入的调用方租户身份；``None`` 表示本 Runnable 不做归属校验。"""
        return self._tenant_context

    @property
    def trim_policy(self) -> ContextWindowPolicy | None:
        """注入的时序裁剪策略；``None`` 表示不裁剪。"""
        return self._trim_policy

    @property
    def token_budget(self) -> TokenBudgetPolicy | None:
        """注入的 Token 预算策略；``None`` 表示不做预算裁剪。"""
        return self._token_budget

    @property
    def token_counter(self) -> TokenCounter:
        """本 Runnable 使用的 token 计数器。"""
        return self._token_counter

    @property
    def summarizer(self) -> SummaryPolicy | None:
        """注入的摘要压缩策略；``None`` 表示不做摘要压缩。"""
        return self._summarizer

    @property
    def summary_chain(self) -> SummaryChain | None:
        """注入的摘要链；``None`` 表示使用确定性摘要链。"""
        return self._summary_chain

    @property
    def lock_manager(self) -> SessionLockManager | None:
        """注入的并发会话锁管理器；``None`` 表示本 Runnable 不加锁。"""
        return self._lock_manager

    # ------------------------------------------------------------------ #
    # 调用辅助
    # ------------------------------------------------------------------ #
    def trim_context(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_ms: int | None = None,
    ) -> list[BaseMessage]:
        """按注入的裁剪策略裁剪上下文消息。

        Args:
            messages: 时序升序的消息序列。
            now_ms: 当前 epoch 毫秒，缺省取系统时间。

        Returns:
            裁剪后的消息列表；未注入策略时返回入参的等价副本。
        """
        if self._trim_policy is None:
            return list(messages)
        return TimeWindowTrimmer(self._trim_policy).trim(messages, now_ms=now_ms)

    def summarize_context(self, messages: Sequence[BaseMessage]) -> SummaryResult:
        """按注入的摘要策略把中间段旧消息折叠为一条 system 摘要消息。

        Args:
            messages: 时序升序的消息序列。

        Returns:
            摘要压缩结果；未注入摘要策略时返回入参的等价副本，且
            ``report.applied`` 为 ``False``。
        """
        if self._summarizer is None:
            return SummaryResult(messages=list(messages), report=SummaryReport())
        return SummaryCompressor(
            self._summarizer,
            chain=self._summary_chain,
            counter=self._token_counter,
        ).compress(messages)

    def build_context(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_ms: int | None = None,
    ) -> TokenTrimResult:
        """按「时序窗口 → 摘要压缩 → Token 预算」三级流水线构建最终上下文。

        依次应用 :attr:`trim_policy`、:attr:`summarizer`、:attr:`token_budget`
        （各自未注入时该级为空操作）；摘要消息是 system 消息，因此天然受预算裁剪的
        系统槽位保护，不会被预算层丢弃。

        Args:
            messages: 时序升序的消息序列。
            now_ms: 当前 epoch 毫秒，缺省取系统时间（仅时序窗口生效时使用）。

        Returns:
            裁剪结果：``messages`` 为最终上下文，``report`` 为预算裁剪报告
            （未注入预算策略时 ``report.applied`` 为 ``False``），``summary`` 为
            摘要压缩报告（未注入摘要策略时为 ``None``）。
        """
        trimmed = self.trim_context(messages, now_ms=now_ms)
        summarized = self.summarize_context(trimmed)
        context = summarized.messages
        summary_report = summarized.report if self._summarizer is not None else None

        if self._token_budget is None:
            return TokenTrimResult(
                messages=context,
                report=TokenTrimReport(),
                summary=summary_report,
            )
        budget_result = TokenBudgetTrimmer(self._token_budget, counter=self._token_counter).trim(
            context
        )
        return TokenTrimResult(
            messages=budget_result.messages,
            report=budget_result.report,
            summary=summary_report,
        )

    def resolve_namespace(self, config: Mapping[str, Any] | None) -> tuple[str, str, str] | None:
        """从调用配置中解析会话命名空间 ``(tenant_id, dept_id, session_id)``。

        解析顺序与 ``get_session_history`` 闭包一致：配置项 → 构造时默认值 →
        注入的租户身份；本方法只做解析，不做归属校验（校验仍由存储句柄闭包完成）。

        Args:
            config: LangChain 调用配置（``{"configurable": {...}}``）。

        Returns:
            三段式命名空间；缺少 ``session_id`` 或 ``config`` 非映射时返回 ``None``。
        """
        if not isinstance(config, Mapping):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, Mapping):
            return None
        session_id = configurable.get("session_id")
        if not session_id or not isinstance(session_id, str):
            return None
        tenant_id = configurable.get("tenant_id") or ""
        dept_id = configurable.get("dept_id") or ""
        if self._tenant_context is not None:
            tenant_id = tenant_id or self._tenant_context.tenant_id
            dept_id = dept_id or self._tenant_context.dept_id
        resolved_tenant = tenant_id or self._default_namespace.get("tenant_id", "")
        resolved_dept = dept_id or self._default_namespace.get("dept_id", "")
        return resolved_tenant, resolved_dept, session_id

    @contextmanager
    def session_lock(
        self,
        session_id: str,
        *,
        tenant_id: str = "",
        dept_id: str = "",
        timeout_ms: int | None = None,
    ) -> Iterator[SessionLock]:
        """以会话锁圈定一段临界区（供调用方跨多次调用自行加锁）。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID；缺省回退到默认命名空间 / 注入的租户身份。
            dept_id: 科室 ID；缺省回退规则同上。
            timeout_ms: 获取锁的最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Yields:
            已持有的锁实例。

        Raises:
            LockError: 本 Runnable 未启用会话锁时。
            LockAcquisitionError: 等待超时仍未获取到锁时。
        """
        manager = self._lock_manager
        if manager is None:
            raise LockError("session locking is not enabled on this runnable")
        resolved_tenant = tenant_id or self._default_namespace.get("tenant_id", "")
        resolved_dept = dept_id or self._default_namespace.get("dept_id", "")
        with manager.hold(
            resolved_tenant,
            resolved_dept,
            session_id,
            timeout_ms=timeout_ms,
        ) as lock:
            yield lock

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """调用下游 Runnable；启用会话锁时在整段临界区外自动加锁。

        加锁范围覆盖「取历史 → 下游调用 → 写回历史」全过程，避免并发问诊下的
        上下文错乱；配置中缺少 ``session_id`` 时不加锁（由父类抛出参数缺失异常）。

        Args:
            input: 下游 Runnable 的输入。
            config: 调用配置（须含 ``configurable.session_id`` 等命名空间字段）。
            **kwargs: 透传给父类的额外关键字参数。

        Returns:
            下游 Runnable 的输出。

        Raises:
            LockAcquisitionError: 启用会话锁且等待超时仍未获取到锁时。
        """
        manager = self._lock_manager
        namespace = self.resolve_namespace(config) if manager is not None else None
        if manager is None or namespace is None:
            return super().invoke(input, config, **kwargs)
        tenant_id, dept_id, session_id = namespace
        with manager.hold(tenant_id, dept_id, session_id):
            return super().invoke(input, config, **kwargs)

    @staticmethod
    def build_config(
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str = "",
    ) -> dict[str, Any]:
        """构造调用本 Runnable 所需的 ``config`` 字典。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID，缺省为空。

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
