"""长会话 LLM 摘要压缩（超长会话触发摘要链，摘要写回 system 槽位并标记已压缩区间）。

D25 的时序窗口与 D26 的 Token 预算解决的是「**丢弃**多少历史」，代价是早期轮次
信息彻底消失。对动辄数百轮的慢病复诊会话，早期轮次往往包含过敏史、既往用药等
关键信息，直接丢弃有医疗风险。本模块提供第三种手段：把中间段旧消息**折叠成一条
摘要**，写回 system 槽位，只保留摘要 + 最近若干轮原文。

* **触发**：:class:`SummaryPolicy` 声明阈值（``max_messages`` 条数 /
  ``max_tokens`` 预算），可压缩消息数或 token 数超过阈值才触发，避免无谓的
  摘要调用。
* **保护位**：系统槽位消息、首条患者主诉消息、以及**历史摘要消息本身**永不被
  压缩（与 D25 / D26 语义一致），保证摘要幂等、不会反复自我压缩。
* **区间标记**：摘要消息的 ``additional_kwargs`` 写入 ``med_summary`` /
  ``compressed_range`` / ``compressed_count``，明确标记「哪一段被折叠了」，
  供上层审计与前端提示「历史已折叠」。
* **摘要链**：:class:`SummaryChain` 协议只要求 ``invoke({"messages": [...]})``，
  因此既可直接用 ``build_summary_prompt() | llm | StrOutputParser()`` 组成 LCEL
  链，也可注入任意替身；未注入时退化为零依赖、零网络的
  :class:`ExtractiveSummaryChain`（结构化拼接 + 截断），保证离线 / CI 可跑通。

设计取舍：所有判定与拼接只依赖消息类型、``additional_kwargs`` 中的结构化字段与
字符级截断，**不含任何文本语义解析、分词或中文文本预处理逻辑**。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, model_validator

from med_langchain_memory.stores.base import MedChatMessageHistory

from .token_budget import HeuristicTokenCounter, TokenCounter, message_text, stored_token_count
from .trimmer import find_chief_complaint, is_system_message, med_role_of

#: ``additional_kwargs`` 中标记「本消息是历史摘要」的键名（值为 ``True``）。
SUMMARY_FLAG_KEY = "med_summary"

#: ``additional_kwargs`` 中记录被折叠区间 ``[起始下标, 结束下标]``（闭区间）的键名。
COMPRESSED_RANGE_KEY = "compressed_range"

#: ``additional_kwargs`` 中记录被折叠消息条数的键名。
COMPRESSED_COUNT_KEY = "compressed_count"

#: 摘要消息正文的默认抬头，便于人工与程序识别摘要边界。
DEFAULT_SUMMARY_HEADER = "[medical session summary]"

#: 默认摘要指令（仅文本模板，不含任何解析逻辑）。
SUMMARY_INSTRUCTION = (
    "You are a medical record assistant. Compress the conversation excerpt below into "
    "a concise clinical summary.\n"
    "Rules: keep patient-reported symptoms, allergy and medication history, key findings, "
    "decisions and pending items; keep structured identifiers (IDs, phone numbers) exactly "
    "as written; never invent facts; answer with the summary text only."
)

#: 截断后缀（纯 ASCII，避免影响下游编码口径）。
_ELLIPSIS = "..."


class SummaryPolicy(BaseModel):
    """摘要压缩触发与保留策略（不可变）。

    Attributes:
        max_messages: 可压缩消息条数超过该值即触发摘要；``None`` 表示不按条数触发。
        max_tokens: 可压缩消息 token 数超过该值即触发摘要；``None`` 表示不按 token 触发。
        keep_recent_messages: 尾部保留原文的对话消息条数（不参与压缩），``0`` 表示
            全部可压缩消息都折叠进摘要。
        keep_system_messages: 是否保护系统槽位消息不被压缩。
        keep_chief_complaint: 是否保护首条患者主诉消息不被压缩。
        prefer_stored_token_count: 消息 ``additional_kwargs["token_count"]`` 为正整数时
            直接复用，避免重复编码；为 0（未计算）时回退到计数器。
        summary_header: 写回 system 槽位时摘要正文的抬头，用于标识摘要边界。

    Raises:
        pydantic.ValidationError: 两个触发阈值都未设置、阈值为非正数或
            ``keep_recent_messages`` 为负数时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_messages: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    keep_recent_messages: int = Field(default=6, ge=0)
    keep_system_messages: bool = True
    keep_chief_complaint: bool = True
    prefer_stored_token_count: bool = True
    summary_header: str = Field(default=DEFAULT_SUMMARY_HEADER, min_length=1)

    @model_validator(mode="after")
    def _validate_trigger(self) -> SummaryPolicy:
        """校验至少配置了一个触发阈值，避免策略静默失效。"""
        if self.max_messages is None and self.max_tokens is None:
            raise ValueError("at least one of max_messages or max_tokens must be configured")
        return self


def is_summary_message(message: BaseMessage) -> bool:
    """判断消息是否为历史摘要消息。

    Args:
        message: LangChain 消息。

    Returns:
        ``additional_kwargs["med_summary"]`` 为真值时返回 ``True``。
    """
    return bool(message.additional_kwargs.get(SUMMARY_FLAG_KEY))


def compressed_range_of(message: BaseMessage) -> tuple[int, int] | None:
    """读取摘要消息标记的被压缩区间。

    Args:
        message: LangChain 消息。

    Returns:
        ``(起始下标, 结束下标)`` 闭区间；缺失或格式非法时返回 ``None``。
    """
    raw = message.additional_kwargs.get(COMPRESSED_RANGE_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:  # noqa: PLR2004 - 区间固定两元素
        return None
    start, end = raw[0], raw[1]
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start:
        return None
    return (start, end)


def compressed_count_of(message: BaseMessage) -> int:
    """读取摘要消息记录的被折叠消息条数。

    Args:
        message: LangChain 消息。

    Returns:
        ``additional_kwargs["compressed_count"]`` 为非负整数时返回该值，否则返回 ``0``。
    """
    raw = message.additional_kwargs.get(COMPRESSED_COUNT_KEY)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def extract_summary_text(raw: Any) -> str:
    """从摘要链返回值中提取摘要正文。

    Args:
        raw: 摘要链输出，支持 ``str``、``BaseMessage`` 或含 ``text`` / ``content``
            字符串字段的映射。

    Returns:
        去除首尾空白后的摘要正文。

    Raises:
        TypeError: 返回值类型不受支持，或映射中不含字符串 ``text`` / ``content``。
        ValueError: 提取到的正文为空串时（视为摘要链失败，不写入空摘要）。
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, BaseMessage):
        text = message_text(raw)
    elif isinstance(raw, Mapping):
        candidate = raw.get("text", raw.get("content"))
        if not isinstance(candidate, str):
            raise TypeError("summary chain mapping output must contain a string 'text' field")
        text = candidate
    else:
        raise TypeError(f"unsupported summary chain output: {type(raw).__name__}")

    stripped = text.strip()
    if not stripped:
        raise ValueError("summary chain returned empty text")
    return stripped


def _truncate(text: str, limit: int) -> str:
    """按字符数截断文本，超长时追加省略号。"""
    if len(text) <= limit:
        return text
    return text[:limit] + _ELLIPSIS


def _coerce_payload(payload: Any) -> list[BaseMessage]:
    """把摘要链输入统一解析为消息列表。

    Args:
        payload: ``{"messages": [...]}`` 形式的映射、消息序列或单条消息。

    Returns:
        消息列表（可能为空）。

    Raises:
        TypeError: 载荷中存在非 ``BaseMessage`` 元素时。
    """
    raw: Any = payload.get("messages") if isinstance(payload, Mapping) else payload
    if raw is None:
        return []
    if isinstance(raw, BaseMessage):
        return [raw]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"unsupported summary chain input: {type(raw).__name__}")
    messages: list[BaseMessage] = []
    for item in raw:
        if not isinstance(item, BaseMessage):
            raise TypeError(f"summary chain input must contain messages, got {type(item).__name__}")
        messages.append(item)
    return messages


class SummaryChain(Protocol):
    """摘要链协议：任何具备 ``invoke`` 的可调用对象（LCEL 链 / 自定义替身）均可。

    约定入参为 ``{"messages": [BaseMessage, ...]}``，返回 ``str`` / ``BaseMessage``
    或含 ``text`` 字段的映射。
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """执行摘要调用并返回摘要结果。"""
        ...  # pragma: no cover - 协议桩，仅用于类型标注


def build_summary_prompt(instruction: str = SUMMARY_INSTRUCTION) -> ChatPromptTemplate:
    """构造摘要提示词模板，供 ``prompt | llm | StrOutputParser()`` 组装 LCEL 摘要链。

    Args:
        instruction: 系统指令文本，缺省为 :data:`SUMMARY_INSTRUCTION`。

    Returns:
        含系统指令与 ``messages`` 占位符的聊天提示词模板；调用时传入
        ``{"messages": [...]}`` 即可。
    """
    return ChatPromptTemplate.from_messages(
        [("system", instruction), ("placeholder", "{messages}")]
    )


class ExtractiveSummaryChain:
    """零依赖、零网络的确定性摘要链（LCEL 摘要链的离线替身）。

    不做任何语义理解，只按「角色标签 + 正文截断」结构化拼接被折叠区间，并做
    总长度截断。用于未注入真实 LLM 链时的降级路径，也便于测试断言。

    Args:
        max_chars_per_message: 单条消息正文保留的最大字符数。
        max_chars: 摘要总长度上限（字符数）。

    Raises:
        ValueError: 任一长度上限为非正数时。
    """

    def __init__(self, *, max_chars_per_message: int = 120, max_chars: int = 800) -> None:
        """初始化确定性摘要链。

        Args:
            max_chars_per_message: 单条消息正文保留的最大字符数。
            max_chars: 摘要总长度上限。

        Raises:
            ValueError: 任一长度上限为非正数时。
        """
        if max_chars_per_message <= 0 or max_chars <= 0:
            raise ValueError("max_chars_per_message and max_chars must be positive")
        self._max_chars_per_message = max_chars_per_message
        self._max_chars = max_chars

    @property
    def max_chars_per_message(self) -> int:
        """单条消息正文保留的最大字符数。"""
        return self._max_chars_per_message

    @property
    def max_chars(self) -> int:
        """摘要总长度上限。"""
        return self._max_chars

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        """按结构化拼接生成摘要文本。

        Args:
            input: ``{"messages": [...]}`` 形式的映射或消息序列。
            config: 兼容 LCEL ``invoke`` 签名的配置参数，本实现忽略。
            **kwargs: 兼容 LCEL ``invoke`` 签名的额外参数，本实现忽略。

        Returns:
            逐行 ``[角色] 正文`` 拼接后的摘要文本（超长时截断）。

        Raises:
            ValueError: 未提供任何消息时（不生成空摘要）。
            TypeError: 载荷中存在非消息元素时。
        """
        messages = _coerce_payload(input)
        if not messages:
            raise ValueError("no messages to summarize")
        lines: list[str] = []
        for message in messages:
            role = med_role_of(message)
            label = role.value if role is not None else message.type
            body = _truncate(message_text(message), self._max_chars_per_message)
            lines.append(f"[{label}] {body}")
        return _truncate("\n".join(lines), self._max_chars)


@dataclass
class SummaryReport:
    """一次摘要压缩的结构化报告。

    Attributes:
        applied: 是否实际执行了摘要压缩（未触发或无内容可压缩时为 ``False``）。
        trigger: 触发压缩的阈值名（``"max_messages"`` / ``"max_tokens"``），未触发为 ``None``。
        compressed_count: 被折叠进摘要的消息条数。
        compressed_range: 被折叠区间 ``(起始下标, 结束下标)``（闭区间），未压缩为 ``None``。
        summary_tokens: 摘要消息占用的 token 数（按本压缩器的计数口径）。
        kept_messages: 压缩后最终上下文的消息条数（含摘要消息）。
        warnings: 告警 / 说明文本列表。
    """

    applied: bool = False
    trigger: str | None = None
    compressed_count: int = 0
    compressed_range: tuple[int, int] | None = None
    summary_tokens: int = 0
    kept_messages: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class SummaryResult:
    """摘要压缩结果（消息 + 报告 + 摘要消息）。

    Attributes:
        messages: 压缩后的消息列表，保持原始时序升序，摘要消息插入在被折叠区间处。
        report: 本次压缩的结构化报告。
        summary_message: 写回 system 槽位的摘要消息；未压缩时为 ``None``。
    """

    messages: list[BaseMessage]
    report: SummaryReport
    summary_message: SystemMessage | None = None


@dataclass(frozen=True)
class _Plan:
    """一次压缩的规划结果（内部使用）。

    Attributes:
        protected: 受保护、不参与压缩的消息下标。
        recent: 尾部保留原文的消息下标。
        compressed: 将被折叠进摘要的消息下标。
        trigger: 命中的触发阈值名，未触发为 ``None``。
    """

    protected: tuple[int, ...]
    recent: tuple[int, ...]
    compressed: tuple[int, ...]
    trigger: str | None


class SummaryCompressor:
    """把长会话中间段旧消息折叠为一条 system 摘要消息的压缩器。

    压缩只做结构化切片与摘要链调用，输出保持原始时序升序，且不修改入参。
    """

    def __init__(
        self,
        policy: SummaryPolicy,
        *,
        chain: SummaryChain | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        """初始化压缩器。

        Args:
            policy: 摘要触发与保留策略。
            chain: 摘要链；``None`` 时使用零依赖的 :class:`ExtractiveSummaryChain`。
            counter: token 计数器；``None`` 时使用启发式计数器（零依赖）。
        """
        self._policy = policy
        self._chain: SummaryChain = ExtractiveSummaryChain() if chain is None else chain
        self._counter = HeuristicTokenCounter() if counter is None else counter

    @property
    def policy(self) -> SummaryPolicy:
        """当前生效的摘要策略。"""
        return self._policy

    @property
    def chain(self) -> SummaryChain:
        """当前生效的摘要链。"""
        return self._chain

    @property
    def counter(self) -> TokenCounter:
        """当前生效的 token 计数器。"""
        return self._counter

    def count(self, messages: Sequence[BaseMessage]) -> int:
        """按本压缩器的计数口径统计消息序列的 token 合计。

        Args:
            messages: 消息序列。

        Returns:
            token 数合计。
        """
        return sum(self._message_cost(message) for message in messages)

    def _message_cost(self, message: BaseMessage) -> int:
        """单条消息的 token 占用（优先复用已存 token 数）。"""
        if self._policy.prefer_stored_token_count:
            stored = stored_token_count(message)
            if stored is not None:
                return stored
        return self._counter.count_message(message)

    def _protected_indices(self, messages: Sequence[BaseMessage]) -> set[int]:
        """标记受保护消息下标（系统槽位 / 主诉 / 已有摘要消息）。"""
        protected: set[int] = {index for index, m in enumerate(messages) if is_summary_message(m)}
        if self._policy.keep_system_messages:
            protected.update(index for index, m in enumerate(messages) if is_system_message(m))
        if self._policy.keep_chief_complaint:
            chief = find_chief_complaint(messages)
            if chief is not None:
                protected.add(chief)
        return protected

    def _trigger(self, messages: Sequence[BaseMessage], candidates: Sequence[int]) -> str | None:
        """判断是否命中触发阈值，返回命中的阈值名。"""
        policy = self._policy
        if policy.max_messages is not None and len(candidates) > policy.max_messages:
            return "max_messages"
        if policy.max_tokens is not None:
            used = sum(self._message_cost(messages[index]) for index in candidates)
            if used > policy.max_tokens:
                return "max_tokens"
        return None

    def _plan(self, messages: Sequence[BaseMessage]) -> _Plan:
        """规划本次压缩：保护位 -> 可压缩候选 -> 尾部保留 -> 触发判定。"""
        protected = self._protected_indices(messages)
        candidates = [index for index in range(len(messages)) if index not in protected]
        keep_recent = self._policy.keep_recent_messages
        recent = tuple(candidates[-keep_recent:]) if keep_recent > 0 else ()
        recent_set = set(recent)
        compressed = tuple(index for index in candidates if index not in recent_set)
        return _Plan(
            protected=tuple(sorted(protected)),
            recent=recent,
            compressed=compressed,
            trigger=self._trigger(messages, candidates),
        )

    def should_compress(self, messages: Sequence[BaseMessage]) -> bool:
        """判断给定消息序列是否**既命中阈值、又确有可折叠内容**。

        Args:
            messages: 时序升序的消息序列。

        Returns:
            命中 ``max_messages`` / ``max_tokens`` 阈值且存在将被折叠的消息时返回 ``True``；
            空序列、未命中阈值或可折叠内容为空时返回 ``False``。
        """
        plan = self._plan(messages)
        return plan.trigger is not None and bool(plan.compressed)

    def compress(self, messages: Sequence[BaseMessage]) -> SummaryResult:
        """按策略把中间段旧消息折叠为一条 system 摘要消息。

        流程：规划保护位与可折叠区间 -> 命中阈值且区间非空时调用摘要链 -> 构造带区间
        标记的摘要消息 -> 按原始时序把「保护消息 + 摘要 + 尾部原文」重新拼装。
        未触发阈值、无内容可折叠或入参为空时原样返回消息并给出说明性告警。

        Args:
            messages: 时序升序的消息序列。

        Returns:
            压缩结果；``messages`` 为最终上下文，``report`` 为结构化报告，
            ``summary_message`` 为写回 system 槽位的摘要消息。

        Raises:
            TypeError: 摘要链返回值类型不受支持时。
            ValueError: 摘要链返回空正文时。
        """
        source = list(messages)
        if not source:
            return SummaryResult(
                messages=[],
                report=SummaryReport(warnings=["empty message list"]),
                summary_message=None,
            )

        plan = self._plan(source)
        if plan.trigger is None:
            return SummaryResult(
                messages=source,
                report=SummaryReport(kept_messages=len(source)),
                summary_message=None,
            )
        if not plan.compressed:
            return SummaryResult(
                messages=source,
                report=SummaryReport(
                    trigger=plan.trigger,
                    kept_messages=len(source),
                    warnings=[f"summary triggered by {plan.trigger} but nothing to compress"],
                ),
                summary_message=None,
            )

        excerpt = [source[index] for index in plan.compressed]
        text = extract_summary_text(self._chain.invoke({"messages": excerpt}))
        summary_message = self._build_summary_message(text, plan.compressed)

        kept = sorted(set(plan.protected) | set(plan.recent))
        result_messages: list[BaseMessage] = [source[index] for index in kept]
        insert_at = sum(1 for index in kept if index < plan.compressed[0])
        result_messages.insert(insert_at, summary_message)

        compressed_range = (plan.compressed[0], plan.compressed[-1])
        report = SummaryReport(
            applied=True,
            trigger=plan.trigger,
            compressed_count=len(plan.compressed),
            compressed_range=compressed_range,
            summary_tokens=self._counter.count_message(summary_message),
            kept_messages=len(result_messages),
            warnings=[
                f"compressed {len(plan.compressed)} message(s) into a session summary "
                f"(range={compressed_range[0]}..{compressed_range[1]})"
            ],
        )
        return SummaryResult(
            messages=result_messages, report=report, summary_message=summary_message
        )

    def compress_history(self, history: MedChatMessageHistory) -> SummaryResult:
        """压缩存储句柄中的全量会话消息。

        Args:
            history: 会话历史句柄。

        Returns:
            压缩结果（消息为 LangChain 格式）。
        """
        return self.compress(history.messages)

    def _build_summary_message(self, text: str, compressed: Sequence[int]) -> SystemMessage:
        """构造带被折叠区间标记的摘要 system 消息。"""
        return SystemMessage(
            content=f"{self._policy.summary_header}\n{text}",
            additional_kwargs={
                SUMMARY_FLAG_KEY: True,
                COMPRESSED_RANGE_KEY: [compressed[0], compressed[-1]],
                COMPRESSED_COUNT_KEY: len(compressed),
            },
        )
