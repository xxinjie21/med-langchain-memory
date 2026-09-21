"""Token 预算裁剪（token 计数 + 预算内贪心保留最近消息）。

D25 的时序窗口解决了「保留多久 / 保留多少条」，但相同条数在不同长度消息下
Token 成本可能相差一个数量级。本模块在时序裁剪之后追加一层**纯结构化**的
Token 预算控制：

* **计数**：:class:`TiktokenTokenCounter` 按需懒加载 ``tiktoken`` 编码表；
  缺包或编码表拉取失败时**自动降级**为纯标准库的 :class:`HeuristicTokenCounter`，
  保证离线 / CI 环境不会因外部依赖而中断；也可直接注入自定义计数器替身。
* **裁剪**：从会话尾部向前**贪心**累加，保留「最近的连续后缀」直到触达有效预算
  （``max_tokens - reserve_tokens``）；某条消息放不下即停止，**不跳条**，
  避免把中间一轮问答挖空导致对话断裂。
* **保护位**：系统槽位与首条患者主诉消息永不被裁掉（与 D25 语义一致），
  且被保护消息同样计入预算占用。
* **告警**：预算用满 / 用超时产出结构化告警（:attr:`TokenTrimReport.warnings`），
  供上层记录日志或触发降级决策。

设计取舍：计数与裁剪只依赖消息类型、``additional_kwargs`` 中的结构化字段
（``med_role`` / ``token_count``）与消息正文的字符构成，**不含任何文本语义
解析或分词预处理逻辑**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from med_langchain_memory.stores.base import MedChatMessageHistory

from .trimmer import find_chief_complaint, is_system_message

#: ``additional_kwargs`` 中承载消息已算好 token 数的键名（写入时为 0 表示未计算）。
TOKEN_COUNT_KEY = "token_count"

#: 默认使用的 tiktoken 编码表（GPT-4 / GPT-3.5 系列通用）。
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"

#: 允许通过名称选择的 tiktoken 编码表白名单（避免拼写错误被静默降级）。
TIKTOKEN_ENCODINGS = frozenset(
    {"cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2"}
)

#: 启发式计数中，平均多少个 ASCII 字符算一个 token。
_ASCII_CHARS_PER_TOKEN = 4


class TokenCounter(ABC):
    """Token 计数器抽象。

    子类只需实现 :meth:`count`（单段文本计数），消息级与序列级计数由基类
    按统一规则组合，保证不同实现之间口径一致。
    """

    @abstractmethod
    def count(self, text: str) -> int:
        """统计一段文本的 token 数。

        Args:
            text: 待统计文本，允许为空串。

        Returns:
            非负 token 数。
        """

    def count_message(self, message: BaseMessage) -> int:
        """统计单条消息正文的 token 数。

        Args:
            message: LangChain 消息。

        Returns:
            该消息正文对应的 token 数（非字符串内容按结构化块拼接后统计）。
        """
        return self.count(message_text(message))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        """统计消息序列的 token 数合计。

        Args:
            messages: 消息序列，允许为空。

        Returns:
            所有消息 token 数之和；空序列返回 0。
        """
        return sum(self.count_message(message) for message in messages)


def message_text(message: BaseMessage) -> str:
    """提取消息中可计数的文本。

    医疗会话正文均为纯文本（``str``）；若上游使用了结构化内容块，则按
    ``text`` 字段拼接，未知块退化为空串。

    Args:
        message: LangChain 消息。

    Returns:
        用于计数的文本，可能为空串。
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            raw = block.get("text")
            if isinstance(raw, str):
                parts.append(raw)
    return "".join(parts)


class HeuristicTokenCounter(TokenCounter):
    """纯标准库启发式计数器（零依赖、零网络、结果确定）。

    规则：非 ASCII 字符（中文/全角标点等）按 1 token/字计，ASCII 字符按
    ``4 字符 ≈ 1 token`` 向上取整。用于 tiktoken 不可用时的降级路径，
    也便于测试中构造可预期的固定计数。
    """

    def count(self, text: str) -> int:
        """按字符构成估算 token 数。

        Args:
            text: 待统计文本。

        Returns:
            空串返回 0；否则返回「非 ASCII 字符数 + ASCII 字符数 / 4 向上取整」。
        """
        if not text:
            return 0
        wide = sum(1 for char in text if not char.isascii())
        narrow = len(text) - wide
        return wide + -(-narrow // _ASCII_CHARS_PER_TOKEN)


class TiktokenTokenCounter(TokenCounter):
    """基于 ``tiktoken`` 的精确计数器（懒加载 + 自动降级）。

    ``tiktoken`` 的编码表在首次使用时才加载，且可能需要联网拉取 BPE 文件。
    为避免把外部依赖的不确定性传导给会话链路，本类在**导入失败**或**编码表
    加载失败**时自动降级为 :class:`HeuristicTokenCounter`，并通过
    :attr:`degraded` 暴露降级状态供上层观测。

    Args:
        encoding_name: tiktoken 编码表名，须在 :data:`TIKTOKEN_ENCODINGS` 白名单内。
        fallback: 降级使用的计数器，缺省为 :class:`HeuristicTokenCounter`。
    """

    def __init__(
        self,
        encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
        *,
        fallback: TokenCounter | None = None,
    ) -> None:
        """初始化计数器（不触发任何网络或重计算）。

        Args:
            encoding_name: tiktoken 编码表名。
            fallback: 降级计数器；``None`` 时使用启发式计数器。
        """
        self._encoding_name = encoding_name
        self._fallback = HeuristicTokenCounter() if fallback is None else fallback
        self._encoding: Any = None
        self._degraded = False

    @property
    def encoding_name(self) -> str:
        """本计数器请求的 tiktoken 编码表名。"""
        return self._encoding_name

    @property
    def degraded(self) -> bool:
        """是否已降级为备用计数器（``tiktoken`` 不可用或编码表加载失败）。"""
        return self._degraded

    def _resolve_encoding(self) -> Any:
        """懒加载编码表；失败时标记降级并返回 ``None``。"""
        if self._encoding is not None or self._degraded:
            return self._encoding
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except Exception:  # noqa: BLE001 - 缺包/网络失败/编码名非法统一降级，不阻断会话链路
            self._degraded = True
        return self._encoding

    def count(self, text: str) -> int:
        """统计文本 token 数；编码表不可用时走降级计数器。

        Args:
            text: 待统计文本。

        Returns:
            非负 token 数。
        """
        encoding = self._resolve_encoding()
        if encoding is None:
            return self._fallback.count(text)
        return int(len(encoding.encode(text)))


def resolve_token_counter(spec: TokenCounter | str | None) -> TokenCounter:
    """把「计数器实例 / 名称 / 空」统一解析为 :class:`TokenCounter`。

    Args:
        spec: ``None`` 或 ``"heuristic"`` 返回启发式计数器；``"tiktoken"`` 返回
            默认编码表的 tiktoken 计数器；:data:`TIKTOKEN_ENCODINGS` 中的编码表名
            返回对应 tiktoken 计数器；已是 :class:`TokenCounter` 实例时原样返回。

    Returns:
        解析后的计数器实例。

    Raises:
        ValueError: ``spec`` 为非空字符串但不在已知名称/编码表白名单内。
    """
    if spec is None:
        return HeuristicTokenCounter()
    if isinstance(spec, TokenCounter):
        return spec
    if spec in {"heuristic", "char", "default"}:
        return HeuristicTokenCounter()
    if spec == "tiktoken":
        return TiktokenTokenCounter()
    if spec in TIKTOKEN_ENCODINGS:
        return TiktokenTokenCounter(spec)
    raise ValueError(f"unknown token counter: {spec!r}")


class TokenBudgetPolicy(BaseModel):
    """Token 预算策略（不可变）。

    Attributes:
        max_tokens: 上下文总预算（含预留输出额度），必须为正。
        reserve_tokens: 为模型输出预留的 token 数，有效预算为
            ``max_tokens - reserve_tokens``，必须小于 ``max_tokens``。
        keep_system_messages: 是否保护系统槽位消息不被裁剪。
        keep_chief_complaint: 是否保护首条患者主诉消息不被裁剪。
        warn_ratio: 有效预算使用率超过该比例时产出「预算将尽」告警，取值 ``(0, 1]``；
            取 1.0 表示仅在真正用满/用超时才告警。
        prefer_stored_token_count: 消息 ``additional_kwargs["token_count"]``
            为正整数时直接复用，避免重复编码；为 0（未计算）时回退到计数器。

    Raises:
        pydantic.ValidationError: 字段越界或 ``reserve_tokens >= max_tokens`` 时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(ge=1)
    reserve_tokens: int = Field(default=0, ge=0)
    keep_system_messages: bool = True
    keep_chief_complaint: bool = True
    warn_ratio: float = Field(default=0.9, gt=0.0, le=1.0)
    prefer_stored_token_count: bool = True

    @model_validator(mode="after")
    def _validate_reserve(self) -> TokenBudgetPolicy:
        """校验预留额度必须小于总预算，保证有效预算为正。"""
        if self.reserve_tokens >= self.max_tokens:
            raise ValueError("reserve_tokens must be smaller than max_tokens")
        return self

    @property
    def effective_budget(self) -> int:
        """可用于承载上下文的有效预算（``max_tokens - reserve_tokens``）。"""
        return self.max_tokens - self.reserve_tokens


@dataclass
class TokenTrimReport:
    """一次 Token 预算裁剪的结构化报告。

    Attributes:
        applied: 是否实际执行了预算裁剪（未注入策略时为 ``False``）。
        budget_tokens: 策略声明的总预算。
        effective_budget: 扣除预留后的有效预算。
        used_tokens: 裁剪后保留消息的 token 合计（含被保护消息）。
        dropped_tokens: 被裁掉消息的 token 合计。
        kept_messages: 保留的消息条数。
        dropped_messages: 被裁掉的消息条数。
        over_budget: 裁剪后仍超预算（仅可能因被保护消息自身过大）。
        warnings: 告警文本列表，按「超限 → 将尽 → 已裁剪」顺序排列。
    """

    applied: bool = False
    budget_tokens: int = 0
    effective_budget: int = 0
    used_tokens: int = 0
    dropped_tokens: int = 0
    kept_messages: int = 0
    dropped_messages: int = 0
    over_budget: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class TokenTrimResult:
    """Token 预算裁剪结果（消息 + 报告）。

    Attributes:
        messages: 裁剪后的消息列表，保持原始时序升序。
        report: 本次裁剪的结构化报告。
    """

    messages: list[BaseMessage]
    report: TokenTrimReport


def stored_token_count(message: BaseMessage) -> int | None:
    """读取消息上已算好的 token 数。

    Args:
        message: LangChain 消息。

    Returns:
        ``additional_kwargs["token_count"]`` 为正整数时返回该值；缺失、为 0
        （表示未计算）或类型非法时返回 ``None``。
    """
    raw = message.additional_kwargs.get(TOKEN_COUNT_KEY)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


class TokenBudgetTrimmer:
    """在 Token 预算内贪心保留最近消息的裁剪器。

    裁剪只做结构化过滤，输出保持原始时序升序，且不修改入参。
    """

    def __init__(
        self,
        policy: TokenBudgetPolicy,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        """初始化裁剪器。

        Args:
            policy: Token 预算策略。
            counter: token 计数器；``None`` 时使用启发式计数器（零依赖）。
        """
        self._policy = policy
        self._counter = HeuristicTokenCounter() if counter is None else counter

    @property
    def policy(self) -> TokenBudgetPolicy:
        """当前生效的预算策略。"""
        return self._policy

    @property
    def counter(self) -> TokenCounter:
        """当前生效的 token 计数器。"""
        return self._counter

    def count(self, messages: Sequence[BaseMessage]) -> int:
        """按本裁剪器的计数口径统计消息序列的 token 合计。

        Args:
            messages: 消息序列。

        Returns:
            token 数合计。
        """
        return sum(self._message_cost(message) for message in messages)

    def _message_cost(self, message: BaseMessage) -> int:
        """单条消息的预算占用（优先复用已存 token 数）。"""
        if self._policy.prefer_stored_token_count:
            stored = stored_token_count(message)
            if stored is not None:
                return stored
        return self._counter.count_message(message)

    def _protected_indices(self, messages: Sequence[BaseMessage]) -> set[int]:
        """标记受保护消息下标（系统槽位 / 首条患者主诉）。"""
        protected: set[int] = set()
        if self._policy.keep_system_messages:
            protected.update(index for index, m in enumerate(messages) if is_system_message(m))
        if self._policy.keep_chief_complaint:
            chief = find_chief_complaint(messages)
            if chief is not None:
                protected.add(chief)
        return protected

    def trim(self, messages: Sequence[BaseMessage]) -> TokenTrimResult:
        """按预算裁剪消息序列。

        流程：标记保护位 -> 统计各消息预算占用 -> 从尾部向前贪心累加保留连续后缀
        -> 汇总告警。被保护消息始终保留且计入预算占用；若保护消息自身已超预算，
        报告 ``over_budget=True`` 并产出超限告警（此时仍不丢弃保护消息）。

        Args:
            messages: 时序升序的消息序列。

        Returns:
            裁剪结果；入参为空时返回空消息列表与零值报告。
        """
        source = list(messages)
        policy = self._policy
        budget = policy.effective_budget

        if not source:
            return TokenTrimResult(
                messages=[],
                report=TokenTrimReport(
                    applied=True,
                    budget_tokens=policy.max_tokens,
                    effective_budget=budget,
                ),
            )

        costs = [self._message_cost(message) for message in source]
        protected = self._protected_indices(source)
        used = sum(costs[index] for index in protected)

        tail: list[int] = []
        for index in range(len(source) - 1, -1, -1):
            if index in protected:
                continue
            cost = costs[index]
            if used + cost > budget:
                break
            used += cost
            tail.append(index)

        keep = protected | set(tail)
        kept = [message for index, message in enumerate(source) if index in keep]
        dropped = [index for index in range(len(source)) if index not in keep]
        dropped_tokens = sum(costs[index] for index in dropped)
        over_budget = used > budget

        warnings: list[str] = []
        if over_budget:
            warnings.append(
                f"token budget exceeded: used={used} budget={budget} over_by={used - budget}"
            )
        elif used > budget * policy.warn_ratio:
            warnings.append(
                f"token budget nearly exhausted: used={used} budget={budget} "
                f"ratio={used / budget:.2f}"
            )
        if dropped:
            warnings.append(
                f"dropped {len(dropped)} message(s) to fit token budget ({dropped_tokens} tokens)"
            )

        report = TokenTrimReport(
            applied=True,
            budget_tokens=policy.max_tokens,
            effective_budget=budget,
            used_tokens=used,
            dropped_tokens=dropped_tokens,
            kept_messages=len(kept),
            dropped_messages=len(dropped),
            over_budget=over_budget,
            warnings=warnings,
        )
        return TokenTrimResult(messages=kept, report=report)

    def trim_history(self, history: MedChatMessageHistory) -> TokenTrimResult:
        """裁剪存储句柄中的全量会话消息。

        Args:
            history: 会话历史句柄。

        Returns:
            裁剪结果（消息为 LangChain 格式）。
        """
        return self.trim(history.messages)
