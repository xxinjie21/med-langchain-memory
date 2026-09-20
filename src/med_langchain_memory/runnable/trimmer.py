"""时序上下文裁剪（滑动窗口）。

一次完整问诊可能长达数百轮，把全量历史直接喂给 LLM 会带来成本与超时风险。
本模块在链路取用历史之后、组装提示词之前做一次**纯结构化**的滑动窗口裁剪：

* **按条数**：只保留会话尾部 ``max_messages`` 条对话消息；
* **按时间窗**：丢弃早于 ``now - max_age_seconds`` 的消息；
* **主诉保护**：患者首条主诉消息永不被裁掉，避免丢失就诊意图；
* **系统槽位保护**：``SystemMessage``（诊疗规范 / 提示词骨架）永不被裁掉。

设计取舍：判定完全基于消息类型与 ``additional_kwargs`` 中的结构化字段
（``med_role`` / ``created_at``），不含任何文本内容解析或语义理解逻辑；
被保护的消息不计入 ``max_messages`` 预算，缺失时间戳的消息在时间窗裁剪下
保守保留（无法判定时序时不丢弃上下文）。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from med_langchain_memory.domain.message import MessageRole, now_millis
from med_langchain_memory.stores.base import MED_ROLE_KEY, MedChatMessageHistory

#: ``additional_kwargs`` 中承载消息创建时间（epoch millis）的键名。
CREATED_AT_KEY = "created_at"

#: 缺少 ``med_role`` 扩展字段时，按 LangChain 消息类型回退推断医疗角色。
_FALLBACK_ROLE: dict[str, MessageRole] = {
    "human": MessageRole.PATIENT,
    "ai": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}

#: 一秒对应的毫秒数（时间窗计算用）。
_MILLIS_PER_SECOND = 1000


class ContextWindowPolicy(BaseModel):
    """时序裁剪策略（不可变）。

    Attributes:
        max_messages: 保留的**对话消息**条数上限（被保护的系统消息与主诉不计入），
            ``None`` 表示不按条数裁剪。
        max_age_seconds: 时间窗长度（秒），早于 ``now - max_age_seconds`` 的消息被丢弃，
            ``None`` 表示不按时间裁剪。
        keep_system_messages: 是否保护系统消息不被裁剪。
        keep_chief_complaint: 是否保护首条患者主诉消息不被裁剪。

    Raises:
        pydantic.ValidationError: ``max_messages`` 小于 1 或 ``max_age_seconds`` 非正数时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_messages: int | None = Field(default=None, ge=1)
    max_age_seconds: int | None = Field(default=None, gt=0)
    keep_system_messages: bool = True
    keep_chief_complaint: bool = True


def extract_created_at(message: BaseMessage) -> int | None:
    """读取消息创建时间（epoch millis）。

    Args:
        message: LangChain 消息。

    Returns:
        ``additional_kwargs["created_at"]`` 为正整数时返回该值，缺失或非法时返回 ``None``。
    """
    raw = message.additional_kwargs.get(CREATED_AT_KEY)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


def med_role_of(message: BaseMessage) -> MessageRole | None:
    """解析消息的医疗角色。

    优先取 ``additional_kwargs["med_role"]``；缺失时按 LangChain 消息类型回退推断。

    Args:
        message: LangChain 消息。

    Returns:
        解析到的医疗角色；无法推断（未知角色值或不支持的消息类型）时返回 ``None``。
    """
    raw = message.additional_kwargs.get(MED_ROLE_KEY)
    if isinstance(raw, str):
        try:
            return MessageRole(raw)
        except ValueError:
            return None
    return _FALLBACK_ROLE.get(message.type)


def is_system_message(message: BaseMessage) -> bool:
    """判断消息是否为系统槽位消息。

    Args:
        message: LangChain 消息。

    Returns:
        ``SystemMessage`` 类型或医疗角色为 ``system`` 时返回 ``True``。
    """
    if isinstance(message, SystemMessage):
        return True
    return med_role_of(message) is MessageRole.SYSTEM


def find_chief_complaint(messages: Sequence[BaseMessage]) -> int | None:
    """定位首条患者主诉消息。

    Args:
        messages: 时序升序的消息序列。

    Returns:
        首条医疗角色为 ``PATIENT`` 的消息下标；不存在时返回 ``None``。
    """
    for index, message in enumerate(messages):
        if med_role_of(message) is MessageRole.PATIENT:
            return index
    return None


def _within_window(message: BaseMessage, cutoff_ms: int) -> bool:
    """判断消息是否落在时间窗内。

    缺失时间戳的消息保守保留（返回 ``True``）。
    """
    created_at = extract_created_at(message)
    return created_at is None or created_at >= cutoff_ms


class TimeWindowTrimmer:
    """按时序窗口裁剪会话上下文。

    裁剪只做结构化过滤，输出保持原始时序升序，且不修改入参。
    """

    def __init__(self, policy: ContextWindowPolicy) -> None:
        """初始化裁剪器。

        Args:
            policy: 时序裁剪策略。
        """
        self._policy = policy

    @property
    def policy(self) -> ContextWindowPolicy:
        """当前生效的裁剪策略。"""
        return self._policy

    def trim(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_ms: int | None = None,
    ) -> list[BaseMessage]:
        """按策略裁剪消息序列。

        流程：标记保护位（系统消息 / 主诉）-> 对候选对话消息先做时间窗过滤，
        再做尾部条数截取 -> 按原始顺序输出保留下来的消息。

        Args:
            messages: 时序升序的消息序列。
            now_ms: 当前 epoch 毫秒，缺省取系统时间（仅 ``max_age_seconds`` 生效时使用）。

        Returns:
            裁剪后的新消息列表；入参为空时返回空列表。
        """
        source = list(messages)
        if not source:
            return []

        protected: set[int] = set()
        if self._policy.keep_system_messages:
            protected.update(index for index, m in enumerate(source) if is_system_message(m))
        if self._policy.keep_chief_complaint:
            chief = find_chief_complaint(source)
            if chief is not None:
                protected.add(chief)

        candidates = [(index, m) for index, m in enumerate(source) if index not in protected]

        max_age = self._policy.max_age_seconds
        if max_age is not None:
            now = now_millis() if now_ms is None else now_ms
            cutoff = now - max_age * _MILLIS_PER_SECOND
            candidates = [pair for pair in candidates if _within_window(pair[1], cutoff)]

        limit = self._policy.max_messages
        if limit is not None and len(candidates) > limit:
            candidates = candidates[-limit:]

        keep_indices = protected | {index for index, _ in candidates}
        return [m for index, m in enumerate(source) if index in keep_indices]

    def trim_history(
        self,
        history: MedChatMessageHistory,
        *,
        now_ms: int | None = None,
    ) -> list[BaseMessage]:
        """裁剪存储句柄中的全量会话消息。

        Args:
            history: 会话历史句柄。
            now_ms: 当前 epoch 毫秒，缺省取系统时间。

        Returns:
            裁剪后的消息列表（LangChain 格式）。
        """
        return self.trim(history.messages, now_ms=now_ms)
