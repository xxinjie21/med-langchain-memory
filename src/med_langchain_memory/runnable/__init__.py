"""医疗增强 Runnable 层（LCEL 上下文工程）。

当前提供基于存储工厂注入的会话历史增强 Runnable、多租户/科室命名空间隔离所需的
身份上下文与越权守卫、时序滑动窗口裁剪策略，以及 Token 预算裁剪（计数 + 预算内
贪心保留 + 超限告警）；后续迭代将叠加 LLM 摘要压缩、并发会话锁与读写降级等能力。
"""

from __future__ import annotations

from .med_history_runnable import MedRunnableWithMessageHistory
from .tenant import (
    NAMESPACE_SEPARATOR,
    STORAGE_KEY_PREFIX,
    SessionNamespace,
    TenantContext,
    TenantGuard,
    build_namespace_key,
    parse_namespace_key,
)
from .token_budget import (
    DEFAULT_TIKTOKEN_ENCODING,
    TIKTOKEN_ENCODINGS,
    TOKEN_COUNT_KEY,
    HeuristicTokenCounter,
    TiktokenTokenCounter,
    TokenBudgetPolicy,
    TokenBudgetTrimmer,
    TokenCounter,
    TokenTrimReport,
    TokenTrimResult,
    message_text,
    resolve_token_counter,
    stored_token_count,
)
from .trimmer import (
    CREATED_AT_KEY,
    ContextWindowPolicy,
    TimeWindowTrimmer,
    extract_created_at,
    find_chief_complaint,
    is_system_message,
    med_role_of,
)

__all__ = [
    "CREATED_AT_KEY",
    "DEFAULT_TIKTOKEN_ENCODING",
    "TIKTOKEN_ENCODINGS",
    "TOKEN_COUNT_KEY",
    "ContextWindowPolicy",
    "HeuristicTokenCounter",
    "MedRunnableWithMessageHistory",
    "NAMESPACE_SEPARATOR",
    "STORAGE_KEY_PREFIX",
    "SessionNamespace",
    "TenantContext",
    "TenantGuard",
    "TiktokenTokenCounter",
    "TimeWindowTrimmer",
    "TokenBudgetPolicy",
    "TokenBudgetTrimmer",
    "TokenCounter",
    "TokenTrimReport",
    "TokenTrimResult",
    "build_namespace_key",
    "extract_created_at",
    "find_chief_complaint",
    "is_system_message",
    "med_role_of",
    "message_text",
    "parse_namespace_key",
    "resolve_token_counter",
    "stored_token_count",
]
