"""医疗增强 Runnable 层（LCEL 上下文工程）。

当前提供基于存储工厂注入的会话历史增强 Runnable；后续迭代将叠加多租户隔离、
时序/Token 预算裁剪、LLM 摘要压缩、并发会话锁与读写降级等能力。
"""

from __future__ import annotations

from .med_history_runnable import MedRunnableWithMessageHistory

__all__ = ["MedRunnableWithMessageHistory"]
