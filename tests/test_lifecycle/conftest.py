"""test_lifecycle 共享测试替身：隔离的内存存储与消息工厂。

注意：本目录不放置 ``__init__.py``，靠 pytest 注入 ``src`` 路径引入被测模块。
"""

from __future__ import annotations

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.stores.base import MedChatMessageHistory

NAMESPACE = {
    "session_id": "s-1",
    "tenant_id": "hosp-a",
    "dept_id": "cardio",
    "patient_id": "p-1",
}


class FakeHistory(MedChatMessageHistory):
    """隔离的内存存储替身：每条实例持有独立数据，不按 storage_key 共享。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data: list[MedMessage] = []

    def _append(self, messages: list[MedMessage]) -> None:
        self._data.extend(messages)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        if limit is None:
            return list(self._data)
        return list(self._data[-limit:])

    def clear(self) -> None:
        self._data.clear()


def make_messages(n: int, **overrides) -> list[MedMessage]:
    """构造 ``n`` 条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    role_cycle = [
        MessageRole.PATIENT,
        MessageRole.DOCTOR,
        MessageRole.ASSISTANT,
        MessageRole.SYSTEM,
    ]
    out: list[MedMessage] = []
    for i in range(n):
        out.append(
            MedMessage(
                session_id=NAMESPACE["session_id"],
                tenant_id=NAMESPACE["tenant_id"],
                dept_id=NAMESPACE["dept_id"],
                patient_id=NAMESPACE["patient_id"],
                role=role_cycle[i % len(role_cycle)],
                content=f"msg-{i}",
                **overrides,
            )
        )
    return out
