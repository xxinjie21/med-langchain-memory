"""MedRunnableWithMessageHistory 单元测试。

覆盖：构建辅助、存储工厂参数透传、端到端持久化（memory 后端）、
default_namespace 回退、未知后端与缺失配置键两类异常。

全部用例零外部依赖：memory 后端与录制式 FakeFactory 即可跑通。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from med_langchain_memory.runnable import MedRunnableWithMessageHistory
from med_langchain_memory.stores import StoreFactory
from med_langchain_memory.stores.factory import StoreNotFoundError


def _make_runnable() -> RunnableLambda:
    """构造一个无需外部 LLM 的下游 Runnable：把历史条数与用户输入回显。"""

    def _model(inp: dict[str, Any]) -> str:
        history: list[BaseMessage] = inp.get("history", [])
        return f"history={len(history)} input={inp['input']}"

    return RunnableLambda(_model)


def test_build_config_contains_all_fields() -> None:
    """build_config 产出含四段命名空间的 configurable 字典。"""
    config = MedRunnableWithMessageHistory.build_config("s1", "t1", "d1", "p1")
    assert config == {
        "configurable": {
            "session_id": "s1",
            "tenant_id": "t1",
            "dept_id": "d1",
            "patient_id": "p1",
        }
    }


def test_history_factory_config_fields() -> None:
    """history_factory_config 必须声明 session_id + 三段式命名空间。"""
    rwh = MedRunnableWithMessageHistory(_make_runnable(), backend="memory")
    ids = [spec.id for spec in rwh.history_factory_config]
    assert ids == ["session_id", "tenant_id", "dept_id", "patient_id"]


def test_invoke_persists_messages_end_to_end() -> None:
    """两轮对话后，memory 后端按命名空间持久化 4 条时序消息。"""
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-e2e", "t-e2e", "d-e2e", "p-e2e")

    first = rwh.invoke({"input": "hello"}, config=cfg)
    assert first == "history=0 input=hello"
    second = rwh.invoke({"input": "hi"}, config=cfg)
    assert second == "history=2 input=hi"

    # 以同命名空间新建历史句柄，验证底层已落库（memory 为类级共享）。
    history = StoreFactory.create(
        "memory",
        session_id="s-e2e",
        tenant_id="t-e2e",
        dept_id="d-e2e",
        patient_id="p-e2e",
    )
    messages = history.messages
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hello"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "history=2 input=hi"


def test_default_namespace_fallback() -> None:
    """config 中留空的命名空间字段应回退到 default_namespace。"""
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        default_namespace={"tenant_id": "def-tenant"},
        input_messages_key="input",
        history_messages_key="history",
    )
    # 调用时 tenant_id 留空，应落到 def-tenant。
    cfg = MedRunnableWithMessageHistory.build_config("s-def", "", "d-def", "p-def")
    rwh.invoke({"input": "x"}, config=cfg)

    history = StoreFactory.create(
        "memory",
        session_id="s-def",
        tenant_id="def-tenant",
        dept_id="d-def",
        patient_id="p-def",
    )
    assert len(history.messages) == 2


class _RecordingFactory:
    """记录 create 调用参数的假工厂，用于验证注入透传。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def create(
        self,
        backend: str,
        *,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        ttl_seconds: int | None = None,
        **options: Any,
    ) -> Any:
        self.calls.append(
            (backend, session_id, tenant_id, dept_id, patient_id, ttl_seconds, options)
        )
        return _StubHistory()


class _StubHistory:
    """满足 BaseChatMessageHistory 最小契约的占位对象（仅供工厂返回）。"""

    @property
    def messages(self) -> list[BaseMessage]:
        return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:  # type: ignore[no-untyped-def]
        pass

    def clear(self) -> None:
        pass


def test_factory_injection_forwards_args() -> None:
    """store_factory / backend / ttl / store_options 全部正确透传给 create。"""
    factory = _RecordingFactory()
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="redis",
        store_factory=factory,  # type: ignore[arg-type]
        ttl_seconds=900,
        store_options={"url": "redis://localhost:6379/0"},
    )
    # 直接调用父类保存的闭包，验证参数组装。
    rwh.get_session_history(  # type: ignore[attr-defined]
        session_id="s9", tenant_id="t9", dept_id="d9", patient_id="p9"
    )
    assert factory.calls == [
        ("redis", "s9", "t9", "d9", "p9", 900, {"url": "redis://localhost:6379/0"})
    ]


def test_default_namespace_fallback_records_default() -> None:
    """default_namespace 在空字段时回退，应体现在透传参数中。"""
    factory = _RecordingFactory()
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        store_factory=factory,  # type: ignore[arg-type]
        default_namespace={"tenant_id": "fallback-t", "dept_id": "fallback-d"},
    )
    rwh.get_session_history(  # type: ignore[attr-defined]
        session_id="s", tenant_id="", dept_id="", patient_id="p"
    )
    assert factory.calls[0][2] == "fallback-t"
    assert factory.calls[0][3] == "fallback-d"
    assert factory.calls[0][4] == "p"


def test_unknown_backend_raises_on_invoke() -> None:
    """未知后端在 invoke 取用历史时抛 StoreNotFoundError。"""
    rwh = MedRunnableWithMessageHistory(_make_runnable(), backend="no-such-backend")
    cfg = MedRunnableWithMessageHistory.build_config("s-x", "t-x", "d-x", "p-x")
    try:
        rwh.invoke({"input": "x"}, config=cfg)
    except StoreNotFoundError:
        return
    raise AssertionError("expected StoreNotFoundError for unknown backend")


def test_incomplete_config_raises() -> None:
    """config 缺少任一命名空间键时应抛异常（KeyError 或 ValueError）。"""
    rwh = MedRunnableWithMessageHistory(_make_runnable(), backend="memory")
    # 故意缺失 patient_id。
    bad_config = {
        "configurable": {
            "session_id": "s-bad",
            "tenant_id": "t-bad",
            "dept_id": "d-bad",
        }
    }
    raised = False
    try:
        rwh.invoke({"input": "x"}, config=bad_config)
    except (KeyError, ValueError):
        raised = True
    assert raised, "expected KeyError/ValueError for incomplete configurable"
