"""多租户 / 科室命名空间隔离单元测试。

覆盖：租户身份上下文（构造校验 / 作用域判定 / 越权拒绝）、会话命名空间值对象
（三段式键与存储键互转 / 解析异常）、命名空间键工具函数、TenantGuard 守卫
（归属校验 / 句柄守卫 / 工厂闭包包装），以及 MedRunnableWithMessageHistory 接入
tenant_context 后的端到端隔离行为。

全部用例零外部依赖：memory 后端与录制式假工厂即可跑通。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

import med_langchain_memory
from med_langchain_memory.exceptions import TenantIsolationError, ValidationError
from med_langchain_memory.runnable import (
    NAMESPACE_SEPARATOR,
    STORAGE_KEY_PREFIX,
    MedRunnableWithMessageHistory,
    SessionNamespace,
    TenantContext,
    TenantGuard,
    build_namespace_key,
    parse_namespace_key,
)
from med_langchain_memory.stores import InMemoryMedHistory, StoreFactory

# --------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------- #


def _make_runnable() -> RunnableLambda:
    """构造无需外部 LLM 的下游 Runnable：回显历史条数与输入。"""

    def _model(inp: dict[str, Any]) -> str:
        history: list[BaseMessage] = inp.get("history", [])
        return f"history={len(history)} input={inp['input']}"

    return RunnableLambda(_model)


class _RecordingFactory:
    """记录 create 调用参数的假工厂，并返回一个最小可用历史句柄。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
            {
                "backend": backend,
                "session_id": session_id,
                "tenant_id": tenant_id,
                "dept_id": dept_id,
                "patient_id": patient_id,
                "ttl_seconds": ttl_seconds,
                "options": options,
            }
        )
        return StoreFactory.create(
            "memory",
            session_id=session_id,
            tenant_id=tenant_id,
            dept_id=dept_id,
            patient_id=patient_id,
        )


@pytest.fixture(autouse=True)
def _clean_memory_store() -> None:
    """memory 后端为类级共享存储，每个用例前后清场避免串扰。"""
    InMemoryMedHistory.reset()
    yield
    InMemoryMedHistory.reset()


# --------------------------------------------------------------------- #
# TenantContext
# --------------------------------------------------------------------- #


def test_context_defaults_to_tenant_wide() -> None:
    """未指定 dept_id 的上下文为租户级视角。"""
    ctx = TenantContext(tenant_id="h-a")
    assert ctx.dept_id == ""
    assert ctx.is_tenant_wide is True


def test_context_with_dept_is_scoped() -> None:
    """指定 dept_id 后上下文收敛到单个科室。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology", actor_id="doc-1")
    assert ctx.is_tenant_wide is False
    assert ctx.actor_id == "doc-1"


@pytest.mark.parametrize("tenant_id", ["", "bad:id", "bad/id", "租户"])
def test_context_rejects_invalid_tenant_id(tenant_id: str) -> None:
    """空值或含分隔符/非 ASCII 的租户 ID 一律拒绝。"""
    with pytest.raises(PydanticValidationError):
        TenantContext(tenant_id=tenant_id)


def test_context_is_frozen() -> None:
    """上下文不可变：赋值触发 pydantic 校验异常。"""
    ctx = TenantContext(tenant_id="h-a")
    with pytest.raises(PydanticValidationError):
        ctx.dept_id = "cardiology"  # type: ignore[misc]


def test_allows_same_tenant_same_dept() -> None:
    """同租户同科室允许访问。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    assert ctx.allows("h-a", "cardiology") is True


def test_allows_rejects_cross_tenant() -> None:
    """跨租户一律拒绝，即便科室名相同。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    assert ctx.allows("h-b", "cardiology") is False


def test_allows_rejects_cross_dept_when_scoped() -> None:
    """限定科室时，同租户的其他科室拒绝。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    assert ctx.allows("h-a", "neurology") is False


def test_allows_any_dept_when_tenant_wide() -> None:
    """租户级上下文（跨科室会诊/院级管理员）可访问本租户任意科室。"""
    ctx = TenantContext(tenant_id="h-a")
    assert ctx.allows("h-a", "cardiology") is True
    assert ctx.allows("h-a", "neurology") is True
    assert ctx.allows("h-b", "cardiology") is False


def test_assert_access_passes_for_owned_namespace() -> None:
    """归属正确的命名空间校验通过（无异常）。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    ctx.assert_access("h-a", "cardiology")


def test_assert_access_rejects_cross_tenant() -> None:
    """跨租户访问抛 TenantIsolationError 且信息含双方命名空间。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    with pytest.raises(TenantIsolationError) as exc:
        ctx.assert_access("h-b", "cardiology")
    assert "h-b" in str(exc.value)


def test_assert_access_rejects_cross_dept() -> None:
    """同租户跨科室访问抛 TenantIsolationError。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    with pytest.raises(TenantIsolationError):
        ctx.assert_access("h-a", "neurology")


@pytest.mark.parametrize(
    ("tenant_id", "dept_id"),
    [("", "cardiology"), ("h-a", "")],
)
def test_assert_access_rejects_incomplete_namespace(tenant_id: str, dept_id: str) -> None:
    """命名空间不完整时无从判定归属，抛 ValidationError。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    with pytest.raises(ValidationError):
        ctx.assert_access(tenant_id, dept_id)


def test_namespace_uses_context_dept() -> None:
    """未显式给科室时，命名空间取上下文科室。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    ns = ctx.namespace("s-1", patient_id="p-1")
    assert ns.key == "h-a:cardiology:s-1"
    assert ns.patient_id == "p-1"


def test_namespace_accepts_explicit_dept() -> None:
    """租户级上下文显式指定目标科室。"""
    ctx = TenantContext(tenant_id="h-a")
    ns = ctx.namespace("s-1", dept_id="neurology")
    assert ns.key == "h-a:neurology:s-1"


def test_namespace_requires_dept_for_tenant_wide() -> None:
    """租户级上下文未指定科室时无法构造存储键，抛 ValidationError。"""
    ctx = TenantContext(tenant_id="h-a")
    with pytest.raises(ValidationError):
        ctx.namespace("s-1")


# --------------------------------------------------------------------- #
# SessionNamespace
# --------------------------------------------------------------------- #


def test_namespace_key_and_storage_key() -> None:
    """三段式键与统一存储键格式正确。"""
    ns = SessionNamespace(session_id="s-1", tenant_id="h-a", dept_id="cardiology")
    assert ns.key == "h-a:cardiology:s-1"
    assert ns.storage_key == "med:chat:h-a:cardiology:s-1"
    assert ns.storage_key == f"{STORAGE_KEY_PREFIX}:{ns.key}"
    assert NAMESPACE_SEPARATOR == ":"


def test_namespace_to_config() -> None:
    """to_config 产出 Runnable 可直接使用的 configurable 字典。"""
    ns = SessionNamespace(session_id="s-1", tenant_id="h-a", dept_id="cardiology", patient_id="p-1")
    assert ns.to_config() == {
        "configurable": {
            "session_id": "s-1",
            "tenant_id": "h-a",
            "dept_id": "cardiology",
            "patient_id": "p-1",
        }
    }


def test_namespace_parse_roundtrip() -> None:
    """键 → 对象 → 键往返一致。"""
    ns = SessionNamespace.parse("h-a:cardiology:s-1", patient_id="p-9")
    assert (ns.tenant_id, ns.dept_id, ns.session_id) == ("h-a", "cardiology", "s-1")
    assert ns.key == "h-a:cardiology:s-1"


def test_namespace_parse_rejects_wrong_segment_count() -> None:
    """段数不为 3 时抛 ValidationError。"""
    with pytest.raises(ValidationError):
        SessionNamespace.parse("h-a:s-1")
    with pytest.raises(ValidationError):
        SessionNamespace.parse("h-a:cardiology:s-1:extra")


def test_namespace_parse_rejects_empty_segment() -> None:
    """含空段的键抛 ValidationError。"""
    with pytest.raises(ValidationError):
        SessionNamespace.parse("h-a::s-1")


def test_namespace_parse_rejects_illegal_id() -> None:
    """段内含非法字符时抛 pydantic 校验异常（段数正确但 ID 不合规）。"""
    with pytest.raises(PydanticValidationError):
        SessionNamespace.parse("h-a:cardiology:s 1")


def test_namespace_from_history() -> None:
    """从存储句柄提取命名空间，与构造参数一致。"""
    history = StoreFactory.create(
        "memory",
        session_id="s-1",
        tenant_id="h-a",
        dept_id="cardiology",
        patient_id="p-1",
    )
    ns = SessionNamespace.from_history(history)
    assert ns.key == "h-a:cardiology:s-1"
    assert ns.patient_id == "p-1"


def test_namespace_is_frozen() -> None:
    """命名空间不可变。"""
    ns = SessionNamespace(session_id="s-1", tenant_id="h-a", dept_id="cardiology")
    with pytest.raises(PydanticValidationError):
        ns.session_id = "s-2"  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 命名空间键工具函数
# --------------------------------------------------------------------- #


def test_build_namespace_key() -> None:
    """三段式键按 tenant:dept:session 顺序拼接。"""
    assert build_namespace_key("h-a", "cardiology", "s-1") == "h-a:cardiology:s-1"


@pytest.mark.parametrize(
    ("tenant_id", "dept_id", "session_id"),
    [("", "d", "s"), ("t", "", "s"), ("t", "d", "")],
)
def test_build_namespace_key_rejects_empty(tenant_id: str, dept_id: str, session_id: str) -> None:
    """任一段为空即抛 ValidationError。"""
    with pytest.raises(ValidationError):
        build_namespace_key(tenant_id, dept_id, session_id)


def test_parse_namespace_key() -> None:
    """解析结果按 (tenant, dept, session) 顺序返回。"""
    assert parse_namespace_key("h-a:cardiology:s-1") == ("h-a", "cardiology", "s-1")


def test_parse_namespace_key_rejects_malformed() -> None:
    """段数异常与空段均抛 ValidationError。"""
    with pytest.raises(ValidationError):
        parse_namespace_key("h-a:cardiology")
    with pytest.raises(ValidationError):
        parse_namespace_key("h-a:cardiology:")


# --------------------------------------------------------------------- #
# TenantGuard
# --------------------------------------------------------------------- #


def _memory_history(tenant_id: str, dept_id: str, session_id: str = "s-1") -> Any:
    """构造一个 memory 后端句柄，用于守卫校验。"""
    return StoreFactory.create(
        "memory",
        session_id=session_id,
        tenant_id=tenant_id,
        dept_id=dept_id,
        patient_id="p-1",
    )


def test_guard_exposes_context() -> None:
    """守卫持有绑定的身份上下文。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    assert TenantGuard(ctx).context is ctx


def test_guard_assert_access_delegates() -> None:
    """守卫的归属校验委托给上下文。"""
    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    guard.assert_access("h-a", "cardiology")
    with pytest.raises(TenantIsolationError):
        guard.assert_access("h-a", "neurology")


def test_guard_history_returns_same_instance() -> None:
    """归属正确的句柄原样返回，便于链式书写。"""
    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    history = _memory_history("h-a", "cardiology")
    assert guard.guard_history(history) is history


def test_guard_history_rejects_foreign_history() -> None:
    """跨租户句柄被守卫拒绝。"""
    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    history = _memory_history("h-b", "cardiology")
    with pytest.raises(TenantIsolationError):
        guard.guard_history(history)


def test_guard_bind_passes_and_forwards() -> None:
    """守卫包装后：允许的访问正常透传参数到底层工厂。"""
    recorded: list[tuple[str, str, str, str]] = []

    def _factory(
        session_id: str, tenant_id: str = "", dept_id: str = "", patient_id: str = ""
    ) -> Any:
        recorded.append((session_id, tenant_id, dept_id, patient_id))
        return _memory_history(tenant_id, dept_id, session_id)

    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    guarded = guard.bind(_factory)
    history = guarded("s-1", tenant_id="h-a", dept_id="cardiology", patient_id="p-1")
    assert history.storage_key == "med:chat:h-a:cardiology:s-1"
    assert recorded == [("s-1", "h-a", "cardiology", "p-1")]


def test_guard_bind_rejects_without_calling_factory() -> None:
    """越权访问在调用底层工厂之前即被拦截。"""
    called = False

    def _factory(
        session_id: str, tenant_id: str = "", dept_id: str = "", patient_id: str = ""
    ) -> Any:
        nonlocal called
        called = True
        return _memory_history(tenant_id, dept_id, session_id)

    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    with pytest.raises(TenantIsolationError):
        guard.bind(_factory)("s-1", tenant_id="h-a", dept_id="neurology")
    assert called is False


def test_guard_bind_falls_back_to_context() -> None:
    """包装后命名空间留空时回退到守卫上下文。"""
    guard = TenantGuard(TenantContext(tenant_id="h-a", dept_id="cardiology"))
    history = guard.bind(
        lambda session_id, tenant_id="", dept_id="", patient_id="": _memory_history(
            tenant_id, dept_id, session_id
        )
    )("s-1")
    assert history.storage_key == "med:chat:h-a:cardiology:s-1"


# --------------------------------------------------------------------- #
# Runnable 集成
# --------------------------------------------------------------------- #


def test_runnable_exposes_tenant_context() -> None:
    """tenant_context 通过属性暴露，未注入时为 None。"""
    rwh = MedRunnableWithMessageHistory(_make_runnable(), backend="memory")
    assert rwh.tenant_context is None

    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    guarded = MedRunnableWithMessageHistory(_make_runnable(), backend="memory", tenant_context=ctx)
    assert guarded.tenant_context is ctx


def test_runnable_without_context_does_not_enforce() -> None:
    """未注入身份时保持 D23 行为：不做归属校验。"""
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-open", "h-b", "d-b", "p-b")
    assert rwh.invoke({"input": "x"}, config=cfg) == "history=0 input=x"


def test_runnable_allows_owned_namespace() -> None:
    """注入身份后，归属正确的会话可正常读写并落库。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        tenant_context=ctx,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-ok", "h-a", "cardiology", "p-1")
    assert rwh.invoke({"input": "hi"}, config=cfg) == "history=0 input=hi"

    history = _memory_history("h-a", "cardiology", "s-ok")
    assert len(history.messages) == 2


def test_runnable_rejects_cross_tenant() -> None:
    """注入身份后，跨租户 config 抛 TenantIsolationError。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        tenant_context=ctx,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-bad", "h-b", "cardiology", "p-1")
    with pytest.raises(TenantIsolationError):
        rwh.invoke({"input": "hi"}, config=cfg)


def test_runnable_rejects_cross_dept() -> None:
    """同租户跨科室 config 抛 TenantIsolationError。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        tenant_context=ctx,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-bad2", "h-a", "neurology", "p-1")
    with pytest.raises(TenantIsolationError):
        rwh.invoke({"input": "hi"}, config=cfg)


def test_runnable_tenant_wide_allows_any_dept() -> None:
    """租户级身份可访问本租户任意科室。"""
    ctx = TenantContext(tenant_id="h-wide")
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        tenant_context=ctx,
        input_messages_key="input",
        history_messages_key="history",
    )
    for dept in ("cardiology", "neurology"):
        cfg = MedRunnableWithMessageHistory.build_config(f"s-{dept}", "h-wide", dept, "p-1")
        assert rwh.invoke({"input": "hi"}, config=cfg) == "history=0 input=hi"


def test_runnable_falls_back_to_context_namespace() -> None:
    """config 中留空租户/科室时回退到 tenant_context。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    factory = _RecordingFactory()
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        store_factory=factory,  # type: ignore[arg-type]
        tenant_context=ctx,
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-fb", "", "", "p-1")
    rwh.invoke({"input": "x"}, config=cfg)
    assert factory.calls[0]["tenant_id"] == "h-a"
    assert factory.calls[0]["dept_id"] == "cardiology"


def test_runnable_default_namespace_conflict_is_rejected() -> None:
    """default_namespace 指向越权租户时，身份校验仍应拒绝。"""
    ctx = TenantContext(tenant_id="h-a", dept_id="cardiology")
    rwh = MedRunnableWithMessageHistory(
        _make_runnable(),
        backend="memory",
        tenant_context=ctx,
        default_namespace={"tenant_id": "h-b", "dept_id": "neurology"},
        input_messages_key="input",
        history_messages_key="history",
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-conflict", "", "", "p-1")
    with pytest.raises(TenantIsolationError):
        rwh.invoke({"input": "x"}, config=cfg)


def test_build_config_patient_id_defaults_to_empty() -> None:
    """build_config 允许省略 patient_id，缺省为空串。"""
    cfg = MedRunnableWithMessageHistory.build_config("s-1", "h-a", "cardiology")
    assert cfg["configurable"]["patient_id"] == ""


def test_tenant_symbols_exported_from_package_root() -> None:
    """租户隔离三件套在包根导出，便于外部按命名空间自助构造。"""
    assert med_langchain_memory.TenantContext is TenantContext
    assert med_langchain_memory.SessionNamespace is SessionNamespace
    assert med_langchain_memory.TenantGuard is TenantGuard
