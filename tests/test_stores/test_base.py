"""MedChatMessageHistory 抽象与 LangChain 消息转换单元测试。

测试不依赖任何真实中间件：用进程内列表实现的替身 Store 覆盖存储原语。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from med_langchain_memory.domain import MedMessage, MessageRole, SessionMeta, SessionStatus
from med_langchain_memory.exceptions import (
    StateTransitionError,
    StorageError,
    TenantIsolationError,
    ValidationError,
)
from med_langchain_memory.stores import (
    MED_ROLE_KEY,
    MedChatMessageHistory,
    from_langchain_message,
    to_langchain_message,
)

NAMESPACE = {
    "session_id": "s-001",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-001"


def make_message(**overrides: object) -> MedMessage:
    """构造一条属于默认命名空间的合法医疗消息。"""
    kwargs: dict = {**NAMESPACE, "role": MessageRole.PATIENT, "content": "chest pain"}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


class FakeStore(MedChatMessageHistory):
    """进程内列表替身存储，仅实现三个存储原语。"""

    def _append(self, messages: list[MedMessage]) -> None:
        self._items.extend(messages)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        items = getattr(self, "_items", [])
        return list(items) if limit is None else list(items[-limit:])

    def clear(self) -> None:
        self._items = []

    def __init__(self, **kwargs: object) -> None:
        self._items: list[MedMessage] = []
        super().__init__(**kwargs)  # type: ignore[arg-type]


class TtlStore(FakeStore):
    """支持原生 TTL 的替身存储，记录 TTL 下发次数。"""

    supports_ttl = True

    def __init__(self, **kwargs: object) -> None:
        self.applied: list[int] = []
        super().__init__(**kwargs)

    def _apply_ttl(self, ttl_seconds: int) -> None:
        self.applied.append(ttl_seconds)


@pytest.fixture
def store() -> FakeStore:
    """默认命名空间的替身存储。"""
    return FakeStore(**NAMESPACE)


# --------------------------------------------------------------------------- #
# 消息转换
# --------------------------------------------------------------------------- #
class TestMessageConversion:
    @pytest.mark.parametrize(
        ("role", "expected_type"),
        [
            (MessageRole.PATIENT, "human"),
            (MessageRole.DOCTOR, "human"),
            (MessageRole.ASSISTANT, "ai"),
            (MessageRole.SYSTEM, "system"),
        ],
    )
    def test_role_maps_to_langchain_type(self, role: MessageRole, expected_type: str) -> None:
        lc = to_langchain_message(make_message(role=role))
        assert lc.type == expected_type
        assert lc.additional_kwargs[MED_ROLE_KEY] == role.value

    def test_to_langchain_carries_medical_fields(self) -> None:
        med = make_message(token_count=7, masked=True, metadata={"triage": "urgent"})
        lc = to_langchain_message(med)
        assert lc.id == med.message_id
        assert lc.content == "chest pain"
        assert lc.additional_kwargs["token_count"] == 7
        assert lc.additional_kwargs["masked"] is True
        assert lc.additional_kwargs["metadata"] == {"triage": "urgent"}

    def test_round_trip_is_lossless(self) -> None:
        med = make_message(
            role=MessageRole.DOCTOR,
            token_count=12,
            masked=True,
            metadata={"icd": "I20"},
        )
        restored = from_langchain_message(to_langchain_message(med), **NAMESPACE)
        assert restored == med

    def test_role_inferred_when_extension_missing(self) -> None:
        restored = from_langchain_message(AIMessage(content="take rest"), **NAMESPACE)
        assert restored.role is MessageRole.ASSISTANT
        assert restored.session_id == "s-001"
        assert restored.token_count == 0

    def test_plain_system_message_inferred(self) -> None:
        restored = from_langchain_message(SystemMessage(content="triage bot"), **NAMESPACE)
        assert restored.role is MessageRole.SYSTEM

    def test_non_uuid_id_is_replaced(self) -> None:
        restored = from_langchain_message(HumanMessage(content="hi", id="run-42"), **NAMESPACE)
        assert restored.message_id != "run-42"

    def test_unsupported_message_type_rejected(self) -> None:
        tool_message = ToolMessage(content="result", tool_call_id="call-1")
        with pytest.raises(ValidationError, match="unsupported langchain message type"):
            from_langchain_message(tool_message, **NAMESPACE)

    def test_unknown_med_role_rejected(self) -> None:
        lc = HumanMessage(content="hi", additional_kwargs={MED_ROLE_KEY: "nurse"})
        with pytest.raises(ValidationError, match="unknown med_role"):
            from_langchain_message(lc, **NAMESPACE)

    def test_non_text_content_rejected(self) -> None:
        lc = HumanMessage(content=[{"type": "text", "text": "hi"}])
        with pytest.raises(ValidationError, match="plain text"):
            from_langchain_message(lc, **NAMESPACE)

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot convert langchain message"):
            from_langchain_message(HumanMessage(content=""), **NAMESPACE)


# --------------------------------------------------------------------------- #
# 命名空间与租户钩子
# --------------------------------------------------------------------------- #
class TestNamespace:
    def test_exposes_namespace_and_storage_key(self, store: FakeStore) -> None:
        assert store.session_id == "s-001"
        assert store.tenant_id == "hospital_a"
        assert store.dept_id == "cardiology"
        assert store.patient_id == "p-1024"
        assert store.storage_key == STORAGE_KEY
        assert isinstance(store.session_meta, SessionMeta)

    @pytest.mark.parametrize("bad_id", ["", "a:b", "含中文", "x" * 65])
    def test_invalid_namespace_rejected(self, bad_id: str) -> None:
        with pytest.raises(ValidationError, match="invalid session namespace"):
            FakeStore(**{**NAMESPACE, "tenant_id": bad_id})

    def test_belongs_to(self, store: FakeStore) -> None:
        assert store.belongs_to("hospital_a") is True
        assert store.belongs_to("hospital_a", "cardiology") is True
        assert store.belongs_to("hospital_a", "oncology") is False
        assert store.belongs_to("hospital_b") is False

    def test_assert_tenant_passes(self, store: FakeStore) -> None:
        store.assert_tenant("hospital_a", "cardiology")

    def test_assert_tenant_rejects_cross_tenant(self, store: FakeStore) -> None:
        with pytest.raises(TenantIsolationError, match="access denied"):
            store.assert_tenant("hospital_b")


# --------------------------------------------------------------------------- #
# 读写契约
# --------------------------------------------------------------------------- #
class TestReadWrite:
    def test_add_messages_via_langchain_contract(self, store: FakeStore) -> None:
        store.add_messages([HumanMessage(content="fever"), AIMessage(content="noted")])
        assert [m.type for m in store.messages] == ["human", "ai"]
        assert store.session_meta.message_count == 2

    def test_add_message_and_helpers(self, store: FakeStore) -> None:
        store.add_message(HumanMessage(content="cough"))
        store.add_user_message("headache")
        store.add_ai_message("please rest")
        assert [m.role for m in store.get_med_messages()] == [
            MessageRole.PATIENT,
            MessageRole.PATIENT,
            MessageRole.ASSISTANT,
        ]

    def test_add_med_messages_updates_meta(self, store: FakeStore) -> None:
        store.add_med_messages([make_message(), make_message(content="second")])
        assert store.session_meta.message_count == 2
        assert store.get_med_messages()[1].content == "second"

    def test_add_empty_sequence_is_noop(self, store: FakeStore) -> None:
        store.add_med_messages([])
        assert store.get_med_messages() == []
        assert store.session_meta.message_count == 0

    def test_cross_session_write_rejected(self, store: FakeStore) -> None:
        foreign = make_message(session_id="s-999")
        with pytest.raises(StorageError, match="not med:chat:hospital_a"):
            store.add_med_messages([foreign])
        assert store.get_med_messages() == []

    def test_cross_tenant_write_rejected(self, store: FakeStore) -> None:
        with pytest.raises(StorageError):
            store.add_med_messages([make_message(tenant_id="hospital_b")])

    def test_get_med_messages_limit(self, store: FakeStore) -> None:
        store.add_med_messages([make_message(content=str(i)) for i in range(5)])
        assert [m.content for m in store.get_med_messages(limit=2)] == ["3", "4"]

    @pytest.mark.parametrize("bad_limit", [0, -1])
    def test_non_positive_limit_rejected(self, store: FakeStore, bad_limit: int) -> None:
        with pytest.raises(ValidationError, match="limit must be"):
            store.get_med_messages(limit=bad_limit)

    def test_clear_removes_all(self, store: FakeStore) -> None:
        store.add_med_messages([make_message()])
        store.clear()
        assert store.messages == []


# --------------------------------------------------------------------------- #
# TTL 钩子
# --------------------------------------------------------------------------- #
class TestTtlHooks:
    def test_default_store_has_no_ttl(self, store: FakeStore) -> None:
        assert store.supports_ttl is False
        assert store.ttl_seconds is None
        assert store.refresh_ttl() is False
        assert store.is_expired() is False

    def test_set_ttl_none_allowed_without_support(self, store: FakeStore) -> None:
        store.set_ttl(None)
        assert store.ttl_seconds is None

    def test_set_ttl_rejected_when_unsupported(self, store: FakeStore) -> None:
        with pytest.raises(StorageError, match="does not support native ttl"):
            store.set_ttl(60)

    @pytest.mark.parametrize("bad_ttl", [0, -5])
    def test_non_positive_ttl_rejected(self, bad_ttl: int) -> None:
        with pytest.raises(ValidationError, match="ttl_seconds must be"):
            TtlStore(**NAMESPACE).set_ttl(bad_ttl)

    def test_ttl_applied_on_construction_and_write(self) -> None:
        ttl_store = TtlStore(**NAMESPACE, ttl_seconds=30)
        assert ttl_store.ttl_seconds == 30
        assert ttl_store.applied == [30]
        ttl_store.add_med_messages([make_message()])
        assert ttl_store.applied == [30, 30]

    def test_refresh_ttl_returns_true_when_supported(self) -> None:
        ttl_store = TtlStore(**NAMESPACE, ttl_seconds=10)
        assert ttl_store.refresh_ttl() is True

    def test_apply_ttl_must_be_overridden(self) -> None:
        class BrokenStore(FakeStore):
            supports_ttl = True

        with pytest.raises(NotImplementedError, match="_apply_ttl"):
            BrokenStore(**NAMESPACE, ttl_seconds=5)

    def test_is_expired_by_logical_clock(self) -> None:
        ttl_store = TtlStore(**NAMESPACE, ttl_seconds=1)
        updated_at = ttl_store.session_meta.updated_at
        assert ttl_store.is_expired(now_ms=updated_at + 999) is False
        assert ttl_store.is_expired(now_ms=updated_at + 1000) is True


# --------------------------------------------------------------------------- #
# 归档钩子
# --------------------------------------------------------------------------- #
class TestArchiveHook:
    def test_archive_exports_meta_and_messages(self, store: FakeStore) -> None:
        store.add_med_messages([make_message(), make_message(content="second")])
        meta, messages = store.archive()
        assert meta.status is SessionStatus.ARCHIVED
        assert len(messages) == 2
        assert store.session_meta.status is SessionStatus.ARCHIVED
        assert store.get_med_messages() == messages

    def test_archive_twice_rejected(self, store: FakeStore) -> None:
        store.archive()
        with pytest.raises(StateTransitionError, match="archived -> archived"):
            store.archive()
