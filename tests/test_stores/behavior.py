"""跨存储后端共享的行为基准测试套件。

新增任何存储适配器（文件 / Redis / MySQL / ES）时，只需在其测试模块中定义::

    from behavior import MedHistoryBehaviorSuite

    class TestFileStore(MedHistoryBehaviorSuite):
        backend_name = "file"

        def make_history(self, **overrides):
            return FileMedHistory(**{**self.NAMESPACE, **overrides})

即可复用本文件的全部通用用例，保证各后端对外语义完全一致。

注意：本模块文件名不以 ``test_`` 开头，pytest 不会直接收集；
同目录测试模块通过 pytest 自动注入的 sys.path 直接 ``from behavior import ...`` 引用。
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from med_langchain_memory.domain import MedMessage, MessageRole, SessionStatus
from med_langchain_memory.exceptions import (
    StateTransitionError,
    StorageError,
    TenantIsolationError,
    ValidationError,
)
from med_langchain_memory.stores import MED_ROLE_KEY, MedChatMessageHistory, StoreFactory


class MedHistoryBehaviorSuite:
    """所有 :class:`MedChatMessageHistory` 实现都必须满足的行为契约。"""

    #: 被测存储的默认会话命名空间。
    NAMESPACE: ClassVar[dict[str, str]] = {
        "session_id": "s-behavior",
        "tenant_id": "hospital_a",
        "dept_id": "cardiology",
        "patient_id": "p-1024",
    }

    #: 与 :attr:`NAMESPACE` 对应的统一存储键。
    STORAGE_KEY: ClassVar[str] = "med:chat:hospital_a:cardiology:s-behavior"

    #: 该后端在 :class:`StoreFactory` 中的注册名；``None`` 表示不校验注册。
    backend_name: ClassVar[str | None] = None

    #: 同一会话的多个实例句柄是否共享同一份数据（远端存储均为 ``True``）。
    shared_across_handles: ClassVar[bool] = True

    # ------------------------------------------------------------------ #
    # 子类必须提供的构造入口
    # ------------------------------------------------------------------ #
    def make_history(self, **overrides: Any) -> MedChatMessageHistory:
        """构造被测存储实例，``overrides`` 覆盖默认命名空间或 TTL。"""
        raise NotImplementedError("behavior suite subclass must implement make_history")

    def make_message(
        self,
        content: str = "chest pain since monday",
        role: MessageRole = MessageRole.PATIENT,
        **overrides: Any,
    ) -> MedMessage:
        """构造一条属于默认命名空间的合法医疗消息。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE, "role": role, "content": content}
        kwargs.update(overrides)
        return MedMessage(**kwargs)

    @pytest.fixture
    def history(self) -> MedChatMessageHistory:
        """默认命名空间下的空会话历史。"""
        return self.make_history()

    # ------------------------------------------------------------------ #
    # 读写原语
    # ------------------------------------------------------------------ #
    def test_new_session_reads_empty(self, history: MedChatMessageHistory) -> None:
        assert history.get_med_messages() == []
        assert history.messages == []
        assert history.session_meta.message_count == 0

    def test_add_and_read_med_messages(self, history: MedChatMessageHistory) -> None:
        first = self.make_message("fever", created_at=1_700_000_000_000)
        second = self.make_message("cough", role=MessageRole.DOCTOR, created_at=1_700_000_001_000)

        history.add_med_messages([first, second])

        stored = history.get_med_messages()
        assert [m.message_id for m in stored] == [first.message_id, second.message_id]
        assert [m.content for m in stored] == ["fever", "cough"]

    def test_add_empty_sequence_is_noop(self, history: MedChatMessageHistory) -> None:
        history.add_med_messages([])

        assert history.get_med_messages() == []
        assert history.session_meta.message_count == 0

    def test_add_message_from_other_session_raises(self, history: MedChatMessageHistory) -> None:
        foreign = self.make_message(session_id="s-other")

        with pytest.raises(StorageError, match="belongs to"):
            history.add_med_messages([foreign])
        assert history.get_med_messages() == []

    def test_read_is_ordered_by_created_at(self, history: MedChatMessageHistory) -> None:
        later = self.make_message("later", created_at=1_700_000_009_000)
        earlier = self.make_message("earlier", created_at=1_700_000_001_000)

        history.add_med_messages([later, earlier])

        assert [m.content for m in history.get_med_messages()] == ["earlier", "later"]

    def test_read_limit_returns_latest_messages(self, history: MedChatMessageHistory) -> None:
        messages = [
            self.make_message(f"msg-{index}", created_at=1_700_000_000_000 + index)
            for index in range(5)
        ]
        history.add_med_messages(messages)

        assert [m.content for m in history.get_med_messages(limit=2)] == ["msg-3", "msg-4"]

    def test_read_limit_larger_than_size_returns_all(self, history: MedChatMessageHistory) -> None:
        history.add_med_messages([self.make_message()])

        assert len(history.get_med_messages(limit=99)) == 1

    @pytest.mark.parametrize("limit", [0, -1])
    def test_read_non_positive_limit_raises(
        self, history: MedChatMessageHistory, limit: int
    ) -> None:
        with pytest.raises(ValidationError, match="limit must be"):
            history.get_med_messages(limit=limit)

    # ------------------------------------------------------------------ #
    # LangChain 契约
    # ------------------------------------------------------------------ #
    def test_langchain_messages_roundtrip(self, history: MedChatMessageHistory) -> None:
        history.add_messages([HumanMessage(content="head ache"), AIMessage(content="since when?")])

        stored = history.messages
        assert [m.content for m in stored] == ["head ache", "since when?"]
        assert [m.additional_kwargs[MED_ROLE_KEY] for m in stored] == ["patient", "assistant"]
        assert all(m.additional_kwargs["session_id"] == history.session_id for m in stored)

    def test_add_single_langchain_message(self, history: MedChatMessageHistory) -> None:
        history.add_message(HumanMessage(content="dizzy"))

        assert [m.content for m in history.get_med_messages()] == ["dizzy"]
        assert history.get_med_messages()[0].role is MessageRole.PATIENT

    def test_add_unsupported_langchain_message_raises(self, history: MedChatMessageHistory) -> None:
        with pytest.raises(ValidationError, match="unsupported langchain message type"):
            history.add_messages([ToolMessage(content="ignored", tool_call_id="call-1")])
        assert history.get_med_messages() == []

    # ------------------------------------------------------------------ #
    # 租户命名空间隔离
    # ------------------------------------------------------------------ #
    def test_storage_key_format(self, history: MedChatMessageHistory) -> None:
        assert history.storage_key == self.STORAGE_KEY
        assert history.belongs_to(history.tenant_id, history.dept_id)

    def test_assert_tenant_rejects_foreign_caller(self, history: MedChatMessageHistory) -> None:
        history.assert_tenant(history.tenant_id, history.dept_id)

        with pytest.raises(TenantIsolationError, match="access denied"):
            history.assert_tenant("hospital_b", history.dept_id)

    def test_different_sessions_do_not_share_data(self, history: MedChatMessageHistory) -> None:
        other = self.make_history(session_id="s-other")
        history.add_med_messages([self.make_message("mine")])

        assert other.get_med_messages() == []
        assert len(history.get_med_messages()) == 1

    def test_second_handle_sees_same_session_data(self, history: MedChatMessageHistory) -> None:
        if not self.shared_across_handles:
            pytest.skip("backend does not share state across handles")
        history.add_med_messages([self.make_message("shared")])

        reopened = self.make_history()
        assert [m.content for m in reopened.get_med_messages()] == ["shared"]

    # ------------------------------------------------------------------ #
    # 清理与会话元数据
    # ------------------------------------------------------------------ #
    def test_clear_removes_all_messages(self, history: MedChatMessageHistory) -> None:
        history.add_med_messages([self.make_message(), self.make_message("second")])

        history.clear()

        assert history.get_med_messages() == []

    def test_clear_empty_session_is_idempotent(self, history: MedChatMessageHistory) -> None:
        history.clear()
        history.clear()

        assert history.get_med_messages() == []

    def test_message_count_and_updated_at_are_tracked(self, history: MedChatMessageHistory) -> None:
        created_at = history.session_meta.created_at

        history.add_med_messages([self.make_message(), self.make_message("second")])

        assert history.session_meta.message_count == 2
        assert history.session_meta.updated_at >= created_at

    # ------------------------------------------------------------------ #
    # 归档钩子
    # ------------------------------------------------------------------ #
    def test_archive_exports_messages_and_marks_status(
        self, history: MedChatMessageHistory
    ) -> None:
        history.add_med_messages([self.make_message("main complaint")])

        meta, messages = history.archive()

        assert meta.status is SessionStatus.ARCHIVED
        assert [m.content for m in messages] == ["main complaint"]
        assert history.get_med_messages() != [], "归档不应清理热存储数据"

    def test_archive_twice_raises(self, history: MedChatMessageHistory) -> None:
        history.archive()

        with pytest.raises(StateTransitionError, match="illegal session transition"):
            history.archive()

    # ------------------------------------------------------------------ #
    # TTL 钩子
    # ------------------------------------------------------------------ #
    def test_set_ttl_matches_backend_capability(self, history: MedChatMessageHistory) -> None:
        if type(history).supports_ttl:
            history.set_ttl(60)
            assert history.ttl_seconds == 60
            assert history.refresh_ttl() is True
        else:
            with pytest.raises(StorageError, match="does not support native ttl"):
                history.set_ttl(60)
            assert history.ttl_seconds is None
            assert history.refresh_ttl() is False

    @pytest.mark.parametrize("ttl", [0, -5])
    def test_set_ttl_rejects_non_positive(self, history: MedChatMessageHistory, ttl: int) -> None:
        with pytest.raises(ValidationError, match="ttl_seconds must be"):
            history.set_ttl(ttl)

    def test_is_expired_without_ttl_is_false(self, history: MedChatMessageHistory) -> None:
        assert history.is_expired() is False
        assert history.is_expired(now_ms=2_000_000_000_000) is False

    def test_is_expired_respects_ttl_window(self, history: MedChatMessageHistory) -> None:
        if not type(history).supports_ttl:
            pytest.skip("backend has no native ttl support")
        history.set_ttl(60)
        updated_at = history.session_meta.updated_at

        assert history.is_expired(now_ms=updated_at + 30_000) is False
        assert history.is_expired(now_ms=updated_at + 61_000) is True

    # ------------------------------------------------------------------ #
    # 工厂注册
    # ------------------------------------------------------------------ #
    def test_backend_is_registered_in_factory(self, history: MedChatMessageHistory) -> None:
        if self.backend_name is None:
            pytest.skip("backend is not exposed through StoreFactory")
        assert StoreFactory.is_registered(self.backend_name)
        assert StoreFactory.get(self.backend_name) is type(history)
