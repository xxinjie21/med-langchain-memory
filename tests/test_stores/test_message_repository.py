"""消息仓储单元测试（D32）。

覆盖 :class:`InMemoryMessageRepository` 的 4 个原语（append / read / count / clear）
的正向路径与边界/异常路径：空批次、命名空间不匹配、跨会话与跨租户隔离、乱序写入后
的时序契约、幂等清空，以及多线程并发追加不丢消息。
"""

from __future__ import annotations

import threading

import pytest

from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.stores.message_repository import (
    InMemoryMessageRepository,
    MessageRepository,
)
from med_langchain_memory.stores.session_repository import SessionScope

SCOPE = SessionScope(tenant_id="hosp-a", dept_id="cardio")
OTHER_TENANT = SessionScope(tenant_id="hosp-b", dept_id="cardio")
OTHER_DEPT = SessionScope(tenant_id="hosp-a", dept_id="neuro")


def _message(
    index: int,
    *,
    session_id: str = "s-1",
    created_at: int | None = None,
    scope: SessionScope = SCOPE,
) -> MedMessage:
    """构造时序可控的测试消息。"""
    return MedMessage(
        session_id=session_id,
        tenant_id=scope.tenant_id,
        dept_id=scope.dept_id,
        patient_id="p-1",
        role=MessageRole.PATIENT,
        content=f"msg-{index}",
        created_at=created_at if created_at is not None else 1000 + index,
    )


@pytest.fixture
def repository() -> InMemoryMessageRepository:
    """每个用例一个全新的内存仓储（避免共享状态串扰）。"""
    return InMemoryMessageRepository()


# --------------------------------------------------------------------------- #
# append / read
# --------------------------------------------------------------------------- #
def test_append_returns_sorted_batch(repository: InMemoryMessageRepository) -> None:
    """``append`` 返回本次写入的消息并按时序升序排列。"""
    batch = repository.append(
        SCOPE,
        "s-1",
        [_message(2, created_at=3000), _message(0, created_at=1000), _message(1, created_at=2000)],
    )
    assert [message.content for message in batch] == ["msg-0", "msg-1", "msg-2"]


def test_read_returns_all_messages_in_time_order(repository: InMemoryMessageRepository) -> None:
    """分批乱序写入后，``read`` 仍返回全局时序升序的完整消息。"""
    repository.append(SCOPE, "s-1", [_message(2, created_at=3000)])
    repository.append(SCOPE, "s-1", [_message(0, created_at=1000), _message(1, created_at=2000)])
    assert [message.content for message in repository.read(SCOPE, "s-1")] == [
        "msg-0",
        "msg-1",
        "msg-2",
    ]


def test_read_unknown_session_returns_empty(repository: InMemoryMessageRepository) -> None:
    """读取不存在的会话返回空列表（边界：未创建会话）。"""
    assert repository.read(SCOPE, "s-none") == []


def test_read_returns_copy_not_internal_list(repository: InMemoryMessageRepository) -> None:
    """``read`` 返回副本，调用方改动不影响仓储内部状态（边界：防御性拷贝）。"""
    repository.append(SCOPE, "s-1", [_message(0)])
    snapshot = repository.read(SCOPE, "s-1")
    snapshot.clear()
    assert repository.count(SCOPE, "s-1") == 1


def test_append_rejects_empty_batch(repository: InMemoryMessageRepository) -> None:
    """空批次 → 领域校验异常（边界：空写入）。"""
    with pytest.raises(ValidationError, match="must not be empty"):
        repository.append(SCOPE, "s-1", [])


def test_append_rejects_foreign_namespace_message(repository: InMemoryMessageRepository) -> None:
    """消息不属于目标命名空间 → :class:`StorageError`（越权写入防护）。"""
    foreign = _message(0, scope=OTHER_TENANT)
    with pytest.raises(StorageError, match="belongs to"):
        repository.append(SCOPE, "s-1", [foreign])


def test_append_rejects_message_of_other_session(repository: InMemoryMessageRepository) -> None:
    """消息归属其他会话 → :class:`StorageError`（边界：会话错配）。"""
    with pytest.raises(StorageError, match="belongs to"):
        repository.append(SCOPE, "s-1", [_message(0, session_id="s-2")])


def test_append_is_atomic_on_validation_failure(repository: InMemoryMessageRepository) -> None:
    """批次中任一消息非法时整批拒绝，不产生部分写入（边界：原子性）。"""
    with pytest.raises(StorageError):
        repository.append(SCOPE, "s-1", [_message(0), _message(1, session_id="s-2")])
    assert repository.count(SCOPE, "s-1") == 0


# --------------------------------------------------------------------------- #
# count / clear
# --------------------------------------------------------------------------- #
def test_count_tracks_appended_messages(repository: InMemoryMessageRepository) -> None:
    """``count`` 随追加累加。"""
    assert repository.count(SCOPE, "s-1") == 0
    repository.append(SCOPE, "s-1", [_message(0), _message(1)])
    assert repository.count(SCOPE, "s-1") == 2


def test_count_unknown_session_is_zero(repository: InMemoryMessageRepository) -> None:
    """未知会话计数为 0（边界：不存在）。"""
    assert repository.count(SCOPE, "s-none") == 0


def test_clear_removes_all_messages(repository: InMemoryMessageRepository) -> None:
    """``clear`` 清空会话消息。"""
    repository.append(SCOPE, "s-1", [_message(0), _message(1)])
    repository.clear(SCOPE, "s-1")
    assert repository.read(SCOPE, "s-1") == []


def test_clear_is_idempotent(repository: InMemoryMessageRepository) -> None:
    """重复清空不报错（边界：幂等）。"""
    repository.clear(SCOPE, "s-none")
    repository.clear(SCOPE, "s-none")
    assert repository.count(SCOPE, "s-none") == 0


# --------------------------------------------------------------------------- #
# 隔离
# --------------------------------------------------------------------------- #
def test_sessions_are_isolated(repository: InMemoryMessageRepository) -> None:
    """同一命名空间下不同会话的消息互不可见。"""
    repository.append(SCOPE, "s-1", [_message(0, session_id="s-1")])
    repository.append(SCOPE, "s-2", [_message(1, session_id="s-2")])
    assert [message.content for message in repository.read(SCOPE, "s-1")] == ["msg-0"]
    assert [message.content for message in repository.read(SCOPE, "s-2")] == ["msg-1"]


@pytest.mark.parametrize("scope", [OTHER_TENANT, OTHER_DEPT])
def test_namespaces_are_isolated(
    repository: InMemoryMessageRepository, scope: SessionScope
) -> None:
    """跨租户 / 跨科室读写互不可见（越权防护）。"""
    repository.append(SCOPE, "s-1", [_message(0)])
    assert repository.read(scope, "s-1") == []
    assert repository.count(scope, "s-1") == 0


def test_clear_only_affects_target_session(repository: InMemoryMessageRepository) -> None:
    """清空某会话不影响同命名空间的其他会话。"""
    repository.append(SCOPE, "s-1", [_message(0, session_id="s-1")])
    repository.append(SCOPE, "s-2", [_message(1, session_id="s-2")])
    repository.clear(SCOPE, "s-1")
    assert repository.count(SCOPE, "s-2") == 1


# --------------------------------------------------------------------------- #
# 并发
# --------------------------------------------------------------------------- #
def test_concurrent_appends_do_not_lose_messages(repository: InMemoryMessageRepository) -> None:
    """多线程并发追加不丢消息（边界：ASGI worker 并发写入）。"""
    threads = [
        threading.Thread(target=repository.append, args=(SCOPE, "s-1", [_message(index)]))
        for index in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert repository.count(SCOPE, "s-1") == 20


# --------------------------------------------------------------------------- #
# 抽象契约
# --------------------------------------------------------------------------- #
def test_repository_is_abstract() -> None:
    """``MessageRepository`` 为抽象类，不可直接实例化（边界：抽象契约）。"""
    with pytest.raises(TypeError):
        MessageRepository()  # type: ignore[abstract]


def test_in_memory_repository_implements_contract() -> None:
    """内存实现满足抽象契约。"""
    assert isinstance(InMemoryMessageRepository(), MessageRepository)
