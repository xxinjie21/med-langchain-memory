"""InMemoryMedHistory 单元测试。

分两部分：
* :class:`TestInMemoryMedHistory` 复用跨后端共享行为套件，校验通用存储契约；
* 其余用例覆盖内存后端专有能力：进程内共享表、逻辑 TTL 惰性淘汰、线程安全。

全部用例零外部依赖，时间通过 monkeypatch 替换 ``now_millis`` 精确控制。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.stores import StoreConfig, StoreFactory
from med_langchain_memory.stores import memory_store as memory_store_module
from med_langchain_memory.stores.memory_store import InMemoryMedHistory

NAMESPACE = {
    "session_id": "s-mem",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-mem"


@pytest.fixture(autouse=True)
def clean_store() -> Iterator[None]:
    """每个用例前后都清空进程内全局会话表。"""
    InMemoryMedHistory.reset()
    yield
    InMemoryMedHistory.reset()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """可手动推进的假时钟，替换内存存储模块内的 ``now_millis``。"""
    state = {"now": 1_700_000_000_000}
    monkeypatch.setattr(memory_store_module, "now_millis", lambda: state["now"])
    return state


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


# --------------------------------------------------------------------------- #
# 共享行为契约
# --------------------------------------------------------------------------- #
class TestInMemoryMedHistory(MedHistoryBehaviorSuite):
    """内存后端必须满足全部通用存储行为契约。"""

    backend_name = "memory"
    shared_across_handles = True

    def make_history(self, **overrides: Any) -> InMemoryMedHistory:
        """构造内存存储实例。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE}
        kwargs.update(overrides)
        return InMemoryMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 进程内全局会话表
# --------------------------------------------------------------------------- #
def test_reset_clears_all_sessions() -> None:
    first = InMemoryMedHistory(**NAMESPACE)
    second = InMemoryMedHistory(**{**NAMESPACE, "session_id": "s-other"})
    first.add_med_messages([make_message()])
    second.add_med_messages([make_message(session_id="s-other")])
    assert len(InMemoryMedHistory.session_keys()) == 2

    InMemoryMedHistory.reset()

    assert InMemoryMedHistory.session_keys() == []
    assert first.get_med_messages() == []


def test_session_keys_lists_only_touched_sessions() -> None:
    history = InMemoryMedHistory(**NAMESPACE)
    assert InMemoryMedHistory.session_keys() == []

    history.add_med_messages([make_message()])

    assert InMemoryMedHistory.session_keys() == [STORAGE_KEY]


def test_size_reflects_stored_message_count() -> None:
    history = InMemoryMedHistory(**NAMESPACE)
    assert history.size == 0

    history.add_med_messages([make_message(), make_message("second")])

    assert history.size == 2


# --------------------------------------------------------------------------- #
# 逻辑 TTL 与惰性淘汰
# --------------------------------------------------------------------------- #
def test_expires_at_is_none_without_ttl() -> None:
    history = InMemoryMedHistory(**NAMESPACE)
    history.add_med_messages([make_message()])

    assert history.ttl_seconds is None
    assert history.expires_at is None


def test_ttl_sets_absolute_expiry_timestamp(clock: dict[str, int]) -> None:
    history = InMemoryMedHistory(**NAMESPACE, ttl_seconds=10)

    assert history.ttl_seconds == 10
    assert history.expires_at == clock["now"] + 10_000


def test_expired_session_is_evicted_on_read(clock: dict[str, int]) -> None:
    history = InMemoryMedHistory(**NAMESPACE, ttl_seconds=10)
    history.add_med_messages([make_message()])

    clock["now"] += 9_000
    assert history.size == 1

    clock["now"] += 2_000
    assert history.get_med_messages() == []
    assert InMemoryMedHistory.session_keys() == []


def test_write_slides_the_expiration_window(clock: dict[str, int]) -> None:
    history = InMemoryMedHistory(**NAMESPACE, ttl_seconds=10)
    history.add_med_messages([make_message("first")])

    clock["now"] += 9_000
    history.add_med_messages([make_message("second")])
    assert history.expires_at == clock["now"] + 10_000

    clock["now"] += 9_000
    assert history.size == 2


def test_expired_session_restarts_on_next_write(clock: dict[str, int]) -> None:
    history = InMemoryMedHistory(**NAMESPACE, ttl_seconds=10)
    history.add_med_messages([make_message("stale")])

    clock["now"] += 20_000
    history.add_med_messages([make_message("fresh")])

    assert [m.content for m in history.get_med_messages()] == ["fresh"]
    assert history.expires_at == clock["now"] + 10_000


def test_clear_restarts_the_expiration_window(clock: dict[str, int]) -> None:
    history = InMemoryMedHistory(**NAMESPACE, ttl_seconds=10)
    history.add_med_messages([make_message()])

    clock["now"] += 5_000
    history.clear()

    assert history.get_med_messages() == []
    assert history.expires_at == clock["now"] + 10_000


def test_expires_at_of_unknown_session_is_none() -> None:
    history = InMemoryMedHistory(**NAMESPACE)

    assert history.expires_at is None


# --------------------------------------------------------------------------- #
# 工厂集成
# --------------------------------------------------------------------------- #
def test_factory_creates_memory_backend() -> None:
    history = StoreFactory.create("memory", **NAMESPACE)

    assert isinstance(history, InMemoryMedHistory)
    assert history.storage_key == STORAGE_KEY


def test_factory_creates_memory_backend_from_config(clock: dict[str, int]) -> None:
    config = StoreConfig(backend="MEMORY", ttl_seconds=30)

    history = StoreFactory.create_from_config(config, **NAMESPACE)

    assert isinstance(history, InMemoryMedHistory)
    assert history.ttl_seconds == 30
    assert history.expires_at == clock["now"] + 30_000


# --------------------------------------------------------------------------- #
# 并发安全
# --------------------------------------------------------------------------- #
def test_concurrent_writes_keep_every_message() -> None:
    writers = 4
    per_writer = 25

    def write(worker: int) -> None:
        handle = InMemoryMedHistory(**NAMESPACE)
        handle.add_med_messages([make_message(f"w{worker}-{index}") for index in range(per_writer)])

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert InMemoryMedHistory(**NAMESPACE).size == writers * per_writer
