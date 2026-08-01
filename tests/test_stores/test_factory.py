"""StoreFactory 注册器与配置驱动实例化单元测试。

测试不依赖任何真实中间件：用进程内列表实现的替身 Store 覆盖注册与实例化路径。
每个用例前后都会快照/还原全局注册表，避免用例之间互相污染。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.domain import MedMessage
from med_langchain_memory.exceptions import (
    StorageError,
    StoreNotFoundError,
    StoreRegistrationError,
)
from med_langchain_memory.stores import MedChatMessageHistory, StoreConfig, StoreFactory

NAMESPACE = {
    "session_id": "s-001",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}


class FakeStore(MedChatMessageHistory):
    """进程内列表替身存储，仅实现三个存储原语。"""

    def __init__(self, **kwargs: object) -> None:
        self._items: list[MedMessage] = []
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def _append(self, messages: list[MedMessage]) -> None:
        self._items.extend(messages)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        return list(self._items) if limit is None else list(self._items[-limit:])

    def clear(self) -> None:
        self._items = []


class OptionStore(FakeStore):
    """接受额外构造参数的替身存储，用于校验 options 透传。"""

    def __init__(self, *, url: str, encoding: str = "protobuf", **kwargs: object) -> None:
        self.url = url
        self.encoding = encoding
        super().__init__(**kwargs)


class TtlStore(FakeStore):
    """支持原生 TTL 的替身存储。"""

    supports_ttl = True

    def __init__(self, **kwargs: object) -> None:
        self.applied: list[int] = []
        super().__init__(**kwargs)

    def _apply_ttl(self, ttl_seconds: int) -> None:
        self.applied.append(ttl_seconds)


class NotAStore:
    """非 MedChatMessageHistory 子类，用于校验注册类型约束。"""


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """每个用例使用独立的注册表副本。"""
    snapshot = dict(StoreFactory._registry)
    StoreFactory._registry.clear()
    yield
    StoreFactory._registry.clear()
    StoreFactory._registry.update(snapshot)


@pytest.fixture
def fake_backend() -> type[FakeStore]:
    """注册名为 ``fake`` 的替身存储。"""
    return StoreFactory.register("fake")(FakeStore)


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #
def test_register_puts_class_into_registry(fake_backend: type[FakeStore]) -> None:
    assert fake_backend is FakeStore
    assert StoreFactory.is_registered("fake")
    assert StoreFactory.get("fake") is FakeStore


def test_register_as_decorator_returns_original_class() -> None:
    @StoreFactory.register("memory")
    class DecoratedStore(FakeStore):
        """装饰器语法注册。"""

    assert StoreFactory.get("memory") is DecoratedStore
    assert DecoratedStore.__name__ == "DecoratedStore"


def test_register_normalizes_name_case_and_spaces() -> None:
    StoreFactory.register("  Redis  ")(FakeStore)
    assert StoreFactory.available() == ["redis"]
    assert StoreFactory.is_registered("REDIS")


def test_register_duplicate_name_raises(fake_backend: type[FakeStore]) -> None:
    with pytest.raises(StoreRegistrationError, match="already registered"):
        StoreFactory.register("fake")(TtlStore)


def test_register_duplicate_with_override_replaces(fake_backend: type[FakeStore]) -> None:
    StoreFactory.register("fake", override=True)(TtlStore)
    assert StoreFactory.get("fake") is TtlStore


@pytest.mark.parametrize("name", ["", "   ", "bad name", "-redis", "redis!", "中文"])
def test_register_invalid_name_raises(name: str) -> None:
    with pytest.raises(StoreRegistrationError, match="invalid backend name"):
        StoreFactory.register(name)


def test_register_non_string_name_raises() -> None:
    with pytest.raises(StoreRegistrationError, match="must be a string"):
        StoreFactory.register(123)  # type: ignore[arg-type]


def test_register_non_store_class_raises() -> None:
    with pytest.raises(StoreRegistrationError, match="not a subclass"):
        StoreFactory.register("bogus")(NotAStore)  # type: ignore[type-var]
    assert not StoreFactory.is_registered("bogus")


def test_register_abstract_class_raises() -> None:
    with pytest.raises(StoreRegistrationError, match="abstract store"):
        StoreFactory.register("abstract")(MedChatMessageHistory)  # type: ignore[type-abstract]


# --------------------------------------------------------------------------- #
# unregister / 查询
# --------------------------------------------------------------------------- #
def test_unregister_removes_backend(fake_backend: type[FakeStore]) -> None:
    StoreFactory.unregister("FAKE")
    assert StoreFactory.available() == []


def test_unregister_unknown_backend_raises() -> None:
    with pytest.raises(StoreNotFoundError, match="unknown store backend"):
        StoreFactory.unregister("ghost")


def test_available_returns_sorted_names() -> None:
    StoreFactory.register("redis")(FakeStore)
    StoreFactory.register("memory")(FakeStore)
    StoreFactory.register("file")(FakeStore)
    assert StoreFactory.available() == ["file", "memory", "redis"]


def test_get_unknown_backend_lists_available(fake_backend: type[FakeStore]) -> None:
    with pytest.raises(StoreNotFoundError, match=r"available: \['fake'\]"):
        StoreFactory.get("ghost")


def test_is_registered_false_for_unknown() -> None:
    assert StoreFactory.is_registered("ghost") is False


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
def test_create_builds_instance_with_namespace(fake_backend: type[FakeStore]) -> None:
    store = StoreFactory.create("fake", **NAMESPACE)  # type: ignore[arg-type]
    assert isinstance(store, FakeStore)
    assert store.storage_key == "med:chat:hospital_a:cardiology:s-001"
    assert store.ttl_seconds is None


def test_create_passes_ttl_to_ttl_capable_store() -> None:
    StoreFactory.register("ttl")(TtlStore)
    store = StoreFactory.create("ttl", ttl_seconds=60, **NAMESPACE)  # type: ignore[arg-type]
    assert isinstance(store, TtlStore)
    assert store.ttl_seconds == 60
    assert store.applied == [60]


def test_create_passes_extra_options_to_implementation() -> None:
    StoreFactory.register("opt")(OptionStore)
    store = StoreFactory.create(
        "opt",
        url="redis://localhost:6379/0",
        encoding="json",
        **NAMESPACE,  # type: ignore[arg-type]
    )
    assert isinstance(store, OptionStore)
    assert store.url == "redis://localhost:6379/0"
    assert store.encoding == "json"


def test_create_unknown_backend_raises() -> None:
    with pytest.raises(StoreNotFoundError, match="unknown store backend"):
        StoreFactory.create("ghost", **NAMESPACE)  # type: ignore[arg-type]


def test_create_with_missing_required_option_raises_storage_error() -> None:
    StoreFactory.register("opt")(OptionStore)
    with pytest.raises(StorageError, match="cannot build OptionStore"):
        StoreFactory.create("opt", **NAMESPACE)  # type: ignore[arg-type]


def test_create_with_unexpected_option_raises_storage_error(
    fake_backend: type[FakeStore],
) -> None:
    with pytest.raises(StorageError, match="cannot build FakeStore"):
        StoreFactory.create("fake", nope=1, **NAMESPACE)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# StoreConfig
# --------------------------------------------------------------------------- #
def test_store_config_defaults_and_normalization() -> None:
    config = StoreConfig(backend="  Memory ")
    assert config.backend == "memory"
    assert config.ttl_seconds is None
    assert config.options == {}


def test_store_config_accepts_ttl_and_options() -> None:
    config = StoreConfig(backend="redis", ttl_seconds=3600, options={"url": "redis://x"})
    assert config.ttl_seconds == 3600
    assert config.options["url"] == "redis://x"


@pytest.mark.parametrize("ttl", [0, -1])
def test_store_config_rejects_non_positive_ttl(ttl: int) -> None:
    with pytest.raises(PydanticValidationError):
        StoreConfig(backend="redis", ttl_seconds=ttl)


def test_store_config_rejects_empty_backend() -> None:
    with pytest.raises(PydanticValidationError, match="must not be empty"):
        StoreConfig(backend="   ")


def test_store_config_forbids_unknown_field() -> None:
    with pytest.raises(PydanticValidationError):
        StoreConfig(backend="redis", unknown=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# create_from_config
# --------------------------------------------------------------------------- #
def test_create_from_config_builds_instance() -> None:
    StoreFactory.register("opt")(OptionStore)
    config = StoreConfig(backend="OPT", options={"url": "redis://cfg", "encoding": "json"})
    store = StoreFactory.create_from_config(config, **NAMESPACE)  # type: ignore[arg-type]
    assert isinstance(store, OptionStore)
    assert store.url == "redis://cfg"
    assert store.ttl_seconds is None


def test_create_from_config_applies_ttl() -> None:
    StoreFactory.register("ttl")(TtlStore)
    config = StoreConfig(backend="ttl", ttl_seconds=120)
    store = StoreFactory.create_from_config(config, **NAMESPACE)  # type: ignore[arg-type]
    assert store.ttl_seconds == 120


def test_create_from_config_unknown_backend_raises() -> None:
    config = StoreConfig(backend="ghost")
    with pytest.raises(StoreNotFoundError):
        StoreFactory.create_from_config(config, **NAMESPACE)  # type: ignore[arg-type]
