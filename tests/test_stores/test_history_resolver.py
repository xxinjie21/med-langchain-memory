"""会话历史解析器单元测试（D33）。

覆盖默认解析器 :class:`StoreFactoryHistoryResolver` 的正向解析（后端名 → 具体实现类、
命名空间映射）、可选参数透传（构造选项 / TTL），以及四类异常与边界路径：

* 未注册后端 → :class:`StoreNotFoundError`；
* 构造签名不匹配（多余选项）→ :class:`StorageError`；
* 后端不支持原生 TTL 却传了 ``ttl_seconds`` → :class:`StorageError`；
* ``options`` 缺省（``None``）时按无额外参数构造。

同时验证抽象基类不可实例化，避免解析器退化为「可被误用」的空壳。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.exceptions import StorageError, StoreNotFoundError
from med_langchain_memory.stores import (
    HistoryResolver,
    InMemoryMedHistory,
    SessionScope,
    StoreFactoryHistoryResolver,
)
from med_langchain_memory.stores.base import MedChatMessageHistory
from med_langchain_memory.stores.file_store import FileMedHistory

#: 测试用命名空间坐标。
SCOPE = SessionScope(tenant_id="hosp-a", dept_id="cardio")


def _resolve(
    backend: str = "memory",
    *,
    session_id: str = "s-1",
    patient_id: str = "p-1",
    ttl_seconds: int | None = None,
    options: dict[str, object] | None = None,
) -> MedChatMessageHistory:
    """用默认解析器解析历史句柄（简化各用例的重复参数）。"""
    return StoreFactoryHistoryResolver().resolve(
        backend,
        scope=SCOPE,
        session_id=session_id,
        patient_id=patient_id,
        ttl_seconds=ttl_seconds,
        options=options,
    )


def _message(session_id: str, content: str, created_at: int) -> MedMessage:
    """构造一条属于 ``SCOPE`` 命名空间的测试消息。"""
    return MedMessage(
        session_id=session_id,
        tenant_id=SCOPE.tenant_id,
        dept_id=SCOPE.dept_id,
        patient_id="p-1",
        role=MessageRole.PATIENT,
        content=content,
        created_at=created_at,
    )


# --------------------------------------------------------------------------- #
# 正向解析
# --------------------------------------------------------------------------- #
def test_resolve_memory_backend_maps_namespace() -> None:
    """``memory`` 后端解析为内存实现，且命名空间字段逐项对齐。"""
    history = _resolve(session_id="s-resolve-1", patient_id="p-9")
    assert isinstance(history, InMemoryMedHistory)
    assert history.session_id == "s-resolve-1"
    assert history.tenant_id == "hosp-a"
    assert history.dept_id == "cardio"
    assert history.patient_id == "p-9"
    assert history.storage_key == "med:chat:hosp-a:cardio:s-resolve-1"


def test_resolve_same_key_shares_storage() -> None:
    """同一后端 + 同一存储键的两次解析指向同一份存储（与远端存储语义一致）。"""
    session_id = "s-resolve-shared"
    writer = _resolve(session_id=session_id)
    writer.add_med_messages([_message(session_id, "主诉：头晕三天", 1000)])
    reader = _resolve(session_id=session_id)
    assert [m.content for m in reader.get_med_messages()] == ["主诉：头晕三天"]


def test_resolve_different_namespace_is_isolated() -> None:
    """不同科室解析出的历史互不可见（命名空间隔离）。"""
    writer = _resolve(session_id="s-resolve-iso")
    writer.add_med_messages([_message("s-resolve-iso", "甲科室消息", 1000)])
    other = StoreFactoryHistoryResolver().resolve(
        "memory",
        scope=SessionScope(tenant_id="hosp-a", dept_id="neuro"),
        session_id="s-resolve-iso",
        patient_id="p-1",
    )
    assert other.get_med_messages() == []


def test_resolve_passes_extra_options(tmp_path: Path) -> None:
    """``options`` 被透传给具体实现（``file`` 后端的 ``base_dir`` 生效）。"""
    history = _resolve(
        "file",
        session_id="s-file",
        options={"base_dir": str(tmp_path)},
    )
    assert isinstance(history, FileMedHistory)
    history.add_med_messages([_message("s-file", "文件后端消息", 1000)])
    assert (tmp_path / "hosp-a" / "cardio" / "s-file.jsonl").is_file()


def test_resolve_without_options_uses_defaults() -> None:
    """``options=None`` 时按无额外参数构造（边界：缺省分支）。"""
    history = _resolve(session_id="s-resolve-default", options=None)
    assert isinstance(history, InMemoryMedHistory)
    assert history.get_med_messages() == []


def test_resolve_passes_ttl_seconds() -> None:
    """``ttl_seconds`` 透传到底层存储（``memory`` 支持原生 TTL）。"""
    history = _resolve(session_id="s-ttl", ttl_seconds=60)
    assert history.ttl_seconds == 60
    assert history.refresh_ttl() is True


# --------------------------------------------------------------------------- #
# 异常与边界
# --------------------------------------------------------------------------- #
def test_resolve_unknown_backend_raises_store_not_found() -> None:
    """未注册后端 → :class:`StoreNotFoundError`（边界：后端名不存在）。"""
    with pytest.raises(StoreNotFoundError):
        _resolve("no-such-backend")


def test_resolve_incompatible_option_raises_storage_error() -> None:
    """实现类不接受该构造参数 → :class:`StorageError`（边界：选项名拼错）。"""
    with pytest.raises(StorageError):
        _resolve(session_id="s-bad-opt", options={"nope": 1})


def test_resolve_unsupported_ttl_raises_storage_error() -> None:
    """``file`` 后端不支持原生 TTL，传入 ``ttl_seconds`` → :class:`StorageError`。"""
    with pytest.raises(StorageError):
        _resolve("file", session_id="s-file-ttl", ttl_seconds=60)


def test_resolver_abstract_cannot_instantiate() -> None:
    """抽象基类不可实例化（边界：防止解析器被误用为空壳）。"""
    with pytest.raises(TypeError):
        HistoryResolver()  # type: ignore[abstract]


def test_default_resolver_implements_contract() -> None:
    """默认解析器满足 :class:`HistoryResolver` 契约。"""
    assert isinstance(StoreFactoryHistoryResolver(), HistoryResolver)
