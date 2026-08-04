"""FileMedHistory 单元测试。

分三部分：

* :class:`TestFileStoreJsonl` / :class:`TestFileStoreBinary` 复用跨后端共享行为套件，
  校验两种落盘格式都满足通用存储契约；
* 落盘格式用例：目录布局、JSONL 行结构、二进制帧结构、UTF-8 与追加语义；
* 健壮性用例：损坏记录、截断帧、非法格式名、路径穿越、工厂集成与并发写。

全部用例零外部依赖，数据写入 pytest ``tmp_path`` 临时目录。
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest
from behavior import MedHistoryBehaviorSuite

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.serde import ProtobufSerializer
from med_langchain_memory.stores import StoreConfig, StoreFactory
from med_langchain_memory.stores.file_store import FileFormat, FileMedHistory

NAMESPACE = {
    "session_id": "s-file",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
LENGTH_PREFIX = struct.Struct(">I")


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


def make_history(tmp_path: Path, **overrides: Any) -> FileMedHistory:
    """在临时目录下构造文件存储实例。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "base_dir": tmp_path}
    kwargs.update(overrides)
    return FileMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 共享行为契约（两种格式各跑一遍）
# --------------------------------------------------------------------------- #
class _FileStoreSuite(MedHistoryBehaviorSuite):
    """文件后端的共享行为契约基类，子类只切换落盘格式。"""

    backend_name = "file"
    shared_across_handles = True
    file_format: ClassVar[FileFormat] = FileFormat.JSONL

    @pytest.fixture(autouse=True)
    def _isolated_base_dir(self, tmp_path: Path) -> None:
        """每个用例使用独立的数据根目录。"""
        self.base_dir = tmp_path

    def make_history(self, **overrides: Any) -> FileMedHistory:
        """构造文件存储实例。"""
        kwargs: dict[str, Any] = {
            **self.NAMESPACE,
            "base_dir": self.base_dir,
            "file_format": self.file_format,
        }
        kwargs.update(overrides)
        return FileMedHistory(**kwargs)


class TestFileStoreJsonl(_FileStoreSuite):
    """JSONL 模式必须满足全部通用存储行为契约。"""

    file_format = FileFormat.JSONL


class TestFileStoreBinary(_FileStoreSuite):
    """protobuf 二进制模式必须满足全部通用存储行为契约。"""

    file_format = FileFormat.BINARY


# --------------------------------------------------------------------------- #
# 目录布局
# --------------------------------------------------------------------------- #
def test_path_layout_is_tenant_dept_session(tmp_path: Path) -> None:
    history = make_history(tmp_path)

    assert history.path == tmp_path / "hospital_a" / "cardiology" / "s-file.jsonl"
    assert history.lock_path == Path(f"{history.path}.lock")
    assert history.file_format is FileFormat.JSONL


def test_binary_format_uses_pb_suffix(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)

    assert history.path.name == "s-file.pb"


def test_format_accepts_plain_string(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format="binary")

    assert history.file_format is FileFormat.BINARY


def test_unknown_format_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown file format"):
        make_history(tmp_path, file_format="yaml")


@pytest.mark.parametrize("field", ["tenant_id", "dept_id", "session_id"])
def test_path_traversal_component_rejected(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValidationError, match="unsafe path component"):
        make_history(tmp_path, **{field: ".."})


def test_file_is_created_lazily_on_first_write(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    assert history.exists is False
    assert history.get_med_messages() == []

    history.add_med_messages([make_message()])

    assert history.exists is True
    assert history.size == 1


def test_two_formats_use_separate_files(tmp_path: Path) -> None:
    jsonl = make_history(tmp_path, file_format=FileFormat.JSONL)
    binary = make_history(tmp_path, file_format=FileFormat.BINARY)

    jsonl.add_med_messages([make_message("text mode")])

    assert binary.get_med_messages() == []
    assert jsonl.path != binary.path


def test_clear_removes_the_log_file(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message()])

    history.clear()

    assert history.exists is False
    assert history.get_med_messages() == []


# --------------------------------------------------------------------------- #
# JSONL 落盘格式
# --------------------------------------------------------------------------- #
def test_jsonl_writes_one_line_per_message(tmp_path: Path) -> None:
    history = make_history(tmp_path)

    history.add_med_messages([make_message("first"), make_message("second")])

    lines = history.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["content"] for line in lines] == ["first", "second"]


def test_jsonl_record_keeps_full_field_set(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    message = make_message("fever", token_count=7, masked=True, metadata={"icd": "R50"})

    history.add_med_messages([message])

    record = json.loads(history.path.read_text(encoding="utf-8").strip())
    assert record["message_id"] == message.message_id
    assert record["role"] == "patient"
    assert record["token_count"] == 7
    assert record["masked"] is True
    assert record["metadata"] == {"icd": "R50"}


def test_jsonl_keeps_non_ascii_readable(tmp_path: Path) -> None:
    history = make_history(tmp_path)

    history.add_med_messages([make_message("胸痛三天")])

    assert "胸痛三天" in history.path.read_text(encoding="utf-8")
    assert history.get_med_messages()[0].content == "胸痛三天"


def test_jsonl_append_does_not_rewrite_existing_records(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message("first")])
    first_bytes = history.path.read_bytes()

    history.add_med_messages([make_message("second")])

    assert history.path.read_bytes().startswith(first_bytes)


def test_jsonl_blank_lines_are_ignored(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message()])
    with open(history.path, "a", encoding="utf-8") as handle:
        handle.write("\n   \n")

    assert len(history.get_med_messages()) == 1


def test_jsonl_corrupted_line_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message()])
    with open(history.path, "a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    with pytest.raises(StorageError, match="corrupted jsonl record at line 2"):
        history.get_med_messages()


def test_jsonl_record_with_invalid_field_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message()])
    with open(history.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"session_id": "s-file"}) + "\n")

    with pytest.raises(StorageError, match="corrupted jsonl record"):
        history.get_med_messages()


def test_jsonl_invalid_utf8_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path)
    history.add_med_messages([make_message()])
    with open(history.path, "ab") as handle:
        handle.write(b"\xff\xfe\n")

    with pytest.raises(StorageError, match="not valid utf-8 jsonl"):
        history.get_med_messages()


# --------------------------------------------------------------------------- #
# 二进制落盘格式
# --------------------------------------------------------------------------- #
def test_binary_writes_length_prefixed_frames(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    serializer = ProtobufSerializer()
    message = make_message("bradycardia")

    history.add_med_messages([message])

    raw = history.path.read_bytes()
    (size,) = LENGTH_PREFIX.unpack_from(raw, 0)
    body = raw[LENGTH_PREFIX.size : LENGTH_PREFIX.size + size]
    assert len(raw) == LENGTH_PREFIX.size + size
    assert serializer.deserialize_message(body).content == "bradycardia"


def test_binary_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    message = make_message(
        "血压偏高", role=MessageRole.DOCTOR, token_count=12, masked=True, metadata={"bp": "150/95"}
    )

    history.add_med_messages([message])

    assert history.get_med_messages() == [message]


def test_binary_truncated_prefix_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    history.add_med_messages([make_message()])
    with open(history.path, "ab") as handle:
        handle.write(b"\x00\x01")

    with pytest.raises(StorageError, match="truncated frame length prefix"):
        history.get_med_messages()


def test_binary_frame_longer_than_file_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    history.add_med_messages([make_message()])
    with open(history.path, "ab") as handle:
        handle.write(LENGTH_PREFIX.pack(4096) + b"partial")

    with pytest.raises(StorageError, match="corrupted frame of declared size 4096"):
        history.get_med_messages()


def test_binary_zero_length_frame_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    history.path.parent.mkdir(parents=True, exist_ok=True)
    history.path.write_bytes(LENGTH_PREFIX.pack(0))

    with pytest.raises(StorageError, match="corrupted frame of declared size 0"):
        history.get_med_messages()


def test_binary_oversized_frame_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    history.path.parent.mkdir(parents=True, exist_ok=True)
    history.path.write_bytes(LENGTH_PREFIX.pack(9 * 1024 * 1024) + b"x")

    with pytest.raises(StorageError, match="corrupted frame of declared size"):
        history.get_med_messages()


def test_binary_corrupted_payload_raises(tmp_path: Path) -> None:
    history = make_history(tmp_path, file_format=FileFormat.BINARY)
    history.path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\xff\xff\xff\xff\xff\xff"
    history.path.write_bytes(LENGTH_PREFIX.pack(len(payload)) + payload)

    with pytest.raises(StorageError, match="corrupted protobuf frame"):
        history.get_med_messages()


# --------------------------------------------------------------------------- #
# 工厂集成
# --------------------------------------------------------------------------- #
def test_factory_creates_file_backend(tmp_path: Path) -> None:
    history = StoreFactory.create("file", **NAMESPACE, base_dir=tmp_path)

    assert isinstance(history, FileMedHistory)
    assert history.path.parent == tmp_path / "hospital_a" / "cardiology"


def test_factory_creates_file_backend_from_config(tmp_path: Path) -> None:
    config = StoreConfig(
        backend="FILE", options={"base_dir": str(tmp_path), "file_format": "binary"}
    )

    history = StoreFactory.create_from_config(config, **NAMESPACE)

    assert isinstance(history, FileMedHistory)
    assert history.file_format is FileFormat.BINARY


def test_factory_rejects_ttl_for_file_backend(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="does not support native ttl"):
        StoreFactory.create("file", **NAMESPACE, ttl_seconds=30, base_dir=tmp_path)


# --------------------------------------------------------------------------- #
# 并发写
# --------------------------------------------------------------------------- #
def test_concurrent_writes_keep_every_message(tmp_path: Path) -> None:
    writers = 4
    per_writer = 10

    def write(worker: int) -> None:
        handle = make_history(tmp_path)
        handle.add_med_messages([make_message(f"w{worker}-{index}") for index in range(per_writer)])

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert make_history(tmp_path).size == writers * per_writer
