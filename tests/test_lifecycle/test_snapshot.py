"""会话快照 :mod:`snapshot` 单元测试。

覆盖文件包字节布局/校验和/损坏检测、导出→导入往返、命名空间隔离、批处理与
便捷函数等场景。仅依赖 ``MedChatMessageHistory`` 公共接口与磁盘临时文件，
无需任何真实中间件即可跑通。
"""

from __future__ import annotations

import hashlib

import pytest
from conftest import NAMESPACE, FakeHistory, make_messages

from med_langchain_memory.exceptions import (
    IntegrityError,
    TenantIsolationError,
)
from med_langchain_memory.lifecycle import (
    DEFAULT_SCHEMA_VERSION,
    SessionSnapshotPackage,
    SessionSnapshotter,
    SnapshotSummary,
    restore_session,
    snapshot_session,
)
from med_langchain_memory.serde import SerializationError
from med_langchain_memory.serde.protobuf_serializer import ProtobufSerializer


# ====================================================================== #
# 文件包 SessionSnapshotPackage
# ====================================================================== #
def test_package_roundtrip_bytes():
    pkg = SessionSnapshotPackage(schema_version="2", payload=b"protobuf-bytes")
    raw = pkg.to_bytes()
    assert raw.startswith(b"MEDSNAP1")
    assert SessionSnapshotPackage.from_bytes(raw) == pkg


def test_package_from_bytes_rejects_short_input():
    with pytest.raises(SerializationError):
        SessionSnapshotPackage.from_bytes(b"MEDSNAP1")


def test_package_from_bytes_rejects_bad_magic():
    bad = b"NOTSNAP" + b"\x00" * 40
    with pytest.raises(SerializationError):
        SessionSnapshotPackage.from_bytes(bad)


def test_package_detects_checksum_tamper():
    pkg = SessionSnapshotPackage(schema_version="1", payload=b"hello")
    raw = bytearray(pkg.to_bytes())
    # 翻转 payload 中一个字节但不更新校验和 -> 校验失败
    raw[20] ^= 0xFF
    with pytest.raises(IntegrityError):
        SessionSnapshotPackage.from_bytes(bytes(raw))


def test_package_save_and_load_disk(tmp_path):
    path = tmp_path / "snap.bin"
    pkg = SessionSnapshotPackage(schema_version="1", payload=b"x" * 10)
    pkg.save(path)
    assert path.is_file()
    assert SessionSnapshotPackage.load(path) == pkg


def test_package_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SessionSnapshotPackage.load(tmp_path / "nope.bin")


# ====================================================================== #
# 导出 export_session
# ====================================================================== #
def test_export_writes_recoverable_file(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(3))
    path = tmp_path / "s.bin"

    summary = SessionSnapshotter().export_session(src, path)

    assert path.is_file()
    assert summary.session_key == src.storage_key
    assert summary.message_count == 3
    assert summary.schema_version == DEFAULT_SCHEMA_VERSION
    assert summary.verified is None
    assert isinstance(summary, SnapshotSummary)


def test_export_empty_session(tmp_path):
    src = FakeHistory(**NAMESPACE)
    path = tmp_path / "empty.bin"
    summary = SessionSnapshotter().export_session(src, path)
    assert summary.message_count == 0
    assert path.is_file()


def test_export_custom_schema_version(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(1))
    path = tmp_path / "v.bin"
    summary = SessionSnapshotter().export_session(src, path, schema_version="9")
    assert summary.schema_version == "9"
    pkg = SessionSnapshotPackage.load(path)
    assert pkg.schema_version == "9"


# ====================================================================== #
# 导入 import_session
# ====================================================================== #
def test_import_roundtrip_preserves_messages(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(5))
    path = tmp_path / "rt.bin"
    SessionSnapshotter().export_session(src, path)

    dst = FakeHistory(**NAMESPACE)
    assert dst.get_med_messages() == []

    summary = SessionSnapshotter().import_session(path, dst)

    restored = dst.get_med_messages()
    assert len(restored) == 5
    assert [m.message_id for m in restored] == [m.message_id for m in src.get_med_messages()]
    assert summary.verified is True


def test_import_roundtrip_preserves_namespace_and_meta(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(2))
    path = tmp_path / "meta.bin"
    SessionSnapshotter().export_session(src, path)

    dst = FakeHistory(**NAMESPACE)
    SessionSnapshotter().import_session(path, dst)

    assert dst.session_meta.tenant_id == NAMESPACE["tenant_id"]
    assert dst.session_meta.dept_id == NAMESPACE["dept_id"]
    assert dst.session_meta.patient_id == NAMESPACE["patient_id"]
    assert dst.session_meta.message_count == 2


def test_import_rejects_cross_tenant(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(1))
    path = tmp_path / "x.bin"
    SessionSnapshotter().export_session(src, path)

    other = FakeHistory(
        session_id="s-1",
        tenant_id="hosp-b",
        dept_id="cardio",
        patient_id="p-1",
    )
    with pytest.raises(TenantIsolationError):
        SessionSnapshotter().import_session(path, other)


def test_import_corrupted_file_raises(tmp_path):
    path = tmp_path / "corrupt.bin"
    path.write_bytes(b"MEDSNAP1" + b"\x00" * 40)
    dst = FakeHistory(**NAMESPACE)
    with pytest.raises((SerializationError, IntegrityError)):
        SessionSnapshotter().import_session(path, dst)


def test_import_invalid_protobuf_raises(tmp_path):
    # 合法文件包但 payload 不是 SessionSnapshot protobuf
    pkg = SessionSnapshotPackage(schema_version="1", payload=b"not-a-proto")
    path = tmp_path / "bad.bin"
    pkg.save(path)
    dst = FakeHistory(**NAMESPACE)
    with pytest.raises(SerializationError):
        SessionSnapshotter().import_session(path, dst)


def test_import_is_idempotent_on_second_run(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(3))
    path = tmp_path / "id.bin"
    SessionSnapshotter().export_session(src, path)

    dst = FakeHistory(**NAMESPACE)
    SessionSnapshotter().import_session(path, dst)
    SessionSnapshotter().import_session(path, dst)  # 重跑不增重复
    assert len(dst.get_med_messages()) == 3


# ====================================================================== #
# 大批量
# ====================================================================== #
def test_export_import_large_batch(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(500))
    path = tmp_path / "big.bin"
    SessionSnapshotter().export_session(src, path)

    dst = FakeHistory(**NAMESPACE)
    SessionSnapshotter().import_session(path, dst)
    assert len(dst.get_med_messages()) == 500
    # 校验和覆盖整段 payload，必须一致还原
    assert [m.content for m in dst.get_med_messages()] == [
        m.content for m in src.get_med_messages()
    ]


# ====================================================================== #
# 便捷函数
# ====================================================================== #
def test_convenience_functions(tmp_path):
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(2))
    path = tmp_path / "conv.bin"

    exp = snapshot_session(src, path)
    assert exp.message_count == 2

    dst = FakeHistory(**NAMESPACE)
    imp = restore_session(path, dst)
    assert imp.verified is True
    assert len(dst.get_med_messages()) == 2


def test_injected_serializer_is_used(tmp_path):
    # 用自定义序列化器可注入；验证其确实被调用（行为一致性由 serde 单测保证）
    class _DummySer(ProtobufSerializer):
        def __init__(self) -> None:
            self.calls = 0
            super().__init__()

        def serialize_snapshot(self, session, messages, schema_version="1"):
            self.calls += 1
            return super().serialize_snapshot(session, messages, schema_version)

    ser = _DummySer()
    src = FakeHistory(**NAMESPACE)
    src.add_med_messages(make_messages(1))
    path = tmp_path / "inj.bin"
    SessionSnapshotter(serializer=ser).export_session(src, path)
    assert ser.calls == 1


def test_checksum_covers_full_payload(tmp_path):
    # 直接验证：文件包尾部 32 字节为 sha256(magic..payload)
    pkg = SessionSnapshotPackage(schema_version="1", payload=b"abcdef")
    raw = pkg.to_bytes()
    body, digest = raw[:-32], raw[-32:]
    assert hashlib.sha256(body).digest() == digest
