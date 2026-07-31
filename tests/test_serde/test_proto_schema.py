"""校验 ``protos/med_session.proto`` 协议文件本身的规范性。

仅用标准库正则做结构断言，不依赖 protobuf 运行时，
保证协议文件字段/枚举与 ROADMAP 跨语言规范保持一致。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROTO_PATH = Path(__file__).resolve().parents[2] / "protos" / "med_session.proto"


@pytest.fixture(scope="module")
def proto_text() -> str:
    return PROTO_PATH.read_text(encoding="utf-8")


def _message_block(text: str, name: str) -> str:
    """提取指定 message 的定义块。"""
    match = re.search(rf"message {name} \{{(.*?)\n\}}", text, re.DOTALL)
    assert match is not None, f"message {name} not found"
    return match.group(1)


def _field_numbers(block: str) -> dict[str, int]:
    """解析 message 块中 字段名 -> 字段号 的映射。"""
    return {name: int(num) for name, num in re.findall(r"(\w+) = (\d+);", block)}


class TestProtoFile:
    def test_file_exists(self) -> None:
        assert PROTO_PATH.is_file()

    def test_syntax_and_package(self, proto_text: str) -> None:
        assert 'syntax = "proto3";' in proto_text
        assert "package med.session.v1;" in proto_text

    def test_med_message_fields_match_spec(self, proto_text: str) -> None:
        """MedMessage 字段名与字段号必须与 ROADMAP 规范逐一对齐。"""
        expected = {
            "message_id": 1,
            "session_id": 2,
            "tenant_id": 3,
            "dept_id": 4,
            "patient_id": 5,
            "role": 6,
            "content": 7,
            "token_count": 8,
            "masked": 9,
            "created_at": 10,
            "metadata": 11,
        }
        fields = _field_numbers(_message_block(proto_text, "MedMessage"))
        assert fields == expected

    def test_session_meta_fields(self, proto_text: str) -> None:
        fields = _field_numbers(_message_block(proto_text, "SessionMeta"))
        assert fields == {
            "session_id": 1,
            "tenant_id": 2,
            "dept_id": 3,
            "patient_id": 4,
            "status": 5,
            "message_count": 6,
            "created_at": 7,
            "updated_at": 8,
            "metadata": 9,
        }

    @pytest.mark.parametrize("name", ["MedMessageBatch", "SessionSnapshot"])
    def test_batch_and_snapshot_present(self, proto_text: str, name: str) -> None:
        assert _message_block(proto_text, name)

    def test_enums_have_unspecified_zero(self, proto_text: str) -> None:
        """proto3 枚举必须保留 0 号 UNSPECIFIED 值。"""
        assert "MESSAGE_ROLE_UNSPECIFIED = 0;" in proto_text
        assert "SESSION_STATUS_UNSPECIFIED = 0;" in proto_text

    def test_field_numbers_unique_within_each_message(self, proto_text: str) -> None:
        """边界：任一 message 内字段号不得重复。"""
        for name in ["MedMessage", "SessionMeta", "MedMessageBatch", "SessionSnapshot"]:
            numbers = list(_field_numbers(_message_block(proto_text, name)).values())
            assert len(numbers) == len(set(numbers)), f"duplicate field number in {name}"
