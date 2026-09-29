"""消息游标分页工具单元测试（D32）。

覆盖 :mod:`med_langchain_memory.api.cursor` 的全部公开函数：

* :func:`encode_cursor` / :func:`decode_cursor` —— 往返一致、URL 安全无填充、
  以及空串 / 非 ASCII / 非法 base64 / 缺分隔符 / 时间戳非数字 / ID 非 UUID / 非正时间戳
  等畸形游标；
* :func:`paginate` —— 排序、翻页、末页判定、游标重放、游标越界，以及 ``limit`` 非正数的异常路径。
"""

from __future__ import annotations

import base64
import uuid

import pytest
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.api.cursor import (
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_MESSAGE_PAGE_SIZE,
    MessageCursor,
    decode_cursor,
    encode_cursor,
    paginate,
)
from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.exceptions import ValidationError


def _message(index: int, *, created_at: int | None = None) -> MedMessage:
    """构造时序可控的测试消息。"""
    return MedMessage(
        session_id="s-1",
        tenant_id="hosp-a",
        dept_id="cardio",
        patient_id="p-1",
        role=MessageRole.PATIENT,
        content=f"msg-{index}",
        created_at=created_at if created_at is not None else 1000 + index,
    )


def _token(raw: str) -> str:
    """把原始字符串编码为不带填充的 base64url 游标（用于构造畸形输入）。"""
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- #
# 常量与游标模型
# --------------------------------------------------------------------------- #
def test_page_size_constants_are_sane() -> None:
    """默认页大小不超过上限，且均为正数。"""
    assert 0 < DEFAULT_MESSAGE_PAGE_SIZE <= MAX_MESSAGE_PAGE_SIZE


def test_cursor_of_message_uses_sort_key() -> None:
    """``MessageCursor.of`` 抽取消息的时序定位键。"""
    message = _message(0, created_at=4242)
    cursor = MessageCursor.of(message)
    assert cursor.created_at == 4242
    assert cursor.message_id == message.message_id
    assert cursor.sort_key == (4242, message.message_id)


def test_cursor_is_frozen() -> None:
    """游标模型不可变（边界：赋值被拒绝）。"""
    cursor = MessageCursor(created_at=1, message_id=str(uuid.uuid4()))
    with pytest.raises(PydanticValidationError):
        cursor.created_at = 2  # type: ignore[misc]


def test_cursor_forbids_extra_field() -> None:
    """游标模型拒绝未声明字段（边界：脏输入）。"""
    with pytest.raises(PydanticValidationError):
        MessageCursor(created_at=1, message_id=str(uuid.uuid4()), extra=1)  # type: ignore[call-arg]


def test_cursor_rejects_non_positive_created_at() -> None:
    """游标时间戳必须为正（边界：0 或负数）。"""
    with pytest.raises(PydanticValidationError):
        MessageCursor(created_at=0, message_id=str(uuid.uuid4()))


# --------------------------------------------------------------------------- #
# 编解码
# --------------------------------------------------------------------------- #
def test_encode_decode_round_trip() -> None:
    """编码后再解码得到等价游标。"""
    cursor = MessageCursor(created_at=1762560000000, message_id=str(uuid.uuid4()))
    assert decode_cursor(encode_cursor(cursor)) == cursor


def test_encoded_cursor_is_urlsafe_without_padding() -> None:
    """编码结果为 URL 安全字符集且不含 ``=`` 填充（可直接放查询参数）。"""
    token = encode_cursor(MessageCursor(created_at=1, message_id=str(uuid.uuid4())))
    assert "=" not in token
    assert all(char.isalnum() or char in "-_" for char in token)


def test_encode_is_deterministic() -> None:
    """同一游标重复编码结果稳定（可安全重放）。"""
    cursor = MessageCursor(created_at=99, message_id=str(uuid.uuid4()))
    assert encode_cursor(cursor) == encode_cursor(cursor)


@pytest.mark.parametrize(
    "token",
    [
        "",
        "%%%",
        "====",
        "é",
        _token("notanumber:" + str(uuid.uuid4())),
        _token("1000:not-a-uuid"),
        _token("1000"),
        _token("0:" + str(uuid.uuid4())),
        _token("-5:" + str(uuid.uuid4())),
        _token("1000:"),
    ],
)
def test_decode_rejects_malformed_cursor(token: str) -> None:
    """空串 / 非法 base64 / 缺分隔符 / 时间戳非法 / ID 非 UUID / 非正时间戳 → 领域校验异常。"""
    with pytest.raises(ValidationError, match="cursor"):
        decode_cursor(token)


# --------------------------------------------------------------------------- #
# 分页
# --------------------------------------------------------------------------- #
def test_paginate_empty_input_returns_no_cursor() -> None:
    """空输入返回空页且无下一页游标（边界：空会话）。"""
    page, next_cursor = paginate([], limit=10)
    assert page == []
    assert next_cursor is None


def test_paginate_sorts_messages_by_time() -> None:
    """输入乱序时按 ``(created_at, message_id)`` 升序输出。"""
    messages = [
        _message(2, created_at=3000),
        _message(0, created_at=1000),
        _message(1, created_at=2000),
    ]
    page, next_cursor = paginate(messages, limit=10)
    assert [message.content for message in page] == ["msg-0", "msg-1", "msg-2"]
    assert next_cursor is None


def test_paginate_first_page_returns_cursor_when_more() -> None:
    """存在剩余消息时返回指向本页末条的游标。"""
    messages = [_message(index, created_at=1000 + index) for index in range(3)]
    page, next_cursor = paginate(messages, limit=2)
    assert [message.content for message in page] == ["msg-0", "msg-1"]
    assert next_cursor is not None
    assert next_cursor == MessageCursor.of(messages[1])


def test_paginate_exact_page_size_has_no_cursor() -> None:
    """恰好取满且无剩余时 ``next_cursor`` 为 ``None``（边界：末页判定）。"""
    messages = [_message(index, created_at=1000 + index) for index in range(2)]
    page, next_cursor = paginate(messages, limit=2)
    assert len(page) == 2
    assert next_cursor is None


def test_paginate_with_cursor_skips_consumed_messages() -> None:
    """携带游标时只返回严格晚于游标定位键的消息。"""
    messages = [_message(index, created_at=1000 + index) for index in range(4)]
    first, cursor = paginate(messages, limit=2)
    assert cursor is not None
    second, last = paginate(messages, limit=2, cursor=cursor)
    assert [message.content for message in second] == ["msg-2", "msg-3"]
    assert last is None
    assert {message.message_id for message in first}.isdisjoint(
        message.message_id for message in second
    )


def test_paginate_cursor_at_last_message_returns_empty() -> None:
    """游标指向末条时返回空页（边界：重复消费同一游标）。"""
    messages = [_message(index, created_at=1000 + index) for index in range(3)]
    page, next_cursor = paginate(messages, limit=3, cursor=MessageCursor.of(messages[-1]))
    assert page == []
    assert next_cursor is None


def test_paginate_walks_all_pages_without_duplicates() -> None:
    """逐页遍历能覆盖全部消息且不重复、不遗漏（分页闭环）。"""
    messages = [_message(index, created_at=1000 + index) for index in range(5)]
    seen: list[str] = []
    cursor: MessageCursor | None = None
    while True:
        page, cursor = paginate(messages, limit=2, cursor=cursor)
        seen.extend(message.content for message in page)
        if cursor is None:
            break
    assert seen == [f"msg-{index}" for index in range(5)]


@pytest.mark.parametrize("limit", [0, -1])
def test_paginate_rejects_non_positive_limit(limit: int) -> None:
    """``limit`` 非正整数 → 领域校验异常（边界：非法分页参数）。"""
    with pytest.raises(ValidationError, match="limit"):
        paginate([_message(0)], limit=limit)


def test_paginate_requires_keyword_limit() -> None:
    """``limit`` 为关键字参数，位置传参即报错（对外契约稳定）。"""
    with pytest.raises(TypeError):
        paginate([_message(0)], 1)  # type: ignore[misc]
