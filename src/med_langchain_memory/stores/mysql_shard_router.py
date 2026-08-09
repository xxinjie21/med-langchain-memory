"""MySQL 分表路由器：基于 crc32 的一致性 hash 分表路由。

按 ROADMAP《对外存储规范》``med_message_{crc32(session_id) % 16}``，
将 ``session_id`` 一致性地映射到 16 张消息分表之一，
保证同一会话的全部消息始终落在同一张分表上，实现物理局部性。

路由函数使用标准库 :func:`zlib.crc32`，返回无符号 32 位整数，
取模后得到 ``[0, SHARD_COUNT)`` 范围的分片编号。

本模块只做路由计算与表对象查找，不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

from sqlalchemy import Table

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import ValidationError

from .mysql_schema import SHARD_COUNT, message_table, message_table_name


class ShardRouter:
    """``session_id`` → 消息分表的一致性 hash 路由器。

    使用 ``crc32(session_id) % shard_count`` 计算分片编号。
    同一 ``session_id`` 永远路由到同一张分表，保证会话内消息的物理局部性。

    Args:
        shard_count: 分表数量，取值范围 ``[1, SHARD_COUNT]``，
            缺省为 :data:`~med_langchain_memory.stores.mysql_schema.SHARD_COUNT`（16）。
            仅允许缩小范围以便单元测试，生产环境始终使用 16 张表。

    Raises:
        ValidationError: ``shard_count`` 不在 ``[1, SHARD_COUNT]`` 范围内时。
    """

    def __init__(self, shard_count: int = SHARD_COUNT) -> None:
        if not 1 <= shard_count <= SHARD_COUNT:
            raise ValidationError(f"shard_count must be in [1, {SHARD_COUNT}], got {shard_count}")
        self._shard_count = shard_count

    @property
    def shard_count(self) -> int:
        """当前路由器使用的分表数量。"""
        return self._shard_count

    def shard_of(self, session_id: str) -> int:
        """返回 ``session_id`` 对应的分片编号。

        Args:
            session_id: 会话 ID。

        Returns:
            分片编号，取值范围 ``[0, shard_count)``。

        Raises:
            ValidationError: ``session_id`` 为空字符串时。
        """
        if not session_id:
            raise ValidationError("session_id must not be empty")
        return zlib.crc32(session_id.encode("utf-8")) % self._shard_count

    def table_name_of(self, session_id: str) -> str:
        """返回 ``session_id`` 应写入的分表名。

        等价于 ``message_table_name(self.shard_of(session_id))``。

        Raises:
            ValidationError: ``session_id`` 为空字符串时。
        """
        return message_table_name(self.shard_of(session_id))

    def table_of(self, session_id: str) -> Table:
        """返回 ``session_id`` 应写入的 SQLAlchemy ``Table`` 对象。

        Raises:
            ValidationError: ``session_id`` 为空字符串时。
        """
        return message_table(self.shard_of(session_id))

    def all_shards(self) -> list[int]:
        """返回全部分片编号列表（``[0, shard_count)``）。"""
        return list(range(self._shard_count))

    def all_tables(self) -> list[Table]:
        """返回路由范围内的全部消息分表，按分片编号升序。

        跨会话的全表扫描（如归档导出、迁移）需要依次访问每张分表，
        本方法给出稳定顺序的表清单。

        Returns:
            长度为 ``shard_count`` 的 ``Table`` 列表。
        """
        return [message_table(shard) for shard in range(self._shard_count)]

    def group_by_shard(self, session_ids: Iterable[str]) -> dict[int, list[str]]:
        """将多个会话 ID 按分片归组，用于批量查询的扇出。

        同一分片内的会话可合并为一条 ``WHERE session_id IN (...)`` 查询，
        把 N 次单表查询压缩为最多 ``shard_count`` 次。
        重复的会话 ID 只保留一次，组内顺序与首次出现顺序一致；
        返回字典按分片编号升序排列，且不含空分组。

        Args:
            session_ids: 待归组的会话 ID 序列。

        Returns:
            ``{分片编号: [会话ID, ...]}``，仅包含实际命中的分片。

        Raises:
            ValidationError: 任一会话 ID 为空字符串时。
        """
        grouped: dict[int, list[str]] = {}
        seen: set[str] = set()
        for session_id in session_ids:
            if session_id in seen:
                continue
            seen.add(session_id)
            grouped.setdefault(self.shard_of(session_id), []).append(session_id)
        return {shard: grouped[shard] for shard in sorted(grouped)}

    def group_messages(self, messages: Iterable[MedMessage]) -> dict[int, list[MedMessage]]:
        """将多条消息按其会话所属分片归组，用于批量写入。

        每个分组可整体交给一次 ``executemany`` 插入对应分表，
        组内顺序与输入顺序一致；返回字典按分片编号升序排列，且不含空分组。

        Args:
            messages: 待落库的医疗消息序列。

        Returns:
            ``{分片编号: [消息, ...]}``，仅包含实际命中的分片。

        Raises:
            ValidationError: 任一消息的 ``session_id`` 为空字符串时。
        """
        grouped: dict[int, list[MedMessage]] = {}
        for message in messages:
            grouped.setdefault(self.shard_of(message.session_id), []).append(message)
        return {shard: grouped[shard] for shard in sorted(grouped)}


#: 进程内默认路由器（16 张分表），生产代码直接复用，避免重复构造。
DEFAULT_ROUTER = ShardRouter()
