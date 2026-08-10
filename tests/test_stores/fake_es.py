"""内存版 Elasticsearch 客户端替身，供 ES 归档存储单测使用。

只实现被 :mod:`med_langchain_memory.stores.es_store` 真正调用的 API 子集：
``bulk`` / ``search`` / ``count`` / ``delete_by_query`` 以及
``indices.put_index_template`` / ``indices.exists_index_template``。

查询侧支持 ``bool.filter`` 中的 ``term`` 与 ``range`` 子句（生产代码只用这两种），
遇到未支持的子句直接抛 ``AssertionError``，避免测试静默放过错误的查询体。

同时记录 bulk 分块大小、refresh 取值与最近一次 search 参数，便于断言写入策略；
``fail_on`` 可对指定动作注入异常，用于校验异常包装。

注意：本模块文件名不以 ``test_`` 开头，pytest 不会收集；
同目录测试模块通过 pytest 自动注入的 sys.path 直接 ``from fake_es import ...`` 引用。
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any


class FakeIndicesClient:
    """``client.indices`` 命名空间替身，仅覆盖索引模板 API。"""

    def __init__(self, owner: FakeElasticsearch) -> None:
        """绑定所属的假客户端。"""
        self._owner = owner

    def exists_index_template(self, *, name: str) -> bool:
        """判断索引模板是否已注册。"""
        self._owner.maybe_fail("exists_index_template")
        self._owner.calls.append(("exists_index_template", {"name": name}))
        return name in self._owner.templates

    def put_index_template(self, *, name: str, **body: Any) -> dict[str, bool]:
        """写入（覆盖）索引模板。"""
        self._owner.maybe_fail("put_index_template")
        self._owner.calls.append(("put_index_template", {"name": name, **body}))
        self._owner.templates[name] = dict(body)
        return {"acknowledged": True}


class FakeElasticsearch:
    """极简内存 ES 替身：``{index: {doc_id: source}}`` 三层字典存储。"""

    def __init__(self) -> None:
        """初始化空存储与调用记录。"""
        self.documents: dict[str, dict[str, dict[str, Any]]] = {}
        self.templates: dict[str, dict[str, Any]] = {}
        self.indices = FakeIndicesClient(self)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.bulk_batches: list[int] = []
        self.refresh_flags: list[Any] = []
        self.last_search: dict[str, Any] = {}
        self.rejected_ids: set[str] = set()
        self._failures: dict[str, Exception] = {}

    # ------------------------------------------------------------------ #
    # 测试控制接口
    # ------------------------------------------------------------------ #
    def fail_on(self, action: str, error: Exception) -> None:
        """让指定动作抛出给定异常（``bulk`` / ``search`` / ``count`` / ...）。"""
        self._failures[action] = error

    def maybe_fail(self, action: str) -> None:
        """若该动作被注入了异常则抛出。"""
        error = self._failures.get(action)
        if error is not None:
            raise error

    def all_sources(self) -> list[dict[str, Any]]:
        """返回全部索引中的文档源，用于断言落库结果。"""
        return [source for docs in self.documents.values() for source in docs.values()]

    # ------------------------------------------------------------------ #
    # ES API 子集
    # ------------------------------------------------------------------ #
    def bulk(self, *, operations: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        """执行 ``index`` 动作的批量写入（动作行 + 文档行成对出现）。"""
        self.maybe_fail("bulk")
        self.refresh_flags.append(refresh)
        self.bulk_batches.append(len(operations) // 2)
        items: list[dict[str, Any]] = []
        errors = False
        for action, document in zip(operations[::2], operations[1::2], strict=True):
            meta = action["index"]
            doc_id = str(meta["_id"])
            index = str(meta["_index"])
            if doc_id in self.rejected_ids:
                errors = True
                items.append({"index": {"_id": doc_id, "error": {"reason": "rejected"}}})
                continue
            self.documents.setdefault(index, {})[doc_id] = dict(document)
            items.append({"index": {"_id": doc_id, "result": "created", "status": 201}})
        return {"errors": errors, "items": items, "took": 1}

    def search(
        self,
        *,
        index: str,
        query: dict[str, Any] | None = None,
        size: int = 10,
        sort: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """按 ``bool.filter`` 过滤 + 多字段排序 + ``size`` 截断返回命中文档。"""
        self.maybe_fail("search")
        self.last_search = {"index": index, "query": query, "size": size, "sort": sort, **kwargs}
        hits = [
            {"_index": idx, "_id": doc_id, "_source": dict(source)}
            for idx, doc_id, source in self._matching(index, query)
        ]
        for spec in reversed(sort or []):
            field, options = next(iter(spec.items()))
            reverse = str(options.get("order", "asc")) == "desc"
            hits.sort(key=lambda hit, key=field: hit["_source"].get(key), reverse=reverse)
        hits = hits[:size]
        return {"hits": {"total": {"value": len(hits), "relation": "eq"}, "hits": hits}}

    def count(
        self, *, index: str, query: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """统计匹配文档数。"""
        self.maybe_fail("count")
        return {"count": len(self._matching(index, query))}

    def delete_by_query(
        self,
        *,
        index: str,
        query: dict[str, Any] | None = None,
        refresh: Any = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """删除全部匹配文档。"""
        self.maybe_fail("delete_by_query")
        self.refresh_flags.append(refresh)
        matched = self._matching(index, query)
        for idx, doc_id, _source in matched:
            self.documents[idx].pop(doc_id, None)
        return {"deleted": len(matched), "failures": []}

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _matching(
        self, index: str, query: dict[str, Any] | None
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """返回匹配索引模式与查询条件的 ``(index, doc_id, source)`` 三元组。"""
        matched: list[tuple[str, str, dict[str, Any]]] = []
        for idx, docs in self.documents.items():
            if not fnmatch(idx, index):
                continue
            matched.extend(
                (idx, doc_id, source) for doc_id, source in docs.items() if _matches(source, query)
            )
        return matched


def _matches(source: dict[str, Any], query: dict[str, Any] | None) -> bool:
    """判断单个文档是否满足 ``bool.filter`` 中的全部子句。"""
    if not query:
        return True
    clauses = query.get("bool", {}).get("filter", [])
    return all(_matches_clause(source, clause) for clause in clauses)


def _matches_clause(source: dict[str, Any], clause: dict[str, Any]) -> bool:
    """判断单个 ``term`` / ``range`` 子句是否命中。"""
    if "term" in clause:
        field, expected = next(iter(clause["term"].items()))
        return bool(source.get(field) == expected)
    if "range" in clause:
        field, bounds = next(iter(clause["range"].items()))
        value = source.get(field)
        if value is None:
            return False
        if "gte" in bounds and value < bounds["gte"]:
            return False
        return not ("lte" in bounds and value > bounds["lte"])
    raise AssertionError(f"unsupported query clause in fake elasticsearch: {clause}")
