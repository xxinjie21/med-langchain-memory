"""文档与代码一致性测试（D35 文档收尾迭代配套单测）。

文档一旦与代码脱节就会误导下游接入方，因此这里把「文档里写的」与「代码里真实存在的」
做机器可校验的对齐，全部用标准库（正则 / tomllib / pathlib）加少量运行时反射完成：

* ``docs/storage-spec.md`` ↔ ``protos/med_session.proto`` / ``domain`` / ``stores`` / ``privacy``
* ``README.md`` ↔ ``api`` 的 OpenAPI 契约 / ``StoreFactory`` 已注册后端 / ``pyproject.toml`` extras
* ``docs/architecture.md`` ↔ 分层目录结构 / 异常体系

每个公开校验方法均含正向用例与边界/异常用例（缺失标记、占位符、失效链接等）。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
README_PATH = PROJECT_ROOT / "README.md"
SPEC_PATH = DOCS_DIR / "storage-spec.md"
ARCH_PATH = DOCS_DIR / "architecture.md"
PROTO_PATH = PROJECT_ROOT / "protos" / "med_session.proto"

#: 文档中不允许出现的占位符标记（文档必须是已完成状态）。
PLACEHOLDER_PATTERN = re.compile(r"\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b|待补充|待填|占位符")

#: 统一存储键模板，与 ``MedMessage.storage_key`` 的实现必须一致。
STORAGE_KEY_TEMPLATE = "med:chat:{tenant_id}:{dept_id}:{session_id}"


@pytest.fixture(scope="module")
def readme_text() -> str:
    """README 全文。"""
    return README_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spec_text() -> str:
    """对外存储规范全文。"""
    return SPEC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def arch_text() -> str:
    """架构文档全文。"""
    return ARCH_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def proto_text() -> str:
    """protobuf 协议源文件全文。"""
    return PROTO_PATH.read_text(encoding="utf-8")


def _proto_block(text: str, name: str) -> str:
    """提取 proto 中指定 message 的定义块。"""
    match = re.search(rf"message {name} \{{(.*?)\n\}}", text, re.DOTALL)
    assert match is not None, f"message {name} not found in proto"
    return match.group(1)


def _proto_fields(text: str, name: str) -> dict[str, int]:
    """解析 proto message 的 字段名 -> 字段号 映射。"""
    return {n: int(num) for n, num in re.findall(r"(\w+) = (\d+);", _proto_block(text, name))}


def _table_row_for(text: str, value: str) -> str:
    """返回以 ``| `value` |`` 开头的表格行（大小写不敏感，不存在时断言失败）。

    文档为可读性使用枚举名（``ACTIVE``），而领域层 ``StrEnum`` 的取值是小写
    （``active``），故此处统一按小写比对。
    """
    needle = f"| `{value}` |".lower()
    for line in text.splitlines():
        if line.lower().startswith(needle):
            return line
    raise AssertionError(f"no table row found for {value!r}")


def _relative_links(text: str) -> list[str]:
    """提取 markdown 中的相对链接目标（忽略 http/https 与纯锚点）。"""
    return [
        target
        for target in re.findall(r"\]\((\./[^)\s]+)\)", text)
        if not target.startswith("http")
    ]


class TestDocsPresent:
    """文档存在性、体量与整洁度。"""

    @pytest.mark.parametrize(
        ("label", "path"),
        [("README", README_PATH), ("storage-spec", SPEC_PATH), ("architecture", ARCH_PATH)],
    )
    def test_doc_file_exists_and_is_substantial(self, label: str, path: Path) -> None:
        """正向：三份文档均存在且内容非空（>1500 字符）。"""
        assert path.is_file(), f"{label} missing at {path}"
        assert len(path.read_text(encoding="utf-8")) > 1500, f"{label} looks truncated"

    @pytest.mark.parametrize(
        "path", [README_PATH, SPEC_PATH, ARCH_PATH], ids=["readme", "spec", "arch"]
    )
    def test_no_placeholder_markers(self, path: Path) -> None:
        """边界：文档不得残留任何占位符标记。"""
        hits = PLACEHOLDER_PATTERN.findall(path.read_text(encoding="utf-8"))
        assert hits == [], f"{path.name} contains placeholders: {hits}"

    def test_placeholder_scanner_detects_marker(self) -> None:
        """边界：占位符扫描器本身有效（正对照，防止空扫描假通过）。"""
        assert PLACEHOLDER_PATTERN.search("this is TODO: finish later") is not None
        assert PLACEHOLDER_PATTERN.search("clean text without markers") is None

    def test_relative_links_resolve(self) -> None:
        """边界：文档中的相对链接必须指向真实存在的文件。"""
        for path in (README_PATH, SPEC_PATH, ARCH_PATH):
            for target in _relative_links(path.read_text(encoding="utf-8")):
                assert (path.parent / target).resolve().exists(), f"{path.name} -> {target} broken"

    def test_readme_links_to_both_docs(self, readme_text: str) -> None:
        """正向：README 必须给出存储规范与架构文档的入口。"""
        assert "docs/storage-spec.md" in readme_text
        assert "docs/architecture.md" in readme_text


class TestStorageSpecMatchesCode:
    """``docs/storage-spec.md`` 与 proto / 领域层 / 存储层的对齐。"""

    def test_proto_source_declared(self, spec_text: str, proto_text: str) -> None:
        """正向：规范声明的包名与语法与 proto 源文件一致。"""
        assert 'syntax = "proto3";' in proto_text
        assert "package med.session.v1;" in proto_text
        assert "med.session.v1" in spec_text

    def test_message_fields_documented_with_numbers(self, spec_text: str, proto_text: str) -> None:
        """正向：MedMessage 每个字段及其字段号都必须出现在规范字段表中。"""
        fields = _proto_fields(proto_text, "MedMessage")
        assert fields, "proto MedMessage block parsed empty"
        for name, number in fields.items():
            assert re.search(rf"\|\s*{number}\s*\|\s*`{name}`", spec_text), (
                f"MedMessage.{name} (#{number}) not documented"
            )

    def test_session_meta_fields_documented_with_numbers(
        self, spec_text: str, proto_text: str
    ) -> None:
        """正向：SessionMeta 每个字段及其字段号都必须出现在规范字段表中。"""
        fields = _proto_fields(proto_text, "SessionMeta")
        assert fields, "proto SessionMeta block parsed empty"
        for name, number in fields.items():
            assert re.search(rf"\|\s*{number}\s*\|\s*`{name}`", spec_text), (
                f"SessionMeta.{name} (#{number}) not documented"
            )

    def test_batch_and_snapshot_messages_documented(self, spec_text: str, proto_text: str) -> None:
        """边界：批量与快照消息也必须被文档覆盖。"""
        for name in ("MedMessageBatch", "SessionSnapshot"):
            assert _proto_block(proto_text, name)
            assert name in spec_text

    def test_enum_values_documented(self, spec_text: str) -> None:
        """正向：MessageRole / SessionStatus 的全部取值都必须写进规范。"""
        from med_langchain_memory.domain.message import MessageRole
        from med_langchain_memory.domain.session import SessionStatus

        for role in MessageRole:
            assert role.value in spec_text, f"role {role.value} not documented"
        for status in SessionStatus:
            assert status.value in spec_text, f"status {status.value} not documented"

    def test_state_machine_table_matches_domain_code(self, spec_text: str) -> None:
        """正向：规范的状态流转表必须与领域层状态机逐条一致。"""
        from med_langchain_memory.domain.session import _ALLOWED_TRANSITIONS

        for source, targets in _ALLOWED_TRANSITIONS.items():
            row = _table_row_for(spec_text, source.value)
            for target in targets:
                assert target.value in row.lower(), (
                    f"transition {source.value} -> {target.value} not documented"
                )

    def test_state_machine_terminal_state_documented(self, spec_text: str) -> None:
        """边界：终态（无出边）必须在规范中显式标注为终态。"""
        from med_langchain_memory.domain.session import _ALLOWED_TRANSITIONS, SessionStatus

        terminal = [s for s, targets in _ALLOWED_TRANSITIONS.items() if not targets]
        assert terminal == [SessionStatus.DELETED]
        assert "终态" in _table_row_for(spec_text, SessionStatus.DELETED.value)

    def test_storage_key_template_matches_code(self, spec_text: str) -> None:
        """正向：规范给出的统一存储键模板必须能由领域模型实例复现。"""
        from med_langchain_memory.domain.session import SessionMeta

        meta = SessionMeta(
            session_id="s-20261005-001",
            tenant_id="hospital-a",
            dept_id="cardiology",
            patient_id="p-0001",
        )
        expected = STORAGE_KEY_TEMPLATE.format(
            tenant_id=meta.tenant_id, dept_id=meta.dept_id, session_id=meta.session_id
        )
        assert meta.storage_key == expected
        assert STORAGE_KEY_TEMPLATE in spec_text

    def test_id_constraint_documented(self, spec_text: str) -> None:
        """正向：业务 ID 的字符集约束必须与领域层实现一致。"""
        from med_langchain_memory.domain.message import IdStr

        pattern = str(IdStr.__metadata__[0].pattern)  # type: ignore[attr-defined]
        assert pattern == r"^[A-Za-z0-9_.-]+$"
        assert pattern in spec_text
        assert "64" in spec_text

    def test_redis_key_suffixes_match_code(self, spec_text: str) -> None:
        """正向：Redis 键后缀常量必须出现在规范中。"""
        redis_store = pytest.importorskip("med_langchain_memory.stores.redis_store")
        assert redis_store.MESSAGES_SUFFIX in spec_text
        assert redis_store.META_SUFFIX in spec_text

    def test_mysql_sharding_documented(self, spec_text: str) -> None:
        """正向：分表数量与首尾表名必须与 mysql_schema 实现一致。"""
        schema = pytest.importorskip("med_langchain_memory.stores.mysql_schema")
        assert str(schema.SHARD_COUNT) in spec_text
        assert schema.message_table_name(0) in spec_text
        assert schema.message_table_name(schema.SHARD_COUNT - 1) in spec_text

    def test_mysql_sharding_boundary_rejects_out_of_range(self) -> None:
        """边界：分表编号越界必须被拒绝（文档描述的 16 张表是硬边界）。"""
        schema = pytest.importorskip("med_langchain_memory.stores.mysql_schema")
        from med_langchain_memory.exceptions import ValidationError

        with pytest.raises(ValidationError):
            schema.message_table_name(schema.SHARD_COUNT)

    def test_es_index_naming_documented(self, spec_text: str) -> None:
        """正向：ES 归档索引前缀与月度滚动格式必须与 es_store 实现一致。"""
        es_store = pytest.importorskip("med_langchain_memory.stores.es_store")
        assert es_store.ARCHIVE_INDEX_PREFIX in spec_text
        assert f"{es_store.ARCHIVE_INDEX_PREFIX}-{{yyyy.MM}}" in spec_text
        assert es_store.monthly_index(1_700_000_000_000).startswith(
            f"{es_store.ARCHIVE_INDEX_PREFIX}-"
        )

    def test_snapshot_package_format_matches_code(self, spec_text: str) -> None:
        """正向：快照文件包魔数与 schema 版本必须与 snapshot 实现一致。"""
        from med_langchain_memory.lifecycle.snapshot import (
            DEFAULT_SCHEMA_VERSION,
            SessionSnapshotPackage,
        )

        package = SessionSnapshotPackage(schema_version=DEFAULT_SCHEMA_VERSION, payload=b"")
        magic = package.to_bytes()[:8].decode("ascii")
        assert magic == "MEDSNAP1"
        assert magic in spec_text
        assert DEFAULT_SCHEMA_VERSION in spec_text

    def test_masking_rules_documented(self, spec_text: str) -> None:
        """正向：内置脱敏规则名必须全部写进规范。"""
        from med_langchain_memory.privacy.masker import BUILTIN_RULES

        assert BUILTIN_RULES, "builtin mask rules must not be empty"
        for rule in BUILTIN_RULES:
            assert f"`{rule.name}`" in spec_text, f"mask rule {rule.name} not documented"

    def test_forbidden_nlp_dependency_not_documented_as_feature(self, spec_text: str) -> None:
        """边界：规范必须声明零 NLP 依赖，不得把文本解析写成能力。"""
        assert "无文本解析" in spec_text or "零 NLP" in spec_text


class TestReadmeMatchesCode:
    """``README.md`` 与 API 契约 / 存储后端 / 打包元数据的对齐。"""

    def test_all_api_endpoints_documented(self, readme_text: str) -> None:
        """正向：OpenAPI 契约中的每个 方法+路径 都必须在 README 端点表中出现。"""
        from med_langchain_memory.api.app import create_app

        spec = create_app().openapi()
        paths = spec["paths"]
        assert paths, "openapi paths must not be empty"
        for path, operations in paths.items():
            for method in operations:
                assert f"{method.upper()} {path}" in readme_text, f"{method.upper()} {path} missing"

    def test_all_store_backends_documented(self, readme_text: str) -> None:
        """正向：工厂已注册的每个后端名都必须出现在 README。"""
        from med_langchain_memory.stores import StoreFactory

        backends = StoreFactory.available()
        assert backends, "no store backend registered"
        for backend in backends:
            assert backend in readme_text, f"backend {backend} not documented in README"

    def test_all_optional_extras_documented(self, readme_text: str) -> None:
        """正向：pyproject 声明的每个 extras 都必须在 README 安装段出现。"""
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        assert extras, "no optional dependency group declared"
        for extra in extras:
            assert f"[{extra}]" in readme_text, f"extra [{extra}] not documented in README"

    def test_ci_badge_present(self, readme_text: str) -> None:
        """正向：README 必须携带 CI 状态徽章。"""
        assert "actions/workflows/ci.yml/badge.svg" in readme_text

    def test_message_fields_listed(self, readme_text: str, spec_text: str) -> None:
        """正向：README 的消息字段清单必须覆盖规范中的全部 MedMessage 字段。"""
        proto_text = PROTO_PATH.read_text(encoding="utf-8")
        for name in _proto_fields(proto_text, "MedMessage"):
            assert name in readme_text, f"MedMessage.{name} not listed in README"
            assert name in spec_text

    def test_quick_start_import_path_is_real(self, readme_text: str) -> None:
        """边界：README 快速开始里的 StoreFactory 入口必须真实存在且可用。"""
        from med_langchain_memory.stores import StoreFactory

        assert "StoreFactory" in readme_text
        assert "memory" in StoreFactory.available()
        history = StoreFactory.create(
            "memory",
            session_id="s-readme-smoke",
            tenant_id="hospital-a",
            dept_id="cardiology",
            patient_id="p-0001",
        )
        assert history.storage_key == STORAGE_KEY_TEMPLATE.format(
            tenant_id="hospital-a", dept_id="cardiology", session_id="s-readme-smoke"
        )


class TestArchitectureDocMatchesCode:
    """``docs/architecture.md`` 与分层结构 / 异常体系的对齐。"""

    def test_all_layer_packages_documented(self, arch_text: str) -> None:
        """正向：src 下每个分层子包都必须在架构文档中出现。"""
        package_dir = PROJECT_ROOT / "src" / "med_langchain_memory"
        layers = sorted(
            p.name for p in package_dir.iterdir() if p.is_dir() and p.name != "__pycache__"
        )
        assert layers, "no layer package found"
        for layer in layers:
            assert f"{layer}/" in arch_text, f"layer {layer} not documented"

    def test_all_exception_types_documented(self, arch_text: str) -> None:
        """正向：异常体系中的每个业务异常都必须写进架构文档错误表。"""
        from med_langchain_memory import exceptions as exc_module

        base = exc_module.MedMemoryError
        names = sorted(
            name
            for name, obj in vars(exc_module).items()
            if isinstance(obj, type) and issubclass(obj, base) and obj is not base
        )
        assert names, "no exception subclass found"
        for name in names:
            assert name in arch_text, f"exception {name} not documented"

    def test_dependency_direction_declared(self, arch_text: str) -> None:
        """边界：架构文档必须声明单向依赖方向，避免循环依赖被文档化掩盖。"""
        assert "依赖方向" in arch_text
        assert "api" in arch_text and "domain" in arch_text
