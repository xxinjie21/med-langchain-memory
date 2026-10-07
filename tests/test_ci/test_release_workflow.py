"""``.github/workflows/release.yml`` 的规范测试。

发布流水线一旦配置错误，代价是「错误的产物被推上 PyPI」或「tag 打了却发不出
Release」，而这两件事都很难回滚。因此这里用纯标准库（正则 + ``build_parser``）
把工作流的关键契约钉死：

* 只由 ``v*`` tag 触发，绝不挂在分支推送或 PR 上；
* 构建前先跑版本守卫（tag ↔ pyproject ↔ ``__version__``），构建后再校验产物；
* 权限最小化：顶层只读，只有建 Release 的作业可写 ``contents``；
* 发布 PyPI 是**显式可选**步骤（仓库变量开关 + OIDC 可信发布，无 token secret）；
* 工作流里用到的命令行选项必须是 ``release.py`` 真实提供的选项；
* 所有 ``uses:`` 引用都锁定版本。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from med_langchain_memory.release import TAG_PREFIX, build_parser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"

#: 形如 ``uses: actions/checkout@v4`` 的动作引用。
USES_PATTERN = re.compile(r"uses:\s*(\S+)")

#: 允许的动作引用形态：``owner/repo@v<major>`` 或 ``owner/repo@release/v<major>``。
PINNED_ACTION_PATTERN = re.compile(r"[\w.-]+/[\w.-]+@(?:v\d+|release/v\d+)$")

#: 工作流中必须出现的 ``release.py`` 选项。
CLI_FLAGS = ("--tag", "--dist", "--expected-version", "--pyproject")


def read_workflow() -> str:
    """返回发布工作流的原始文本。

    Raises:
        AssertionError: 文件不存在或为空。
    """
    assert WORKFLOW_PATH.is_file(), f"missing workflow file: {WORKFLOW_PATH}"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert text.strip(), "workflow file must not be empty"
    return text


class TestWorkflowFile:
    def test_workflow_exists_and_readable(self) -> None:
        """正向：工作流文件存在且以 ``name:`` 开头。"""
        assert read_workflow().startswith("name:")

    def test_no_tab_indentation(self) -> None:
        """边界：YAML 不得使用制表符缩进。"""
        offending = [
            index + 1 for index, line in enumerate(read_workflow().splitlines()) if "\t" in line
        ]
        assert offending == [], f"tab characters found on lines: {offending}"

    def test_release_workflow_is_separate_from_ci_gate(self) -> None:
        """边界：发布流水线必须与必过 CI 门禁分离，避免拖慢日常反馈。"""
        assert CI_WORKFLOW_PATH.is_file()
        assert "release.yml" not in CI_WORKFLOW_PATH.read_text(encoding="utf-8")


class TestTriggers:
    def test_triggered_only_by_version_tags(self) -> None:
        """正向：仅 ``v*`` tag 触发，不挂分支推送与 PR。"""
        text = read_workflow()
        assert re.search(r"^on:", text, re.MULTILINE)
        assert re.search(r'^ {4}tags:\s*\["v\*"\]', text, re.MULTILINE)
        assert "pull_request:" not in text
        assert re.search(r"^\s{2}push:\s*\n\s{4}tags:", text, re.MULTILINE)

    def test_no_branch_trigger(self) -> None:
        """边界：不得出现 ``branches:``，否则分支推送会误发版本。"""
        assert "branches:" not in read_workflow()

    def test_concurrency_does_not_cancel_releases(self) -> None:
        """边界：发布过程不能被并发取消（``cancel-in-progress: false``）。"""
        text = read_workflow()
        assert "concurrency:" in text
        assert "cancel-in-progress: false" in text


class TestVerifyJob:
    def test_job_declares_verify_pipeline(self) -> None:
        """正向：存在 ``verify`` 作业，串起守卫 → 测试 → 构建 → 产物校验。"""
        text = read_workflow()
        assert re.search(r"^ {2}verify:", text, re.MULTILINE)
        assert "Check tag against package version" in text
        assert "Run test suite" in text
        assert "Build sdist and wheel" in text
        assert "Verify distribution artifacts" in text

    def test_version_guard_runs_before_build(self) -> None:
        """正向：版本守卫的步骤必须排在构建之前（否则白构建）。"""
        text = read_workflow()
        guard = text.index("Check tag against package version")
        build = text.index("Build sdist and wheel")
        assert guard < build

    def test_artifact_check_runs_after_build(self) -> None:
        """正向：产物校验必须排在构建之后。"""
        text = read_workflow()
        assert text.index("Build sdist and wheel") < text.index("Verify distribution artifacts")

    def test_uses_release_guard_cli_with_github_ref_name(self) -> None:
        """正向：守卫用 ``github.ref_name``（tag 名）驱动，而非硬编码版本。"""
        text = read_workflow()
        assert 'python -m med_langchain_memory.release --tag "${{ github.ref_name }}"' in text
        assert '--expected-version "${{ github.ref_name }}"' in text
        assert re.search(r"^\s+run:\s*pytest\s*$", text, re.MULTILINE)
        assert re.search(r"^\s+run:\s*python -m build\s*$", text, re.MULTILINE)

    def test_installs_dev_extras(self) -> None:
        """正向：测试依赖走 ``.[dev]``，与 ci.yml 保持一致。"""
        assert 'pip install -e ".[dev]"' in read_workflow()

    def test_uploads_distributions_for_downstream_jobs(self) -> None:
        """正向：产物上传为 artifact 供发布作业复用（不重复构建）。"""
        text = read_workflow()
        assert "actions/upload-artifact@v4" in text
        assert re.search(r"^\s+name:\s*distributions\s*$", text, re.MULTILINE)
        assert text.count("actions/download-artifact@v4") == 2


class TestLeastPrivilege:
    def test_top_level_permissions_are_read_only(self) -> None:
        """正向：顶层权限只读，写权限下放到具体作业。"""
        assert re.search(
            r"^permissions:\s*\n\s{2}contents:\s*read\s*$", read_workflow(), re.MULTILINE
        )

    def test_contents_write_only_for_release_job(self) -> None:
        """边界：``contents: write`` 只能出现一次（建 Release 的作业）。"""
        text = read_workflow()
        assert text.count("contents: write") == 1
        release_job = text.index("github-release:")
        assert text.index("contents: write") > release_job

    def test_id_token_write_only_for_pypi_job(self) -> None:
        """边界：OIDC ``id-token: write`` 只能出现在 PyPI 发布作业里。"""
        text = read_workflow()
        assert text.count("id-token: write") == 1
        assert text.index("id-token: write") > text.index("publish-pypi:")


class TestReleaseJobs:
    def test_github_release_creates_release_with_artifacts(self) -> None:
        """正向：用官方 ``gh`` 建 Release，附上 sdist/wheel 并自动生成说明。"""
        text = read_workflow()
        assert re.search(r"^ {2}github-release:", text, re.MULTILINE)
        assert "gh release create" in text
        assert "--verify-tag" in text
        assert "--generate-notes" in text
        assert "dist/*" in text
        assert "GH_TOKEN: ${{ github.token }}" in text

    def test_release_job_depends_on_verify(self) -> None:
        """正向：发布作业必须等 ``verify`` 通过。"""
        text = read_workflow()
        assert re.search(r"^ {4}needs:\s*verify\s*$", text, re.MULTILINE)

    def test_pypi_publish_is_opt_in_via_repository_variable(self) -> None:
        """正向：PyPI 发布默认关闭，需显式仓库变量开关。"""
        text = read_workflow()
        assert re.search(r"^ {2}publish-pypi:", text, re.MULTILINE)
        assert "if: vars.PUBLISH_TO_PYPI == 'true'" in text

    def test_pypi_publish_uses_trusted_publishing(self) -> None:
        """边界：可信发布（OIDC）不应依赖任何 token secret。"""
        text = read_workflow()
        assert "pypa/gh-action-pypi-publish@release/v1" in text
        assert "secrets." not in text
        assert "PYPI_TOKEN" not in text

    def test_pypi_publish_uses_protected_environment(self) -> None:
        """边界：发布作业应绑定受保护环境，便于人工审批。"""
        assert re.search(r"^ {4}environment:\s*pypi\s*$", read_workflow(), re.MULTILINE)


class TestActionPinning:
    def test_all_actions_pin_a_version(self) -> None:
        """边界：所有 ``uses:`` 引用都必须锁定版本。"""
        uses = USES_PATTERN.findall(read_workflow())
        assert uses, "workflow must reference at least one action"
        unpinned = [ref for ref in uses if PINNED_ACTION_PATTERN.search(ref) is None]
        assert unpinned == [], f"unpinned action references: {unpinned}"

    def test_pinning_scanner_rejects_floating_refs(self) -> None:
        """边界：锁版本扫描器本身有效（正对照，防止空扫描假通过）。"""
        assert PINNED_ACTION_PATTERN.search("actions/checkout@v4") is not None
        assert PINNED_ACTION_PATTERN.search("pypa/gh-action-pypi-publish@release/v1") is not None
        assert PINNED_ACTION_PATTERN.search("actions/checkout@main") is None


class TestConsistencyWithCode:
    def test_tag_prefix_matches_workflow_trigger(self) -> None:
        """正向：工作流的 tag 通配符由 ``release.py::TAG_PREFIX`` 决定。"""
        assert f'["{TAG_PREFIX}*"]' in read_workflow()

    @pytest.mark.parametrize("flag", CLI_FLAGS)
    def test_cli_flags_exist_in_release_module(self, flag: str) -> None:
        """正向：工作流引用的每个选项都必须是 ``release.py`` 真实提供的选项。"""
        parser = build_parser()
        # 参数值随便给一个：不抛 SystemExit 即说明该选项被识别。
        parser.parse_args([flag, "placeholder"])

    def test_workflow_only_uses_documented_cli_flags(self) -> None:
        """边界：守卫调用区段里的选项必须都在已知清单内。"""
        text = read_workflow()
        guard_region = text[
            text.index("Check tag against package version") : text.index("Upload distributions")
        ]
        used = set(re.findall(r"--[a-z][a-z-]+", guard_region))
        assert used, "workflow must call the release guard CLI"
        assert used <= set(CLI_FLAGS), f"unexpected flags: {used - set(CLI_FLAGS)}"

    def test_project_version_matches_package_version(self) -> None:
        """边界：发布所依赖的「版本单一事实源」在仓库内确实是自洽的。"""
        import med_langchain_memory

        text = PYPROJECT_PATH.read_text(encoding="utf-8")
        assert f'version = "{med_langchain_memory.__version__}"' in text


class TestDocumentation:
    def test_readme_documents_release_flow(self) -> None:
        """正向：README 必须给出 tag 发布入口、守卫命令与开关变量三要素。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert WORKFLOW_PATH.name in readme
        assert "python -m med_langchain_memory.release" in readme
        assert "PUBLISH_TO_PYPI" in readme

    def test_readme_documents_tag_convention(self) -> None:
        """正向：README 必须说明 tag 形态约定，避免打错 tag 才发现。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert f"git tag {TAG_PREFIX}" in readme
