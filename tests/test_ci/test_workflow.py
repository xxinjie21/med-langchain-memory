"""Specification tests for the GitHub Actions CI workflow.

These tests parse ``.github/workflows/ci.yml`` with stdlib-only tooling
(regular expressions) and assert the structural rules the project relies on:
trigger branches, job presence, Python version matrix alignment with
``pyproject.toml``, coverage gate, and pinned action versions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def read_workflow() -> str:
    """Return the raw text of the CI workflow file.

    Raises:
        AssertionError: If the workflow file does not exist or is empty.
    """
    assert WORKFLOW_PATH.is_file(), f"missing workflow file: {WORKFLOW_PATH}"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert text.strip(), "workflow file must not be empty"
    return text


def extract_matrix_versions(workflow_text: str) -> list[str]:
    """Extract the Python version matrix entries from the workflow text.

    Args:
        workflow_text: Raw YAML text of the workflow.

    Returns:
        The list of quoted version strings declared under ``python-version``.
    """
    match = re.search(r"python-version:\s*\[([^\]]+)\]", workflow_text)
    if match is None:
        return []
    return re.findall(r"\"([\d.]+)\"", match.group(1))


class TestWorkflowFile:
    def test_workflow_exists_and_readable(self) -> None:
        text = read_workflow()
        assert text.startswith("name:")

    def test_no_tab_indentation(self) -> None:
        """YAML must never be indented with tab characters."""
        text = read_workflow()
        offending = [i + 1 for i, line in enumerate(text.splitlines()) if "\t" in line]
        assert offending == [], f"tab characters found on lines: {offending}"


class TestTriggers:
    def test_triggers_on_push_and_pull_request_to_main(self) -> None:
        text = read_workflow()
        assert re.search(r"^on:", text, re.MULTILINE)
        assert re.search(r"push:\s*\n\s*branches:\s*\[main\]", text)
        assert re.search(r"pull_request:\s*\n\s*branches:\s*\[main\]", text)

    def test_concurrency_cancels_in_progress_runs(self) -> None:
        text = read_workflow()
        assert "concurrency:" in text
        assert "cancel-in-progress: true" in text


class TestJobs:
    def test_lint_job_runs_ruff_and_mypy(self) -> None:
        text = read_workflow()
        assert re.search(r"^\s{2}lint:", text, re.MULTILINE)
        assert "ruff check ." in text
        assert re.search(r"^\s*run: mypy\s*$", text, re.MULTILINE)

    def test_test_job_runs_pytest_with_coverage_gate(self) -> None:
        text = read_workflow()
        assert re.search(r"^\s{2}test:", text, re.MULTILINE)
        assert "--cov=med_langchain_memory" in text
        assert re.search(r"--cov-fail-under=\d+", text)

    def test_matrix_does_not_fail_fast(self) -> None:
        """One failing Python version must not cancel the other matrix legs."""
        text = read_workflow()
        assert "fail-fast: false" in text


class TestPythonVersionMatrix:
    def test_matrix_covers_supported_versions(self) -> None:
        versions = extract_matrix_versions(read_workflow())
        assert versions == ["3.11", "3.12", "3.13"]

    def test_matrix_aligned_with_pyproject_classifiers(self) -> None:
        """Every matrix version must be declared as a trove classifier."""
        versions = extract_matrix_versions(read_workflow())
        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
        for version in versions:
            classifier = f"Programming Language :: Python :: {version}"
            assert classifier in pyproject, f"missing classifier for {version}"

    def test_matrix_minimum_matches_requires_python(self) -> None:
        versions = extract_matrix_versions(read_workflow())
        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
        match = re.search(r"requires-python\s*=\s*\">=([\d.]+)\"", pyproject)
        assert match is not None, "requires-python missing from pyproject.toml"
        assert versions, "workflow matrix must not be empty"
        assert versions[0] == match.group(1)


class TestActionPinning:
    def test_all_actions_pin_a_major_version(self) -> None:
        """Every ``uses:`` reference must pin at least a major version tag."""
        text = read_workflow()
        uses = re.findall(r"uses:\s*(\S+)", text)
        assert uses, "workflow must reference at least one action"
        unpinned = [ref for ref in uses if not re.search(r"@v\d+$", ref)]
        assert unpinned == [], f"actions without pinned major version: {unpinned}"

    @pytest.mark.parametrize(
        "action",
        ["actions/checkout@v4", "actions/setup-python@v5"],
    )
    def test_expected_core_actions_present(self, action: str) -> None:
        assert action in read_workflow()


class TestCommitLintJob:
    """``commit-lint`` 作业契约（D51 提交信息守门）。"""

    def test_job_only_runs_on_pull_requests(self) -> None:
        """PR 之外的推送不重 lint 已落地历史，作业挂在 ``pull_request`` 条件上。"""
        text = read_workflow()
        assert re.search(r"^\s{2}commit-lint:", text, re.MULTILINE)
        assert "if: github.event_name == 'pull_request'" in text

    def test_checkout_fetches_full_history_for_the_range(self) -> None:
        """范围 ``origin/<base>..HEAD`` 需要完整历史，checkout 必须 ``fetch-depth: 0``。"""
        text = read_workflow()
        assert re.search(r"^\s{2}commit-lint:", text, re.MULTILINE)
        assert "fetch-depth: 0" in text

    def test_job_installs_package_without_dependencies(self) -> None:
        """守门是纯标准库工具，装包不拉第三方依赖（``--no-deps``）。"""
        assert "pip install -e . --no-deps" in read_workflow()

    def test_job_runs_the_real_cli_entry(self) -> None:
        """运行命令必须是真实存在的 ``python -m`` 入口与 PR 范围表达式。"""
        text = read_workflow()
        assert "python -m med_langchain_memory.testing.commitlint" in text
        assert "origin/${{ github.base_ref }}..HEAD" in text

    def test_cli_options_exist_for_the_documented_usage(self) -> None:
        """交叉校验：工作流用到的模块与选项在 CLI 解析器里真实存在。"""
        from med_langchain_memory.testing import commitlint

        parser = commitlint.build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        assert "--message" in options
        assert "--message-file" in options
        assert commitlint.__name__ == "med_langchain_memory.testing.commitlint"

    def test_pre_commit_hook_wired_at_commit_msg_stage(self) -> None:
        """pre-commit 的 ``commit-msg`` 钩子与 CI 作业共用同一个入口。"""
        config_path = REPO_ROOT / ".pre-commit-config.yaml"
        assert config_path.is_file(), "missing .pre-commit-config.yaml"
        config = config_path.read_text(encoding="utf-8")
        assert "stages: [commit-msg]" in config
        assert "python -m med_langchain_memory.testing.commitlint --message-file" in config
