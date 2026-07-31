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
