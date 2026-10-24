"""``med_langchain_memory.testing.commitlint`` 的离线单测（D51 提交信息守门）。

全部用例只依赖标准库与合成文本：git 调用通过 monkeypatch 替换 ``subprocess.run``
覆盖，**不需要网络或真实 git 仓库**，因此这些用例在默认 ``pytest`` 下就会执行
（不带 ``integration`` 标记）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from med_langchain_memory.testing import commitlint
from med_langchain_memory.testing.commitlint import (
    ALLOWED_SCOPES,
    ALLOWED_TYPES,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VIOLATIONS,
    MAX_SUBJECT_LENGTH,
    LintFinding,
    LintReport,
    iter_commit_messages,
    lint_commits,
    lint_message,
    main,
    parse_git_log_messages,
    render_report,
)

#: 仓库根目录（子进程用例的 cwd 与 PYTHONPATH 基准）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 被守门的模块路径（``python -m`` 入口）。
COMMITLINT_MODULE = "med_langchain_memory.testing.commitlint"

#: 恰好 72 字符的合规主题行（边界：等于上限时通过）。
SUBJECT_AT_LIMIT = "feat: " + "x" * (MAX_SUBJECT_LENGTH - len("feat: "))


def _fake_git_log(stdout: str) -> object:
    """构造替身 ``subprocess.run``：固定返回给定 stdout。"""

    def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout)

    return _run


class TestLintMessage:
    """单条消息校验：每个公开方法至少一个正向 + 一个边界/异常用例。"""

    @pytest.mark.parametrize(
        "message",
        [
            "feat: add session ttl",
            "fix(stores): repair redis pipeline",
            "feat(runnable)!: drop legacy context builder",
            "chore: bump deps",
            "feat: add x\n\nbody line one\nbody line two",
            "feat: add x\n\n# please enter the commit message",
            "feat: add x\r\n\r\nwindows crlf body",
            SUBJECT_AT_LIMIT,
        ],
    )
    def test_positive_conforming_messages(self, message: str) -> None:
        """正向：规范消息（含 scope / 破坏标记 / 正文 / 注释行 / CRLF / 长度上限）零违规。"""
        report = lint_message(message)
        assert report.ok, report.findings
        assert report.commit is None

    def test_positive_commit_label_is_kept(self) -> None:
        """正向：``commit`` 标签原样进入报告，仅用于展示。"""
        report = lint_message("feat: add x", commit="abc1234")
        assert report.ok
        assert report.commit == "abc1234"

    def test_positive_finding_carries_code_and_message(self) -> None:
        """正向：``LintFinding`` 携带规则码与人类可读信息，报告按序收集。"""
        finding = LintFinding(code="type-invalid", message="unknown type 'wip'")
        assert finding.code == "type-invalid"
        assert finding.message == "unknown type 'wip'"
        report = LintReport(commit=None, subject="wip: x", findings=(finding,))
        assert not report.ok
        assert report.findings == (finding,)

    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("", "empty-message"),
            ("   \n\n  ", "empty-message"),
            ("# only a comment", "empty-message"),
            ("feat:add x", "subject-format"),
            ("feat add x", "subject-format"),
            ("Feat: add x", "subject-format"),
            ("feat(): add x", "subject-format"),
            ("feature: add x", "type-invalid"),
            ("feat(unknown): add x", "scope-invalid"),
            ("feat: ", "subject-format"),
            ("feat: add x ", "description-trailing-whitespace"),
            ("feat: add x.", "description-trailing-period"),
            ("feat: " + "x" * (MAX_SUBJECT_LENGTH - len("feat: ") + 1), "subject-too-long"),
            ("feat: add x\nbody without blank separator", "body-separator"),
        ],
    )
    def test_boundary_violations(self, message: str, code: str) -> None:
        """边界：每种违规规则各有一个反例，发现码精确匹配。"""
        report = lint_message(message)
        assert not report.ok
        assert [finding.code for finding in report.findings] == [code]

    def test_boundary_multiple_findings_in_one_message(self) -> None:
        """边界：一条消息可同时触发多条违规（type 与句点），全部收集不截断。"""
        report = lint_message("wip: ship it.")
        codes = {finding.code for finding in report.findings}
        assert codes == {"type-invalid", "description-trailing-period"}

    def test_boundary_allowed_sets_are_frozen_tuples(self) -> None:
        """边界：type / scope 白名单是不可变元组，防止运行期被意外改写。"""
        assert isinstance(ALLOWED_TYPES, tuple)
        assert isinstance(ALLOWED_SCOPES, tuple)
        assert "feat" in ALLOWED_TYPES
        assert "stores" in ALLOWED_SCOPES


class TestParseGitLogMessages:
    """git log 文本解析。"""

    def test_positive_records_with_and_without_body(self) -> None:
        """正向：有正文与无正文的记录都解析为 (哈希, 消息)，正文以空行接回主题。"""
        text = (
            "a" * 40
            + "\x1f"
            + "feat: add x"
            + "\x1f"
            + "body text\x1e"
            + "b" * 40
            + "\x1f"
            + "fix: repair y"
            + "\x1f"
            + "\x1e"
        )
        messages = parse_git_log_messages(text)
        assert messages == [
            ("a" * 40, "feat: add x\n\nbody text"),
            ("b" * 40, "fix: repair y"),
        ]

    def test_boundary_malformed_records_are_skipped(self) -> None:
        """边界：字段数不符、哈希非法、空正文分隔符形态（历史坑位）都不误判。"""
        text = (
            "onlyonfield\x1e"
            + "short\x1ffeat: x\x1fbody\x1e"
            + "c" * 40
            + "\x1ffeat: ok\x1f\x1e"
            + "d" * 40
            + "\x1f\x1f\x1e"
        )
        messages = parse_git_log_messages(text)
        assert messages == [("c" * 40, "feat: ok")]


class TestLintCommits:
    """版本范围校验：git 调用经替身覆盖。"""

    def test_positive_range_lints_every_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：范围内每条消息各出一份报告，commit 为 7 位短哈希。"""
        stdout = "a" * 40 + "\x1ffeat: add x\x1f\x1e" + "b" * 40 + "\x1fbad subject\x1f\x1e"
        monkeypatch.setattr(commitlint.subprocess, "run", _fake_git_log(stdout))
        reports = lint_commits("origin/main..HEAD")
        assert [report.commit for report in reports] == ["a" * 7, "b" * 7]
        assert reports[0].ok
        assert not reports[1].ok

    def test_boundary_git_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """边界：git 退出码非零时异常向上传播（由 CLI 转换为退出码 2）。"""

        def _boom(command: list[str], **kwargs: object) -> object:
            raise subprocess.CalledProcessError(128, command)

        monkeypatch.setattr(commitlint.subprocess, "run", _boom)
        with pytest.raises(subprocess.CalledProcessError):
            lint_commits("bad..range")

    def test_boundary_iter_passes_range_and_returns_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """边界：``iter_commit_messages`` 把范围原样透传给 git，并解析其输出。"""
        seen: list[list[str]] = []

        def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.append(command)
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="e" * 40 + "\x1ffeat: x\x1f\x1e"
            )

        monkeypatch.setattr(commitlint.subprocess, "run", _run)
        messages = iter_commit_messages("v0.1.0..HEAD")
        assert seen == [["git", "log", f"--format={commitlint.GIT_LOG_FORMAT}", "v0.1.0..HEAD"]]
        assert messages == [("e" * 40, "feat: x")]


class TestRenderReport:
    """报告渲染。"""

    def test_positive_all_conforming(self) -> None:
        """正向：全部合规时只输出汇总行，不出现 FAIL。"""
        text = render_report([lint_message("feat: add x"), lint_message("fix: repair y")])
        assert text == "2 commit message(s) conform to Conventional Commits"

    def test_positive_violations_are_listed(self) -> None:
        """正向：违规逐条列出（FAIL 行 + 缩进明细），汇总行给出违规计数。"""
        report = lint_message("nope", commit="abc1234")
        text = render_report([lint_message("feat: add x"), report])
        lines = text.splitlines()
        assert lines[0] == "FAIL abc1234 nope"
        assert lines[1] == '  - subject-format: subject must match "type(scope)!: description"'
        assert lines[-1] == "1/2 commit message(s) violate Conventional Commits"

    def test_boundary_empty_report_list(self) -> None:
        """边界：空输入列表渲染 0/0 合规汇总，不抛异常。"""
        assert render_report([]) == "0 commit message(s) conform to Conventional Commits"


class TestBuildParser:
    """CLI 参数解析器。"""

    def test_positive_defaults(self) -> None:
        """正向：位置参数范围可选，三个输入源默认均为空。"""
        args = commitlint.build_parser().parse_args([])
        assert args.range is None
        assert args.message is None
        assert args.message_file is None

    def test_positive_options_parse(self) -> None:
        """正向：``--message`` / ``--message-file`` / 位置范围都能解析。"""
        parser = commitlint.build_parser()
        assert parser.parse_args(["--message", "feat: x"]).message == "feat: x"
        assert parser.parse_args(["--message-file", "msg.txt"]).message_file == Path("msg.txt")
        assert parser.parse_args(["origin/main..HEAD"]).range == "origin/main..HEAD"


class TestMain:
    """CLI 入口（进程内调用；git 经 monkeypatch 替换）。"""

    def test_positive_single_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """正向：``--message`` 合规 → 退出码 0，stdout 给出合规汇总。"""
        code = main(["--message", "feat: add x"])
        assert code == EXIT_OK
        assert "1 commit message(s) conform" in capsys.readouterr().out

    def test_positive_message_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：``--message-file`` 读文件（含 git 注释行）→ 退出码 0。"""
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text("feat: add x\n\n# comment\n", encoding="utf-8")
        code = main(["--message-file", str(path)])
        assert code == EXIT_OK
        assert "conform" in capsys.readouterr().out

    def test_positive_message_file_dash_reads_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：``--message-file -`` 从 stdin 读取。"""

        class _Stdin:
            def read(self) -> str:
                return "fix: repair y"

        monkeypatch.setattr(commitlint.sys, "stdin", _Stdin())
        code = main(["--message-file", "-"])
        assert code == EXIT_OK
        assert "conform" in capsys.readouterr().out

    def test_boundary_violating_message_returns_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``--message`` 违规 → 退出码 1，stdout 含 FAIL 明细。"""
        code = main(["--message", "nope"])
        assert code == EXIT_VIOLATIONS
        out = capsys.readouterr().out
        assert "FAIL <message> nope" in out
        assert "1/1 commit message(s) violate" in out

    def test_boundary_no_source_returns_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """边界：未指定任何输入源 → 退出码 2，stderr 提示。"""
        code = main([])
        assert code == EXIT_USAGE
        assert "exactly one of" in capsys.readouterr().err

    def test_boundary_multiple_sources_return_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：同时指定 ``--message`` 与范围 → 退出码 2，stderr 提示。"""
        code = main(["--message", "feat: x", "origin/main..HEAD"])
        assert code == EXIT_USAGE
        assert "exactly one of" in capsys.readouterr().err

    def test_boundary_range_with_violations(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：范围内有违规 → 退出码 1。"""
        stdout = "f" * 40 + "\x1fbad subject\x1f\x1e"
        monkeypatch.setattr(commitlint.subprocess, "run", _fake_git_log(stdout))
        code = main(["origin/main..HEAD"])
        assert code == EXIT_VIOLATIONS
        assert "FAIL " + "f" * 7 in capsys.readouterr().out

    def test_boundary_range_git_failure_returns_usage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：git 调用失败 → 退出码 2，stderr 说明原因。"""

        def _boom(command: list[str], **kwargs: object) -> object:
            raise subprocess.CalledProcessError(128, command)

        monkeypatch.setattr(commitlint.subprocess, "run", _boom)
        code = main(["bad..range"])
        assert code == EXIT_USAGE
        assert "git log failed" in capsys.readouterr().err

    def test_boundary_unreadable_message_file_returns_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：消息文件不存在 → 退出码 2，stderr 说明原因。"""
        code = main(["--message-file", "no-such-file.txt"])
        assert code == EXIT_USAGE
        assert "cannot read message file" in capsys.readouterr().err


def test_subprocess_entry_point_conforming_message() -> None:
    """正向（子进程）：``python -m ...commitlint`` 真实入口对合规消息返回 0。"""
    result = subprocess.run(
        [sys.executable, "-m", COMMITLINT_MODULE, "--message", "feat: add x"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "conform" in result.stdout


def test_subprocess_entry_point_rejects_violation() -> None:
    """边界（子进程）：真实入口对违规消息返回退出码 1。"""
    result = subprocess.run(
        [sys.executable, "-m", COMMITLINT_MODULE, "--message", "nope"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == EXIT_VIOLATIONS, result.stderr
    assert "FAIL" in result.stdout
