"""``med_langchain_memory.changelog`` 变更日志生成器单测。

全部离线：git log 文本为合成字符串，``collect_commits`` 的 ``subprocess.run``
用 monkeypatch 替身替换；每个公开函数均含正向用例与边界 / 异常用例，
另保留两条子进程用例验证 ``python -m`` 真实入口与 git 失败退出码。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from med_langchain_memory.changelog import (
    BREAKING_SECTION_TITLE,
    GIT_LOG_FORMAT,
    SHORT_HASH_LENGTH,
    CommitEntry,
    build_parser,
    collect_commits,
    group_by_type,
    main,
    parse_commit,
    parse_git_log,
    render_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 记录字段分隔符（与 GIT_LOG_FORMAT 对应）。
FIELD_SEP = "\x1f"
#: 记录分隔符。
RECORD_SEP = "\x1e"

HASH_A = "a1b2c3d4e5f6" + "0" * 28
HASH_B = "ff00ff00ff00" + "1" * 28
HASH_C = "0" * 12 + "deadbeef" + "2" * 20


def make_record(commit_hash: str, subject: str, body: str = "") -> str:
    """按 git log 记录格式拼一条记录（含尾部记录分隔符）。"""
    return f"{commit_hash}{FIELD_SEP}{subject}{FIELD_SEP}{body}{RECORD_SEP}"


def entry(
    commit_hash: str = HASH_A,
    commit_type: str = "feat",
    scope: str | None = "stores",
    description: str = "add redis store",
    breaking: bool = False,
) -> CommitEntry:
    """构造测试用提交条目。"""
    return CommitEntry(
        hash=commit_hash,
        type=commit_type,
        scope=scope,
        description=description,
        breaking=breaking,
    )


class TestParseCommit:
    """``parse_commit``：单条记录解析。"""

    def test_parses_conventional_commit_with_scope(self) -> None:
        """正向：带 scope 的规范提交被完整解析。"""
        parsed = parse_commit(make_record(HASH_A, "feat(stores): add redis store"))
        assert parsed == entry()

    def test_parses_conventional_commit_without_scope(self) -> None:
        """正向：无 scope 提交的 scope 为 ``None``。"""
        parsed = parse_commit(make_record(HASH_B, "fix: correct ttl renewal"))
        assert parsed is not None
        assert parsed.scope is None
        assert parsed.type == "fix"
        assert parsed.description == "correct ttl renewal"

    def test_type_is_lowercased(self) -> None:
        """边界：type 大小写归一为小写。"""
        parsed = parse_commit(make_record(HASH_A, "Feat(API): add endpoint"))
        assert parsed is not None
        assert parsed.type == "feat"

    def test_bang_marks_breaking(self) -> None:
        """正向：主题行 ``!`` 标记为破坏性变更。"""
        parsed = parse_commit(make_record(HASH_A, "feat!: drop python 3.9"))
        assert parsed is not None
        assert parsed.breaking is True

    def test_breaking_footer_marks_breaking(self) -> None:
        """正向：正文 ``BREAKING CHANGE:`` 脚注同样标记破坏性变更。"""
        parsed = parse_commit(
            make_record(HASH_A, "feat: rework keys", "BREAKING CHANGE: new key layout")
        )
        assert parsed is not None
        assert parsed.breaking is True

    def test_plain_body_is_not_breaking(self) -> None:
        """边界：普通正文不触发破坏性标记。"""
        parsed = parse_commit(make_record(HASH_A, "docs: update readme", "Note: see ADR-1"))
        assert parsed is not None
        assert parsed.breaking is False

    def test_description_keeps_inner_colon(self) -> None:
        """边界：描述内部的冒号被保留。"""
        parsed = parse_commit(make_record(HASH_A, "fix: handle a:b edge case"))
        assert parsed is not None
        assert parsed.description == "handle a:b edge case"

    def test_scope_allows_slash_and_dot(self) -> None:
        """边界：scope 允许斜杠与点（如 ``api/routers``）。"""
        parsed = parse_commit(make_record(HASH_A, "feat(api/routers): add sessions"))
        assert parsed is not None
        assert parsed.scope == "api/routers"

    def test_unicode_description_is_preserved(self) -> None:
        """边界：中文 / Unicode 描述原样保留（无任何文本预处理）。"""
        parsed = parse_commit(make_record(HASH_A, "feat(privacy): 新增床号脱敏规则"))
        assert parsed is not None
        assert parsed.description == "新增床号脱敏规则"

    def test_short_hash_property(self) -> None:
        """正向：``short_hash`` 取前 7 位。"""
        parsed = parse_commit(make_record(HASH_A, "chore: tidy"))
        assert parsed is not None
        assert parsed.short_hash == HASH_A[:SHORT_HASH_LENGTH]
        assert len(parsed.short_hash) == SHORT_HASH_LENGTH

    def test_returns_none_for_merge_commit(self) -> None:
        """边界：合并提交（非规范主题）返回 ``None``。"""
        assert parse_commit(make_record(HASH_A, "Merge branch 'main' into dev")) is None

    def test_returns_none_for_non_conventional_subject(self) -> None:
        """边界：不符合 Conventional Commits 的主题返回 ``None``。"""
        assert parse_commit(make_record(HASH_A, "update readme")) is None

    def test_returns_none_without_space_after_colon(self) -> None:
        """边界：冒号后缺少空格（规范要求 ``: ``）返回 ``None``。"""
        assert parse_commit(make_record(HASH_A, "feat:no space")) is None

    def test_returns_none_for_empty_record(self) -> None:
        """边界：空记录返回 ``None``。"""
        assert parse_commit("") is None

    def test_returns_none_for_whitespace_record(self) -> None:
        """边界：纯空白记录返回 ``None``。"""
        assert parse_commit("   \n  ") is None

    def test_returns_none_for_malformed_fields(self) -> None:
        """边界：字段不足（缺正文分隔段）返回 ``None``。"""
        assert parse_commit(f"{HASH_A}{FIELD_SEP}feat: x") is None

    def test_returns_none_for_invalid_hash(self) -> None:
        """边界：非法哈希（非十六进制 / 过短）返回 ``None``。"""
        assert parse_commit(make_record("zzz", "feat: x")) is None
        assert parse_commit(make_record("abc12", "feat: x")) is None

    def test_tolerates_surrounding_whitespace_and_separator(self) -> None:
        """边界：记录首尾空白与尾部记录分隔符被容忍。"""
        parsed = parse_commit(f"  {make_record(HASH_A, 'chore: tidy')}\n")
        assert parsed is not None
        assert parsed.hash == HASH_A


class TestParseGitLog:
    """``parse_git_log``：全文解析与记录跳过。"""

    def test_parses_multiple_records_in_order(self) -> None:
        """正向：多条记录按 git 输出顺序（新→旧）解析。"""
        text = (
            make_record(HASH_A, "feat: newer")
            + make_record(HASH_B, "fix: older")
            + make_record(HASH_C, "docs: oldest")
        )
        entries = parse_git_log(text)
        assert [item.hash for item in entries] == [HASH_A, HASH_B, HASH_C]

    def test_skips_non_conventional_records(self) -> None:
        """边界：合并提交等不合规记录被跳过，不影响其余记录。"""
        text = (
            make_record(HASH_A, "feat: keep me")
            + make_record(HASH_B, "Merge branch 'main'")
            + make_record(HASH_C, "fix: keep me too")
        )
        entries = parse_git_log(text)
        assert [item.hash for item in entries] == [HASH_A, HASH_C]

    def test_empty_text_returns_empty_list(self) -> None:
        """边界：空文本返回空列表。"""
        assert parse_git_log("") == []

    def test_handles_trailing_record_separator(self) -> None:
        """边界：末尾记录分隔符产生的空片段被安全跳过。"""
        text = make_record(HASH_A, "chore: tidy") + RECORD_SEP
        entries = parse_git_log(text)
        assert len(entries) == 1

    def test_handles_text_without_trailing_separator(self) -> None:
        """边界：最后一条不带记录分隔符也能解析。"""
        text = f"{HASH_A}{FIELD_SEP}chore: tidy{FIELD_SEP}"
        entries = parse_git_log(text)
        assert len(entries) == 1


class TestGroupByType:
    """``group_by_type``：分组与章节顺序。"""

    def test_groups_and_orders_by_canonical_order(self) -> None:
        """正向：分组键序遵循 ``TYPE_TITLES`` 规范章节序。"""
        entries = [
            entry(HASH_A, "docs"),
            entry(HASH_B, "fix"),
            entry(HASH_C, "feat"),
        ]
        grouped = group_by_type(entries)
        assert list(grouped) == ["feat", "fix", "docs"]
        assert [item.hash for item in grouped["feat"]] == [HASH_C]

    def test_omits_empty_groups(self) -> None:
        """边界：没有条目的 type 不出现在结果里。"""
        grouped = group_by_type([entry(HASH_A, "chore")])
        assert list(grouped) == ["chore"]

    def test_unknown_types_come_after_known_sorted(self) -> None:
        """边界：未知 type 排在已知类型之后并按字母序排列。"""
        entries = [
            entry(HASH_A, "style"),
            entry(HASH_B, "feat"),
            entry(HASH_C, "build"),
        ]
        grouped = group_by_type(entries)
        assert list(grouped) == ["feat", "build", "style"]

    def test_empty_input_returns_empty_dict(self) -> None:
        """边界：空输入返回空字典。"""
        assert group_by_type([]) == {}


class TestRenderMarkdown:
    """``render_markdown``：Markdown 渲染。"""

    def test_header_contains_version_and_date(self) -> None:
        """正向：标题行包含版本与日期。"""
        markdown = render_markdown([entry()], version="1.2.3", date="2026-10-23")
        assert markdown.startswith("## [1.2.3] - 2026-10-23\n")

    def test_default_version_is_unreleased(self) -> None:
        """边界：缺省版本渲染为 ``Unreleased``。"""
        markdown = render_markdown([entry()], date="2026-10-23")
        assert markdown.startswith("## [Unreleased] - 2026-10-23\n")

    def test_default_date_is_today_utc(self) -> None:
        """边界：缺省日期取当前 UTC 日期（形态校验）。"""
        markdown = render_markdown([entry()])
        first_line = markdown.splitlines()[0]
        assert first_line.startswith("## [Unreleased] - ")
        assert len(first_line.removeprefix("## [Unreleased] - ")) == 10

    def test_breaking_section_comes_first(self) -> None:
        """正向：Breaking Changes 章节渲染在最前。"""
        entries = [
            entry(HASH_A, "feat", description="normal feature"),
            entry(HASH_B, "feat", description="breaking change", breaking=True),
        ]
        markdown = render_markdown(entries, version="2.0.0", date="2026-10-23")
        breaking_index = markdown.index(f"### {BREAKING_SECTION_TITLE}")
        features_index = markdown.index("### Features")
        assert breaking_index < features_index
        assert "- **stores**: breaking change (ff00ff0)" in markdown

    def test_entry_renders_scope_bold_and_short_hash(self) -> None:
        """正向：条目渲染为 ``- **scope**: description (短哈希)``。"""
        markdown = render_markdown([entry()], version="1.0.0", date="2026-10-23")
        assert "- **stores**: add redis store (a1b2c3d)" in markdown

    def test_entry_without_scope_has_no_scope_prefix(self) -> None:
        """边界：无 scope 条目只有描述与短哈希。"""
        markdown = render_markdown(
            [entry(HASH_B, "fix", scope=None, description="correct ttl")],
            version="1.0.0",
            date="2026-10-23",
        )
        assert "- correct ttl (ff00ff0)" in markdown
        assert "**" not in markdown

    def test_sections_follow_type_order(self) -> None:
        """正向：章节顺序遵循规范类型序，且组间以空行分隔。"""
        entries = [
            entry(HASH_A, "docs", description="write docs"),
            entry(HASH_B, "fix", description="fix bug"),
            entry(HASH_C, "feat", description="add feature"),
        ]
        markdown = render_markdown(entries, version="1.0.0", date="2026-10-23")
        assert (
            markdown.index("### Features")
            < markdown.index("### Bug Fixes")
            < markdown.index("### Documentation")
        )
        assert "\n\n### Features\n" in markdown

    def test_unknown_type_section_title_capitalized(self) -> None:
        """边界：未知 type 的章节标题用首字母大写兜底。"""
        markdown = render_markdown(
            [entry(HASH_A, "style", description="tidy imports")],
            version="1.0.0",
            date="2026-10-23",
        )
        assert "### Style" in markdown

    def test_empty_entries_render_no_changes_note(self) -> None:
        """边界：无条目时给出「无变更」说明且不产生任何章节。"""
        markdown = render_markdown([], version="1.0.0", date="2026-10-23")
        assert markdown == "## [1.0.0] - 2026-10-23\n\nNo changes recorded.\n"
        assert "###" not in markdown


class TestCollectCommits:
    """``collect_commits``：git 子进程调用（monkeypatch 替身，不碰真实 git）。"""

    def test_invokes_git_log_with_format_and_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：以 ``git log --format=... <range>`` 调用并解析输出。"""
        calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(
                command, 0, stdout=make_record(HASH_A, "feat: from git"), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        entries = collect_commits("v0.1.0..HEAD")
        assert len(calls) == 1
        assert calls[0][:3] == ["git", "log", f"--format={GIT_LOG_FORMAT}"]
        assert calls[0][-1] == "v0.1.0..HEAD"
        assert [item.description for item in entries] == ["from git"]

    def test_omits_range_when_not_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """边界：缺省范围时不追加范围参数（取全部历史）。"""
        calls: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert collect_commits() == []
        assert calls[0] == ["git", "log", f"--format={GIT_LOG_FORMAT}"]

    def test_passes_cwd(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """边界：``cwd`` 透传给子进程。"""
        seen: dict[str, object] = {}

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.update(kwargs)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        collect_commits("HEAD", cwd=tmp_path)
        assert seen["cwd"] == tmp_path

    def test_propagates_called_process_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """异常：git 退出码非零时向上抛 ``CalledProcessError``。"""

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(128, command, stderr="fatal: bad revision")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            collect_commits("nope..HEAD")


class TestMain:
    """``main``：CLI 行为与退出码。"""

    def test_writes_output_file_and_returns_zero(self, tmp_path: Path) -> None:
        """正向：``--log-file`` + ``--output`` 写文件并返回 0。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(
            make_record(HASH_A, "feat(stores): add redis store")
            + make_record(HASH_B, "fix: correct ttl"),
            encoding="utf-8",
        )
        output = tmp_path / "notes.md"
        exit_code = main(["--log-file", str(log_file), "--tag", "v1.2.3", "--output", str(output)])
        assert exit_code == 0
        text = output.read_text(encoding="utf-8")
        assert text.startswith("## [1.2.3] - ")
        assert "- **stores**: add redis store (a1b2c3d)" in text
        assert "- correct ttl (ff00ff0)" in text

    def test_prints_to_stdout_when_no_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：缺省 ``--output`` 时打印到标准输出。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(make_record(HASH_A, "chore: tidy"), encoding="utf-8")
        exit_code = main(["--log-file", str(log_file)])
        assert exit_code == 0
        assert "## [Unreleased] - " in capsys.readouterr().out

    def test_tag_prefix_is_stripped_in_header(self, tmp_path: Path) -> None:
        """正向：``--tag v1.2.3`` 的 ``v`` 前缀在标题中被剥离。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(make_record(HASH_A, "feat: x"), encoding="utf-8")
        output = tmp_path / "notes.md"
        assert main(["--log-file", str(log_file), "--tag", "v1.2.3", "--output", str(output)]) == 0
        assert output.read_text(encoding="utf-8").startswith("## [1.2.3] - ")

    def test_date_override(self, tmp_path: Path) -> None:
        """正向：``--date`` 覆盖发布日期。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(make_record(HASH_A, "feat: x"), encoding="utf-8")
        output = tmp_path / "notes.md"
        assert (
            main(
                [
                    "--log-file",
                    str(log_file),
                    "--tag",
                    "v1.0.0",
                    "--date",
                    "2026-01-31",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        assert "## [1.0.0] - 2026-01-31" in output.read_text(encoding="utf-8")

    def test_missing_log_file_returns_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：log 文件不存在时返回 2 并打印原因。"""
        exit_code = main(["--log-file", str(tmp_path / "absent.txt")])
        assert exit_code == 2
        assert "cannot read log file" in capsys.readouterr().err

    def test_git_failure_returns_two(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """异常：git 调用失败时返回 2 并打印原因。"""

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(128, command, stderr="fatal: not a git repository")

        monkeypatch.setattr(subprocess, "run", fake_run)
        exit_code = main(["--tag", "v1.0.0"])
        assert exit_code == 2
        assert "git log failed" in capsys.readouterr().err

    def test_output_write_failure_returns_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """异常：输出目录不存在时返回 2 并打印原因。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(make_record(HASH_A, "feat: x"), encoding="utf-8")
        exit_code = main(
            ["--log-file", str(log_file), "--output", str(tmp_path / "absent" / "notes.md")]
        )
        assert exit_code == 2
        assert "cannot write output" in capsys.readouterr().err

    def test_parser_accepts_documented_options(self) -> None:
        """正向：全部文档化选项均可被解析。"""
        args = build_parser().parse_args(
            [
                "--tag",
                "v1.0.0",
                "--rev-range",
                "v0.9.0..HEAD",
                "--log-file",
                "commits.txt",
                "--output",
                "notes.md",
                "--date",
                "2026-10-23",
            ]
        )
        assert args.tag == "v1.0.0"
        assert args.rev_range == "v0.9.0..HEAD"
        assert args.log_file == Path("commits.txt")
        assert args.output == Path("notes.md")
        assert args.date == "2026-10-23"

    def test_parser_rejects_unknown_option(self) -> None:
        """边界：未知选项触发 ``SystemExit``。"""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--unknown"])


class TestCommandLineEntry:
    """``python -m med_langchain_memory.changelog`` 真实入口（子进程）。"""

    #: 子进程环境：``PYTHONPATH`` 指向 src，与 pytest 的 ``pythonpath`` 配置一致。
    SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    def test_module_runs_as_script(self, tmp_path: Path) -> None:
        """正向：``python -m`` 入口端到端渲染并退出 0。"""
        log_file = tmp_path / "commits.txt"
        log_file.write_text(
            make_record(HASH_A, "feat(stores): add redis store")
            + make_record(HASH_B, "docs: update spec"),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "med_langchain_memory.changelog",
                "--log-file",
                str(log_file),
                "--tag",
                "v9.9.9",
            ],
            cwd=PROJECT_ROOT,
            env=self.SUBPROCESS_ENV,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "## [9.9.9] - " in completed.stdout
        assert "- **stores**: add redis store (a1b2c3d)" in completed.stdout

    def test_module_reports_git_failure_exit_code(self, tmp_path: Path) -> None:
        """边界：非 git 目录下真实 git 失败，进程以退出码 2 结束。"""
        completed = subprocess.run(
            [sys.executable, "-m", "med_langchain_memory.changelog", "--tag", "v1.0.0"],
            cwd=tmp_path,
            env=self.SUBPROCESS_ENV,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2
        assert "git log failed" in completed.stderr
