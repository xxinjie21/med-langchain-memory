"""Conventional Commits 变更日志生成器（纯标准库）。

把 ``git log`` 输出里符合 `Conventional Commits`_ 规范的提交按 type 分组，渲染成
Markdown 发布说明（GitHub Release notes / CHANGELOG 片段），供
``.github/workflows/release.yml`` 在建 Release 时使用。

设计约束：

* **只用标准库**（argparse / re / subprocess / dataclasses / datetime），零第三方依赖，
  也不参与运行时链路（与 ``release.py`` 同级的发布工程工具）；
* 不调用任何中文文本预处理或语义解析——只对「提交主题行」这一结构化字段做正则匹配；
* 输入是 ``git log`` 文本（或 ``--log-file`` 提供的同格式文本），单测全程离线。

git log 记录格式（字段间以 ``\\x1f`` 分隔，记录间以 ``\\x1e`` 分隔）::

    %H%x1f%s%x1f%b%x1e

命令行用法::

    python -m med_langchain_memory.changelog --tag v0.1.0 --rev-range v0.0.9..HEAD
    python -m med_langchain_memory.changelog --log-file commits.txt --output notes.md

.. _Conventional Commits: https://www.conventionalcommits.org/
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

#: ``git log --format`` 记录格式：完整哈希、主题、正文，记录以 ``\\x1e`` 结尾。
GIT_LOG_FORMAT = "%H%x1f%s%x1f%b%x1e"

#: Conventional Commits 主题行：``type(scope)!: description``。
SUBJECT_PATTERN = re.compile(
    r"^(?P<type>[A-Za-z]+)(?:\((?P<scope>[^()\s]*)\))?(?P<breaking>!)?:\s+(?P<description>\S.*)$"
)

#: 正文里的 ``BREAKING CHANGE:`` 脚注（与主题行 ``!`` 标记等价）。
BREAKING_FOOTER_PATTERN = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)

#: 完整（或缩写）提交哈希：十六进制，7–64 位。
HASH_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")

#: 章节顺序：type → 人类可读标题（与 ROADMAP 提交规范的类型集合一致）。
TYPE_TITLES: tuple[tuple[str, str], ...] = (
    ("feat", "Features"),
    ("fix", "Bug Fixes"),
    ("perf", "Performance Improvements"),
    ("refactor", "Refactoring"),
    ("test", "Tests"),
    ("docs", "Documentation"),
    ("ci", "CI"),
    ("chore", "Chores"),
)

#: Breaking Changes 章节标题（固定渲染在最前）。
BREAKING_SECTION_TITLE = "Breaking Changes"

#: 短哈希长度（与 ``git log --oneline`` 一致）。
SHORT_HASH_LENGTH = 7


@dataclass(frozen=True)
class CommitEntry:
    """一条解析后的 Conventional Commit。"""

    hash: str
    type: str
    scope: str | None
    description: str
    breaking: bool

    @property
    def short_hash(self) -> str:
        """7 位短哈希（与 ``git log --oneline`` 一致）。"""
        return self.hash[:SHORT_HASH_LENGTH]


def parse_commit(record: str) -> CommitEntry | None:
    """解析单条 git log 记录；不符合 Conventional Commits 时返回 ``None``。

    Args:
        record: 一条 ``hash\\x1fsubject\\x1fbody`` 记录（可带尾部 ``\\x1e`` 与空白）。

    Returns:
        解析成功的条目；合并提交、非规范主题、字段缺失或哈希非法时返回 ``None``。

    Note:
        先 ``rstrip`` 记录分隔符再按字段分隔符切分——``\\x1e`` / ``\\x1f`` 会被
        ``str.strip()`` 当作空白字符，直接 ``strip()`` 会连正文为空时的字段分隔符
        一起吃掉，导致字段数误判。
    """
    fields = record.rstrip("\x1e").split("\x1f")
    if len(fields) != 3:
        return None
    commit_hash, subject, body = (field.strip() for field in fields)
    if HASH_PATTERN.fullmatch(commit_hash) is None:
        return None
    match = SUBJECT_PATTERN.match(subject)
    if match is None:
        return None
    breaking = (
        match.group("breaking") is not None or BREAKING_FOOTER_PATTERN.search(body) is not None
    )
    return CommitEntry(
        hash=commit_hash,
        type=match.group("type").lower(),
        scope=match.group("scope") or None,
        description=match.group("description").strip(),
        breaking=breaking,
    )


def parse_git_log(text: str) -> list[CommitEntry]:
    """解析 ``git log`` 全文。

    Args:
        text: ``GIT_LOG_FORMAT`` 格式的多条记录文本。

    Returns:
        解析成功的条目列表，保持 git 输出顺序（新→旧）；不合规记录被跳过。
    """
    entries: list[CommitEntry] = []
    for record in text.split("\x1e"):
        entry = parse_commit(record)
        if entry is not None:
            entries.append(entry)
    return entries


def group_by_type(entries: Iterable[CommitEntry]) -> dict[str, list[CommitEntry]]:
    """按 type 分组。

    Args:
        entries: 提交条目。

    Returns:
        ``type -> 条目列表`` 的字典；键序为规范章节序（``TYPE_TITLES``），
        未知 type 排在已知类型之后并按字母序排列，空分组不出现。
    """
    grouped: dict[str, list[CommitEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.type, []).append(entry)
    known = {commit_type for commit_type, _title in TYPE_TITLES}
    ordered: dict[str, list[CommitEntry]] = {}
    for commit_type, _title in TYPE_TITLES:
        if commit_type in grouped:
            ordered[commit_type] = grouped[commit_type]
    for commit_type in sorted(set(grouped) - known):
        ordered[commit_type] = grouped[commit_type]
    return ordered


def render_markdown(
    entries: Sequence[CommitEntry],
    *,
    version: str | None = None,
    date: str | None = None,
) -> str:
    """把提交列表渲染为 Markdown 发布说明。

    Args:
        entries: 提交条目（传入顺序即渲染顺序，通常为 git log 的新→旧顺序）。
        version: 版本标签（如 ``0.1.0``）；缺省渲染为 ``Unreleased``。
        date: 发布日期（``YYYY-MM-DD``）；缺省取当前 UTC 日期。

    Returns:
        Markdown 文本：``## [版本] - 日期`` 开头，Breaking Changes 章节在最前，
        其余按 ``TYPE_TITLES`` 顺序；条目为空时给出「无变更」说明。
    """
    label = version or "Unreleased"
    stamp = date or datetime.now(UTC).date().isoformat()
    lines = [f"## [{label}] - {stamp}", ""]
    if not entries:
        lines.extend(["No changes recorded.", ""])
        return "\n".join(lines)
    breaking = [entry for entry in entries if entry.breaking]
    if breaking:
        lines.append(f"### {BREAKING_SECTION_TITLE}")
        lines.append("")
        lines.extend(_render_entries(breaking))
        lines.append("")
    for commit_type, group in group_by_type(entries).items():
        lines.append(f"### {_section_title(commit_type)}")
        lines.append("")
        lines.extend(_render_entries(group))
        lines.append("")
    return "\n".join(lines)


def collect_commits(rev_range: str | None = None, *, cwd: Path | None = None) -> list[CommitEntry]:
    """调用 ``git log`` 收集提交（本模块唯一触及外部进程的地方）。

    Args:
        rev_range: git 版本范围（如 ``v0.1.0..HEAD``）；缺省取全部历史。
        cwd: 执行 git 的工作目录；缺省为当前目录。

    Returns:
        解析后的提交列表（新→旧）。

    Raises:
        subprocess.CalledProcessError: git 退出码非零（如范围非法、不是 git 仓库）。
        OSError: 找不到 git 可执行文件。
    """
    command = ["git", "log", f"--format={GIT_LOG_FORMAT}"]
    if rev_range:
        command.append(rev_range)
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
    return parse_git_log(completed.stdout)


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器（选项供工作流静态测试交叉校验）。"""
    parser = argparse.ArgumentParser(
        prog="python -m med_langchain_memory.changelog",
        description="Render Conventional Commits as a Markdown changelog / release notes.",
    )
    parser.add_argument("--tag", help="release tag, e.g. v0.1.0 (a leading 'v' is stripped)")
    parser.add_argument(
        "--rev-range", help='git revision range, e.g. "v0.1.0..HEAD" (default: full history)'
    )
    parser.add_argument(
        "--log-file", type=Path, help="read git-log text from a file instead of invoking git"
    )
    parser.add_argument(
        "--output", type=Path, help="write the Markdown to this file instead of stdout"
    )
    parser.add_argument("--date", help="release date as YYYY-MM-DD (default: today, UTC)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv: 命令行参数；缺省取 ``sys.argv``。

    Returns:
        ``0`` 成功；``2`` 输入或输出不可用（log 文件读不了 / git 调用失败 /
        输出写不了），原因打印到 stderr。
    """
    args = build_parser().parse_args(argv)
    if args.log_file is not None:
        try:
            raw = args.log_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"changelog: cannot read log file: {exc}", file=sys.stderr)
            return 2
        entries = parse_git_log(raw)
    else:
        try:
            entries = collect_commits(args.rev_range)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"changelog: git log failed: {exc}", file=sys.stderr)
            return 2
    version = args.tag.removeprefix("v") if args.tag else None
    markdown = render_markdown(entries, version=version, date=args.date)
    if args.output is not None:
        try:
            args.output.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            print(f"changelog: cannot write output: {exc}", file=sys.stderr)
            return 2
    else:
        print(markdown)
    return 0


def _section_title(commit_type: str) -> str:
    """type → 章节标题；未知 type 用首字母大写兜底，不丢分组。"""
    return dict(TYPE_TITLES).get(commit_type, commit_type.capitalize())


def _render_entries(entries: Sequence[CommitEntry]) -> list[str]:
    """渲染条目列表：``- **scope**: description (短哈希)``。"""
    return [f"- {_format_entry(entry)}" for entry in entries]


def _format_entry(entry: CommitEntry) -> str:
    """渲染单条：带 scope 时加粗前缀，无 scope 时只有描述与短哈希。"""
    scope = f"**{entry.scope}**: " if entry.scope else ""
    return f"{scope}{entry.description} ({entry.short_hash})"


if __name__ == "__main__":  # pragma: no cover - 仅 `python -m` 入口
    raise SystemExit(main())
