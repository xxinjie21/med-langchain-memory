"""``med_langchain_memory.release`` 发布守卫单测。

覆盖两条发布门禁：

1. **三方版本一致性**：git tag ↔ ``pyproject.toml`` ↔ 包内 ``__version__``；
2. **分发物完整性**：``python -m build`` 产出的 sdist / wheel 是否齐全、
   wheel 内必需成员与元数据版本是否正确。

全部使用合成 wheel / sdist（``zipfile`` + ``tarfile``）与临时 pyproject，
不执行真实构建、不触碰网络；每个公开函数均含正向用例与边界 / 异常用例。
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from med_langchain_memory import __version__ as PACKAGE_VERSION
from med_langchain_memory.release import (
    REQUIRED_SDIST_MEMBERS,
    REQUIRED_WHEEL_MEMBERS,
    TAG_PREFIX,
    build_parser,
    check_release,
    default_pyproject_path,
    inspect_dist,
    is_semver,
    main,
    normalize_version,
    parse_tag,
    read_project_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

#: 与当前包版本一致的合法发布 tag。
VALID_TAG = f"{TAG_PREFIX}{PACKAGE_VERSION}"

WHEEL_NAME = f"med_langchain_memory-{PACKAGE_VERSION}-py3-none-any.whl"
SDIST_NAME = f"med_langchain_memory-{PACKAGE_VERSION}.tar.gz"
SDIST_ROOT = f"med_langchain_memory-{PACKAGE_VERSION}"


def write_wheel(
    dist_dir: Path,
    *,
    name: str = WHEEL_NAME,
    version: str | None = PACKAGE_VERSION,
    members: Sequence[str] = REQUIRED_WHEEL_MEMBERS,
) -> Path:
    """生成最小可用的合成 wheel。

    Args:
        dist_dir: 输出目录（必须已存在）。
        name: wheel 文件名。
        version: ``METADATA`` 中声明的版本；``None`` 表示不写 ``METADATA``。
        members: 写入 wheel 的成员路径。

    Returns:
        生成的 wheel 路径。
    """
    path = dist_dir / name
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "")
        if version is not None:
            archive.writestr(
                f"med_langchain_memory-{PACKAGE_VERSION}.dist-info/METADATA",
                f"Metadata-Version: 2.3\nName: med-langchain-memory\nVersion: {version}\n",
            )
    return path


def write_sdist(
    dist_dir: Path,
    *,
    name: str = SDIST_NAME,
    members: Sequence[str] = REQUIRED_SDIST_MEMBERS,
) -> Path:
    """生成最小可用的合成 sdist（``tar.gz``）。"""
    path = dist_dir / name
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            info = tarfile.TarInfo(name=f"{SDIST_ROOT}/{member}")
            info.size = 0
            archive.addfile(info, io.BytesIO(b""))
    return path


def write_dist(dist_dir: Path, *, wheel_version: str | None = PACKAGE_VERSION) -> None:
    """写出「一份 wheel + 一份 sdist」的完整产物集。"""
    write_wheel(dist_dir, version=wheel_version)
    write_sdist(dist_dir)


class TestTagParsing:
    """``parse_tag`` 与 ``TAG_PREFIX``。"""

    def test_tag_prefix_is_v(self) -> None:
        """正向：发布 tag 前缀固定为 ``v``（工作流触发器与之绑定）。"""
        assert TAG_PREFIX == "v"

    def test_parse_tag_strips_prefix(self) -> None:
        """正向：``v0.1.0`` → ``0.1.0``。"""
        assert parse_tag(f"{TAG_PREFIX}{PACKAGE_VERSION}") == PACKAGE_VERSION

    def test_parse_tag_ignores_surrounding_whitespace(self) -> None:
        """边界：tag 前后的空白字符不应导致校验失败。"""
        assert parse_tag(f"  {TAG_PREFIX}{PACKAGE_VERSION}\n") == PACKAGE_VERSION

    @pytest.mark.parametrize(
        "tag",
        ["0.1.0", "v1.0", "v1", "v1.0.0-rc1", "release-1.0.0", "v1.0.0.0", "v", ""],
        ids=[
            "no-prefix",
            "two-segments",
            "one-segment",
            "prerelease",
            "wrong-prefix",
            "four-segments",
            "prefix-only",
            "empty",
        ],
    )
    def test_parse_tag_rejects_malformed_tags(self, tag: str) -> None:
        """边界：非法 tag 一律抛 ``ValueError``（不允许静默放过）。"""
        with pytest.raises(ValueError, match="invalid release tag"):
            parse_tag(tag)


class TestVersionHelpers:
    """``is_semver`` 与 ``normalize_version``。"""

    @pytest.mark.parametrize("value", ["0.1.0", "1.2.3", "10.20.30"], ids=["patch", "small", "big"])
    def test_is_semver_accepts_three_segments(self, value: str) -> None:
        """正向：三段式数字版本合法。"""
        assert is_semver(value) is True

    @pytest.mark.parametrize("value", ["1.0", "v1.0.0", "1.0.0.0", "1.0.0-rc1", "latest", ""])
    def test_is_semver_rejects_other_shapes(self, value: str) -> None:
        """边界：带前缀 / 段数不符 / 预发布后缀均不合法。"""
        assert is_semver(value) is False

    def test_normalize_version_accepts_both_forms(self) -> None:
        """正向：带前缀与不带前缀都能归一化到同一版本。"""
        assert normalize_version(f"{TAG_PREFIX}{PACKAGE_VERSION}") == PACKAGE_VERSION
        assert normalize_version(PACKAGE_VERSION) == PACKAGE_VERSION

    def test_normalize_version_rejects_garbage(self) -> None:
        """边界：非版本字符串抛 ``ValueError``。"""
        with pytest.raises(ValueError, match="invalid version"):
            normalize_version("v1.0")


class TestReadProjectVersion:
    """``read_project_version`` 与 ``default_pyproject_path``。"""

    def test_reads_real_pyproject(self) -> None:
        """正向：仓库 pyproject 的版本与包内 ``__version__`` 一致。"""
        assert read_project_version(PYPROJECT_PATH) == PACKAGE_VERSION

    def test_default_path_points_to_repo_root(self) -> None:
        """正向：默认路径解析到仓库根的真实 pyproject。"""
        path = default_pyproject_path()
        assert path == PYPROJECT_PATH
        assert path.is_file()

    def test_missing_version_key_raises(self, tmp_path: Path) -> None:
        """边界：pyproject 缺少 ``[project].version`` 时抛 ``ValueError``。"""
        broken = tmp_path / "pyproject.toml"
        broken.write_text('[project]\nname = "x"\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r"missing \[project\].version"):
            read_project_version(broken)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """边界：文件不存在时抛 ``FileNotFoundError``。"""
        with pytest.raises(FileNotFoundError):
            read_project_version(tmp_path / "nope.toml")


class TestCheckRelease:
    """``check_release`` 三方一致性判定。"""

    def test_matching_versions_pass(self) -> None:
        """正向：tag / pyproject / 包版本三者一致时通过。"""
        check = check_release(VALID_TAG)
        assert check.ok is True
        assert check.errors == ()
        assert check.version == PACKAGE_VERSION
        assert check.project_version == PACKAGE_VERSION
        assert check.package_version == PACKAGE_VERSION

    def test_pyproject_mismatch_is_reported(self, tmp_path: Path) -> None:
        """边界：pyproject 版本与 tag 不一致时给出明确差异。"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "9.9.9"\n', encoding="utf-8")
        check = check_release(VALID_TAG, pyproject_path=pyproject)
        assert check.ok is False
        assert len(check.errors) == 1
        assert "pyproject version '9.9.9'" in check.errors[0]

    def test_package_version_mismatch_is_reported(self) -> None:
        """边界：包内 ``__version__`` 与 tag 不一致时给出明确差异。"""
        check = check_release(VALID_TAG, package_version="9.9.9")
        assert check.ok is False
        assert len(check.errors) == 1
        assert "package __version__ '9.9.9'" in check.errors[0]

    def test_all_mismatches_are_collected(self, tmp_path: Path) -> None:
        """边界：多处不一致时全部报出，而不是只报第一条。"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nversion = "9.9.8"\n', encoding="utf-8")
        check = check_release(VALID_TAG, pyproject_path=pyproject, package_version="9.9.9")
        assert check.ok is False
        assert len(check.errors) == 2

    def test_invalid_tag_raises_before_reading_files(self, tmp_path: Path) -> None:
        """边界：tag 形态非法时直接抛 ``ValueError``（不读 pyproject）。"""
        with pytest.raises(ValueError, match="invalid release tag"):
            check_release("0.1.0", pyproject_path=tmp_path / "missing.toml")


class TestSummaries:
    """``ReleaseCheck.summary`` / ``DistCheck.summary`` 的可读报告。"""

    def test_release_summary_ok_lists_all_versions(self) -> None:
        """正向：通过时报告带 OK 标记与三方版本。"""
        summary = check_release(VALID_TAG).summary()
        assert summary.startswith("release check OK")
        assert f"tag={VALID_TAG}" in summary
        assert f"pyproject={PACKAGE_VERSION}" in summary
        assert f"package={PACKAGE_VERSION}" in summary
        assert "  ! " not in summary

    def test_release_summary_failure_lists_errors(self) -> None:
        """边界：失败时报告带 FAILED 标记与逐条差异。"""
        summary = check_release(VALID_TAG, package_version="9.9.9").summary()
        assert summary.startswith("release check FAILED")
        assert "  ! package __version__ '9.9.9'" in summary

    def test_dist_summary_ok_lists_artifacts(self, tmp_path: Path) -> None:
        """正向：通过时报告列出产物文件名。"""
        write_dist(tmp_path)
        summary = inspect_dist(tmp_path).summary()
        assert summary.startswith("dist check OK")
        assert WHEEL_NAME in summary
        assert SDIST_NAME in summary

    def test_dist_summary_failure_lists_errors(self, tmp_path: Path) -> None:
        """边界：失败时报告带 FAILED 标记与逐条问题。"""
        summary = inspect_dist(tmp_path).summary()
        assert summary.startswith("dist check FAILED")
        assert "no wheel" in summary
        assert "no sdist" in summary


class TestInspectDist:
    """``inspect_dist`` 分发物完整性判定。"""

    def test_complete_dist_passes(self, tmp_path: Path) -> None:
        """正向：齐全的 sdist + wheel 通过，并读出元数据版本。"""
        write_dist(tmp_path)
        check = inspect_dist(tmp_path, expected_version=f"{TAG_PREFIX}{PACKAGE_VERSION}")
        assert check.ok is True
        assert check.errors == ()
        assert check.metadata_version == PACKAGE_VERSION
        assert check.artifacts == (WHEEL_NAME, SDIST_NAME)

    def test_missing_directory_is_reported(self, tmp_path: Path) -> None:
        """边界：dist 目录不存在时给出明确错误（而非静默通过）。"""
        check = inspect_dist(tmp_path / "nope")
        assert check.ok is False
        assert check.artifacts == ()
        assert "dist directory not found" in check.errors[0]

    def test_empty_directory_reports_both_artifacts(self, tmp_path: Path) -> None:
        """边界：空目录同时缺 wheel 与 sdist。"""
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert len(check.errors) == 2
        assert check.metadata_version is None

    def test_wheel_only_reports_missing_sdist(self, tmp_path: Path) -> None:
        """边界：只有 wheel 时仍报缺 sdist。"""
        write_wheel(tmp_path)
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert check.errors == ("no sdist (*.tar.gz) found",)

    def test_sdist_only_reports_missing_wheel(self, tmp_path: Path) -> None:
        """边界：只有 sdist 时仍报缺 wheel。"""
        write_sdist(tmp_path)
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert check.errors == ("no wheel (*.whl) found",)
        assert check.metadata_version is None

    def test_missing_wheel_member_is_reported(self, tmp_path: Path) -> None:
        """边界：wheel 缺必需成员（如 ``py.typed``）时必须失败。"""
        members = [m for m in REQUIRED_WHEEL_MEMBERS if not m.endswith("py.typed")]
        write_wheel(tmp_path, members=members)
        write_sdist(tmp_path)
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert any("py.typed" in error for error in check.errors)

    def test_metadata_version_mismatch_is_reported(self, tmp_path: Path) -> None:
        """边界：wheel 元数据版本与期望版本不一致时必须失败。"""
        write_dist(tmp_path, wheel_version="0.0.1")
        check = inspect_dist(tmp_path, expected_version=PACKAGE_VERSION)
        assert check.ok is False
        assert any("metadata version '0.0.1'" in error for error in check.errors)

    def test_expected_version_without_metadata_is_reported(self, tmp_path: Path) -> None:
        """边界：要求比对版本但 wheel 里没有 METADATA 时失败。"""
        write_dist(tmp_path, wheel_version=None)
        check = inspect_dist(tmp_path, expected_version=PACKAGE_VERSION)
        assert check.ok is False
        assert "wheel METADATA Version not found" in check.errors

    def test_invalid_expected_version_raises(self, tmp_path: Path) -> None:
        """边界：期望版本本身非法时抛 ``ValueError``。"""
        write_dist(tmp_path)
        with pytest.raises(ValueError, match="invalid version"):
            inspect_dist(tmp_path, expected_version="v1.0")

    def test_multiple_wheels_are_reported(self, tmp_path: Path) -> None:
        """边界：dist 里躺着两个 wheel（旧构建残留）时必须失败。"""
        write_dist(tmp_path)
        write_wheel(tmp_path, name="med_langchain_memory-0.0.1-py3-none-any.whl")
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert any("multiple wheels" in error for error in check.errors)

    def test_sdist_missing_pyproject_is_reported(self, tmp_path: Path) -> None:
        """边界：sdist 缺 ``pyproject.toml`` 时失败（源码包不可重建）。"""
        write_wheel(tmp_path)
        write_sdist(tmp_path, members=[m for m in REQUIRED_SDIST_MEMBERS if "pyproject" not in m])
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert any("pyproject.toml" in error for error in check.errors)

    def test_sdist_missing_proto_is_reported(self, tmp_path: Path) -> None:
        """边界：sdist 缺跨语言协议源文件时失败。"""
        write_wheel(tmp_path)
        write_sdist(tmp_path, members=[m for m in REQUIRED_SDIST_MEMBERS if "proto" not in m])
        check = inspect_dist(tmp_path)
        assert check.ok is False
        assert any("med_session.proto" in error for error in check.errors)

    def test_ignores_directories_in_artifact_list(self, tmp_path: Path) -> None:
        """边界：产物清单只统计文件，目录不参与计数。"""
        write_dist(tmp_path)
        (tmp_path / "subdir").mkdir()
        check = inspect_dist(tmp_path)
        assert check.ok is True
        assert check.artifacts == (WHEEL_NAME, SDIST_NAME)


class TestMainCli:
    """``main`` / ``build_parser`` 命令行契约（工作流直接调用）。"""

    def test_parser_requires_no_positional_arguments(self) -> None:
        """正向：解析器仅接受选项参数，默认值全为空。"""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.tag is None
        assert args.dist is None
        assert args.expected_version is None
        assert args.pyproject is None

    def test_tag_check_succeeds(self, capsys: pytest.CaptureFixture[str]) -> None:
        """正向：tag 与包版本一致时返回 0。"""
        assert main(["--tag", VALID_TAG]) == 0
        assert "release check OK" in capsys.readouterr().out

    def test_tag_check_fails_on_version_drift(self, capsys: pytest.CaptureFixture[str]) -> None:
        """边界：tag 版本漂移时返回 1 并打印差异。"""
        assert main(["--tag", f"{TAG_PREFIX}9.9.9"]) == 1
        assert "release check FAILED" in capsys.readouterr().out

    def test_malformed_tag_reports_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """边界：tag 形态非法时返回 1，错误走 stderr。"""
        assert main(["--tag", "0.1.0"]) == 1
        captured = capsys.readouterr()
        assert "release check FAILED" in captured.err
        assert captured.out == ""

    def test_missing_pyproject_reports_to_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：pyproject 路径不存在时返回 1，不抛栈。"""
        assert main(["--tag", VALID_TAG, "--pyproject", str(tmp_path / "nope.toml")]) == 1
        assert "release check FAILED" in capsys.readouterr().err

    def test_dist_check_succeeds(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """正向：产物齐全时返回 0。"""
        write_dist(tmp_path)
        code = main(["--dist", str(tmp_path), "--expected-version", VALID_TAG])
        assert code == 0
        assert "dist check OK" in capsys.readouterr().out

    def test_dist_check_fails_on_missing_artifacts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：dist 目录为空时返回 1。"""
        assert main(["--dist", str(tmp_path)]) == 1
        assert "dist check FAILED" in capsys.readouterr().out

    def test_dist_check_reports_broken_wheel(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：wheel 不是合法 zip 时返回 1，异常被翻译为报告。"""
        (tmp_path / "broken.whl").write_text("not a zip", encoding="utf-8")
        assert main(["--dist", str(tmp_path)]) == 1
        assert "dist check FAILED" in capsys.readouterr().err

    def test_dist_check_reports_invalid_expected_version(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``--expected-version`` 非法时返回 1，异常被翻译为报告。"""
        write_dist(tmp_path)
        assert main(["--dist", str(tmp_path), "--expected-version", "v1.0"]) == 1
        assert "dist check FAILED" in capsys.readouterr().err

    def test_requires_at_least_one_check(self, capsys: pytest.CaptureFixture[str]) -> None:
        """边界：既不给 ``--tag`` 也不给 ``--dist`` 时以退出码 2 结束。"""
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
        assert "at least one of --tag / --dist is required" in capsys.readouterr().err

    def test_tag_and_dist_checks_compose(self, tmp_path: Path) -> None:
        """正向：两条门禁可在同一次调用中串联执行。"""
        write_dist(tmp_path)
        code = main(
            [
                "--tag",
                VALID_TAG,
                "--dist",
                str(tmp_path),
                "--expected-version",
                VALID_TAG,
            ]
        )
        assert code == 0
