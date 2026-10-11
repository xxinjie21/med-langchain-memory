"""``med_langchain_memory.testing.preflight`` 单元测试。

守卫本身只依赖标准库，因此**不需要任何中间件**：正向用例在回环地址上临时监听
一个端口并把它伪装成 Redis 地址，异常用例用刚释放的空闲端口，
边界用例覆盖「未声明必需服务」「服务名拼写错误」「环境变量缺失 / 空白 / 重复」等分支。

每个公开成员均含正向与边界用例：
``parse_required`` / ``PreflightReport.ok`` / ``PreflightReport.failures`` /
``PreflightReport.render`` / ``preflight`` / ``build_parser`` / ``main``。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from med_langchain_memory.testing import ENV_PREFIX, ENV_REQUIRED, PreflightReport
from med_langchain_memory.testing.preflight import (
    build_parser,
    main,
    parse_required,
    preflight,
)
from med_langchain_memory.testing.services import IntegrationService, ServiceStatus

HOST = "127.0.0.1"

#: 守卫的模块入口（``python -m`` 调用形态）。
PREFLIGHT_MODULE = "med_langchain_memory.testing.preflight"


@contextmanager
def listening_port() -> Iterator[int]:
    """在回环地址上临时监听一个随机端口，产出端口号，退出时关闭。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, 0))
    sock.listen(1)
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def free_port() -> int:
    """返回一个刚被释放、当前无人监听的端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def ready_env(port: int) -> dict[str, str]:
    """构造「开关已开 + Redis 指向回环监听端口」的环境变量映射。"""
    return {ENV_PREFIX: "1", f"{ENV_PREFIX}_REDIS_URL": f"redis://{HOST}:{port}/0"}


def make_status(*, ready: bool, reason: str | None) -> ServiceStatus:
    """手工构造一个服务检查结果（用于直接驱动 :class:`PreflightReport`）。"""
    service = IntegrationService("redis", "redis://localhost:16379/15", "localhost", 16379, "X_URL")
    return ServiceStatus(service, True, ready, reason)


class TestParseRequired:
    """必需服务清单解析：逗号分隔、去空白、去重保序。"""

    def test_parses_comma_separated_names(self) -> None:
        """正向：逗号分隔的清单按声明顺序解析。"""
        env = {ENV_REQUIRED: "redis,mysql,elasticsearch,redis-cluster"}
        assert parse_required(env) == ("redis", "mysql", "elasticsearch", "redis-cluster")

    def test_trims_whitespace_and_drops_blanks(self) -> None:
        """边界：首尾空白被去掉，空项被丢弃。"""
        env = {ENV_REQUIRED: " redis , , mysql ,, "}
        assert parse_required(env) == ("redis", "mysql")

    def test_deduplicates_keeping_first_seen_order(self) -> None:
        """边界：重复服务名只保留首次出现的位置。"""
        assert parse_required({ENV_REQUIRED: "mysql,redis,mysql"}) == ("mysql", "redis")

    @pytest.mark.parametrize("raw", ["", "   ", ",", ", ,"])
    def test_empty_values_yield_no_requirements(self, raw: str) -> None:
        """边界：空串 / 全空白 / 只有分隔符时不产生任何必需服务。"""
        assert parse_required({ENV_REQUIRED: raw}) == ()

    def test_missing_variable_yields_no_requirements(self) -> None:
        """边界：变量缺失时返回空元组（本地手工跑 pytest 不受影响）。"""
        assert parse_required({}) == ()

    def test_reads_os_environ_when_env_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：``env`` 为 ``None`` 时读取真实环境变量。"""
        monkeypatch.setenv(ENV_REQUIRED, "redis, mysql")
        assert parse_required() == ("redis", "mysql")


class TestPreflightReport:
    """报告对象的判定与渲染。"""

    def test_ok_when_every_service_is_ready(self) -> None:
        """正向：全部就绪时为 ``True``。"""
        report = PreflightReport((make_status(ready=True, reason=None),))
        assert report.ok is True

    def test_ok_when_nothing_was_declared(self) -> None:
        """边界：未声明任何服务视为通过（空集合的 ``all`` 为真）。"""
        assert PreflightReport().ok is True

    def test_not_ok_when_a_service_is_unreachable(self) -> None:
        """异常：存在未就绪服务时为 ``False``。"""
        report = PreflightReport((make_status(ready=False, reason="not reachable"),))
        assert report.ok is False

    def test_not_ok_when_a_service_name_is_unknown(self) -> None:
        """异常：存在无法解析的服务名时为 ``False``（即使没有任何状态条目）。"""
        assert PreflightReport((), ("unknown integration service 'kafka'",)).ok is False

    def test_failures_collects_errors_then_reasons(self) -> None:
        """正向：问题列表 = 解析错误 + 未就绪服务的原因，顺序稳定。"""
        report = PreflightReport(
            (make_status(ready=False, reason="boom"), make_status(ready=True, reason=None)),
            ("unknown integration service 'kafka'",),
        )
        assert report.failures == ("unknown integration service 'kafka'", "boom")

    def test_failures_falls_back_to_service_description(self) -> None:
        """边界：未就绪但未附原因时退回服务描述，绝不产出 ``None``。"""
        report = PreflightReport((make_status(ready=False, reason=None),))
        assert report.failures == ("redis @ localhost:16379 (override via X_URL)",)

    def test_failures_is_empty_when_all_ready(self) -> None:
        """边界：全部就绪时没有问题条目。"""
        assert PreflightReport((make_status(ready=True, reason=None),)).failures == ()

    def test_render_lists_every_service_and_verdict(self) -> None:
        """正向：逐服务一行，末行给出总判定；未就绪原因不重复服务地址。"""
        report = PreflightReport(
            (make_status(ready=True, reason=None), make_status(ready=False, reason="boom"))
        )
        lines = report.render().splitlines()
        assert lines[0] == "preflight: 2 required service(s)"
        assert "[ok]   redis @ localhost:16379 (override via X_URL)" in lines[1]
        assert lines[2] == "  [fail] boom"
        assert lines[-1] == "preflight failed: 1 problem(s)"

    def test_render_includes_parse_errors(self) -> None:
        """异常：服务名非法时把错误信息也渲染出来。"""
        text = PreflightReport((), ("unknown integration service 'kafka'",)).render()
        assert "unknown integration service 'kafka'" in text
        assert text.endswith("preflight failed: 1 problem(s)")

    def test_render_marks_a_clean_run_as_passed(self) -> None:
        """正向：全部就绪时末行为 ``preflight passed``。"""
        text = PreflightReport((make_status(ready=True, reason=None),)).render()
        assert text.endswith("preflight passed")


class TestPreflight:
    """逐服务检查：开关、可达性与服务名解析。"""

    def test_ready_service_passes(self) -> None:
        """正向：开关已开且端口可连时整体通过。"""
        with listening_port() as port:
            report = preflight(["redis"], env=ready_env(port))
        assert report.ok is True
        assert report.failures == ()

    def test_switch_off_fails_with_reason(self) -> None:
        """异常：总开关未打开时判定失败，并点明环境变量名。"""
        with listening_port() as port:
            report = preflight(["redis"], env={f"{ENV_PREFIX}_REDIS_URL": f"redis://{HOST}:{port}"})
        assert report.ok is False
        assert ENV_PREFIX in report.failures[0]

    def test_unreachable_service_fails(self) -> None:
        """异常：开关已开但端口不可连时判定失败。"""
        env = {ENV_PREFIX: "1", f"{ENV_PREFIX}_REDIS_URL": f"redis://{HOST}:{free_port()}"}
        assert preflight(["redis"], env=env).ok is False

    def test_unknown_service_is_collected_without_aborting(self) -> None:
        """异常：未知服务名记进错误列表，但不中断对其余服务的检查。"""
        with listening_port() as port:
            report = preflight(["kafka", "redis"], env=ready_env(port))
        assert report.ok is False
        assert len(report.statuses) == 1
        assert "unknown integration service" in report.errors[0]

    def test_no_requirements_is_a_pass(self) -> None:
        """边界：未声明任何服务时直接通过（守卫可安全地无条件执行）。"""
        assert preflight([]).ok is True

    def test_wait_seconds_polls_before_probing(self) -> None:
        """正向：给定 ``wait_seconds`` 时走轮询路径，可达即通过。"""
        with listening_port() as port:
            report = preflight(["redis"], env=ready_env(port), wait_seconds=5.0)
        assert report.ok is True


class TestCli:
    """命令行入口：参数、必需清单来源与退出码。"""

    def test_parser_defaults(self) -> None:
        """正向：默认不传服务、不等待。"""
        args = build_parser().parse_args([])
        assert (args.services, args.wait) == ([], None)

    def test_parser_accepts_services_and_wait(self) -> None:
        """正向：位置参数收集服务名，``--wait`` 解析为浮点秒数。"""
        args = build_parser().parse_args(["redis", "mysql", "--wait", "2.5"])
        assert (args.services, args.wait) == (["redis", "mysql"], 2.5)

    def test_main_exits_zero_when_requirements_are_met(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """正向：显式传入服务名且可达时退出码为 0，报告写到标准输出。"""
        with listening_port() as port:
            monkeypatch.setenv(f"{ENV_PREFIX}_REDIS_URL", f"redis://{HOST}:{port}/0")
            monkeypatch.setenv(ENV_PREFIX, "1")
            assert main(["redis"]) == 0
        assert "preflight passed" in capsys.readouterr().out

    def test_main_exits_one_when_a_service_is_missing(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """异常：服务不可达时退出码为 1，报告点出问题数。"""
        monkeypatch.setenv(f"{ENV_PREFIX}_REDIS_URL", f"redis://{HOST}:{free_port()}")
        monkeypatch.setenv(ENV_PREFIX, "1")
        assert main(["redis"]) == 1
        assert "preflight failed" in capsys.readouterr().out

    def test_main_reads_required_list_from_env(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：未传位置参数时回落到 ``MED_MEMORY_IT_REQUIRED``。"""
        with listening_port() as port:
            monkeypatch.setenv(ENV_PREFIX, "1")
            monkeypatch.setenv(f"{ENV_PREFIX}_REDIS_URL", f"redis://{HOST}:{port}/0")
            monkeypatch.setenv(ENV_REQUIRED, "redis")
            assert main([]) == 0
        assert "1 required service(s)" in capsys.readouterr().out

    def test_main_defaults_argv_to_sys_argv(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``argv`` 为 ``None`` 时读取 ``sys.argv``（真实 CLI 调用路径）。"""
        monkeypatch.setattr(sys, "argv", ["med-memory-preflight"])
        monkeypatch.delenv(ENV_REQUIRED, raising=False)
        assert main() == 0
        assert "0 required service(s)" in capsys.readouterr().out

    @pytest.mark.parametrize(("required", "expected"), [(None, 0), ("redis", 1)])
    def test_module_entry_point_really_runs(self, required: str | None, expected: int) -> None:
        """正向 / 异常：``python -m ...preflight`` 必须真的执行守卫。

        这是**子进程级**的端到端校验：模块若漏写 ``if __name__ == "__main__"``
        守卫，``-m`` 调用会「静默成功、什么都不做」——单元测试直接调 ``main()``
        永远发现不了，只有真跑一次命令行才会暴露。
        """
        project_root = Path(__file__).resolve().parents[2]
        env = {**os.environ, "PYTHONPATH": str(project_root / "src")}
        env.pop(ENV_REQUIRED, None)
        if required is not None:
            env[ENV_REQUIRED] = required
        result = subprocess.run(
            [sys.executable, "-m", PREFLIGHT_MODULE],
            capture_output=True,
            text=True,
            env=env,
            cwd=project_root,
            check=False,
        )
        assert result.returncode == expected, result.stderr
        assert "preflight" in result.stdout
