import signal
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from unirl.rollout.engine.sglang.backends import http


def test_launch_server_owns_process_group_before_importing_runtime(monkeypatch) -> None:
    events: list[object] = []
    result = object()

    def fake_setsid() -> None:
        events.append("setsid")

    def fake_launch_server(server_args: object) -> object:
        events.append(("launch_server", server_args))
        return result

    fake_modules = {
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.entrypoints": ModuleType("sglang.srt.entrypoints"),
        "sglang.srt.entrypoints.http_server": ModuleType("sglang.srt.entrypoints.http_server"),
    }
    fake_modules["sglang.srt.entrypoints.http_server"].launch_server = fake_launch_server
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(http.os, "setsid", fake_setsid)

    server_args = object()
    assert http._launch_server_with_env(server_args, {}) is result
    assert events == ["setsid", ("launch_server", server_args)]


def test_boot_reaps_started_server_when_health_check_fails(monkeypatch) -> None:
    @dataclass
    class FakeServerArgs:
        model_path: str
        tp_size: int = 1
        port: int = 30000
        nccl_port: int = 30001

    processes = []

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.pid = 4242
            self.started = False
            processes.append(self)

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return True

    cleaned = []
    monkeypatch.setattr(http, "_import_sglang_runtime", lambda: {"ServerArgs": FakeServerArgs})
    monkeypatch.setattr(http.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(http.multiprocessing, "Process", FakeProcess)

    def fail_health_check(*args, **kwargs) -> None:
        del args, kwargs
        raise TimeoutError("boot timeout")

    monkeypatch.setattr(http, "wait_server_healthy", fail_health_check)
    monkeypatch.setattr(http, "_terminate_server_process", lambda process: cleaned.append(process))

    with pytest.raises(TimeoutError, match="boot timeout"):
        http.HTTPBackend.boot(
            {
                "model_path": "dummy",
                "tp_size": 1,
                "port": 30000,
                "nccl_port": 30001,
            },
            advertise_host="127.0.0.1",
            concurrency=1,
        )

    assert len(processes) == 1
    assert processes[0].started
    assert cleaned == processes


def test_process_tree_signal_does_not_fan_out_before_setsid(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(http.os, "getpgid", lambda pid: 7)
    monkeypatch.setattr(http.os, "kill", lambda pid, sig: sent.append(("pid", pid, sig)))
    monkeypatch.setattr(http.os, "killpg", lambda pgid, sig: sent.append(("group", pgid, sig)))

    http._signal_process_tree(42, signal.SIGTERM)

    assert sent == [("pid", 42, signal.SIGTERM)]


def test_process_tree_signal_fans_out_after_setsid(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(http.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(http.os, "kill", lambda pid, sig: sent.append(("pid", pid, sig)))
    monkeypatch.setattr(http.os, "killpg", lambda pgid, sig: sent.append(("group", pgid, sig)))

    http._signal_process_tree(42, signal.SIGTERM)

    assert sent == [("group", 42, signal.SIGTERM)]


def test_process_tree_signal_reaps_group_after_session_leader_exits(monkeypatch) -> None:
    sent = []

    def missing_leader(pid):
        del pid
        raise ProcessLookupError

    monkeypatch.setattr(http.os, "getpgid", missing_leader)
    monkeypatch.setattr(http.os, "kill", lambda pid, sig: sent.append(("pid", pid, sig)))
    monkeypatch.setattr(http.os, "killpg", lambda pgid, sig: sent.append(("group", pgid, sig)))

    http._signal_process_tree(42, signal.SIGTERM)

    assert sent == [("group", 42, signal.SIGTERM)]


def test_terminate_server_process_escalates_when_sigterm_is_ignored(monkeypatch) -> None:
    class HungProcess:
        pid = 42

        def __init__(self) -> None:
            self.join_timeouts = []

        def join(self, timeout) -> None:
            self.join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return True

    process = HungProcess()
    term = []
    forced = []
    monkeypatch.setattr(http, "kill_process_tree", lambda pid: term.append(pid))
    monkeypatch.setattr(http, "_signal_process_tree", lambda pid, sig: forced.append((pid, sig)))

    http._terminate_server_process(process, timeout_s=0.25)

    assert term == [42]
    assert forced == [(42, signal.SIGKILL)]
    assert process.join_timeouts == [0.25, 1.0]
