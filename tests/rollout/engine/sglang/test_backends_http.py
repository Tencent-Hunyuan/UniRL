import sys
from types import ModuleType

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
