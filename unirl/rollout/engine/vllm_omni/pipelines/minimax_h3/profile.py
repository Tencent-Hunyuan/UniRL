"""Env-gated rollout profiling for the H3 worker — inert unless ``H3_ROLLOUT_PROFILE_DIR`` is set."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

import torch


def _local_ip() -> str:
    """Best-effort host address used only to name trace files."""
    try:
        out = subprocess.run(["hostname", "-i"], capture_output=True, text=True, timeout=5).stdout
        parts = out.split()
        if parts:
            return parts[-1]
    except Exception:
        pass
    try:
        import socket

        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


class RolloutProfile:
    """Chrome trace plus CUDA memory history for the first few requests on selected hosts/GPUs."""

    def __init__(self, out_dir: str, host_ip: str, gpu_id: int, max_requests: int) -> None:
        self.out_dir = out_dir
        self.host_ip = host_ip
        self.gpu_id = gpu_id
        self.max_requests = max_requests
        self.requests_done = 0
        self.memory_history_on = False
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def from_env() -> Optional["RolloutProfile"]:
        """Build from the ``H3_ROLLOUT_PROFILE_*`` variables, or ``None`` when unset."""
        out_dir = os.environ.get("H3_ROLLOUT_PROFILE_DIR") or ""
        if not out_dir:
            return None
        first = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        if not first.isdigit():
            return None
        gpu_id = int(first)
        if gpu_id >= int(os.environ.get("H3_ROLLOUT_PROFILE_MAX_GPU", "4")):
            return None
        host_ip = _local_ip()
        want_host = os.environ.get("H3_ROLLOUT_PROFILE_HOST") or ""
        if want_host and want_host != host_ip:
            return None
        max_requests = int(os.environ.get("H3_ROLLOUT_PROFILE_REQUESTS", "1"))
        return RolloutProfile(out_dir, host_ip, gpu_id, max_requests)

    def enable_memory_history(self) -> None:
        """Start recording allocations; call before weights load so the snapshot attributes them."""
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=400000,
                clear_history=True,
            )
            self.memory_history_on = True
        except Exception as exc:
            print(f"[h3-profile] memory history enable failed on {self.tag()}: {exc}", flush=True)

    def tag(self) -> str:
        """Per-worker file tag."""
        # The engine's workers share one multi-GPU CUDA_VISIBLE_DEVICES, so the env-derived
        # gpu_id collides across them; the pid and selected device disambiguate.
        try:
            dev = torch.cuda.current_device()
        except Exception:
            dev = self.gpu_id
        return f"{self.host_ip}_pid{os.getpid()}_dev{dev}"

    def active(self) -> bool:
        """True while requests remain within the configured budget."""
        return self.requests_done < self.max_requests

    def request_profiler(self) -> Any:
        """A ``torch.profiler`` context for one request."""
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            on_trace_ready=self._export_trace,
        )

    def _export_trace(self, prof: Any) -> None:
        path = os.path.join(self.out_dir, f"trace_{self.tag()}_req{self.requests_done}.json")
        try:
            prof.export_chrome_trace(path)
            subprocess.Popen(["gzip", "-f", path])
            print(f"[h3-profile] trace exported: {path}.gz", flush=True)
        except Exception as exc:
            print(f"[h3-profile] trace export failed on {self.tag()}: {exc}", flush=True)

    def finish_request(self) -> None:
        """Count the request and dump the memory snapshot once the budget is spent."""
        self.requests_done += 1
        if self.requests_done >= self.max_requests and self.memory_history_on:
            path = os.path.join(self.out_dir, f"memsnap_{self.tag()}.pickle")
            try:
                torch.cuda.memory._dump_snapshot(path)
                print(f"[h3-profile] memory snapshot dumped: {path}", flush=True)
            except Exception as exc:
                print(f"[h3-profile] memory snapshot failed on {self.tag()}: {exc}", flush=True)
            finally:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception:
                    pass
                self.memory_history_on = False


__all__ = ["RolloutProfile"]
