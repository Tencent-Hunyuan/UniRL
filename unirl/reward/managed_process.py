"""Rank-affine managed image scorer in an explicit child Python environment."""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO

import requests
from omegaconf import DictConfig, OmegaConf

from unirl.config.require import require
from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


@dataclass
class ManagedProcessConfig:
    python_executable: str = ""
    service_root: str = ""
    startup_timeout: float = 1200.0
    shutdown_timeout: float = 30.0
    log_dir: str = "/tmp"
    offline: bool = True
    allow_multiple_visible_devices: bool = False


@dataclass
class ManagedScorerConfig:
    name: str = ""
    history_kind: str = "image"
    params: dict[str, Any] = field(default_factory=dict)
    version: str | None = None


@dataclass
class ManagedScorerProcessSpec(BaseRewardComponentSpec):
    process: ManagedProcessConfig = field(default_factory=ManagedProcessConfig)
    scorer: ManagedScorerConfig = field(default_factory=ManagedScorerConfig)
    client: RemoteRewardSpec = field(
        default_factory=lambda: RemoteRewardSpec(
            base_url="managed://rank-affine",
            required_rewards=("placeholder",),
        )
    )
    gpu_residency: str = "resident"
    idempotency_cache_size: int = 1024

    def __post_init__(self) -> None:
        if isinstance(self.scorer.params, DictConfig):
            resolved = OmegaConf.to_container(self.scorer.params, resolve=True)
            require(isinstance(resolved, dict), "managed scorer params must resolve to a mapping")
            self.scorer.params = dict(resolved)
        require(
            Path(self.process.python_executable).is_file(),
            f"reward python not found: {self.process.python_executable}",
        )
        require(Path(self.process.service_root).is_dir(), f"reward service root not found: {self.process.service_root}")
        require(bool(self.scorer.name.strip()), "managed scorer name must be non-empty")
        require(
            self.scorer.history_kind in {"image", "image_edit"},
            f"managed scorer history_kind must be image or image_edit; got {self.scorer.history_kind!r}",
        )
        require(
            tuple(self.client.required_rewards) == (self.scorer.name,),
            "managed single-scorer client.required_rewards must equal (scorer.name,)",
        )
        require(
            self.client.input_kind == "image",
            f"managed image scorer requires client.input_kind='image'; got {self.client.input_kind!r}",
        )
        require(self.gpu_residency in {"resident", "per_call"}, "gpu_residency must be resident or per_call")
        require(self.process.startup_timeout > 0, "startup_timeout must be positive")
        require(self.process.shutdown_timeout > 0, "shutdown_timeout must be positive")
        require(self.idempotency_cache_size > 0, "idempotency_cache_size must be positive")


def _validate_visible_device(env: dict[str, str], *, allow_multiple: bool, device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    visible = [item.strip() for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not visible:
        raise RuntimeError("managed CUDA scorer requires CUDA_VISIBLE_DEVICES to identify its rank-local device")
    if len(visible) != 1 and not allow_multiple:
        raise RuntimeError(
            f"managed scorer expected exactly one visible GPU, got CUDA_VISIBLE_DEVICES={visible!r}; "
            "set allow_multiple_visible_devices=true only for an intentional multi-GPU scorer"
        )


class ManagedScorerProcessBackend(RemoteRewardBackend):
    """Own one persistent loopback scorer child on this reward worker's GPU."""

    def __init__(self, *, config: ManagedScorerProcessSpec, base_device: str) -> None:
        self.spec = config
        self._process: subprocess.Popen[bytes] | None = None
        self._process_log: BinaryIO | None = None
        self._disposed = False
        self._atexit_registered = False
        self._residency_lock = threading.RLock()
        try:
            base_url = self._start_child()
            remote_spec = replace(
                config.client,
                base_url=base_url,
                required_rewards=(config.scorer.name,),
                input_kind="image",
                require_identity_echo=True,
                expected_scorer_version=config.scorer.version,
            )
            super().__init__(config=remote_spec, base_device=base_device)
            # per_call means "on GPU only while scoring" — including BEFORE the
            # first score call and before the child is even ready. The child is
            # started with --boot-offloaded, so it leaves the GPU (or, for
            # scorers honoring UNIRL_SCORER_BOOT_OFFLOADED, never touches it)
            # BEFORE _start_child sees it ready; a parent-side offload here
            # would reopen the bootstrap window it is meant to close.
        except Exception:
            self._stop_child()
            raise
        atexit.register(self._stop_child)
        self._atexit_registered = True

    def _child_env(self) -> dict[str, str]:
        cfg = self.spec
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = cfg.process.service_root + (
            f"{os.pathsep}{existing_pythonpath}" if existing_pythonpath else ""
        )
        env["TOKENIZERS_PARALLELISM"] = "false"
        if cfg.process.offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        env.pop("RAY_ADDRESS", None)
        env["UNIRL_REWARD_PARENT_PID"] = str(os.getpid())
        if cfg.gpu_residency == "per_call":
            # Construction-time hint: scorers that can build on CPU (e.g.
            # EditReward) read this and never touch the GPU during boot, closing
            # the bootstrap window entirely instead of merely shrinking it.
            env["UNIRL_SCORER_BOOT_OFFLOADED"] = "1"
        device = str(cfg.scorer.params.get("device", "cuda"))
        _validate_visible_device(
            env,
            allow_multiple=cfg.process.allow_multiple_visible_devices,
            device=device,
        )
        return env

    def _start_child(self) -> str:
        cfg = self.spec
        env = self._child_env()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.set_inheritable(True)
            port = int(listener.getsockname()[1])

            log_dir = Path(cfg.process.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{cfg.scorer.name}-child-{os.getpid()}-{port}.log"
            self._process_log = log_path.open("ab", buffering=0)

            command = [
                cfg.process.python_executable,
                "-m",
                "reward_service.direct_server",
                "--fd",
                str(listener.fileno()),
                "--scorer",
                cfg.scorer.name,
                "--history-kind",
                cfg.scorer.history_kind,
                "--cache-size",
                str(cfg.idempotency_cache_size),
                "--params-json",
                json.dumps(cfg.scorer.params, separators=(",", ":")),
            ]
            if cfg.gpu_residency == "per_call":
                command.append("--boot-offloaded")
            logger.info(
                "starting managed reward child scorer=%s port=%d cuda_visible=%s python=%s log=%s",
                cfg.scorer.name,
                port,
                env.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                cfg.process.python_executable,
                log_path,
            )
            self._process = subprocess.Popen(
                command,
                env=env,
                pass_fds=(listener.fileno(),),
                stdout=self._process_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        base_url = f"http://127.0.0.1:{port}"
        session = requests.Session()
        session.trust_env = False
        # A per_call child boots offloaded (--boot-offloaded): the scorer must be
        # off the GPU by the time it first reports ready, so that is the ready
        # state to wait for. Resident children report "resident" as before.
        expected_state = "offloaded" if cfg.gpu_residency == "per_call" else "resident"
        deadline = time.monotonic() + float(cfg.process.startup_timeout)
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    # A service tree predating --boot-offloaded exits at argparse
                    # ("unrecognized arguments") before ever binding the socket.
                    hint = (
                        " (an 'unrecognized arguments: --boot-offloaded' log means "
                        f"unirl-reward-service at {cfg.process.service_root!r} predates "
                        "the per_call boot protocol)"
                        if cfg.gpu_residency == "per_call"
                        else ""
                    )
                    raise RuntimeError(
                        f"reward child exited during startup with code {self._process.returncode}; log={log_path}{hint}"
                    )
                try:
                    response = session.get(f"{base_url}/health", timeout=2.0)
                    if response.status_code == 200:
                        body = response.json()
                        scorer = body.get("scorer") or {}
                        if (
                            cfg.scorer.name in dict(body.get("rewards") or {})
                            and body.get("state") == expected_state
                            and scorer.get("input_kind") == "image"
                        ):
                            if cfg.gpu_residency == "per_call" and not scorer.get("supports_offload"):
                                raise RuntimeError(
                                    f"managed scorer {cfg.scorer.name!r} does not support gpu_residency='per_call'"
                                )
                            if cfg.scorer.version is not None and scorer.get("version") != cfg.scorer.version:
                                raise RuntimeError(
                                    f"reward child scorer version {scorer.get('version')!r} "
                                    f"!= expected {cfg.scorer.version!r}"
                                )
                            logger.info("managed reward child ready at %s", base_url)
                            return base_url
                except requests.RequestException:
                    pass
                time.sleep(1.0)
        finally:
            session.close()
        self._stop_child()
        raise TimeoutError(f"reward child did not become ready within {cfg.process.startup_timeout}s; log={log_path}")

    def _post_lifecycle(self, action: str, *, timeout: float = 30.0, tolerate_failure: bool = False) -> bool:
        try:
            response = self._session.post(f"{self.base_url}/lifecycle/{action}", timeout=timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            if tolerate_failure:
                logger.warning("managed scorer lifecycle %s failed during cleanup: %s", action, exc)
                return False
            raise

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if self.spec.gpu_residency != "per_call":
            return super().compute_rewards(request)
        with self._residency_lock:
            try:
                self._post_lifecycle("onload")
                response = super().compute_rewards(request)
            except BaseException:
                # A failed onload or score can also make cleanup fail; preserve
                # the operation that caused the request to fail.
                self._post_lifecycle("offload", tolerate_failure=True)
                raise
            # The direct server reports scorer failures as HTTP 200 with
            # per-item errors. Preserve those details if cleanup also fails.
            self._post_lifecycle("offload", tolerate_failure=not all(response.successes))
            return response

    def is_available(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return False
        return super().is_available()

    def offload(self) -> None:
        with self._residency_lock:
            self._post_lifecycle("offload")

    def onload(self) -> None:
        with self._residency_lock:
            self._post_lifecycle("onload")

    def dispose(self) -> None:
        with self._residency_lock:
            if self._disposed:
                return
            self._disposed = True
            if self._atexit_registered:
                atexit.unregister(self._stop_child)
                self._atexit_registered = False
            try:
                self._post_lifecycle(
                    "shutdown",
                    timeout=float(self.spec.process.shutdown_timeout),
                    tolerate_failure=True,
                )
                super().dispose()
            finally:
                self._stop_child()

    def _stop_child(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=float(self.spec.process.shutdown_timeout))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self._process_log is not None:
            self._process_log.close()
            self._process_log = None


__all__ = [
    "ManagedProcessConfig",
    "ManagedScorerConfig",
    "ManagedScorerProcessBackend",
    "ManagedScorerProcessSpec",
]
