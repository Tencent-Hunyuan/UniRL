"""Fail-closed compatibility checks for the FastVideo runtime patch suite."""

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import fields, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

_ALLOWED_COMMITS = {
    # hao-ai-lab/FastVideo PR #1222 source snapshot pinned by pyproject.toml.
    "2095477eac7e289c7a7ab13acb367ca60687c304",
    # Temporary migration compatibility for Zcchill/FastVideo PR #1-#3 main.
    "e7fafe6d4cfcb6a34735a8d6b767b27b8de09465",
}


class FastVideoCompatibilityError(RuntimeError):
    """Raised when the installed FastVideo cannot be patched safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FastVideoCompatibilityError(message)


def _require_parameters(callable_obj: Any, *, symbol: str, names: set[str]) -> None:
    try:
        actual = set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError) as exc:
        raise FastVideoCompatibilityError(f"cannot inspect FastVideo symbol {symbol}: {exc}") from exc
    missing = names - actual
    _require(not missing, f"incompatible FastVideo {symbol}: missing parameters {sorted(missing)}")


def _installed_commit() -> str | None:
    """Resolve VCS provenance from PEP 610 metadata or a source checkout."""

    try:
        direct_url = metadata.distribution("fastvideo").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        payload = json.loads(direct_url)
        commit = (payload.get("vcs_info") or {}).get("commit_id")
        if commit:
            return str(commit).lower()

    import fastvideo

    package_file = Path(fastvideo.__file__).resolve()
    for parent in package_file.parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip().lower()
    return None


def verify_fastvideo_compatibility() -> None:
    """Verify the narrow upstream seams used by UniRL before patching.

    FastVideo does not currently expose a stable plugin API or a reliable
    package version for these RL seams. We therefore validate symbols,
    dataclass fields, signatures, and the denoising source markers that the
    around-patch relies on. Any mismatch aborts engine boot.
    """

    commit = _installed_commit()
    _require(
        commit in _ALLOWED_COMMITS,
        "FastVideo provenance is not in UniRL's verified allowlist: "
        f"commit={commit!r}, allowed={sorted(_ALLOWED_COMMITS)}. "
        "Install the `fastvideo` extra or use an explicitly verified checkout.",
    )

    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages import denoising
    from fastvideo.worker.multiproc_executor import MultiprocExecutor, WorkerMultiprocProc

    _require(is_dataclass(ForwardBatch), "incompatible FastVideo ForwardBatch: expected dataclass")
    _require(is_dataclass(ForwardBatch.RLData), "incompatible FastVideo ForwardBatch.RLData: expected dataclass")
    rl_fields = {field.name for field in fields(ForwardBatch.RLData)}
    _require(
        {"enabled", "collect_log_probs", "store_trajectory", "keep_trajectory_on_cpu"} <= rl_fields,
        f"incompatible FastVideo RLData fields: {sorted(rl_fields)}",
    )

    _require_parameters(
        denoising.sde_step_with_logprob,
        symbol="denoising.sde_step_with_logprob",
        names={"scheduler", "model_output", "timestep", "sample"},
    )
    _require_parameters(
        denoising.DenoisingStage.forward,
        symbol="DenoisingStage.forward",
        names={"self", "batch", "fastvideo_args"},
    )
    try:
        forward_source = inspect.getsource(denoising.DenoisingStage.forward)
    except (OSError, TypeError) as exc:
        raise FastVideoCompatibilityError(f"cannot inspect FastVideo DenoisingStage.forward: {exc}") from exc
    for marker in ("rl_data", "sde_step_with_logprob", "scheduler.step"):
        _require(
            marker in forward_source,
            f"incompatible FastVideo DenoisingStage.forward: required source marker {marker!r} is absent",
        )

    _require_parameters(
        MultiprocExecutor.execute_forward,
        symbol="MultiprocExecutor.execute_forward",
        names={"self", "forward_batch", "fastvideo_args"},
    )
    _require_parameters(
        WorkerMultiprocProc.make_worker_process,
        symbol="WorkerMultiprocProc.make_worker_process",
        names={"fastvideo_args", "local_rank", "rank", "distributed_init_method"},
    )
    _require(
        callable(getattr(VideoGenerator, "from_fastvideo_args", None)), "FastVideo VideoGenerator boot API is absent"
    )


def verify_fastvideo_capabilities() -> None:
    """Assert every correctness-critical UniRL capability is installed."""

    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages import denoising
    from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage
    from fastvideo.worker.multiproc_executor import MultiprocExecutor, WorkerMultiprocProc

    rl_fields = {field.name for field in fields(ForwardBatch.RLData)}
    _require(
        {"sde_step_indices", "sde_type"} <= rl_fields,
        f"FastVideo RLData patch incomplete: fields are {sorted(rl_fields)}",
    )
    _require(
        bool(getattr(denoising.sde_step_with_logprob, "_unirl_fastvideo_sde", False)),
        "FastVideo denoising SDE patch was not installed",
    )
    _require(
        bool(getattr(denoising.DenoisingStage.forward, "_unirl_fastvideo_denoising", False)),
        "FastVideo denoising context patch was not installed",
    )
    _require(
        bool(getattr(TimestepPreparationStage.forward, "_unirl_fastvideo_timesteps", False)),
        "FastVideo custom-sigma timestep patch was not installed",
    )
    _require(
        bool(getattr(MultiprocExecutor.execute_forward, "_unirl_fastvideo_conditions", False)),
        "FastVideo worker response patch was not installed",
    )
    _require(
        callable(getattr(VideoGenerator, "update_transformer_weights_from_path", None)),
        "FastVideo full-weight update entrypoint is absent",
    )
    _require(
        bool(getattr(WorkerMultiprocProc.worker_main, "_unirl_fastvideo_spawn", False)),
        "FastVideo spawn propagation patch was not installed",
    )
    _require(
        bool(getattr(MultiprocExecutor._init_executor, "_unirl_fastvideo_port_retry", False)),
        "FastVideo TCPStore startup retry patch was not installed",
    )


__all__ = [
    "FastVideoCompatibilityError",
    "verify_fastvideo_capabilities",
    "verify_fastvideo_compatibility",
]
