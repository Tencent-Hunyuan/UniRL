"""Fail-fast runtime contracts for Qwen3.5 AR training."""

from __future__ import annotations

from importlib import metadata
from typing import Any

from packaging.version import InvalidVersion, Version

from unirl.config.require import require

_FSDP2_BACKEND_SUFFIXES = (".FSDPBackend", ".VeOmniBackend")
_SGLANG_ROLLOUT_SUFFIX = ".SGLangRolloutEngine"
_MIN_TRANSFORMERS = Version("5.0.0")
_MIN_SP_TRANSFORMERS = Version("5.9.0")
_MIN_FLASH_LINEAR_ATTENTION = Version("0.4.2")
_MIN_SGLANG = Version("0.5.12.post1")
_MISSING = object()


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _target(cfg: Any) -> str:
    return str(_get(cfg, "_target_", "") or "")


def _installed_version(distribution: str) -> Version:
    try:
        raw = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Qwen3.5 requires {distribution!r}, but it is not installed in the training environment."
        ) from exc
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise RuntimeError(f"Qwen3.5 could not parse installed {distribution} version {raw!r}.") from exc


def _require_version(distribution: str, minimum: Version, *, reason: str) -> None:
    installed = _installed_version(distribution)
    if installed < minimum:
        raise RuntimeError(f"Qwen3.5 {reason} requires {distribution}>={minimum} (installed: {installed}).")


def is_qwen3_5_pipeline_config(pipeline_cfg: Any) -> bool:
    """Whether a Hydra pipeline config targets the Qwen3.5 package."""
    return "unirl.models.qwen3_5." in _target(pipeline_cfg)


def validate_qwen3_5_training_contract(
    *,
    pipeline_cfg: Any,
    backend_cfg: Any,
    rollout_cfg: Any,
    stack_cfg: Any = None,
) -> None:
    """Validate only the Qwen3.5 FSDP2/VeOmni + SGLang execution path.

    The check runs on the driver before any model or rollout actor is created,
    so prompt-policy and dependency mistakes fail before allocating GPUs.
    Other model families are untouched.
    """
    if not is_qwen3_5_pipeline_config(pipeline_cfg):
        return

    backend_target = _target(backend_cfg)
    require(
        backend_target.endswith(_FSDP2_BACKEND_SUFFIXES),
        "Qwen3.5 training is supported only on the FSDP2 backends "
        "(unirl.train.backend.fsdp.FSDPBackend or VeOmniBackend); "
        f"got backend._target_={backend_target!r}.",
    )

    _require_version("transformers", _MIN_TRANSFORMERS, reason="model loading")

    fsdp_cfg = _get(backend_cfg, "fsdp_cfg", {})
    sp_size = int(_get(fsdp_cfg, "sp_size", 1) or 1)
    if backend_target.endswith(".VeOmniBackend") and sp_size > 1:
        _require_version(
            "transformers",
            _MIN_SP_TRANSFORMERS,
            reason=f"VeOmni sequence parallelism (sp_size={sp_size})",
        )
        _require_version(
            "flash-linear-attention",
            _MIN_FLASH_LINEAR_ATTENTION,
            reason=f"GDN sequence parallelism (sp_size={sp_size})",
        )

    # TokenBudgetPlanner only changes CPU-side micro-batch grouping for UniRL's
    # dense replay path; it does not enable Transformers padding_free / THD packed
    # attention. Keep the >=5.9 guard only for true VeOmni sequence parallelism
    # above, where Qwen3.5 GDN sequence-parallel kernels have a validated floor.
    micro_planner_target = _target(_get(stack_cfg, "micro_planner", {}))
    require(
        not micro_planner_target.endswith(".TokenBudgetPlanner") or sp_size == 1,
        "Qwen3.5 TokenBudgetPlanner is validated only with sp_size=1 in this path; "
        "VeOmni sequence parallelism keeps its transformers>=5.9.0 guard.",
    )

    rollout_target = _target(rollout_cfg)
    if not rollout_target.endswith(_SGLANG_ROLLOUT_SUFFIX):
        return

    _require_version("sglang", _MIN_SGLANG, reason="SGLang rollout")

    engine_cfg = _get(rollout_cfg, "config", {})
    chat_kwargs = _get(engine_cfg, "chat_template_kwargs", None)
    rollout_thinking = _get(chat_kwargs, "enable_thinking", _MISSING)
    require(
        rollout_thinking is not _MISSING,
        "Qwen3.5 SGLang rollout must set "
        "rollout.config.chat_template_kwargs.enable_thinking explicitly so it can be checked "
        "against pipeline.enable_thinking.",
    )
    require(
        isinstance(rollout_thinking, bool),
        f"Qwen3.5 rollout.config.chat_template_kwargs.enable_thinking must be a bool; got {rollout_thinking!r}.",
    )
    train_thinking = _get(pipeline_cfg, "enable_thinking", False)
    require(
        isinstance(train_thinking, bool),
        f"Qwen3.5 pipeline.enable_thinking must be a bool; got {train_thinking!r}.",
    )
    require(
        train_thinking == rollout_thinking,
        "Qwen3.5 prompt mismatch: pipeline.enable_thinking="
        f"{train_thinking!r} but rollout.config.chat_template_kwargs.enable_thinking="
        f"{rollout_thinking!r}. Rollout/train prompt IDs would diverge and corrupt the GRPO ratio.",
    )

    forbidden_tokens = set(_get(engine_cfg, "response_forbidden_tokens", None) or [])
    required_placeholders = {"<|image_pad|>", "<|video_pad|>"}
    require(
        required_placeholders.issubset(forbidden_tokens),
        "Qwen3.5 SGLang rollout must exclude multimodal placeholders from "
        "response sampling: set rollout.config.response_forbidden_tokens to "
        f"{sorted(required_placeholders)}. Replaying a placeholder as a text "
        "action is undefined.",
    )


__all__ = ["is_qwen3_5_pipeline_config", "validate_qwen3_5_training_contract"]
