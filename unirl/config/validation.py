"""Shared validation helpers for component configs.

Two flavors of validator live here:

- **Per-field helpers** (e.g. :func:`validate_precision_type`) are called from
  individual ``__post_init__`` bodies so every dataclass that owns the same
  kind of field validates it the same way.
- **Cross-component validators** (``validate_weight_sync_contract``,
  ``validate_offload_contract``, ...) take the full ``cfg`` and enforce
  rules that span multiple resolved sections. They run on the driver against
  the composed ``cfg`` before Ray actors are created.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import torch
from omegaconf import DictConfig

from unirl.config.require import require
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class PrecisionName(str, Enum):
    """Canonical precision aliases accepted by config fields."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


_CANONICAL_BY_DTYPE = {
    torch.bfloat16: PrecisionName.BF16,
    torch.float16: PrecisionName.FP16,
    torch.float32: PrecisionName.FP32,
}


def validate_precision_type(value: Any, *, field: str) -> str:
    """Return the canonical precision alias (``bf16``/``fp16``/``fp32``).

    Delegates alias expansion to ``parse_torch_dtype`` so all precision fields
    accept the same inputs (``bf16``/``bfloat16``, ``fp16``/``float16``/``half``,
    ``fp32``/``float32``/``float``) and raise the same ``ValueError`` on unknown
    names. Caller supplies ``field`` for error-message attribution.
    """
    dtype = parse_torch_dtype(value, field_name=field)
    return _CANONICAL_BY_DTYPE[dtype].value


_SGLANG_ENGINE_TARGET_SUFFIX = "SGLangRolloutEngine"
_VLLM_OMNI_ENGINE_TARGET_SUFFIX = "VLLMOmniRolloutEngine"
_TRAINSIDE_ENGINE_TARGET_SUFFIX = "TrainsideRolloutEngine"
_DIRECT_SAMPLING_ENGINE_SUFFIXES: tuple = (_TRAINSIDE_ENGINE_TARGET_SUFFIX,)
# Sync handlers that only one engine implements. Listed here so the validator
# can fail fast on a mismatched pairing. UpdateWeightFromTensor /
# UpdateWeightFromDistributed work on BOTH sglang and vllm-omni — they're
# transport-shape contracts, not engine-specific (vllm-omni's receivers live
# in unirl.rollout.engine.vllm_omni.weight_sync.{ipc,nccl}_receive_mixin).
_IPC_SYNC_SUFFIXES = frozenset({"UpdateWeightFromIPC"})  # vllm-omni only


def is_direct_sampling(cfg: DictConfig) -> bool:
    """Training-actor-sampling mode is derived from the selected engine.

    ``rollout/engine: trainside`` → ``TrainsideRolloutEngine`` (the
    in-process Pipeline adapter; see ``unirl/rollout/engine/trainside``)
    is the only direct-sampling engine. All other engines (sglang, vllm-omni)
    run dedicated rollout actors.
    """
    target = str(cfg.rollout.engine.get("_target_") or "")
    return target.endswith(_DIRECT_SAMPLING_ENGINE_SUFFIXES)


def validate_dynamic_dotpaths(cfg: DictConfig) -> None:
    """Fail-fast import of every dynamic dotpath the driver will later resolve."""
    from unirl.utils import load_function

    dotpath = str(cfg.run.data_source_dotpath or "").strip()
    require(
        bool(dotpath), f"cfg.run.data_source_dotpath must be a non-empty dotpath; got {cfg.run.data_source_dotpath!r}"
    )
    try:
        load_function(dotpath)
    except Exception as exc:
        raise ValueError(f"cfg.run.data_source_dotpath={dotpath!r} failed to import: {exc}") from exc


def validate_training_batch_geometry(cfg: DictConfig) -> None:
    """Cross-section: training plan's global batch size must divide by DP sizes.

    ``dp_size`` is ``Optional[int]`` on ``TrainTopology``; ``None`` means
    "derive from ``dist.get_world_size()`` at runtime" and is not checkable
    at cfg time.
    """
    global_batch = int(cfg.training.plan.global_batch_size)
    raw_dp_size = cfg.training.topology.dp_size
    dp_replicate_size = int(cfg.training.topology.dp_replicate_size)
    if raw_dp_size is not None:
        dp_size = int(raw_dp_size)
        require(
            dp_size <= 0 or global_batch % dp_size == 0,
            f"cfg.training.plan.global_batch_size ({global_batch}) must be divisible by cfg.training.topology.dp_size ({dp_size})",
        )
    require(
        dp_replicate_size <= 0 or global_batch % dp_replicate_size == 0,
        f"cfg.training.plan.global_batch_size ({global_batch}) must be divisible by cfg.training.topology.dp_replicate_size ({dp_replicate_size})",
    )


def validate_weight_sync_contract(cfg: DictConfig) -> None:
    """Weight-sync section presence + variant must match rollout engine."""
    has_sync = cfg.get("sync") is not None
    is_direct = is_direct_sampling(cfg)
    require(
        not is_direct or not has_sync,
        f"direct_sampling mode forbids a sync section; got cfg.sync={cfg.get('sync')!r}",
    )
    require(
        is_direct or has_sync,
        "dedicated-rollout mode (rollout/engine=sglang or vllm_omni) requires a sync variant; got no sync section",
    )
    if has_sync:
        sync_target = str(cfg.sync.get("_target_") or "")
        sync_name = sync_target.rsplit(".", 1)[-1]
        if sync_name in _IPC_SYNC_SUFFIXES:
            engine_target = str(cfg.rollout.engine.get("_target_") or "")
            require(
                engine_target.endswith(_VLLM_OMNI_ENGINE_TARGET_SUFFIX),
                f"sync={sync_name} (bucketed CUDA-IPC) is only implemented by the "
                f"vllm-omni rollout engine; got rollout.engine._target_={engine_target!r}",
            )


def validate_rollout_layout(cfg: DictConfig) -> None:
    """Multi-GPU colocated rollout requires the sglang engine."""
    num_gpus_per_actor = int(cfg.placement.num_rollout_gpus_per_actor)
    if bool(cfg.placement.colocate) and num_gpus_per_actor > 1:
        engine_target = str(cfg.rollout.engine.get("_target_") or "")
        require(
            engine_target.endswith(_SGLANG_ENGINE_TARGET_SUFFIX),
            f"multi-GPU colocated rollout (num_rollout_gpus_per_actor={num_gpus_per_actor}, colocate=True) requires the sglang engine; got rollout.engine._target_={engine_target!r}",
        )


def validate_offload_contract(cfg: DictConfig) -> None:
    """Direct-sampling mode forbids GPU offloading (there is no paired rollout actor)."""
    if not is_direct_sampling(cfg):
        return
    require(
        not bool(cfg.training.execution.offload_train),
        "direct_sampling mode is incompatible with cfg.training.execution.offload_train=True",
    )
    require(
        not bool(cfg.training.execution.offload_rollout),
        "direct_sampling mode is incompatible with cfg.training.execution.offload_rollout=True",
    )


def validate_keep_local_contract(cfg: DictConfig) -> None:
    """Keep-local data plane is direct-sampling-only and excludes TransferQueue.

    ``cfg.training.execution.keep_local=True`` makes each train actor cache the
    rollout it produced and train on it in place, so heavy tensors never reach the
    driver. That requires producer==consumer (direct sampling), and is mutually
    exclusive with TransferQueue — the other off-driver data plane.

    It is byte-equivalent to the gathered path only when the rollout's prompt
    groups divide evenly across the train actors (enforced below); otherwise the
    per-actor partition — and hence the FSDP-averaged gradient — differs, so
    keep-local would be a distinct training run rather than a transparent
    optimization.
    """
    if not bool(cfg.training.execution.get("keep_local", False)):
        return
    require(
        is_direct_sampling(cfg),
        "cfg.training.execution.keep_local=True requires direct sampling "
        "(rollout/engine=trainside): in separate-sampling mode the rollout "
        "producer is not the train consumer, so payloads cannot stay local.",
    )
    require(
        cfg.get("transfer_queue") is None,
        "cfg.training.execution.keep_local=True is mutually exclusive with "
        "transfer_queue (both move data off the driver); enable exactly one.",
    )
    # Keep-local shards each rollout by prompt-group across the train actors;
    # the gathered path instead re-balances
    # samples evenly on the driver. The two partitions — and thus each rank's
    # mean loss and the FSDP-averaged gradient — coincide only when the prompt
    # groups split evenly across actors, so require that here.
    actor_count = cfg.training.topology.get("actor_count", None)
    if actor_count is not None:
        n = int(actor_count)
        prompts = int(getattr(cfg.algorithm, "prompts_per_rollout", 1))
        require(
            n <= 0 or prompts % n == 0,
            "cfg.training.execution.keep_local=True requires "
            f"cfg.algorithm.prompts_per_rollout ({prompts}) divisible by "
            f"cfg.training.topology.actor_count ({n}): keep-local shards rollouts "
            "by prompt-group across train actors, so an indivisible split gives "
            "unequal per-actor batches and a different gradient than the gathered "
            "path.",
        )


def validate_lora_target_modules(cfg: DictConfig) -> None:
    """Materialize ``cfg.model.lora_target_modules`` from the bundle's class default.

    When LoRA is requested but no explicit target list was supplied, resolve the
    model class via ``cfg.model._target_`` and call its
    ``default_lora_target_modules()`` classmethod. Mutates ``cfg.model`` in
    place (the model config is registered ``mutable=True``) so PEFT (training
    side) and SGLang ``ServerArgs.lora_target_modules`` (rollout side) see the
    same list. Without this materializer, PEFT injects LoRA into a model-class
    default subset while SGLang receives ``None`` and wraps every linear layer,
    producing a wall of "LoRA adapter None does not contain the weights for layer ..."
    warnings and silently disabling LoRA on unmatched layers.

    Priority: explicit ``cfg.model.lora_target_modules`` > model class default
    > ``None`` (warn).
    """
    if not bool(cfg.model.get("use_lora", False)):
        return
    if cfg.model.get("lora_target_modules") is not None:
        return

    target_dotpath = str(cfg.model.get("_target_") or "")
    if not target_dotpath:
        return

    try:
        from unirl.utils.misc import load_function

        model_cls = load_function(target_dotpath)
    except (ImportError, AttributeError, KeyError, ValueError) as exc:
        logger.debug(
            "Could not resolve model class %r for LoRA target lookup: %s",
            target_dotpath,
            exc,
        )
        return

    fn = getattr(model_cls, "default_lora_target_modules", None)
    if not callable(fn):
        return

    try:
        resolved = fn()
    except (TypeError, NotImplementedError) as exc:
        logger.warning(
            "Model class %s.default_lora_target_modules() raised %s; "
            "falling back to None (SGLang will wrap every linear layer).",
            model_cls.__name__,
            exc,
        )
        return

    if resolved is None:
        logger.warning(
            "%s.default_lora_target_modules() returned None and no explicit "
            "cfg.model.lora_target_modules was provided; SGLang will wrap every "
            "linear layer and silently disable LoRA on unmatched ones.",
            model_cls.__name__,
        )
        return
    if not isinstance(resolved, (list, tuple)) or not resolved:
        logger.warning(
            "%s.default_lora_target_modules() returned %r; expected a non-empty list. Falling back to None.",
            model_cls.__name__,
            resolved,
        )
        return

    materialised = [str(item) for item in resolved]
    cfg.model.lora_target_modules = materialised
    logger.info(
        "LoRA target modules materialised from %s.default_lora_target_modules(): %s",
        model_cls.__name__,
        materialised,
    )


def validate_multi_track_mini_batch_geometry(cfg: DictConfig) -> None:
    """Multi-track mini-batching requires per-actor sample counts divisible by num_updates.

    In a multi-track PE joint setup (ar + diffusion), the train actor splits the
    rollout response into ``num_updates_per_batch`` mini-batches along the root
    track (ar).  The root track's per-actor batch size is
    ``P * N / actor_count`` and must divide evenly by ``num_updates_per_batch``;
    otherwise the lineage-aware split cannot produce equal-sized chunks.

    Skips validation when:
    - ``num_updates_per_batch <= 1`` (no splitting)
    - ``cfg.training.tracks`` is absent or has <= 1 track (single-track mode)
    """
    num_updates = int(cfg.training.plan.get("num_updates_per_batch", 1))
    if num_updates <= 1:
        return

    tracks = cfg.training.get("tracks")
    if tracks is None or len(tracks) <= 1:
        return

    # Compute root (ar) track per-actor batch size: P * N / actor_count.
    P = int(cfg.algorithm.get("prompts_per_rollout", 1))
    N = int(cfg.algorithm.get("pe_rewrites_per_prompt", 1))
    actor_count = int(cfg.training.topology.get("actor_count", 1))

    ar_per_actor = (P * N) // max(actor_count, 1)
    require(
        ar_per_actor > 0,
        f"Multi-track mini-batch validation: P*N/actor_count = {P}*{N}/{actor_count} = {ar_per_actor} must be > 0.",
    )
    require(
        ar_per_actor % num_updates == 0,
        f"Multi-track mini-batch geometry: root track (ar) per-actor batch size = "
        f"P*N/actor_count = {P}*{N}/{actor_count} = {ar_per_actor}, which is not "
        f"divisible by num_updates_per_batch={num_updates}. "
        f"Adjust prompts_per_rollout, pe_rewrites_per_prompt, actor_count, or "
        f"num_updates_per_batch to satisfy ar_per_actor % num_updates == 0.",
    )

    # Also check diffusion track: P * N * M / actor_count.
    M = int(cfg.algorithm.get("samples_per_prompt", 1))
    diff_per_actor = (P * N * M) // max(actor_count, 1)
    require(
        diff_per_actor % num_updates == 0,
        f"Multi-track mini-batch geometry: diffusion track per-actor batch size = "
        f"P*N*M/actor_count = {P}*{N}*{M}/{actor_count} = {diff_per_actor}, which is not "
        f"divisible by num_updates_per_batch={num_updates}. "
        f"Adjust samples_per_prompt or num_updates_per_batch.",
    )


__all__ = [
    "PrecisionName",
    "is_direct_sampling",
    "validate_dynamic_dotpaths",
    "validate_lora_target_modules",
    "validate_multi_track_mini_batch_geometry",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_rollout_layout",
    "validate_training_batch_geometry",
    "validate_weight_sync_contract",
]
