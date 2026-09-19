"""Dependency-free contracts spanning sections of a composed Hydra recipe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from unirl.config.require import require

logger = logging.getLogger(__name__)

ENGINE_PACKAGE = "unirl.rollout.engine."

SYNC_VIA_IPC = "update_weights_from_ipc"
SYNC_VIA_TENSOR = "update_weights_from_tensor"
SYNC_VIA_NCCL = "init_weights_update_group"
SYNC_VIA_NCCL_UPDATE = "update_weights_from_distributed"
SYNC_VIA_LORA = "set_lora_from_tensors"
SYNC_VIA_LORA_COPY = "set_lora_from_tensors_copy"
SYNC_VIA_CHECKPOINT = "update_weights_from_path"
SYNC_VIA_CHECKPOINT_ENGINE_IPC = "update_weights_from_checkpoint_engine_ipc"
SYNC_MARK_CHECKPOINT_ENGINE_FAILURE = "mark_checkpoint_engine_sync_failed"
SYNC_VERIFY_TOPOLOGY = "tp_per_stage"
SYNC_VERIFY_LORA = "loaded_lora_checksums"
ENGINE_SHUTDOWN = "shutdown"

ENGINE_CAPABILITY_METHODS = (
    SYNC_VIA_IPC,
    SYNC_VIA_TENSOR,
    SYNC_VIA_NCCL,
    SYNC_VIA_NCCL_UPDATE,
    SYNC_VIA_LORA,
    SYNC_VIA_LORA_COPY,
    SYNC_VIA_CHECKPOINT,
    SYNC_VIA_CHECKPOINT_ENGINE_IPC,
    SYNC_MARK_CHECKPOINT_ENGINE_FAILURE,
    SYNC_VERIFY_TOPOLOGY,
    SYNC_VERIFY_LORA,
    ENGINE_SHUTDOWN,
)

SYNC_LOCAL = "local"
SYNC_REMOTE = "remote"

ENTRYPOINT_AR = "train_ar"
ENTRYPOINT_ASYNC_AR = "train_async_ar"
ENTRYPOINT_DIFFUSION = "train_diffusion"
ENTRYPOINT_ASYNC_DIFFUSION = "train_async_diffusion"
ENTRYPOINT_PE = "train_pe"
ENTRYPOINT_SFT = "train_sft"
ENTRYPOINT_UNIFIED = "train_unified_model"
ENTRYPOINT_AGENTIC = "train_agentic"

SAMPLING_AR = "ar"
SAMPLING_DIFFUSION = "diffusion"
AR_SAMPLING_SUFFIXES = ("ARSamplingParams",)
DIFFUSION_SAMPLING_SUFFIXES = ("DiffusionSamplingParams", "DiffusionParams")

KNOWN_ENTRYPOINTS = frozenset(
    {
        ENTRYPOINT_AR,
        ENTRYPOINT_ASYNC_AR,
        ENTRYPOINT_DIFFUSION,
        ENTRYPOINT_ASYNC_DIFFUSION,
        ENTRYPOINT_PE,
        ENTRYPOINT_SFT,
        ENTRYPOINT_UNIFIED,
        ENTRYPOINT_AGENTIC,
    }
)
LAYOUT_ENTRYPOINTS = frozenset({ENTRYPOINT_DIFFUSION})
ENTRYPOINT_SAMPLING_DOMAINS = {
    ENTRYPOINT_AR: SAMPLING_AR,
    ENTRYPOINT_ASYNC_AR: SAMPLING_AR,
    ENTRYPOINT_DIFFUSION: SAMPLING_DIFFUSION,
    ENTRYPOINT_ASYNC_DIFFUSION: SAMPLING_DIFFUSION,
    ENTRYPOINT_AGENTIC: SAMPLING_AR,
}


@dataclass(frozen=True)
class EngineFamily:
    """Recipe-relevant capabilities shared by one rollout-engine package."""

    direct_sampling: bool
    entrypoints: frozenset[str]
    capabilities: frozenset[str]


_BASE_ENGINE_CAPABILITIES = frozenset({ENGINE_SHUTDOWN})
_IN_MEMORY_CAPABILITIES = frozenset(
    {
        *_BASE_ENGINE_CAPABILITIES,
        SYNC_VIA_IPC,
        SYNC_VIA_TENSOR,
        SYNC_VIA_NCCL,
        SYNC_VIA_NCCL_UPDATE,
        SYNC_VIA_LORA,
    }
)
_AR_ENTRYPOINTS = frozenset({ENTRYPOINT_AR, ENTRYPOINT_ASYNC_AR})
_DIFFUSION_ENTRYPOINTS = frozenset({ENTRYPOINT_DIFFUSION, ENTRYPOINT_ASYNC_DIFFUSION})

ENGINE_FAMILIES: Mapping[str, EngineFamily] = {
    "trainside": EngineFamily(
        direct_sampling=True,
        entrypoints=frozenset({ENTRYPOINT_AR, ENTRYPOINT_DIFFUSION, ENTRYPOINT_PE, ENTRYPOINT_UNIFIED}),
        capabilities=_BASE_ENGINE_CAPABILITIES,
    ),
    "sglang": EngineFamily(
        direct_sampling=False,
        entrypoints=_AR_ENTRYPOINTS,
        capabilities=frozenset(
            {
                *_BASE_ENGINE_CAPABILITIES,
                SYNC_VIA_TENSOR,
                SYNC_VIA_NCCL,
                SYNC_VIA_NCCL_UPDATE,
                SYNC_VIA_LORA,
                SYNC_VIA_CHECKPOINT_ENGINE_IPC,
                SYNC_MARK_CHECKPOINT_ENGINE_FAILURE,
            }
        ),
    ),
    "sglang_diffusion": EngineFamily(
        direct_sampling=False,
        entrypoints=_DIFFUSION_ENTRYPOINTS,
        capabilities=frozenset(
            {
                *_BASE_ENGINE_CAPABILITIES,
                SYNC_VIA_TENSOR,
                SYNC_VIA_NCCL,
                SYNC_VIA_NCCL_UPDATE,
                SYNC_VIA_LORA,
            }
        ),
    ),
    "vllm_omni": EngineFamily(
        direct_sampling=False,
        entrypoints=frozenset({*_AR_ENTRYPOINTS, *_DIFFUSION_ENTRYPOINTS, ENTRYPOINT_UNIFIED}),
        capabilities=frozenset({*_IN_MEMORY_CAPABILITIES, SYNC_VIA_LORA_COPY, SYNC_VERIFY_TOPOLOGY, SYNC_VERIFY_LORA}),
    ),
    "composed": EngineFamily(
        direct_sampling=False,
        entrypoints=frozenset({ENTRYPOINT_PE}),
        capabilities=_IN_MEMORY_CAPABILITIES,
    ),
    "agentic": EngineFamily(
        direct_sampling=False,
        entrypoints=frozenset({ENTRYPOINT_AGENTIC}),
        capabilities=_IN_MEMORY_CAPABILITIES,
    ),
    "fastvideo": EngineFamily(
        direct_sampling=False,
        entrypoints=frozenset({ENTRYPOINT_DIFFUSION}),
        capabilities=frozenset({*_BASE_ENGINE_CAPABILITIES, SYNC_VIA_CHECKPOINT}),
    ),
}


@dataclass(frozen=True)
class SyncHandler:
    """Required engine methods and placement boundary owned by a sync handler."""

    required_methods: frozenset[str]
    topology: str


SYNC_HANDLERS: Mapping[str, SyncHandler] = {
    "IPCWeightSync": SyncHandler(frozenset({SYNC_VIA_IPC}), SYNC_LOCAL),
    "TensorWeightSync": SyncHandler(frozenset({SYNC_VIA_TENSOR}), SYNC_LOCAL),
    "NCCLWeightSync": SyncHandler(frozenset({SYNC_VIA_NCCL, SYNC_VIA_NCCL_UPDATE}), SYNC_REMOTE),
    "LocalLoraWeightSync": SyncHandler(frozenset({SYNC_VIA_LORA}), SYNC_LOCAL),
    "RemoteLoraWeightSync": SyncHandler(frozenset({SYNC_VIA_LORA}), SYNC_REMOTE),
    "CheckpointWeightSync": SyncHandler(frozenset({SYNC_VIA_CHECKPOINT}), SYNC_LOCAL),
    "CkptEngineIPCWeightSync": SyncHandler(
        frozenset({SYNC_VIA_CHECKPOINT_ENGINE_IPC, SYNC_MARK_CHECKPOINT_ENGINE_FAILURE, ENGINE_SHUTDOWN}),
        SYNC_LOCAL,
    ),
}

ENGINE_SECTIONS = ("rollout", "ar_rollout", "dit_rollout")
LAYOUTS = ("colocate", "separate")


def engine_family_name(target: str) -> Optional[str]:
    """Return the declared package family for an engine or engine-config target."""
    if not target.startswith(ENGINE_PACKAGE):
        return None
    family = target[len(ENGINE_PACKAGE) :].split(".", 1)[0]
    return family if family in ENGINE_FAMILIES else None


def engine_family(target: str) -> Optional[EngineFamily]:
    """Return the declared capabilities for an engine or engine-config target."""
    family = engine_family_name(target)
    return None if family is None else ENGINE_FAMILIES[family]


@dataclass(frozen=True)
class Block:
    """A targeted recipe block tagged with its source path and sync options."""

    path: str
    target: str
    track: Optional[str] = None
    track_prefix: Optional[str] = None
    copy: Optional[bool] = None
    verify: Optional[bool] = None

    @property
    def class_name(self) -> str:
        return self.target.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class RecipeFacts:
    """Normalized facts consumed by every cross-component contract."""

    engines: tuple[Block, ...]
    nested_engines: tuple[Block, ...]
    syncs: tuple[Block, ...]
    has_sync_section: bool
    has_layout: bool
    layout_is_dynamic: bool
    layout: Optional[str]
    rollout_anchor_device: Any
    freeze_llm: bool
    samplings: tuple[Block, ...]
    rollout_backend: Optional[str]
    rollout_dp_size: Any
    rollout_engine_kwargs: Any

    @classmethod
    def from_cfg(cls, cfg: Any) -> "RecipeFacts":
        """Read normalized facts from a DictConfig or plain mapping."""
        engines = tuple(block for path in ENGINE_SECTIONS if (block := _read_block(cfg, path)) is not None)
        nested_paths = (
            "rollout.config.ar",
            "rollout.config.diffusion",
            "rollout.config.inner",
        )
        nested_engines = tuple(block for path in nested_paths if (block := _read_block_path(cfg, path)) is not None)
        sync_section = _get(cfg, "sync")
        raw_layout = _get(cfg, "layout")
        raw_anchor = _get(cfg, "rollout_anchor_device")
        raw_backend = _get_path(cfg, "rollout.config.backend")
        return cls(
            engines=engines,
            nested_engines=nested_engines,
            syncs=tuple(_read_sync_blocks(sync_section)),
            has_sync_section=sync_section is not None,
            has_layout=_has(cfg, "layout"),
            layout_is_dynamic=_is_interpolation(raw_layout),
            layout=None if raw_layout is None or _is_interpolation(raw_layout) else str(raw_layout),
            rollout_anchor_device=raw_anchor,
            freeze_llm=bool(_get(cfg, "freeze_llm")),
            samplings=tuple(_read_sampling_blocks(_get(cfg, "sampling"))),
            rollout_backend=(
                None if raw_backend is None or _is_interpolation(raw_backend) else str(raw_backend).strip().lower()
            ),
            rollout_dp_size=_get_path(cfg, "rollout.config.dp_size"),
            rollout_engine_kwargs=_get_path(cfg, "rollout.config.engine_kwargs"),
        )

    def families(self, blocks: Optional[tuple[Block, ...]] = None) -> tuple[tuple[Block, EngineFamily], ...]:
        """Pair known engine blocks with their declared families."""
        known = []
        for block in self.engines if blocks is None else blocks:
            family = engine_family(block.target)
            if family is None:
                logger.info(
                    "cfg.%s._target_=%r has no declared engine family; skipping dependent checks.",
                    block.path,
                    block.target,
                )
                continue
            known.append((block, family))
        return tuple(known)


def _get(cfg: Any, key: str) -> Any:
    """Read one optional mapping key without importing OmegaConf."""
    if cfg is None:
        return None
    getter = getattr(cfg, "get", None)
    return getter(key) if callable(getter) else None


def _is_interpolation(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("${") and value.strip().endswith("}")


def validate_checkpoint_engine_ipc_options(
    *,
    backend: Any,
    dp_size: Any,
    engine_kwargs: Any,
    allow_interpolation: bool = False,
) -> None:
    """Validate config-dependent checkpoint-engine IPC requirements."""

    def dynamic(value: Any) -> bool:
        return allow_interpolation and _is_interpolation(value)

    normalized_backend = None if backend is None or dynamic(backend) else str(backend).strip().lower()
    require(
        normalized_backend in (None, "http"),
        f"CkptEngineIPCWeightSync requires SGLang backend='http'; got {normalized_backend!r}.",
    )

    server_dp = dp_size if dp_size is not None else _get(engine_kwargs, "dp_size")
    require(
        server_dp is None or dynamic(server_dp) or (type(server_dp) is int and server_dp == 1),
        f"CkptEngineIPCWeightSync requires integer SGLang server dp_size=1; got {server_dp!r}.",
    )

    keys = engine_kwargs.keys() if hasattr(engine_kwargs, "keys") else ()
    speculative = [
        str(key)
        for key in keys
        if str(key).startswith("speculative")
        and (value := _get(engine_kwargs, key))
        and not dynamic(value)
        and (not isinstance(value, str) or value.strip().lower() != "none")
    ]
    require(
        not speculative,
        f"CkptEngineIPCWeightSync does not support speculative decoding options {sorted(speculative)}.",
    )


def _has(cfg: Any, key: str) -> bool:
    """Return whether a mapping explicitly contains one key."""
    if cfg is None:
        return False
    try:
        return key in cfg
    except TypeError:
        return False


def _get_path(cfg: Any, path: str) -> Any:
    """Read a dotted mapping path without importing OmegaConf."""
    value = cfg
    for key in path.split("."):
        value = _get(value, key)
    return value


def _read_block(cfg: Any, path: str) -> Optional[Block]:
    target = _target(_get(cfg, path))
    return None if target is None else Block(path=path, target=str(target))


def _read_block_path(cfg: Any, path: str) -> Optional[Block]:
    target = _target(_get_path(cfg, path))
    return None if target is None else Block(path=path, target=str(target), track=path.rsplit(".", 1)[-1])


def _target(block: Any) -> Optional[str]:
    target = _get(block, "_target_")
    return None if target is None else str(target)


def _read_sampling_blocks(sampling_section: Any) -> list[Block]:
    """Normalize single-domain and modality-keyed sampling sections."""
    if sampling_section is None:
        return []
    if (target := _target(sampling_section)) is not None:
        return [Block(path="sampling", target=target)]
    tracks = sampling_section.keys() if hasattr(sampling_section, "keys") else ()
    return [
        Block(path=f"sampling.{track}", target=target, track=str(track))
        for track in tracks
        if (target := _target(_get(sampling_section, track))) is not None
    ]


def _read_sync_blocks(sync_section: Any) -> list[Block]:
    """Normalize a single handler or per-track handler map without losing track names."""
    if sync_section is None:
        return []
    if (target := _get(sync_section, "_target_")) is not None:
        prefix = _get(sync_section, "track_prefix")
        return [
            Block(
                path="sync",
                target=str(target),
                track_prefix=None if prefix is None else str(prefix),
                copy=bool(_get(sync_section, "copy")) if _has(sync_section, "copy") else None,
                verify=bool(_get(sync_section, "verify")) if _has(sync_section, "verify") else None,
            )
        ]
    tracks = sync_section.keys() if hasattr(sync_section, "keys") else ()
    blocks = []
    for track in tracks:
        handler = _get(sync_section, track)
        target = _get(handler, "_target_")
        if target is None:
            continue
        prefix = _get(handler, "track_prefix")
        blocks.append(
            Block(
                path=f"sync.{track}",
                target=str(target),
                track=str(track),
                track_prefix=None if prefix is None else str(prefix),
                copy=bool(_get(handler, "copy")) if _has(handler, "copy") else None,
                verify=bool(_get(handler, "verify")) if _has(handler, "verify") else None,
            )
        )
    return blocks


def _effective_sync_topology(facts: RecipeFacts, entrypoint: str) -> Optional[str]:
    if entrypoint in (ENTRYPOINT_ASYNC_AR, ENTRYPOINT_ASYNC_DIFFUSION):
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_AR and facts.rollout_anchor_device is not None:
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_UNIFIED and {block.path for block in facts.engines} == {
        "ar_rollout",
        "dit_rollout",
    }:
        return SYNC_REMOTE
    if entrypoint == ENTRYPOINT_DIFFUSION:
        if facts.layout_is_dynamic:
            return None
        return SYNC_REMOTE if facts.layout == "separate" else SYNC_LOCAL
    return SYNC_LOCAL


def _nested_engine(facts: RecipeFacts, track: str) -> tuple[Block, ...]:
    suffix = f".{track}"
    return tuple(block for block in facts.nested_engines if block.path.endswith(suffix))


def _sync_engine_blocks(facts: RecipeFacts, sync: Block) -> tuple[Block, ...]:
    rollout = next((block for block in facts.engines if block.path == "rollout"), None)
    if rollout is None:
        return facts.engines
    family = engine_family_name(rollout.target)
    if family == "composed" and sync.track is not None:
        return (rollout, *_nested_engine(facts, sync.track))
    if family == "agentic":
        return (rollout, *_nested_engine(facts, "inner"))
    return facts.engines


def _validate_sync_shape(facts: RecipeFacts, *, entrypoint: str) -> None:
    """Validate the handler-map shape consumed by each trainer."""
    if entrypoint == ENTRYPOINT_PE:
        required = {"diffusion"} if facts.freeze_llm else {"ar", "diffusion"}
        actual = {sync.track for sync in facts.syncs}
        require(
            actual == required, f"train_pe requires sync tracks {sorted(required)}; found {sorted(actual, key=str)}."
        )
        for sync in facts.syncs:
            require(
                sync.track_prefix == sync.track,
                f"cfg.{sync.path}.track_prefix must be {sync.track!r}; got {sync.track_prefix!r}.",
            )
            require(
                bool(_nested_engine(facts, str(sync.track))),
                f"cfg.rollout.config.{sync.track} must declare an engine config for cfg.{sync.path}.",
            )
        return
    require(
        all(sync.track is None for sync in facts.syncs),
        f"{entrypoint} expects one cfg.sync handler, not a per-track map.",
    )
    if entrypoint == ENTRYPOINT_AGENTIC:
        require(bool(_nested_engine(facts, "inner")), "cfg.rollout.config.inner must declare an engine config.")


def _validate_engine_shape(facts: RecipeFacts, *, entrypoint: str) -> None:
    """Require the exact engine sections consumed by each entrypoint."""
    paths = {block.path for block in facts.engines}
    if entrypoint == ENTRYPOINT_UNIFIED:
        require(
            paths in ({"rollout"}, {"ar_rollout", "dit_rollout"}),
            "train_unified_model requires either cfg.rollout or both cfg.ar_rollout and cfg.dit_rollout.",
        )
    elif entrypoint == ENTRYPOINT_SFT:
        require(not paths, "train_sft does not consume rollout engine sections.")
    elif entrypoint in KNOWN_ENTRYPOINTS:
        require(paths == {"rollout"}, f"{entrypoint} requires exactly one cfg.rollout engine section.")


def _validate_handler_topology(facts: RecipeFacts, *, entrypoint: str) -> None:
    topology = _effective_sync_topology(facts, entrypoint)
    for sync in facts.syncs:
        if sync.copy is not None:
            require(sync.class_name == "RemoteLoraWeightSync", f"cfg.{sync.path}.copy belongs to RemoteLoraWeightSync.")
        if sync.verify is not None:
            require(
                sync.class_name in ("LocalLoraWeightSync", "RemoteLoraWeightSync"),
                f"cfg.{sync.path}.verify belongs to a LoRA weight-sync handler.",
            )
        if entrypoint == ENTRYPOINT_ASYNC_AR:
            require(sync.class_name == "NCCLWeightSync", "train_async_ar supports only NCCLWeightSync.")
        if entrypoint == ENTRYPOINT_AGENTIC:
            require(sync.class_name == "TensorWeightSync", "train_agentic supports only TensorWeightSync.")
        if entrypoint == ENTRYPOINT_AR and facts.rollout_anchor_device is not None:
            require(
                sync.class_name == "RemoteLoraWeightSync",
                "anchored train_ar rollout supports only RemoteLoraWeightSync.",
            )
        if entrypoint == ENTRYPOINT_UNIFIED and len(facts.engines) == 2:
            require(
                sync.class_name == "RemoteLoraWeightSync",
                "two-engine train_unified_model supports only RemoteLoraWeightSync.",
            )
            require(
                sync.copy is True, "two-engine train_unified_model requires cfg.sync.copy=true for TP-safe LoRA sync."
            )
        handler = SYNC_HANDLERS.get(sync.class_name)
        if handler is None:
            continue
        require(
            topology is None or handler.topology == topology,
            f"cfg.{sync.path}={sync.class_name} is a {handler.topology}-engine handler, but "
            f"{entrypoint} wires rollout through the {topology} boundary.",
        )


def _sampling_domain(target: str) -> Optional[str]:
    class_name = target.rsplit(".", 1)[-1]
    if class_name.endswith(AR_SAMPLING_SUFFIXES):
        return SAMPLING_AR
    if class_name.endswith(DIFFUSION_SAMPLING_SUFFIXES):
        return SAMPLING_DIFFUSION
    return None


def validate_sampling_contract(facts: RecipeFacts, *, entrypoint: str) -> None:
    """Require the sampling-parameter type consumed by an entrypoint."""
    if entrypoint == ENTRYPOINT_SFT:
        return
    require(bool(facts.samplings), f"{entrypoint} requires a targeted cfg.sampling section.")
    top_level = [block for block in facts.samplings if block.track is None]
    if top_level:
        require(len(top_level) == 1, "cfg.sampling must declare exactly one top-level _target_.")
        expected = ENTRYPOINT_SAMPLING_DOMAINS.get(entrypoint)
        require(expected is not None, f"{entrypoint} requires modality-keyed cfg.sampling tracks.")
        actual = _sampling_domain(top_level[0].target)
        require(
            actual in (None, expected),
            f"{entrypoint} requires {expected} sampling parameters; got cfg.sampling._target_={top_level[0].target!r}.",
        )
        return

    if entrypoint == ENTRYPOINT_AGENTIC:
        raise ValueError("train_agentic requires a top-level cfg.sampling._target_, not modality-keyed tracks.")

    expected_tracks = {
        ENTRYPOINT_AR: {SAMPLING_AR},
        ENTRYPOINT_ASYNC_AR: {SAMPLING_AR},
        ENTRYPOINT_DIFFUSION: {SAMPLING_DIFFUSION},
        ENTRYPOINT_ASYNC_DIFFUSION: {SAMPLING_DIFFUSION},
        ENTRYPOINT_PE: {SAMPLING_AR, SAMPLING_DIFFUSION},
        ENTRYPOINT_UNIFIED: {SAMPLING_AR, SAMPLING_DIFFUSION},
    }[entrypoint]
    actual_tracks = {block.track for block in facts.samplings}
    require(
        actual_tracks == expected_tracks,
        f"{entrypoint} requires sampling tracks {sorted(expected_tracks)}; found {sorted(actual_tracks, key=str)}.",
    )
    for block in facts.samplings:
        actual = _sampling_domain(block.target)
        require(
            actual in (None, block.track),
            f"cfg.{block.path} must use {block.track} sampling parameters; got {block.target!r}.",
        )


def validate_weight_sync_contract(facts: RecipeFacts, *, entrypoint: str) -> None:
    """Validate sampling mode, handler shape, topology, and receiver capabilities."""
    _validate_engine_shape(facts, entrypoint=entrypoint)
    anchor = facts.rollout_anchor_device
    if entrypoint != ENTRYPOINT_AR:
        require(anchor is None, f"{entrypoint} does not consume cfg.rollout_anchor_device; remove it.")
    elif anchor is not None:
        if _is_interpolation(anchor):
            pass
        else:
            require(
                not isinstance(anchor, bool) and isinstance(anchor, (int, str)),
                f"cfg.rollout_anchor_device must be a positive integer; got {anchor!r}.",
            )
            try:
                normalized_anchor = int(anchor)
            except ValueError as exc:
                raise ValueError(f"cfg.rollout_anchor_device must be a positive integer; got {anchor!r}.") from exc
            require(normalized_anchor > 0, f"cfg.rollout_anchor_device must be positive; got {normalized_anchor}.")
    families = facts.families()
    if not families:
        if not facts.engines:
            require(not facts.has_sync_section, f"{entrypoint} does not consume cfg.sync; remove it.")
        return
    for block, family in families:
        require(
            entrypoint in family.entrypoints,
            f"cfg.{block.path}._target_={block.target!r} does not support {entrypoint}; "
            f"expected one of {sorted(family.entrypoints)}.",
        )

    direct = [block.path for block, family in families if family.direct_sampling]
    dedicated = [block.path for block, family in families if not family.direct_sampling]
    require(
        not (direct and dedicated),
        f"recipe mixes direct engines at cfg.{', cfg.'.join(direct)} with dedicated engines at "
        f"cfg.{', cfg.'.join(dedicated)}.",
    )

    if direct:
        require(
            entrypoint != ENTRYPOINT_AR or facts.rollout_anchor_device is None,
            f"anchored train_ar rollout requires a dedicated engine; got direct-sampling cfg.{direct[0]}.",
        )
        found = _describe(facts.syncs) or "a handler-less sync section"
        require(
            not facts.has_sync_section,
            f"cfg.{direct[0]} samples on the train actors and cannot use cfg.sync; found {found}.",
        )
        return

    if entrypoint == ENTRYPOINT_UNIFIED and [block.path for block in facts.engines] == ["rollout"]:
        raise ValueError(
            "train_unified_model single-engine mode does not wire weight sync for a dedicated rollout engine; "
            "use the trainside engine or the ar_rollout + dit_rollout mode."
        )

    require(
        bool(facts.syncs),
        f"cfg.{dedicated[0]} runs a dedicated rollout copy and requires a cfg.sync handler.",
    )
    _validate_sync_shape(facts, entrypoint=entrypoint)
    _validate_handler_topology(facts, entrypoint=entrypoint)

    for sync in facts.syncs:
        handler = SYNC_HANDLERS.get(sync.class_name)
        if handler is None:
            continue
        if sync.class_name == "CkptEngineIPCWeightSync":
            validate_checkpoint_engine_ipc_options(
                backend=facts.rollout_backend,
                dp_size=facts.rollout_dp_size,
                engine_kwargs=facts.rollout_engine_kwargs,
                allow_interpolation=True,
            )
        needed = handler.required_methods
        if sync.class_name == "RemoteLoraWeightSync" and sync.copy is True:
            needed = frozenset({SYNC_VIA_LORA_COPY})
        if sync.verify is True:
            needed = frozenset({*needed, SYNC_VERIFY_TOPOLOGY, SYNC_VERIFY_LORA})
        for block, family in facts.families(_sync_engine_blocks(facts, sync)):
            require(
                needed <= family.capabilities,
                f"cfg.{sync.path}={sync.class_name} needs {sorted(needed)}, but "
                f"cfg.{block.path}._target_={block.target!r} supports {sorted(family.capabilities)}.",
            )


def validate_rollout_layout(facts: RecipeFacts, *, entrypoint: str) -> None:
    """Validate rollout layout values only where the selected entrypoint consumes them."""
    if entrypoint not in LAYOUT_ENTRYPOINTS:
        require(not facts.has_layout, f"{entrypoint} does not consume cfg.layout; remove it.")
    if facts.has_layout:
        if facts.layout_is_dynamic:
            return
        require(
            facts.layout is not None and facts.layout in LAYOUTS,
            f"cfg.layout={facts.layout!r} is invalid; expected one of {list(LAYOUTS)}.",
        )

    if entrypoint == ENTRYPOINT_ASYNC_DIFFUSION:
        effective = "separate"
    else:
        effective = facts.layout or "colocate"
    if effective != "separate":
        return
    for block, family in facts.families():
        require(
            not family.direct_sampling,
            f"cfg.layout='separate' cannot place direct-sampling cfg.{block.path}._target_={block.target!r}.",
        )


CONTRACTS = (
    validate_sampling_contract,
    validate_weight_sync_contract,
    validate_rollout_layout,
)


def validate_recipe(cfg: Any, *, entrypoint: str) -> None:
    """Run every cross-component contract before trainer construction."""
    require(entrypoint in KNOWN_ENTRYPOINTS, f"unknown training entrypoint {entrypoint!r}")
    try:
        facts = RecipeFacts.from_cfg(cfg)
        for contract in CONTRACTS:
            contract(facts, entrypoint=entrypoint)
    except ValueError as exc:
        raise ValueError(f"{entrypoint}: invalid recipe. {exc}") from exc


def _describe(blocks: tuple[Block, ...]) -> str:
    return ", ".join(f"cfg.{block.path}={block.class_name}" for block in blocks)


__all__ = [
    "ENGINE_CAPABILITY_METHODS",
    "ENGINE_FAMILIES",
    "KNOWN_ENTRYPOINTS",
    "SYNC_HANDLERS",
    "SYNC_LOCAL",
    "SYNC_REMOTE",
    "validate_checkpoint_engine_ipc_options",
    "validate_recipe",
]
