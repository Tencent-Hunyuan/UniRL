"""Cross-component recipe contracts — the rules that span recipe sections.

A recipe is one flat YAML that Hydra type-checks nothing about, so the rules
relating one section to another (does the rollout engine need a ``sync``
handler? may it offload? may it live on its own device slab?) have to be
enforced somewhere. :func:`validate_recipe` is that gate, and every
``unirl/train_*.py`` calls it on the driver before constructing its trainer —
so a contradictory recipe dies on the launching process in under a second
instead of somewhere inside a half-built Ray cluster.

Two properties are deliberate:

- **Stdlib only** (like ``require.py``). This keeps the contract predicates
  lightweight and lets the same code run as a static guard over every shipped
  recipe without importing torch / sglang / vllm — see
  ``scripts/check_recipe_contracts.py``, which is how this layer is kept honest
  in a lint-only CI. Runtime entrypoints still import their trainer modules
  before ``main`` executes; the gate promises to run before trainer
  construction and Ray startup, not before all heavy Python imports.
- **Recipe shape knowledge lives in one place** — :class:`RecipeFacts`. Every
  contract is a predicate over normalized facts, never a hard-coded dotpath,
  because the shapes differ per entry point: a diffusion recipe puts its engine
  at ``rollout``, ``train_unified_model``'s two-engine mode uses ``ar_rollout``
  + ``dit_rollout``, ``train_pe`` gives ``sync`` a per-track map instead of a
  single block, and ``train_sft`` has no rollout at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from unirl.config.require import require

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout engine families
#
# An engine is identified by the package it lives in --
# ``unirl.rollout.engine.<family>.`` -- rather than by its class name, so
# renaming a class cannot silently flip a recipe into the wrong mode.
# ``scripts/check_recipe_contracts.py`` reads the engine classes statically and
# fails if the declarations below drift from the code.
# ---------------------------------------------------------------------------

ENGINE_PACKAGE = "unirl.rollout.engine."

#: Weight-sync receive entry points, named after the ``BaseRolloutEngine``
#: methods a sync handler drives, so a declared capability can be checked
#: against the engine class that is supposed to implement it.
SYNC_VIA_IPC = "update_weights_from_ipc"
SYNC_VIA_TENSOR = "update_weights_from_tensor"
SYNC_VIA_NCCL = "init_weights_update_group"
SYNC_VIA_LORA = "set_lora_from_tensors"
SYNC_VIA_CHECKPOINT = "update_weights_from_path"

SYNC_RECEIVE_METHODS = (
    SYNC_VIA_IPC,
    SYNC_VIA_TENSOR,
    SYNC_VIA_NCCL,
    SYNC_VIA_LORA,
    SYNC_VIA_CHECKPOINT,
)


@dataclass(frozen=True)
class EngineFamily:
    """What a rollout engine family is, as far as recipe contracts care.

    ``direct_sampling`` means the engine samples inside the train actor and
    takes the training ``pipeline`` as a local sibling (``trainside``). It is
    the same distinction the trainers make at build time by checking the engine
    class for a ``pipeline`` parameter.

    ``weight_sync`` is the set of receive entry points the engine implements. A
    recipe that pairs a sync handler with an engine implementing neither side
    of that transport is a run that dies on the first weight push.
    """

    direct_sampling: bool
    weight_sync: frozenset[str]


ENGINE_FAMILIES: Mapping[str, EngineFamily] = {
    "trainside": EngineFamily(direct_sampling=True, weight_sync=frozenset()),
    "sglang": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    "sglang_diffusion": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    "vllm_omni": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_IPC, SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    # PE coordinator: owns an ar + a diffusion child and forwards each push to
    # the child named by the handler's ``track_prefix``.
    "composed": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_IPC, SYNC_VIA_TENSOR, SYNC_VIA_NCCL, SYNC_VIA_LORA}),
    ),
    # Reloads from a checkpoint path; implements no in-memory receive path.
    "fastvideo": EngineFamily(
        direct_sampling=False,
        weight_sync=frozenset({SYNC_VIA_CHECKPOINT}),
    ),
}

#: Which receive entry point each shipped sync handler drives.
SYNC_HANDLER_NEEDS: Mapping[str, Optional[str]] = {
    "IPCWeightSync": SYNC_VIA_IPC,
    "TensorWeightSync": SYNC_VIA_TENSOR,
    "NCCLWeightSync": SYNC_VIA_NCCL,
    "LocalLoraWeightSync": SYNC_VIA_LORA,
    "RemoteLoraWeightSync": SYNC_VIA_LORA,
    "CheckpointWeightSync": SYNC_VIA_CHECKPOINT,
}

#: Recipe sections that may hold a rollout engine ``_target_`` block. Flat
#: recipes use ``rollout``; ``train_unified_model``'s two-engine mode splits it
#: into ``ar_rollout`` + ``dit_rollout``.
ENGINE_SECTIONS = ("rollout", "ar_rollout", "dit_rollout")

LAYOUTS = ("colocate", "separate")


def engine_family(target: str) -> Optional[EngineFamily]:
    """Look up the family of an engine ``_target_``, or ``None`` if unknown.

    ``None`` for any dotpath outside ``unirl.rollout.engine.<family>`` — an
    out-of-tree engine, whose mode this module cannot know. Callers skip the
    engine-dependent contracts for it rather than guess.
    """
    if not target.startswith(ENGINE_PACKAGE):
        return None
    family = target[len(ENGINE_PACKAGE) :].split(".", 1)[0]
    return ENGINE_FAMILIES.get(family)


# ---------------------------------------------------------------------------
# Normalized recipe view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A ``_target_`` block found in the recipe, tagged with where it was read."""

    path: str
    target: str

    @property
    def class_name(self) -> str:
        return self.target.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class RecipeFacts:
    """The cross-cutting facts a recipe states, normalized across entry points.

    :meth:`from_cfg` is the only code here that knows *where* an entry point
    puts things; every contract reads these fields instead.
    """

    #: Engine blocks in section order. Empty for a recipe with no rollout at all
    #: (``train_sft``) or one that names its sampler differently (``train_refl``
    #: uses ``policy``) — such recipes have no engine contracts to check.
    engines: tuple[Block, ...]
    #: Sync handler blocks; more than one when ``sync`` is a per-track map.
    syncs: tuple[Block, ...]
    #: Whether a ``sync`` section is present at all, even one declaring no
    #: handler — distinct from ``syncs`` being empty, and worth distinguishing
    #: in the error message.
    has_sync_section: bool
    layout: str
    #: ``None`` when the recipe is silent and the entry point's default applies
    #: (``train_unified_model`` defaults it to ``True``, the rest to ``False``),
    #: so a contract can judge only what the recipe itself asked for.
    offload: Optional[bool]

    @classmethod
    def from_cfg(cls, cfg: Any) -> "RecipeFacts":
        """Read the facts out of a composed recipe (a ``DictConfig`` or a dict)."""
        engines = tuple(block for path in ENGINE_SECTIONS if (block := _read_block(cfg, path)) is not None)
        sync_section = _get(cfg, "sync")
        offload = _get(cfg, "enable_fsdp_offload")
        return cls(
            engines=engines,
            syncs=tuple(_read_sync_blocks(sync_section)),
            has_sync_section=sync_section is not None,
            layout=str(_get(cfg, "layout") or "colocate"),
            offload=None if offload is None else bool(offload),
        )

    def families(self) -> tuple[tuple[Block, EngineFamily], ...]:
        """Engine blocks paired with their family, dropping unknown dotpaths."""
        known = []
        for block in self.engines:
            family = engine_family(block.target)
            if family is None:
                logger.info(
                    "cfg.%s._target_=%r is not a %s* dotpath; skipping the engine-dependent recipe contracts for it.",
                    block.path,
                    block.target,
                    ENGINE_PACKAGE,
                )
                continue
            known.append((block, family))
        return tuple(known)


def _get(cfg: Any, key: str) -> Any:
    """``cfg.get(key)`` that tolerates a missing section or a non-mapping value."""
    if cfg is None:
        return None
    getter = getattr(cfg, "get", None)
    return getter(key) if callable(getter) else None


def _read_block(cfg: Any, path: str) -> Optional[Block]:
    target = _get(_get(cfg, path), "_target_")
    return None if target is None else Block(path=path, target=str(target))


def _read_sync_blocks(sync_section: Any) -> list[Block]:
    """Normalize both ``sync`` shapes into a flat list of handler blocks.

    A single handler (``sync: {_target_: ...}``) yields one block; the per-track
    map ``train_pe`` uses (``sync: {diffusion: {...}, ar: {...}}``) yields one
    block per track.
    """
    if sync_section is None:
        return []
    if (target := _get(sync_section, "_target_")) is not None:
        return [Block(path="sync", target=str(target))]
    tracks = sync_section.keys() if hasattr(sync_section, "keys") else ()
    return [
        Block(path=f"sync.{track}", target=str(target))
        for track in tracks
        if (target := _get(_get(sync_section, track), "_target_")) is not None
    ]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def is_direct_sampling(cfg: Any) -> bool:
    """Whether the recipe samples inside the train actors.

    True iff the recipe names at least one recognized engine and *every* one of
    them is direct-sampling — today only ``TrainsideRolloutEngine``, the
    in-process Pipeline adapter (see ``unirl/rollout/engine/trainside``). Every
    other engine runs dedicated rollout actors.

    A recipe mixing the two modes across its engine sections is neither;
    :func:`validate_weight_sync_contract` rejects it by name.
    """
    families = RecipeFacts.from_cfg(cfg).families()
    return bool(families) and all(family.direct_sampling for _, family in families)


def validate_weight_sync_contract(cfg: Any) -> None:
    """A ``sync`` section must be present iff rollout runs in its own actor.

    Direct sampling shares the trainable module with the sampler, so there are
    no separate rollout weights and a ``sync`` block is a contradiction. A
    dedicated engine holds its own copy, so omitting ``sync`` silently trains
    the policy while rollout keeps sampling the initial weights.

    Also checks the pairing: a handler is only usable on an engine implementing
    the receive path it drives — ``IPCWeightSync`` needs
    ``update_weights_from_ipc``, which only the vllm-omni and composed engines
    have.
    """
    facts = RecipeFacts.from_cfg(cfg)
    families = facts.families()
    if not families:
        return

    direct = [block.path for block, family in families if family.direct_sampling]
    dedicated = [block.path for block, family in families if not family.direct_sampling]
    require(
        not (direct and dedicated),
        f"recipe mixes sampling modes: cfg.{', cfg.'.join(direct)} sample inside the train actors "
        f"while cfg.{', cfg.'.join(dedicated)} run dedicated rollout actors. All engine sections "
        "must be on the same side — the weight-sync and placement contracts differ between them.",
    )

    if direct:
        found = _describe(facts.syncs) or "a sync section declaring no handler"
        require(
            not facts.has_sync_section,
            f"cfg.{direct[0]} samples inside the train actors, which share the trainable module "
            f"with the sampler — there are no rollout weights to push. Remove the sync section "
            f"(found {found}).",
        )
        return

    require(
        bool(facts.syncs),
        f"cfg.{dedicated[0]} runs a dedicated rollout actor holding its own copy of the weights, "
        f"so the recipe must declare a cfg.sync handler to push updates to it; found "
        f"{'a sync section declaring no handler' if facts.has_sync_section else 'no sync section'}. "
        "Without one the policy trains while rollout samples the initial weights forever.",
    )
    for sync in facts.syncs:
        needed = SYNC_HANDLER_NEEDS.get(sync.class_name)
        if needed is None:
            continue
        for block, family in families:
            supported = ", ".join(sorted(family.weight_sync)) or "none — it reloads from a checkpoint dir"
            require(
                needed in family.weight_sync,
                f"cfg.{sync.path}={sync.class_name} pushes weights via {needed}(), which "
                f"cfg.{block.path}._target_={block.target!r} does not implement. That engine's "
                f"receive paths: {supported}.",
            )


def validate_rollout_layout(cfg: Any) -> None:
    """``layout`` must name a real layout, and direct sampling implies colocate.

    A direct-sampling engine reaches the training pipeline as a local sibling,
    so it cannot be placed on a disjoint device slab. The trainers reject this
    too, but only once the train slab's actors are already up; catching it here
    keeps the failure on the driver, before Ray is touched.

    An unrecognized ``layout`` is rejected because the trainers compare it
    against ``"separate"`` exactly, so a typo silently means colocate.
    """
    facts = RecipeFacts.from_cfg(cfg)
    require(
        facts.layout in LAYOUTS,
        f"cfg.layout={facts.layout!r} is not a layout; expected one of {list(LAYOUTS)}. The "
        "trainers compare against 'separate' exactly, so a typo silently colocates.",
    )
    if facts.layout != "separate":
        return
    for block, family in facts.families():
        require(
            not family.direct_sampling,
            f"cfg.layout='separate' puts rollout on its own device slab, but "
            f"cfg.{block.path}._target_={block.target!r} samples inside the train actors and needs "
            "the training pipeline as a local sibling. Use a dedicated engine (sglang / vllm-omni) "
            "for a separate slab, or drop layout back to colocate.",
        )


def validate_offload_contract(cfg: Any) -> None:
    """Direct sampling forbids ``enable_fsdp_offload``.

    Offload parks the base weights on CPU while a dedicated rollout actor
    samples. Under direct sampling the sampler *is* the trainable module, so
    there is no window in which the weights may leave the GPU; the trainers
    force the flag off. Reject the recipe that asks for it instead of silently
    ignoring what it asked for.

    Only an explicit ``enable_fsdp_offload: true`` is rejected — when the recipe
    is silent the value comes from the entry point's default, which is not the
    recipe author's statement.
    """
    facts = RecipeFacts.from_cfg(cfg)
    if facts.offload is not True:
        return
    for block, family in facts.families():
        require(
            not family.direct_sampling,
            f"cfg.enable_fsdp_offload=true is incompatible with "
            f"cfg.{block.path}._target_={block.target!r}: direct sampling generates on the live "
            "FSDP modules, so the base weights can never sit on CPU. Drop the flag or set it "
            "false — the trainers force it off for this engine anyway.",
        )


CONTRACTS = (
    validate_weight_sync_contract,
    validate_rollout_layout,
    validate_offload_contract,
)


def validate_recipe(cfg: Any, *, entrypoint: str) -> None:
    """Run every cross-component contract against a composed recipe.

    The single gate each ``unirl/train_*.py`` calls before building its trainer.
    A recipe with no rollout engine section (``train_sft``) simply has no engine
    contracts to check.
    """
    for contract in CONTRACTS:
        try:
            contract(cfg)
        except ValueError as exc:
            raise ValueError(f"{entrypoint}: invalid recipe. {exc}") from exc


def _describe(blocks: tuple[Block, ...]) -> str:
    return ", ".join(f"cfg.{block.path}={block.class_name}" for block in blocks)


__all__ = [
    "Block",
    "CONTRACTS",
    "ENGINE_FAMILIES",
    "ENGINE_PACKAGE",
    "ENGINE_SECTIONS",
    "EngineFamily",
    "LAYOUTS",
    "SYNC_HANDLER_NEEDS",
    "SYNC_RECEIVE_METHODS",
    "SYNC_VIA_CHECKPOINT",
    "SYNC_VIA_IPC",
    "SYNC_VIA_LORA",
    "SYNC_VIA_NCCL",
    "SYNC_VIA_TENSOR",
    "RecipeFacts",
    "engine_family",
    "is_direct_sampling",
    "validate_offload_contract",
    "validate_recipe",
    "validate_rollout_layout",
    "validate_weight_sync_contract",
]
