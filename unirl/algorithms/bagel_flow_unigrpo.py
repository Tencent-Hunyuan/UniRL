"""BagelFlowUniGRPO — FlowGRPO + velocity-MSE regularization (UniGRPO image side).

UniGRPO replaces FlowGRPO's latent-KL penalty with an unweighted MSE on the
predicted velocity field::

    L_MSE(theta) = || v_theta(x_t, t, y) - v_ref(x_t, t, y) ||^2

evaluated at the SDE-trained timesteps, where ``v_ref`` is the frozen pre-trained
base reference: under LoRA the policy with adapters disabled, under full
fine-tuning a frozen snapshot of the base weights swapped in for the v_ref
forward. This pulls the RL-tuned vector field back toward the base across all
noise levels, which mitigates reward hacking better than the timestep-weighted KL.

Subclasses :class:`FlowGRPO`: the clipped surrogate is inherited; the MSE term
adds its own backward into the same optimizer step. GRPO-Guard RatioNorm
(per-SDE-step ratio normalization) is optional via ``ratio_norm=True``.

Compute note: the default ``context_gradient_mode="full"`` keeps the established
separate RatioNorm and velocity-MSE backwards. The opt-in ``"stage_boundary"``
mode shares one detached Stage-0 context and reuses replay's ``v_theta`` values
for MSE, matching the native rollout boundary while avoiding duplicate policy
forwards.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.replay_result import ReplayResult
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    gather_sde_field,
    typed_conditions,
)
from .flowgrpo import FlowGRPO

_CONTEXT_GRADIENT_MODES = ("full", "stage_boundary")


@dataclass
class _PreparedMSEBatch:
    """Detached MSE inputs prepared at one optimizer-update boundary."""

    target_steps: Tuple[int, ...]
    forward_kwargs: Dict[str, Any]
    reference_velocities: List[torch.Tensor]
    surrogate_result: Optional[AlgorithmStepResult] = None


@dataclass
class _PreparedAnchor:
    """One exact pre-update anchor, or an update-0 current-replay marker."""

    target_steps: Tuple[int, ...]
    replay: Optional[ReplayResult]
    derive_from_current: bool


@contextmanager
def _disable_lora(module: Any) -> Iterator[bool]:
    """Temporarily disable LoRA adapters so a forward runs the base model.

    FSDPBackend injects LoRA via ``inject_adapter_in_model`` (not ``get_peft_model``),
    so target modules are ``peft.tuners.lora.LoraLayer``s exposing
    ``enable_adapters(bool)``. Walk the tree, flip every LoRA layer off for the
    scope, restore on exit. Yields ``True`` when at least one LoRA layer was found
    (so the forward really is the base = v_ref), ``False`` otherwise (no-op) so the
    caller can refuse rather than use the policy as its own reference.

    Self-contained here so the BAGEL UniGRPO task stays independent of any other
    algorithm module.
    """
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        yield False
        return
    layers = [m for m in module.modules() if isinstance(m, LoraLayer)]
    if not layers:
        yield False
        return
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield True
    finally:
        for layer in layers:
            layer.enable_adapters(True)


class BagelFlowUniGRPO(FlowGRPO):
    """FlowGRPO with UniGRPO's velocity-MSE regularization (BAGEL image side)."""

    prepares_update_batch = True
    prepares_phased_update_batch = True
    prepares_indexed_update_batch = True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
        mse_weight: float = 0.0,
        ratio_norm: bool = False,
        grad_reweight: bool = False,
        reuse_ratio_context_for_mse: bool = False,
        context_gradient_mode: str = "full",
        lazy_first_update_anchor: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            stage=stage,
            pipeline=pipeline,
            stage_attr=stage_attr,
            clip_range=clip_range,
            clip_schedule=clip_schedule,
            old_logp_source=old_logp_source,
            conditions_cls=conditions_cls,
        )
        self.mse_weight = float(mse_weight)
        # ratio_norm (GRPO-Guard): normalize the flow importance ratio per SDE step
        # so PPO clipping actually engages (the ratio is otherwise left-shifted,
        # mean<1). grad_reweight (×1/|dt|) is the optional 2nd component, off by default.
        self.ratio_norm = bool(ratio_norm)
        self.grad_reweight = bool(grad_reweight)
        self.reuse_ratio_context_for_mse = bool(reuse_ratio_context_for_mse)
        self.context_gradient_mode = str(context_gradient_mode).strip().lower()
        self.lazy_first_update_anchor = bool(lazy_first_update_anchor)
        if self.context_gradient_mode not in _CONTEXT_GRADIENT_MODES:
            raise ValueError(
                "BagelFlowUniGRPO.context_gradient_mode must be one of "
                f"{_CONTEXT_GRADIENT_MODES}; got {context_gradient_mode!r}."
            )
        if self.context_gradient_mode == "stage_boundary" and not self.ratio_norm:
            raise ValueError("context_gradient_mode='stage_boundary' requires ratio_norm=True.")
        if self.reuse_ratio_context_for_mse and not self.ratio_norm:
            raise ValueError("reuse_ratio_context_for_mse requires ratio_norm=True.")
        if self.context_gradient_mode == "stage_boundary" and self.reuse_ratio_context_for_mse:
            raise ValueError(
                "context_gradient_mode='stage_boundary' already shares one detached context between "
                "RatioNorm and MSE; reuse_ratio_context_for_mse must remain false."
            )
        if self.lazy_first_update_anchor and (
            self.context_gradient_mode != "stage_boundary" or not self.ratio_norm or self.old_logp_source != "replay"
        ):
            raise ValueError(
                "lazy_first_update_anchor=True requires context_gradient_mode='stage_boundary', "
                "ratio_norm=True, and old_logp_source='replay'."
            )
        self.prepares_anchor_plan = self.lazy_first_update_anchor
        # Under old_logp_source="replay" the train stack recomputes these per 1-sample
        # micro-slice and cats them back (UnifiedModelTrainStack.prepare_segment). RatioNorm
        # needs μ_old (sde_means) refreshed at the SAME replay geometry as π_old (sde_logp)
        # — base FlowGRPO only refreshes sde_logp — so declare both when ratio_norm is on.
        self.anchor_fields = ("sde_logp", "sde_means") if self.ratio_norm else ("sde_logp",)
        # Full-FT v_ref: a frozen bf16 snapshot of the base (pre-training) weights, captured
        # lazily on the first v_ref swap (before the first optimizer step) from each trainable
        # param's local shard, keyed by stable parameter name, and swapped in per step via in-place copy.
        # Stays None under LoRA (v_ref = adapters off) or mse_weight=0 (no MSE). See
        # _reference_weights.
        self._ref_snapshot: Optional[Dict[str, torch.Tensor]] = None
        # Set by prepare_update_batch and consumed in the exact micro-batch order
        # supplied by UnifiedModelTrainStack. None keeps direct algorithm callers
        # on the legacy per-micro fallback.
        self._prepared_mse_batches: Optional[List[Optional[_PreparedMSEBatch]]] = None
        # Whole-rollout anchor plan. Update 0 uses its own exact current replay
        # (before any optimizer step); later disjoint updates are replayed eagerly
        # and held as small CPU log-prob/mean tensors.
        self._prepared_anchor_updates: Optional[List[Tuple[int, List[_PreparedAnchor]]]] = None
        self._active_anchor_entries: Optional[List[_PreparedAnchor]] = None

    @staticmethod
    def _has_lora(transformer: Any) -> bool:
        """True if the transformer carries peft LoRA layers (LoRA training)."""
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return False
        return any(isinstance(m, LoraLayer) for m in transformer.modules())

    def _snapshot_reference(self, transformer: Any) -> None:
        """Deprecated shim — the v_ref base snapshot is now captured lazily inside
        :meth:`_reference_weights` (at the swap site, so the shard state matches every
        step). Kept as a no-op for any external caller; safe to remove once none remain.
        """
        return None

    def _full_ft_reference_params(self) -> List[tuple[str, torch.nn.Parameter]]:
        transformer = self.stage.model.transformer
        return [(name, param) for name, param in transformer.named_parameters() if param.requires_grad]

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save_reference_checkpoint(self, path: str) -> None:
        """Persist the immutable full-FT MSE reference beside a trainer checkpoint."""
        transformer = self.stage.model.transformer
        if self.mse_weight <= 0.0 or self._has_lora(transformer):
            return
        if self._ref_snapshot is None:
            raise RuntimeError("BagelFlowUniGRPO.save_reference_checkpoint: the base reference has not been captured.")
        rank = int(getattr(self.rank_info, "rank", 0))
        world_size = int(getattr(self.rank_info, "world_size", 1))
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "world_size": world_size,
                "rank": rank,
                "tensors": {name: tensor.detach().cpu() for name, tensor in self._ref_snapshot.items()},
            },
            os.path.join(path, f"bagel_image_reference_rank{rank:05d}.pt"),
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load_reference_checkpoint(self, path: str) -> None:
        """Restore the original full-FT MSE reference before resumed training."""
        transformer = self.stage.model.transformer
        if self.mse_weight <= 0.0 or self._has_lora(transformer):
            return
        rank = int(getattr(self.rank_info, "rank", 0))
        world_size = int(getattr(self.rank_info, "world_size", 1))
        snapshot_path = os.path.join(path, f"bagel_image_reference_rank{rank:05d}.pt")
        if not os.path.isfile(snapshot_path):
            raise RuntimeError(
                "BagelFlowUniGRPO.load_reference_checkpoint: checkpoint is missing the immutable base reference "
                f"for rank {rank}: {snapshot_path}."
            )
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise RuntimeError("BagelFlowUniGRPO.load_reference_checkpoint: malformed reference payload.")
        if (
            int(payload.get("format_version", -1)) != 1
            or int(payload.get("world_size", -1)) != world_size
            or int(payload.get("rank", -1)) != rank
        ):
            raise RuntimeError(
                "BagelFlowUniGRPO.load_reference_checkpoint: reference topology/version does not match "
                f"rank {rank}/{world_size}."
            )
        tensors = payload.get("tensors")
        if not isinstance(tensors, Mapping) or not all(torch.is_tensor(tensor) for tensor in tensors.values()):
            raise RuntimeError("BagelFlowUniGRPO.load_reference_checkpoint: malformed reference tensor payload.")
        expected_names = {name for name, _ in self._full_ft_reference_params()}
        actual_names = {str(name) for name in tensors}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise RuntimeError(
                "BagelFlowUniGRPO.load_reference_checkpoint: reference parameter names do not match the model; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
        self._ref_snapshot = {
            str(name): tensor.detach().to(dtype=torch.bfloat16).clone() for name, tensor in tensors.items()
        }

    @contextmanager
    def _reference_weights(self, transformer: Any) -> Iterator[None]:
        """Swap the frozen base weights into the trainable params for a v_ref forward.

        Full-FT analog of :func:`_disable_lora`. Swaps by **in-place copy of the local
        shard**, NOT a ``.data`` pointer swap: under ``fully_shard`` the forward's
        all-gather reads FSDP2's captured shard storage, so reassigning ``param.data`` is
        silently ignored (verified on a 2-GPU repro — the swapped forward equalled the
        live one), whereas ``local_view(p).copy_(...)`` writes that storage and IS honored.

        On the FIRST call it captures the base snapshot (the pre-trained weights, before
        the first optimizer step) **at this swap site** — the same shard state every
        subsequent step sees, so the copy sizes always match (a pre-loop snapshot would be
        sharded while the swap site, right after the v_theta forward, is unsharded → size
        mismatch). Stored as bf16 (the forward computes in bf16; halves the ~3.5→1.75
        GiB/GPU footprint) keyed by stable parameter name. A resumed run loads
        that same snapshot from the checkpoint instead of recapturing tuned weights.

        Per scope: stash each live local shard, copy the base in (cast to the live fp32
        master dtype), run the no-grad v_ref forward(s), then copy the trained weights back
        before any backward — so v_theta's autograd graph (recomputed under activation
        checkpointing at the post-loop backward) reads the trained weights. The unified
        stack opens one scope per optimizer update; direct callers retain a per-micro
        fallback. In-place copy+restore is autograd-safe here (verified on a 2-GPU
        backward repro).
        """
        from unirl.train.ema import local_view

        live = [(name, param) for name, param in transformer.named_parameters() if param.requires_grad]
        if not live:
            raise RuntimeError(
                "BagelFlowUniGRPO: mse_weight > 0 with no LoRA and no trainable params to snapshot "
                "as the v_ref base — the transformer is fully frozen. Enable full fine-tuning "
                "(use_lora=false unfreezes the decoder blocks) or set mse_weight=0."
            )
        if self._ref_snapshot is None:
            self._ref_snapshot = {
                name: local_view(param).detach().to(dtype=torch.bfloat16).clone() for name, param in live
            }

        prepared: List[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
        for name, param in live:
            lv = local_view(param)
            reference = self._ref_snapshot.get(name)
            if reference is None or reference.shape != lv.shape:
                shape = None if reference is None else tuple(reference.shape)
                raise RuntimeError(
                    "BagelFlowUniGRPO: persisted base reference is incompatible with the live parameter "
                    f"{name!r}: reference shape={shape}, live shape={tuple(lv.shape)}. "
                    "Resume with the same FSDP topology used to save the checkpoint."
                )
            prepared.append((name, param, reference))

        stash: List[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            with torch.no_grad():
                for _, param, reference in prepared:
                    lv = local_view(param)
                    stash.append((param, lv.detach().clone()))
                    lv.copy_(reference)
            yield
        finally:
            with torch.no_grad():
                for param, saved in stash:
                    local_view(param).copy_(saved)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
    ) -> None:
        """Freeze the π_old anchor; under ``old_logp_source="replay"`` + RatioNorm refresh
        μ_old (``sde_means``) alongside π_old (``sde_logp``) from ONE replay.

        Base FlowGRPO recomputes only ``sde_logp`` from the pre-update replay. RatioNorm
        also reads ``segment.sde_means`` as μ_old; leaving it at the rollout (pack-B,
        bf16-packing) geometry while ``sde_logp`` is recomputed at the bs=1 replay geometry
        makes Δμ ≠ 0 at update 0 → ratio ≠ 1. So do one replay and write BOTH. The train
        stack calls this per 1-sample micro-slice (so a bs=1 replay suffices) and cats the
        declared ``anchor_fields`` back. Other modes defer to FlowGRPO unchanged.
        """
        if not (self.ratio_norm and self.old_logp_source == "replay"):
            super().prepare_segment(conditions=conditions, segment=segment)
            return
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        segment.sde_logp = result.log_probs.detach().cpu()
        segment.sde_means = result.prev_sample_means.detach().cpu()

    def prepare_anchor_batch(
        self,
        *,
        updates: Sequence[Sequence[Tuple[Mapping[str, Condition], LatentSegment]]],
    ) -> None:
        """Freeze only the anchors that precede a weight-changing update.

        The unified stack partitions a rollout into disjoint optimizer updates.
        Update 0 runs at the same pre-update weights as the eager anchor, so its
        exact current replay can also serve as a detached old-policy anchor.
        Every later update is still replayed here, before optimizer 0, and stored
        on CPU. This preserves the policy state and exact bs=1 replay geometry
        while removing one anchor replay for every update-0 sample.
        """
        if not self.lazy_first_update_anchor:
            raise RuntimeError("prepare_anchor_batch requires lazy_first_update_anchor=True.")
        if self._prepared_anchor_updates is not None or self._active_anchor_entries is not None:
            raise RuntimeError("BagelFlowUniGRPO.prepare_anchor_batch: previous anchor state was not released.")
        if not updates:
            raise ValueError("BagelFlowUniGRPO.prepare_anchor_batch requires at least one optimizer update.")

        prepared_updates: List[Tuple[int, List[_PreparedAnchor]]] = []
        for update_index, micro_batches in enumerate(updates):
            entries: List[_PreparedAnchor] = []
            for conditions, segment in micro_batches:
                if int(segment.batch_size) != 1:
                    raise ValueError(
                        "BagelFlowUniGRPO.prepare_anchor_batch requires one image per micro-batch "
                        f"(BAGEL navit bs=1); got batch_size={segment.batch_size}."
                    )
                target_steps = tuple(self._resolve_target_steps(segment))
                if update_index == 0 or not target_steps:
                    entries.append(
                        _PreparedAnchor(
                            target_steps=target_steps,
                            replay=None,
                            derive_from_current=update_index == 0,
                        )
                    )
                    continue

                typed_conds = typed_conditions(conditions, self.conditions_cls)
                with torch.no_grad():
                    replay = self.stage.replay(
                        typed_conds,
                        segment=segment,
                        params=self.params,
                        step_indices=list(target_steps),
                    )
                if replay.prev_sample_means is None:
                    raise RuntimeError(
                        "BagelFlowUniGRPO.prepare_anchor_batch: exact anchor replay returned no prev_sample_means."
                    )
                entries.append(
                    _PreparedAnchor(
                        target_steps=target_steps,
                        replay=ReplayResult(
                            log_probs=replay.log_probs.detach().cpu(),
                            prev_sample_means=replay.prev_sample_means.detach().cpu(),
                        ),
                        derive_from_current=False,
                    )
                )
            prepared_updates.append((update_index, entries))
        self._prepared_anchor_updates = prepared_updates

    def _activate_anchor_update(self, *, update_index: int, expected_count: int) -> None:
        if not self.lazy_first_update_anchor:
            return
        if self._active_anchor_entries is not None:
            raise RuntimeError("BagelFlowUniGRPO: previous update left active anchor entries.")
        updates = self._prepared_anchor_updates
        if not updates:
            raise RuntimeError("BagelFlowUniGRPO: no prepared anchor update is available.")
        prepared_index, entries = updates.pop(0)
        if prepared_index != int(update_index):
            raise RuntimeError(
                "BagelFlowUniGRPO: prepared anchor update order mismatch: "
                f"prepared={prepared_index}, requested={int(update_index)}."
            )
        if len(entries) != int(expected_count):
            raise RuntimeError(
                "BagelFlowUniGRPO: prepared anchor micro-batch count mismatch: "
                f"prepared={len(entries)}, requested={int(expected_count)}."
            )
        self._active_anchor_entries = entries

    def _take_prepared_anchor(self, target_steps: Sequence[int]) -> Optional[_PreparedAnchor]:
        if not self.lazy_first_update_anchor:
            return None
        entries = self._active_anchor_entries
        if entries is None or not entries:
            raise RuntimeError("BagelFlowUniGRPO: prepared anchor queue was exhausted early.")
        prepared = entries.pop(0)
        expected = tuple(int(step) for step in target_steps)
        if prepared.target_steps != expected:
            raise RuntimeError(
                "BagelFlowUniGRPO: prepared anchor step indices do not match the consumed micro-batch: "
                f"prepared={prepared.target_steps}, current={expected}."
            )
        return prepared

    def finish_anchor_batch(self, *, succeeded: bool) -> None:
        remaining_updates = len(self._prepared_anchor_updates or ())
        remaining_active = len(self._active_anchor_entries or ())
        self._prepared_anchor_updates = None
        self._active_anchor_entries = None
        if succeeded and (remaining_updates or remaining_active):
            raise RuntimeError(
                "BagelFlowUniGRPO.finish_anchor_batch: training completed with unconsumed anchors: "
                f"updates={remaining_updates}, active_entries={remaining_active}."
            )

    def prepare_update_batch(
        self,
        *,
        micro_batches: Sequence[Tuple[Mapping[str, Condition], LatentSegment, torch.Tensor]],
        training_progress: float,
        loss_scale: float,
        update_index: int = 0,
    ) -> None:
        """Prepare detached MSE references under one weight swap per update.

        BAGEL image replay is constrained to one sample per micro-batch. The old
        path stashed, replaced, and restored every trainable full-FT shard for
        each sample. Here all current-policy text contexts are built first, then
        every ``v_ref`` is evaluated inside one reference-weight scope. The
        contexts and detached reference velocities are consumed later in the
        same order by :meth:`compute_loss_and_backward`.

        With ``reuse_ratio_context_for_mse``, the first phase runs each
        RatioNorm replay and backward, then retains only graph-free terminal K/V
        views. The reference and current-policy MSE phases reuse those exact
        contexts, removing the otherwise duplicated exact Stage-0 reconstruction.

        This hook runs once per optimizer update rather than once per rollout:
        under ``num_updates_per_batch > 1`` the later update therefore rebuilds
        its detached context after the preceding optimizer step, preserving the
        prior policy-state semantics.
        """
        if self._prepared_mse_batches is not None:
            raise RuntimeError(
                "BagelFlowUniGRPO.prepare_update_batch: the previous update left "
                f"{len(self._prepared_mse_batches)} unconsumed MSE micro-batches."
            )
        self._activate_anchor_update(update_index=update_index, expected_count=len(micro_batches))
        if self.mse_weight <= 0.0:
            self._prepared_mse_batches = None
            return

        # The expensive operation is the full-FT local-shard stash/copy/restore.
        # LoRA's adapter toggle is cheap and retains no fp32 model-sized stash, so
        # keep its established per-micro path and avoid retaining a whole update's
        # contexts.
        transformer = self.stage.model.transformer
        if self._has_lora(transformer):
            self._prepared_mse_batches = None
            return

        entries: List[Optional[_PreparedMSEBatch]] = [None] * len(micro_batches)
        pending: List[
            Tuple[
                int,
                LatentSegment,
                Tuple[int, ...],
                Dict[str, Any],
                torch.device,
                Optional[AlgorithmStepResult],
            ]
        ] = []
        for index, (conditions, segment, advantages) in enumerate(micro_batches):
            if int(segment.batch_size) != 1:
                raise ValueError(
                    "BagelFlowUniGRPO.prepare_update_batch requires one image per micro-batch "
                    f"(BAGEL navit bs=1); got batch_size={segment.batch_size}."
                )
            target_steps = tuple(self._resolve_target_steps(segment))
            if not target_steps or segment.sigmas is None:
                continue
            typed_conds = typed_conditions(conditions, self.conditions_cls)
            device = torch.device(self.stage.model.device)
            surrogate_result: Optional[AlgorithmStepResult] = None
            if self.reuse_ratio_context_for_mse:
                surrogate_result, forward_kwargs = self._ratio_norm_surrogate_with_context(
                    conditions=conditions,
                    segment=segment,
                    advantages=advantages,
                    training_progress=float(training_progress),
                    loss_scale=float(loss_scale),
                )
            else:
                with torch.no_grad():
                    forward_kwargs = self.stage.build_forward_kwargs(typed_conds, params=self.params, device=device)
            if self.context_gradient_mode == "stage_boundary":
                # no_grad prevents new graph construction, but does not detach
                # graph-bearing leaves supplied by an existing replay tree.
                forward_kwargs = self.stage.detach_forward_kwargs(forward_kwargs)
            pending.append((index, segment, target_steps, forward_kwargs, device, surrogate_result))

        if pending:
            with torch.no_grad():
                with self._reference_weights(transformer):
                    for index, segment, target_steps, forward_kwargs, device, surrogate_result in pending:
                        schedule = segment.sigmas.to(device)
                        v_refs = [
                            self.stage.predict_velocity_at(
                                forward_kwargs,
                                sample=segment.latents_at(step_idx)[0].to(device),
                                sigma=schedule[step_idx],
                                params=self.params,
                            ).detach()
                            for step_idx in target_steps
                        ]
                        entries[index] = _PreparedMSEBatch(
                            target_steps=target_steps,
                            forward_kwargs=forward_kwargs,
                            reference_velocities=v_refs,
                            surrogate_result=surrogate_result,
                        )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._prepared_mse_batches = entries

    def _take_prepared_mse(self, target_steps: Sequence[int]) -> Optional[_PreparedMSEBatch]:
        """Consume one prepared entry, or return None for direct-call fallback."""
        queue = self._prepared_mse_batches
        if queue is None:
            return None
        if not queue:
            raise RuntimeError("BagelFlowUniGRPO: prepared MSE micro-batch queue was exhausted early.")
        prepared = queue.pop(0)
        if not queue:
            # Restore the documented direct-call fallback immediately after the
            # last stacked micro-batch releases its cached context/reference.
            self._prepared_mse_batches = None
        if prepared is None:
            if target_steps:
                raise RuntimeError("BagelFlowUniGRPO: missing prepared MSE data for a trainable micro-batch.")
            return None
        if prepared.target_steps != tuple(int(step) for step in target_steps):
            raise RuntimeError(
                "BagelFlowUniGRPO: prepared MSE step indices do not match the consumed micro-batch: "
                f"prepared={prepared.target_steps}, current={tuple(target_steps)}."
            )
        return prepared

    def finish_update_batch(self, *, succeeded: bool) -> None:
        """Release prepared KV/reference tensors, including failed updates."""
        remaining = len(self._prepared_mse_batches or ())
        remaining_anchors = len(self._active_anchor_entries or ())
        self._prepared_mse_batches = None
        self._active_anchor_entries = None
        if succeeded and (remaining or remaining_anchors):
            raise RuntimeError(
                "BagelFlowUniGRPO.finish_update_batch: optimizer update completed with unconsumed state: "
                f"mse_batches={remaining}, anchor_entries={remaining_anchors}."
            )

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        needs_boundary_context = self.ratio_norm and self.context_gradient_mode == "stage_boundary"
        target_steps = self._resolve_target_steps(segment) if self.mse_weight > 0.0 or needs_boundary_context else []
        prepared_anchor = self._take_prepared_anchor(target_steps)
        prepared_mse = self._take_prepared_mse(target_steps) if self.mse_weight > 0.0 else None

        # The native Stage 0 -> Stage 1 engine boundary transfers values, not an
        # autograd graph. In the opt-in stage-boundary mode, mirror that contract:
        # use the exact current-policy context built under no_grad for both image
        # losses. Unified training supplies it from prepare_update_batch; direct
        # callers build it once here and retain the existing serial fallback.
        boundary_forward_kwargs: Optional[Dict[str, Any]] = None
        if needs_boundary_context and target_steps:
            if prepared_mse is not None:
                boundary_forward_kwargs = prepared_mse.forward_kwargs
            else:
                typed_conds = typed_conditions(conditions, self.conditions_cls)
                device = torch.device(self.stage.model.device)
                with torch.no_grad():
                    boundary_forward_kwargs = self.stage.build_forward_kwargs(
                        typed_conds,
                        params=self.params,
                        device=device,
                    )
                boundary_forward_kwargs = self.stage.detach_forward_kwargs(boundary_forward_kwargs)

        if needs_boundary_context:
            if not target_steps or segment.sigmas is None:
                return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
            if boundary_forward_kwargs is None:
                raise RuntimeError("Stage-boundary RatioNorm requires a prepared detached context.")

            v_refs: Optional[List[torch.Tensor]] = None
            if self.mse_weight > 0.0:
                if prepared_mse is not None:
                    v_refs = prepared_mse.reference_velocities
                else:
                    device = torch.device(self.stage.model.device)
                    schedule = segment.sigmas.to(device)
                    transformer = self.stage.model.transformer
                    full_ft_ref = not self._has_lora(transformer)
                    with torch.no_grad():
                        ref_ctx = self._reference_weights(transformer) if full_ft_ref else _disable_lora(transformer)
                        with ref_ctx as disabled:
                            if not full_ft_ref and not disabled:
                                raise RuntimeError(
                                    "BagelFlowUniGRPO: mse_weight > 0 but found neither peft LoRA layers "
                                    "to disable nor trainable params to snapshot as v_ref on "
                                    "stage.model.transformer. Train with a lora_cfg or full fine-tuning, "
                                    "or set mse_weight=0."
                                )
                            v_refs = [
                                self.stage.predict_velocity_at(
                                    boundary_forward_kwargs,
                                    sample=segment.latents_at(step_idx)[0].to(device),
                                    sigma=schedule[step_idx],
                                    params=self.params,
                                ).detach()
                                for step_idx in target_steps
                            ]
                    if full_ft_ref and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return self._stage_boundary_loss_and_backward(
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
                forward_kwargs=boundary_forward_kwargs,
                target_steps=target_steps,
                reference_velocities=v_refs,
                prepared_anchor=prepared_anchor,
            )

        # 1. Clipped surrogate (own backward). RatioNorm (GRPO-Guard) replaces the
        #    plain FlowGRPO ratio with the per-step normalized one when enabled;
        #    otherwise the inherited FlowGRPO surrogate.
        if prepared_mse is not None and prepared_mse.surrogate_result is not None:
            result = prepared_mse.surrogate_result
        elif self.ratio_norm:
            result = self._ratio_norm_surrogate(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
        else:
            result = super().compute_loss_and_backward(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
        if self.mse_weight <= 0.0 or not result.has_backward:
            return result
        if not target_steps or segment.sigmas is None:
            return result

        # 2. Velocity-MSE regularizer toward the LoRA-disabled base, at the SDE
        #    steps. Separate backward -> grads accumulate into the same step.
        # FSDP2 CPUOffloadPolicy keeps the decoder's parameter shards on CPU;
        # BAGEL's bundle device is the execution device used by the live
        # embeddings, heads, and FSDP all-gathers.
        device = torch.device(self.stage.model.device)
        schedule = segment.sigmas.to(device)
        # Reuse one detached conditioning context across every SDE step and both
        # v_theta / v_ref. Unified full-FT training prepares it before this call;
        # direct callers and LoRA build it in the fallback below. In either case it
        # is built at the live policy weights before entering the reference scope.
        # The surrogate replay above already trains through the reconstructed
        # T2TI text cache. Keep the MSE context detached: full-FT v_ref swaps
        # parameter shards in place, and mutating them after a grad-carrying
        # context prefill would invalidate autograd's version counters.
        if prepared_mse is not None:
            forward_kwargs = prepared_mse.forward_kwargs
            v_refs = prepared_mse.reference_velocities
        else:
            typed_conds = typed_conditions(conditions, self.conditions_cls)
            with torch.no_grad():
                forward_kwargs = self.stage.build_forward_kwargs(typed_conds, params=self.params, device=device)
            transformer = self.stage.model.transformer
            full_ft_ref = not self._has_lora(transformer)
            with torch.no_grad():
                ref_ctx = self._reference_weights(transformer) if full_ft_ref else _disable_lora(transformer)
                with ref_ctx as disabled:
                    if not full_ft_ref and not disabled:
                        raise RuntimeError(
                            "BagelFlowUniGRPO: mse_weight > 0 but found neither peft LoRA layers "
                            "to disable nor trainable params to snapshot as v_ref on "
                            "stage.model.transformer. Train with a lora_cfg or full fine-tuning, "
                            "or set mse_weight=0."
                        )
                    v_refs = [
                        self.stage.predict_velocity_at(
                            forward_kwargs,
                            sample=segment.latents_at(step_idx)[0].to(device),
                            sigma=schedule[step_idx],
                            params=self.params,
                        ).detach()
                        for step_idx in target_steps
                    ]
            if full_ft_ref and torch.cuda.is_available():
                torch.cuda.empty_cache()

        policy_velocities: List[torch.Tensor] = []
        for step_idx, v_ref in zip(target_steps, v_refs):
            x_t = segment.latents_at(step_idx)[0].to(device)  # [seq, C] (navit bs=1)
            sigma = schedule[step_idx]
            v_theta = self.stage.predict_velocity_at(forward_kwargs, sample=x_t, sigma=sigma, params=self.params)
            policy_velocities.append(v_theta)

        mse = self._velocity_mse(
            policy_velocities=policy_velocities,
            reference_velocities=v_refs,
            target_steps=target_steps,
        )
        (self.mse_weight * mse * loss_scale).backward()

        mse_val = float(mse.detach().item())
        return AlgorithmStepResult(
            loss=result.loss + self.mse_weight * mse_val,
            metrics={**dict(result.metrics), "velocity_mse": mse_val, "mse_weight": self.mse_weight},
            num_steps_or_tokens=result.num_steps_or_tokens,
            has_backward=True,
        )

    # ------------------------------------------------------------------
    # GRPO-Guard RatioNorm surrogate
    # ------------------------------------------------------------------

    @staticmethod
    def _velocity_mse(
        *,
        policy_velocities: Sequence[torch.Tensor],
        reference_velocities: Sequence[torch.Tensor],
        target_steps: Sequence[int],
    ) -> torch.Tensor:
        """Compute velocity MSE after validating one exact-shaped pair per step."""
        if len(policy_velocities) != len(reference_velocities) or len(policy_velocities) != len(target_steps):
            raise RuntimeError(
                "BAGEL velocity MSE count mismatch: "
                f"policy={len(policy_velocities)}, reference={len(reference_velocities)}, "
                f"steps={len(target_steps)}."
            )
        terms: List[torch.Tensor] = []
        for step_idx, v_theta, v_ref in zip(target_steps, policy_velocities, reference_velocities):
            if v_theta.shape != v_ref.shape:
                raise RuntimeError(
                    f"BAGEL velocity MSE shape mismatch at SDE step {int(step_idx)}: "
                    f"policy={tuple(v_theta.shape)}, reference={tuple(v_ref.shape)}. "
                    "Exact shape equality is required; broadcasting is not supported."
                )
            terms.append(((v_theta - v_ref) ** 2).mean())
        return torch.stack(terms).mean()

    def _stage_boundary_loss_and_backward(
        self,
        *,
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
        forward_kwargs: Dict[str, Any],
        target_steps: List[int],
        reference_velocities: Optional[List[torch.Tensor]],
        prepared_anchor: Optional[_PreparedAnchor] = None,
    ) -> AlgorithmStepResult:
        """Joint RatioNorm + velocity MSE at the detached native stage boundary."""
        replay, policy_velocities = self.stage.replay_from_forward_kwargs_with_velocities(
            forward_kwargs,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        anchor_replay: Optional[ReplayResult] = None
        if prepared_anchor is not None:
            anchor_replay = replay if prepared_anchor.derive_from_current else prepared_anchor.replay
            if anchor_replay is None:
                raise RuntimeError("Prepared BAGEL anchor contains no replay values for a trainable micro-batch.")
        policy_loss, metrics = self._ratio_norm_loss(
            replay=replay,
            segment=segment,
            advantages=advantages,
            training_progress=training_progress,
            target_steps=target_steps,
            anchor_replay=anchor_replay,
        )

        mse: Optional[torch.Tensor] = None
        if self.mse_weight > 0.0:
            if reference_velocities is None:
                raise RuntimeError("Stage-boundary velocity MSE requires prepared reference velocities.")
            mse = self._velocity_mse(
                policy_velocities=policy_velocities,
                reference_velocities=reference_velocities,
                target_steps=target_steps,
            )

        total_loss = policy_loss if mse is None else policy_loss + self.mse_weight * mse
        (total_loss * loss_scale).backward()

        result_metrics = dict(metrics)
        if mse is not None:
            result_metrics.update(velocity_mse=float(mse.detach().item()), mse_weight=self.mse_weight)
        return AlgorithmStepResult(
            loss=float(total_loss.detach().item()),
            metrics=result_metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _ratio_norm_surrogate(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        result, _ = self._ratio_norm_surrogate_impl(
            conditions=conditions,
            segment=segment,
            advantages=advantages,
            training_progress=training_progress,
            loss_scale=loss_scale,
            retain_forward_kwargs=False,
        )
        return result

    def _ratio_norm_surrogate_with_context(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> Tuple[AlgorithmStepResult, Dict[str, Any]]:
        result, forward_kwargs = self._ratio_norm_surrogate_impl(
            conditions=conditions,
            segment=segment,
            advantages=advantages,
            training_progress=training_progress,
            loss_scale=loss_scale,
            retain_forward_kwargs=True,
        )
        if forward_kwargs is None:
            raise RuntimeError("RatioNorm context reuse requested but replay produced no forward kwargs.")
        return result, forward_kwargs

    def _ratio_norm_surrogate_impl(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
        retain_forward_kwargs: bool,
    ) -> Tuple[AlgorithmStepResult, Optional[Dict[str, Any]]]:
        """FlowGRPO clipped surrogate with GRPO-Guard RatioNorm.

        The flow importance ratio is left-shifted (mean < 1) and step-inconsistent,
        so plain clipping fails. RatioNorm normalizes the per-SDE-step log-ratio::

            log r̂ = std_var · ( log r + mean(Δμ²) / (2·std_var²) )

        where ``std_var = σ_t·√(-dt)`` is exactly FlowSDEStrategy's per-step SDE std,
        ``Δμ = μ_old − μ_θ`` (``μ_old`` = rollout SDE mean ``segment.sde_means``;
        ``μ_θ`` = replay mean), and ``mean(Δμ²)`` is over elements to match the
        mean-reduced ``log r``. The additive term cancels the ``−‖Δμ‖²/(2σ²dt)`` bias
        (mean → 0, i.e. ``r̂`` mean → 1); the ``σ_t√dt`` factor unifies variance
        across steps. The clip then runs on ``r̂``. With ``grad_reweight`` each step's
        loss is also scaled by the normalized ``1/|dt|`` (GRPO-Guard gradient
        balancing). Mirrors :meth:`FlowGRPO.compute_loss_and_backward` otherwise.

        Logs ``rn_raw_ratio_mean`` (the PRE-RatioNorm ratio): on an off-policy update
        it should be < 1 while ``ratio_mean`` (post-RatioNorm) ≈ 1 — the smoke check
        that RatioNorm is centering correctly. (On the on-policy update both ≈ 1.)
        """
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False), None
        if segment.sde_means is None:
            raise RuntimeError(
                "BagelFlowUniGRPO(ratio_norm=True): segment.sde_means is None. RatioNorm needs the rollout "
                "to store per-SDE-step μ_old; ensure BagelDiffusionStage.diffuse records sde_means."
            )
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        forward_kwargs: Optional[Dict[str, Any]] = None
        if retain_forward_kwargs:
            replay, forward_kwargs = self.stage.replay_with_detached_forward_kwargs(
                typed_conds,
                segment=segment,
                params=self.params,
                step_indices=target_steps,
            )
        else:
            replay = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        loss, metrics = self._ratio_norm_loss(
            replay=replay,
            segment=segment,
            advantages=advantages,
            training_progress=training_progress,
            target_steps=target_steps,
        )
        (loss * loss_scale).backward()
        return (
            AlgorithmStepResult(
                loss=float(loss.detach().item()),
                metrics=metrics,
                num_steps_or_tokens=len(target_steps),
                has_backward=True,
            ),
            forward_kwargs,
        )

    def _ratio_norm_loss(
        self,
        *,
        replay: Any,
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        target_steps: List[int],
        anchor_replay: Optional[ReplayResult] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Build the RatioNorm loss without choosing a backward schedule."""
        new_logp = replay.log_probs  # [1, S']
        mu_theta = replay.prev_sample_means  # [1, S', seq, C]
        if mu_theta is None:
            raise RuntimeError("BagelFlowUniGRPO(ratio_norm=True): stage.replay returned no prev_sample_means (μ_θ).")
        if anchor_replay is None:
            old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
                dtype=new_logp.dtype, device=new_logp.device
            )
            mu_old = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
                dtype=mu_theta.dtype, device=mu_theta.device
            )
        else:
            if anchor_replay.prev_sample_means is None:
                raise RuntimeError("Prepared BAGEL anchor replay returned no prev_sample_means (μ_old).")
            old_logp = anchor_replay.log_probs.detach().to(dtype=new_logp.dtype, device=new_logp.device)
            mu_old = anchor_replay.prev_sample_means.detach().to(dtype=mu_theta.dtype, device=mu_theta.device)
            if old_logp.shape != new_logp.shape or mu_old.shape != mu_theta.shape:
                raise RuntimeError(
                    "Prepared BAGEL anchor shape mismatch: "
                    f"old_logp={tuple(old_logp.shape)}, new_logp={tuple(new_logp.shape)}, "
                    f"mu_old={tuple(mu_old.shape)}, mu_theta={tuple(mu_theta.shape)}."
                )
        # std_var must use the same sigma_max as the SDE step that produced old/new log_probs
        # (diffuse/replay pass schedule[1]); otherwise the two disagree at the σ=1 step.
        sde_sigma_max = float(segment.sigmas[1]) if int(segment.sigmas.shape[0]) > 1 else float(segment.sigmas[0])
        std_var = self._sde_std_var(
            segment.sigmas,
            target_steps,
            eta=float(self.params.eta),
            device=new_logp.device,
            dtype=new_logp.dtype,
            sigma_max=sde_sigma_max,
        )  # [1, S']

        log_r = new_logp - old_logp  # [1, S']
        delta_mu = mu_old - mu_theta  # [1, S', seq, C]
        mean_dmu2 = (delta_mu**2).mean(dim=tuple(range(2, delta_mu.ndim)))  # [1, S'] mean over elements
        log_r_hat = std_var * (log_r + mean_dmu2 / (2.0 * std_var**2))  # [1, S']

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
        # Feed the RatioNorm'd ratio: new' − old = log r̂, so _grpo_clip_loss uses exp(log r̂) = r̂.
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=old_logp + log_r_hat, old_logp=old_logp, advantages=adv_b, clip_range=clip_range
        )
        if self.grad_reweight:
            inv_dt = self._sde_inv_dt(segment.sigmas, target_steps, device=new_logp.device, dtype=new_logp.dtype)
            weight = inv_dt / inv_dt.mean().clamp_min(1e-12)  # normalize to mean 1 (keep loss scale)
            loss = (loss_per_elem * weight).mean()
        else:
            loss = loss_per_elem.mean()

        with torch.no_grad():
            raw_ratio_mean = float(torch.exp(log_r).mean().item())
        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
            "rn_raw_ratio_mean": raw_ratio_mean,  # pre-RatioNorm (off-policy: <1); ratio_mean is post (≈1)
            "rn_delta_mu_sq_mean": float(mean_dmu2.mean().item()),
            "ratio_norm": 1.0,
            "grad_reweight": float(bool(self.grad_reweight)),
        }
        return loss, metrics

    @staticmethod
    def _sde_std_var(
        sigmas: torch.Tensor,
        target_steps: List[int],
        *,
        eta: float,
        device: Any,
        dtype: Any,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        """Per-SDE-step ``std_var = σ_t·√(-dt)`` — byte-matches ``FlowSDEStrategy.step``.

        ``σ_t = η·√(σ/(1-σ))`` (σ=1 clamped via ``sigma_max``), ``dt = σ_next − σ`` (<0).
        Returns ``[1, len(target_steps)]`` so it broadcasts against the ``[1, S']`` ratios.
        """
        sig = sigmas.to(device=device, dtype=torch.float32)
        vals: List[torch.Tensor] = []
        for s in target_steps:
            sigma = sig[s]
            sigma_next = sig[s + 1]
            dt = sigma_next - sigma  # negative (sigma decreases)
            denom = 1.0 - (sigma_max if float(sigma) == 1.0 else float(sigma))
            std_dev_t = torch.sqrt(sigma / denom) * float(eta)
            vals.append(std_dev_t * torch.sqrt(-dt))
        return torch.stack(vals).to(dtype=dtype).reshape(1, -1)

    @staticmethod
    def _sde_inv_dt(
        sigmas: torch.Tensor,
        target_steps: List[int],
        *,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        """Per-SDE-step ``1/|dt| = 1/(σ − σ_next)`` for the GRPO-Guard gradient reweight."""
        sig = sigmas.to(device=device, dtype=torch.float32)
        vals = [1.0 / float(sig[s] - sig[s + 1]) for s in target_steps]
        return torch.tensor(vals, device=device, dtype=dtype).reshape(1, -1)


__all__ = ["BagelFlowUniGRPO"]
