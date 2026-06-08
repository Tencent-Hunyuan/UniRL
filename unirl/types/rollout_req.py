"""RolloutReq — top-level SoA container for one rollout's worth of inputs.

Pairs with ``RolloutResp`` (in ``unirl/types/rollout_resp.py``). Carries:

- ``primitives: Dict[str, Texts | Images | Videos | Audios]`` — raw inputs
  keyed by modality-slot name (``"text"``, ``"image"``, ...). The pipeline
  encodes each via the relevant ``EncodeStage`` / ``EmbedStage`` before
  generation.
- ``request_conditions: Dict[str, Condition]`` — precomputed encoded inputs
  the engine should consume verbatim instead of (re-)deriving from
  ``primitives``. Symmetric with ``RolloutResp.tracks[<slot>].conditions``. Key convention:
  ``"initial_latents"`` → ``ImageLatentCondition(latents=x_T_or_init_img)``
  for engines that accept a precomputed start-of-denoising tensor (SGLang's
  ``Req.latents``, vllm-omni's per-stage init-latents). Future keys: other
  typed engine-bound inputs land under their own slot.
- ``sampling_params: Optional[BaseSamplingParams]`` — typed sampling config.
  Holds ``DiffusionSamplingParams`` for pure diffusion pipelines,
  ``ARSamplingParams`` for pure AR pipelines, or
  ``ComposedSamplingParams`` for composed (PE) pipelines.
  Use ``get_diffusion_params()`` / ``get_ar_params()`` to extract the
  relevant sub-config.
- ``stage_config: Dict[str, Any]`` — model-specific routing metadata
  (``"task"``, ``"bot_task"``, ``"sys_type"``, ``"chat"``).
- ``collect_media_preview`` / ``media_max_items`` — operational flags for
  media preview collection on the actor side.
- ``sigmas: Optional[torch.Tensor]`` — the σ schedule for this rollout,
  computed main-side via
  :func:`unirl.sde.runtime.ensure_req_sigmas` (which applies the
  engine's :class:`FlowMatchSchedulePolicy` to the per-request
  ``(T, H, W)`` triple) and populated by the rollout-engine adapter
  just before dispatch. This is
  the **single source of truth** for σ across all rollout backends
  (trainside / sglang / vllm-omni): every engine MUST consume this
  schedule rather than computing its own, and the response handler
  asserts the schedule the engine actually used (echoed back via
  ``LatentSegment.sigmas``) matches what was sent. ``None`` only at
  request-construction time (driver-side, in the trainer's ``_build_req``);
  engines populate it before forwarding. Shape ``[T+1]`` (length includes the
  terminal 0), values in ``[0, 1]``, ``float32``, host-device-agnostic
  (engines move to worker device when serializing).
- ``sample_ids`` / ``group_ids`` — mirror ``RolloutResp`` so request and
  response can be correlated by ID.

Tracks fan-out helper: :meth:`RolloutReq.make_root_track` produces the
first ``RolloutTrack`` of a fan-out tree (``x → x*branch`` samples). The
companion :meth:`RolloutTrack.fork_track` (in ``rollout_resp.py``) produces
subsequent levels (``N → N*branch`` samples per call). Both use group-by-
parent ordering so per-group ops in downstream code reduce to single
``view(n_parent, branch).reduce(dim=1)`` reshapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field, shared_field
from unirl.types.conditions.base import Condition
from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.sampling import BaseSamplingParams

PrimitiveValue = Union[Texts, Images, Videos, Audios]


@dataclass
class RolloutReq(Batch):
    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)
    primitives: Dict[str, PrimitiveValue] = field(kind=FieldKind.CONCAT, default_factory=dict)
    request_conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, default_factory=dict)
    sampling_params: Optional[BaseSamplingParams] = shared_field(default=None)
    stage_config: Dict[str, Any] = shared_field(default_factory=dict)
    collect_media_preview: bool = shared_field(default=False)
    media_max_items: int = shared_field(default=8)
    # σ schedule is shared across all samples in the request — every
    # sample runs the same num_inference_steps / shift / dynamic-shift μ
    # by construction (geometry varies per-sample only via height/width,
    # but the driver fixes those per-batch at request construction). Hence ``shared_field``.
    sigmas: Optional[torch.Tensor] = shared_field(default=None)
    metadata: List[Optional[Dict[str, Any]]] = concat_field(default_factory=list)
    # Driver-authored x_T RECIPE: per-sample INITIAL-noise group ids (rollout-keyed
    # on the STABLE sample id, e.g. "r5:prompt:42:sample:3") + the latent shape.
    # Each engine regenerates the same x_T via generate_shared_noise(CPU-fp32) keyed
    # on these; base_seed rides on sampling_params.seed. This makes the driver the
    # single source of initial noise, so every engine starts each rollout from a
    # byte-identical x_T (otherwise each engine draws its own RNG → cross-engine
    # divergence). Named ``init_noise_*`` to stay distinct from the SGLang on-wire
    # ``noise_group_ids`` kwarg, which groups per-STEP SDE noise (sourced from
    # ``group_ids``), NOT the initial latent. CONCAT so per-sample ids slice per DP
    # shard exactly like sample_ids (the id string keys noise deterministically, so
    # sharding is order-independent). Empty/None ⇒ no driver recipe (engine draws
    # its own).
    init_noise_group_ids: List[str] = concat_field(default_factory=list)
    init_noise_latent_shape: Optional[List[int]] = shared_field(default=None)

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    # ---- tracks fan-out helper ---------------------------------------------

    def make_root_track(
        self,
        track_name: str,
        branch: int,
        decode_to_condition: Optional[Callable[["RolloutReq"], Dict[str, Condition]]] = None,
        new_segment: Optional[Any] = None,
    ) -> Any:
        """First fan-out: ``x`` request prompts → ``x*branch`` root-track samples.

        The new track has ``parent_track = None`` (parent is this request)
        and ``parent_ids = self.sample_ids`` repeated ``branch`` times in
        group-by-parent order (``[p0, p0, …, p0, p1, p1, …, p1, …]``).
        Hierarchical sample IDs: ``f"{self.sample_ids[i]}/{track_name[0]}{j}"``.

        :param track_name: Name to register the new track under in the
            enclosing ``RolloutResp.tracks``. Also drives the ID prefix
            (first character).
        :param branch: Replication factor (``y`` in the prompt-enhancement
            use case: each prompt → ``y`` refined-prompt candidates).
        :param decode_to_condition: Callable mapping ``self`` to a
            ``Dict[str, Condition]`` at request batch_size (one entry per
            request prompt). Each condition is replicated ``branch``× via
            :meth:`Batch.repeat_interleave`. If ``None`` (default), uses
            ``self.request_conditions`` verbatim.
        :param new_segment: Optional initial segment to set on the new
            track. Most callers leave this ``None`` and let the rollout
            pipeline populate the segment after generation.
        :return: A new :class:`RolloutTrack` of size ``len(self.sample_ids) * branch``.
        """
        # Deferred import to avoid a top-level circular dep edge between
        # rollout_req.py and rollout_resp.py — both are commonly imported
        # near the same package init seam, and the type-only annotations
        # above use ``from __future__ import annotations`` so the runtime
        # import only needs to resolve when this method is actually called.
        from unirl.types.rollout_resp import RolloutTrack

        if not self.sample_ids:
            raise ValueError("RolloutReq.make_root_track: req has no sample_ids")
        if branch < 1:
            raise ValueError(f"RolloutReq.make_root_track: branch must be >= 1, got {branch}")

        prefix = track_name[0] if track_name else "c"
        child_sample_ids = [f"{pid}/{prefix}{j}" for pid in self.sample_ids for j in range(branch)]
        child_parent_ids = [pid for pid in self.sample_ids for _ in range(branch)]

        raw_conditions = decode_to_condition(self) if decode_to_condition is not None else dict(self.request_conditions)
        child_conditions = {k: cond.repeat_interleave(branch) for k, cond in raw_conditions.items()}

        return RolloutTrack(
            sample_ids=child_sample_ids,
            parent_ids=child_parent_ids,
            parent_track=None,
            conditions=child_conditions,
            segment=new_segment,
            decoded=None,
        )


__all__ = ["RolloutReq", "PrimitiveValue"]
