"""RolloutResp + RolloutTrack — typed containers for one rollout's worth of data.

Multi-track architecture:

- ``RolloutResp.tracks: Dict[str, RolloutTrack]`` — top-level container.
- ``RolloutTrack`` — coherent rollout slice; one modality, one lifecycle stage.
  Tracks within a ``RolloutResp`` are linked by per-track ``parent_track`` +
  per-sample ``parent_ids``, forming a fan-out tree.

Use case: prompt-enhancement RL (x prompts → x*y refined → x*y*z images),
where the refined and image tracks have different decoding modalities and
need separate trainers. The track structure makes lineage explicit and
keeps each track self-contained for trainer-side consumption.

Per-track invariants enforced in ``RolloutResp.__post_init__``:

- ``parent_track`` must reference an existing sibling track key (when set).
- ``len(parent_ids) == len(sample_ids)`` (when both set).
- ``parent_ids`` ⊆ parent track's ``sample_ids`` (when both ``parent_track``
  and ``parent_ids`` are set; foreign-key check).

Concat: both ``RolloutResp.concat`` and ``RolloutTrack.concat`` are the
default ``Batch.concat`` (dict-union with per-value concat). Segment rows
are 1:1 with track samples by construction, so shard merge needs no index
remapping.

Per-sample / per-track access (no resp-level shim — read directly off the
track). Segment rows align 1:1 with track samples, so per-segment-row
fields (like LatentSegment's latents) are plain row reads::

    track = resp.tracks["image"]
    latents_i = track.segment.latents[i]

For packed varlen fields (like TextSegment's tokens), use the framework-
managed ``cu_seqlens`` to slice each sample's chunk::

    track = resp.tracks["ar"]
    cu = track.segment.cu_seqlens
    tokens_i = track.segment.tokens[cu[i]:cu[i + 1]]

Pairs with ``RolloutReq`` (in ``unirl/types/rollout_req.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch

from unirl.distributed.tensor.batch import (
    Batch,
    FieldKind,
    concat_field,
    field,
    max_field,
    shared_field,
)
from unirl.distributed.tensor.ref import hydrate
from unirl.types.conditions import Condition
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.segments import Segment
from unirl.utils.shard_balance import lpt_shard_permutation, shard_token_spread

logger = logging.getLogger(__name__)

TR = TypeVar("TR", bound="RolloutTrack")
TT = TypeVar("TT", bound="RolloutResp")

Decoded = Union[Texts, Images, Videos, Audios]


@dataclass
class RolloutTrack(Batch):
    """SoA container for one coherent rollout — one modality, one lifecycle stage.

    Lineage is per-track: ``parent_track`` names this track's parent in the
    enclosing ``RolloutResp.tracks`` dict (or ``None`` if the parent is the
    request, e.g. for a root track produced via ``RolloutReq.make_root_track``).
    ``parent_ids[i]`` is the parent sample id for sample ``i``; sibling samples
    sharing the same ``parent_ids[i]`` form a group for GRPO-style normalization.

    Conditions are self-contained — when forking a child track from a parent
    track, the parent's decoded outputs are replicated into the child's
    ``conditions`` (no parent-id resolution at trainer time).

    ``decoded`` is this track's decoded output (``Texts`` / ``Images`` /
    ``Videos`` / ``Audios``) — or ``None`` when nothing has been decoded yet
    for this track. Each track holds one modality, so a single value suffices;
    different modalities live in different tracks.
    """

    sample_ids: List[str] = concat_field(default_factory=list)
    parent_ids: Optional[List[str]] = concat_field(default=None)
    parent_track: Optional[str] = shared_field(default=None)

    conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, default_factory=dict)
    segment: Optional[Segment] = field(kind=FieldKind.CONCAT, default=None)
    decoded: Optional[Decoded] = field(kind=FieldKind.CONCAT, default=None)
    # Parallel secondary media decoded alongside ``decoded`` for the SAME samples
    # (not a separate track — keeps GRPO grouping / advantage alignment intact).
    # First consumer: LTX-2.3 T2AV, where ``decoded`` holds the video and
    # ``decoded_audio`` the jointly-generated audio waveform. The reward service
    # injects it into the reward request as a side-channel so a composite scorer
    # can read video + audio together. ``None`` for every single-modality track.
    decoded_audio: Optional[Audios] = field(kind=FieldKind.CONCAT, default=None)
    # Source sample rate (Hz) of ``decoded_audio`` waveforms (e.g. the LTX-2
    # vocoder's output rate). Batch-shared metadata — one rate per track. The
    # reward service forwards it to audio scorers via RewardRequest.audio_sample_rate.
    audio_sample_rate: Optional[int] = shared_field(default=None)
    media_preview: Optional[MediaPreview] = concat_field(default=None)

    rewards: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    status: Optional[torch.Tensor] = concat_field(default=None)

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    def metadata_only(self) -> "RolloutTrack":
        """Return a light copy with heavy payload fields reset to defaults.

        Drops ``conditions``, ``segment``, and ``decoded`` — the heavy
        per-sample data — while preserving lineage metadata
        (``sample_ids``, ``parent_ids``, ``parent_track``), rewards,
        advantages, and status.
        """
        import copy

        light = copy.copy(self)
        light.conditions = {}
        light.segment = None
        light.decoded = None
        light.decoded_audio = None
        return light

    @property
    def group_ids(self) -> List[str]:
        """Equivalence-class labels for grouping (e.g. GRPO normalization).

        For tracks with explicit lineage, this is ``parent_ids`` — siblings
        sharing a parent are in one group. For root tracks (parent_ids=None),
        each sample is its own group.
        """
        if self.parent_ids is not None:
            return list(self.parent_ids)
        return list(self.sample_ids)

    def split(self) -> List["RolloutTrack"]:
        """Split into one ``RolloutTrack`` per group-id equivalence class.

        Reads :attr:`group_ids` (derived from ``parent_ids`` or
        ``sample_ids``); per-group shards built via :meth:`Batch.select`.
        """
        gids = self.group_ids
        if not gids:
            return [self]
        groups: Dict[str, List[int]] = {}
        for i, gid in enumerate(gids):
            groups.setdefault(gid, []).append(i)
        results: List[RolloutTrack] = []
        for gid in dict.fromkeys(gids):
            indices = torch.tensor(groups[gid], dtype=torch.long)
            results.append(self.select(indices))
        return results

    def balance_shards(self, num_shards: int, *, min_spread: float = 0.05) -> "RolloutTrack":
        """Reorder samples so ``num_shards`` equal contiguous shards carry ~equal tokens.

        verl ``trainer.balance_batch`` parity. The consumer (``DP_SCATTER``) cuts
        the batch into ``num_shards`` equal contiguous chunks, so every shard keeps
        the same SAMPLE count — this only equalizes the per-shard TOKEN count, via
        greedy LPT (:func:`unirl.utils.shard_balance.lpt_shard_permutation`).
        Reordering is safe because advantages are already attached per sample.

        Returns ``self`` unchanged when balancing cannot apply (no segment lengths,
        ``num_shards <= 1``, batch size not divisible by ``num_shards``) or when the
        shards are already within ``min_spread`` of balanced — permuting an
        already-balanced batch only forces needless cross-shard row movement at
        ``DP_SCATTER``. The reorder is applied via native zero-copy :meth:`select`,
        so data stays worker-resident and materializes on the destination worker.

        Args:
            num_shards: Number of equal contiguous shards (the DP size).
            min_spread: Skip balancing when the current per-shard token spread
                (max-min over mean) is already below this fraction.

        Returns:
            A token-balanced copy of the track, or ``self`` if no reorder applies.
        """
        if self.segment is None or self.segment.lengths is None or num_shards <= 1:
            return self
        total = self.batch_size
        if total % num_shards != 0:
            return self
        lengths = [int(x) for x in self.segment.lengths.tolist()]
        if len(lengths) != total:
            logger.warning("balance_shards: lengths (%d) != batch_size (%d); skipping.", len(lengths), total)
            return self

        before = shard_token_spread(lengths, num_shards)
        if before < min_spread:
            return self

        perm = lpt_shard_permutation(lengths, num_shards)
        after = shard_token_spread([lengths[i] for i in perm], num_shards)
        logger.info("balance_shards: token spread %.1f%% -> %.1f%%", 100 * before, 100 * after)
        return self.select(perm)

    # ---- track-to-track fan-out helper -------------------------------------

    def fork_track(
        self,
        parent_name: str,
        child_name: str,
        branch: int,
        decode_to_condition: Optional[Callable[["RolloutTrack"], Dict[str, Condition]]] = None,
        new_segment: Optional[Segment] = None,
    ) -> "RolloutTrack":
        """Track-to-track fan-out: ``N`` self-samples → ``N*branch`` child samples.

        The new track has ``parent_track = parent_name`` and ``parent_ids =
        self.sample_ids`` repeated ``branch`` times in group-by-parent order
        (``[s0, s0, …, s0, s1, s1, …, s1, …]``). Hierarchical sample IDs:
        ``f"{self.sample_ids[i]}/{child_name[0]}{j}"``.

        :param parent_name: Name self is registered under in the enclosing
            ``RolloutResp.tracks`` dict. Becomes the child's ``parent_track``
            string. Caller is responsible for ensuring this matches the dict
            key when assembling the resp; the foreign-key check in
            ``RolloutResp.__post_init__`` will fire if it doesn't.
        :param child_name: Name for the new child track. First character is
            the ID prefix in hierarchical sample_ids.
        :param branch: Replication factor (``z`` in the prompt-enhancement
            use case: each refined-prompt → ``z`` image candidates).
        :param decode_to_condition: Optional callable mapping ``self`` to a
            ``Dict[str, Condition]`` at self's batch_size (one entry per
            parent sample). Each condition is replicated ``branch``× via
            :meth:`Batch.repeat_interleave`. If ``None``, the child track
            has empty conditions; the caller is expected to populate them
            later (e.g. once a real text encoder is run on this track's
            decoded outputs).
        :param new_segment: Optional initial segment. Most callers leave
            ``None`` and let the rollout pipeline populate it.
        :return: A new :class:`RolloutTrack` of size ``len(self.sample_ids) * branch``.
        """
        if not self.sample_ids:
            raise ValueError("RolloutTrack.fork_track: track has no sample_ids")
        if branch < 1:
            raise ValueError(f"RolloutTrack.fork_track: branch must be >= 1, got {branch}")

        prefix = child_name[0] if child_name else "c"
        child_sample_ids = [f"{pid}/{prefix}{j}" for pid in self.sample_ids for j in range(branch)]
        child_parent_ids = [pid for pid in self.sample_ids for _ in range(branch)]

        if decode_to_condition is None:
            child_conditions: Dict[str, Condition] = {}
        else:
            raw_conditions = decode_to_condition(self)
            child_conditions = {k: cond.repeat_interleave(branch) for k, cond in raw_conditions.items()}

        return RolloutTrack(
            sample_ids=child_sample_ids,
            parent_ids=child_parent_ids,
            parent_track=parent_name,
            conditions=child_conditions,
            segment=new_segment,
            decoded=None,
        )

    # ---- per-group advantage computation -----------------------------------

    def compute_advantages(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
        scope: str = "group",
        use_global_std: bool = False,
        group_ids: Optional[List[str]] = None,
    ) -> "RolloutTrack":
        """GRPO-style per-group advantage: ``(reward - group_mean) / (group_std + eps)``.

        Groups are equivalence classes of :attr:`group_ids` (i.e. ``parent_ids``,
        falling back to per-sample groups when ``parent_ids is None``). Group-by-
        parent ordering is required — sibling samples must be consecutive — so
        the computation reduces to a single ``view(n_groups, branch).reduce(dim=1)``
        reshape rather than scatter ops.

        :param normalize: ``True`` (default) divides by ``group_std + eps``;
            ``False`` returns ``reward - group_mean`` (mean-centering only).
        :param eps: Numerical floor on ``group_std`` to prevent division by
            zero on uniform-reward groups (e.g. all rewards equal).
        :param scope: ``"group"`` (default) centers/normalizes within each
            prompt's sibling group (textbook GRPO). ``"global"`` centers and
            normalizes across the whole batch — ``(r - mean_all)/(std_all + eps)``
            — matching the v1 ``adv_normalization_scope=global`` baseline. Global
            scope gives every sample a nonzero signal vs the batch mean, whereas
            group scope zeroes out all-correct/all-wrong prompts (std=0 → adv=0).
        :param use_global_std: Only meaningful with ``scope="group"``. When
            ``True``, keep the per-group mean but divide every group by ONE
            batch-wide std (unbiased/Bessel, ``eps`` outside the sqrt) instead of
            each group's own std. Same *formula* as v1
            ``normalize_grouped(use_global_std=True)`` (``algorithms/normalizers.py``),
            but reduced over the **full** driver-side batch — NOT a bit-for-bit
            reproduction of the v1 run. v1 computed advantages per rollout actor,
            so its std spanned a single shard (``global_batch / actor_count``
            prompts); the v2 single-controller reduces over all groups at once.
            The two share an expectation (both estimate the population reward std),
            but the full-batch scope is intentional — topology-independent and
            lower-variance. Used by the FlowDPPO recipe; left ``False`` elsewhere.
        :param group_ids: Optional per-sample grouping labels (length =
            ``batch_size``) that OVERRIDE the track's own ``parent_ids`` for the
            ``scope="group"`` computation. Used to group at a different lineage
            level than the immediate parent — e.g. PE's prompt-level grouping
            groups the diffusion track by its ROOT prompt id (all N×M images of a
            prompt) instead of by rewrite (M images), so a rewrite that
            systematically beats the prompt-wide mean gets non-zero advantage.
            Resolve these via :meth:`RolloutResp.compute_track_advantages` rather
            than by hand. Must already be in group-by-parent contiguous order (the
            same invariant ``parent_ids`` satisfies — PE's lineage keeps a
            prompt's images consecutive). ``None`` (default) ⇒ use ``parent_ids``,
            exactly today's behavior. Ignored when ``scope="global"``.
        :return: A new :class:`RolloutTrack` with ``advantages`` set.

        Population std (``unbiased=False``) is used so the math degenerates
        gracefully on single-sample groups (``branch=1``): variance is 0, and
        ``adv = (r - r) / sqrt(eps) = 0``.
        """
        if self.rewards is None:
            raise ValueError("RolloutTrack.compute_advantages: track has no rewards")
        n = len(self.sample_ids)
        if n == 0:
            return self  # trivially nothing to do

        # The reward service runs on workers; its returned ``rewards`` arrives
        # at the driver as a TensorRef proxy (Worker._pack_output dehydrates
        # every Tensor leaf). Driver-side arithmetic below needs a real Tensor.
        rewards_local = hydrate(self.rewards)

        # Global scope: normalize across the whole batch, ignoring group
        # structure (reproduces v1 normalize_global). std() is unbiased (Bessel)
        # with eps added outside the sqrt, matching algorithms/normalizers.py.
        if scope == "global":
            rewards_g = rewards_local.to(torch.float32)
            if normalize:
                adv_g = (rewards_g - rewards_g.mean()) / (rewards_g.std() + eps)
            else:
                adv_g = rewards_g - rewards_g.mean()
            return _track_with_field(self, "advantages", adv_g)

        # Grouping labels: an explicit ``group_ids`` override (e.g. root-prompt
        # ids resolved by RolloutResp.compute_track_advantages) wins; otherwise
        # the track's own ``parent_ids`` (today's default). The override must
        # align 1:1 with samples and obey the same group-by-parent contiguity
        # invariant checked below.
        if group_ids is not None and len(group_ids) != n:
            raise ValueError(
                f"compute_advantages: group_ids length {len(group_ids)} != "
                f"sample count {n}; the override must be one label per sample."
            )
        group_labels = self.parent_ids if group_ids is None else list(group_ids)

        # Root track (no grouping labels) — each sample is its own group, so
        # advantage = 0 for every sample (a single-sample group's mean equals
        # itself; centered = 0).
        if group_labels is None:
            return _track_with_field(
                self,
                "advantages",
                torch.zeros_like(rewards_local, dtype=torch.float32),
            )

        # Detect uniform group sizes via group-by-parent contiguous ordering.
        unique_pids = list(dict.fromkeys(group_labels))
        n_groups = len(unique_pids)
        if n_groups == 0 or n % n_groups != 0:
            raise ValueError(
                f"compute_advantages: non-uniform group sizes (n={n}, "
                f"n_groups={n_groups}). Expected uniform branching factor with "
                f"group-by-parent ordering — use fork_track / make_root_track to "
                f"build the track."
            )
        branch = n // n_groups
        expected_parent_ids = [pid for pid in unique_pids for _ in range(branch)]
        if list(group_labels) != expected_parent_ids:
            raise ValueError(
                "compute_advantages: grouping labels not in group-by-parent contiguous "
                "order. Siblings must be consecutive (use fork_track / "
                "make_root_track), got interleaved ordering."
            )

        rewards = rewards_local.to(torch.float32)
        reshaped = rewards.view(n_groups, branch)
        mean = reshaped.mean(dim=1, keepdim=True)
        if normalize:
            if use_global_std:
                # ``use_global_std``: per-group mean, but ONE batch-wide std
                # (unbiased/Bessel, eps OUTSIDE the sqrt) shared across groups, so
                # every prompt stays on a single reward scale instead of being
                # unit-normalized per group. Same formula as v1 normalize_grouped
                # (algorithms/normalizers.py), but reduced over the full driver
                # batch — not v1's per-actor shard. Scalar broadcasts [n_groups, branch].
                std = rewards.std() + eps
            else:
                # Population std (unbiased=False) handles branch=1 cleanly: var=0,
                # adv = 0 / sqrt(eps) = 0. unbiased=True would NaN on single samples.
                std = (reshaped.var(dim=1, unbiased=False, keepdim=True) + eps).sqrt()
            adv = (reshaped - mean) / std
        else:
            adv = reshaped - mean
        return _track_with_field(self, "advantages", adv.flatten())


def _root_group_per_sample(resp: "RolloutResp", track_name: str) -> List[str]:
    """Return the root-track group_id corresponding to each sample of ``track_name``.

    Walks the lineage up via ``parent_track`` + ``parent_ids`` until reaching
    the root (the unique track with ``parent_track=None``), then reads
    ``root.group_ids`` at the resolved index. The root's own group_ids are
    its ``parent_ids`` (set by ``make_root_track`` to the request prompt
    IDs) or its ``sample_ids`` for fully-root tracks. Used by
    :meth:`RolloutResp.split` to partition descendant tracks by root group.
    """
    track = resp.tracks[track_name]
    if track.parent_track is None:
        return list(track.group_ids)
    if track.parent_ids is None:
        raise RuntimeError(
            f"_root_group_per_sample: track {track_name!r} has parent_track "
            f"{track.parent_track!r} but parent_ids is None; lineage broken."
        )
    parent_root_groups = _root_group_per_sample(resp, track.parent_track)
    parent = resp.tracks[track.parent_track]
    parent_sid_to_idx = {sid: i for i, sid in enumerate(parent.sample_ids)}
    return [parent_root_groups[parent_sid_to_idx[pid]] for pid in track.parent_ids]


def _track_with_field(track: TR, field_name: str, value: Any) -> TR:
    """Return a copy of ``track`` with one field replaced (other fields preserved)."""
    kwargs: Dict[str, Any] = {f.name: getattr(track, f.name) for f in dc_fields(track)}
    kwargs[field_name] = value
    return type(track)(**kwargs)


@dataclass
class RolloutResp(Batch):
    """Top-level rollout response container — keyed dict of ``RolloutTrack``.

    See module docstring for the multi-track architecture. Each track's
    fields are read/written directly via ``resp.tracks[<name>].field``;
    there is no resp-level shim that fans out to / aggregates from tracks.
    """

    tracks: Dict[str, RolloutTrack] = field(kind=FieldKind.CONCAT, default_factory=dict)
    reward_compute_s: float = max_field(default=0.0)

    def __post_init__(self) -> None:
        # Validate per-track invariants (length consistency, lineage foreign-key).
        for name, t in self.tracks.items():
            n = len(t.sample_ids)
            if t.parent_ids is not None and len(t.parent_ids) != n:
                raise ValueError(
                    f"RolloutResp.tracks[{name!r}]: parent_ids length {len(t.parent_ids)} != sample_ids length {n}"
                )
            if t.parent_track is not None and t.parent_track not in self.tracks:
                raise ValueError(
                    f"RolloutResp.tracks[{name!r}].parent_track={t.parent_track!r} "
                    f"not in tracks (have {sorted(self.tracks.keys())})"
                )
            if t.parent_track is not None and t.parent_ids is not None:
                parent_set = set(self.tracks[t.parent_track].sample_ids)
                missing = [p for p in t.parent_ids if p not in parent_set]
                if missing:
                    raise ValueError(
                        f"RolloutResp.tracks[{name!r}].parent_ids: {len(missing)} ids not in "
                        f"parent track {t.parent_track!r} sample_ids; first missing: "
                        f"{missing[:3]!r}"
                    )

    # ---- batch_size --------------------------------------------------------

    @property
    def batch_size(self) -> int:
        # Single-track convention: forward to the only track. Multi-track:
        # batch_size is ambiguous (per-track sizes differ); return the max so
        # downstream code can still infer "is there data here?". Explicit
        # per-track logic should use ``track.batch_size`` directly.
        if not self.tracks:
            return 0
        if len(self.tracks) == 1:
            return next(iter(self.tracks.values())).batch_size
        return max(t.batch_size for t in self.tracks.values())

    # ---- light (metadata-only) view ----------------------------------------

    def metadata_only(self) -> "RolloutResp":
        """Per-track metadata-only view (keep-local light data plane).

        Recurses into each track, replacing it with its own
        ``metadata_only`` (which drops that track's heavy ``conditions`` /
        ``segment`` / ``decoded`` and keeps the light + lineage fields).
        ``parent_track`` / ``parent_ids`` / ``sample_ids`` are preserved,
        so the per-track lineage invariants still hold.
        """
        return RolloutResp(
            tracks={name: track.metadata_only() for name, track in self.tracks.items()},
            reward_compute_s=self.reward_compute_s,
        )

    # ---- structural lookups ------------------------------------------------

    def root_track(self) -> "RolloutTrack":
        """Return the unique root track (the one with ``parent_track is None``).

        Raises if zero or multiple roots exist — multi-root resps don't have
        a well-defined "first dimension" and downstream operations that need
        a sharding identity (e.g. :meth:`split`) can't proceed.
        """
        roots = [t for t in self.tracks.values() if t.parent_track is None]
        if len(roots) != 1:
            root_names = [name for name, t in self.tracks.items() if t.parent_track is None]
            raise RuntimeError(
                f"RolloutResp.root_track: expected exactly one root track "
                f"(parent_track=None), got {sorted(root_names)} out of "
                f"{sorted(self.tracks.keys())}."
            )
        return roots[0]

    def tracks_with_segment_types(self, segment_types: Iterable[Type[Segment]]) -> List[Tuple[str, "RolloutTrack"]]:
        """Return ``(name, track)`` pairs whose ``segment`` matches one of ``segment_types``.

        ``type(track.segment)`` exact match (no ``isinstance``) — subclassing
        a registered segment type is opt-in, not implicit. Tracks with
        ``segment is None`` are skipped. Insertion order of ``self.tracks``
        is preserved, so callers iterating the result see parents before
        children when the resp was built that way.
        """
        wanted = set(segment_types)
        return [(name, t) for name, t in self.tracks.items() if t.segment is not None and type(t.segment) in wanted]

    # ---- per-group split ---------------------------------------------------

    def split(self) -> List["RolloutResp"]:
        """Split into one ``RolloutResp`` per root-track group, tree-complete.

        Splits along the "first dimension" — the root track's group equivalence
        classes. Each shard contains one root group's whole subtree across all
        tracks (e.g. for refined+image with `x` prompts, `y` refined per prompt,
        `z` images per refined: each shard holds the `y` refined samples plus
        the `y*z` image samples for one prompt).

        Requires exactly one root track (the unique track with
        ``parent_track is None``); raises otherwise. Descendant tracks are
        sliced by walking ``parent_ids`` + ``parent_track`` back to determine
        which samples belong to each root group.

        Single-track resps reduce to splitting that one track by its
        ``group_ids`` — same behavior as the pre-multi-track contract.
        """
        if not self.tracks:
            return [self]

        root_gids = self.root_track().group_ids
        if not root_gids:
            return [self]

        per_track_root_groups: Dict[str, List[str]] = {name: _root_group_per_sample(self, name) for name in self.tracks}

        results: List[RolloutResp] = []
        for rgid in dict.fromkeys(root_gids):
            shard_tracks: Dict[str, RolloutTrack] = {}
            for tname, track in self.tracks.items():
                indices = [i for i, rg in enumerate(per_track_root_groups[tname]) if rg == rgid]
                if not indices:
                    raise RuntimeError(
                        f"RolloutResp.split: track {tname!r} has no samples in "
                        f"root group {rgid!r}; lineage tree is malformed."
                    )
                shard_tracks[tname] = track.select(torch.tensor(indices, dtype=torch.long))
            results.append(type(self)(tracks=shard_tracks))
        return results

    # ---- leaf-to-root reward propagation -----------------------------------

    def propagate_rewards(
        self,
        op: Literal["mean", "max", "sum"] = "mean",
    ) -> "RolloutResp":
        """Aggregate child rewards up through the parent_track tree.

        Walks ``self.tracks`` reverse-topologically (leaves first). For each
        track whose ``rewards is None`` and which has at least one child
        track in this resp, sets ``rewards`` to the per-group aggregation of
        its child's rewards.

        Group-by-parent ordering (guaranteed by ``fork_track`` /
        ``make_root_track``) means aggregation is one reshape:
        ``child.rewards.view(n_parent, branch).reduce(dim=1)``.

        :param op: Reduction op — ``"mean"`` (default; standard GRPO),
            ``"max"`` (best-of-z signal), or ``"sum"`` (cumulative reward).
        :return: A new ``RolloutResp`` with rewards filled in. Tracks
            whose ``rewards`` were already set are reused unchanged
            (direct rewards win over inherited).

        Raises ``ValueError`` if a child track has ``rewards=None`` (cannot
        aggregate from a not-yet-scored child) or if the branching factor
        is non-uniform (would violate the reshape invariant).

        Multi-children-per-parent is not yet supported (single-child trees
        only — the prompt-enhancement workload has exactly that shape). The
        method raises ``NotImplementedError`` if a parent has multiple
        children.
        """
        # Build parent_name → [child track names].
        children_of: Dict[str, List[str]] = {}
        for child_name, child in self.tracks.items():
            if child.parent_track is not None:
                children_of.setdefault(child.parent_track, []).append(child_name)

        # Topological sort: leaves (no children) first, then their parents, etc.
        # In a tree (each node has at most one parent), depth-from-leaves is a
        # valid reverse-topo order.
        def _depth(name: str) -> int:
            children = children_of.get(name, [])
            if not children:
                return 0
            return 1 + max(_depth(c) for c in children)

        sorted_names = sorted(self.tracks.keys(), key=_depth)

        # Build new tracks dict, propagating bottom-up.
        new_tracks: Dict[str, RolloutTrack] = {}
        for name in sorted_names:
            track = self.tracks[name]
            if track.rewards is not None:
                # Direct rewards win. Reuse the track unchanged.
                new_tracks[name] = track
                continue
            children_names = children_of.get(name, [])
            if not children_names:
                # Leaf with no rewards — nothing to aggregate. Pass through.
                new_tracks[name] = track
                continue
            if len(children_names) > 1:
                raise NotImplementedError(
                    f"propagate_rewards: track {name!r} has multiple children "
                    f"{sorted(children_names)}; aggregation across multiple "
                    f"children not yet implemented (single-child trees only)."
                )
            child_name = children_names[0]
            # Use the already-propagated child if any, falling back to the
            # input — for the single-parent tree we have, this is equivalent.
            child = new_tracks.get(child_name, self.tracks[child_name])
            if child.rewards is None:
                raise ValueError(
                    f"propagate_rewards: cannot aggregate from child track "
                    f"{child_name!r} to parent {name!r} — child.rewards is None. "
                    f"Score the leaf tracks first."
                )
            n_parent = len(track.sample_ids)
            n_child = len(child.sample_ids)
            if n_parent == 0 or n_child % n_parent != 0:
                raise ValueError(
                    f"propagate_rewards: non-uniform branching from {child_name!r} "
                    f"({n_child} samples) to {name!r} ({n_parent} samples). "
                    f"Group-by-parent ordering requires n_child % n_parent == 0."
                )
            branch = n_child // n_parent
            reshaped = child.rewards.view(n_parent, branch)
            if op == "mean":
                aggregated = reshaped.mean(dim=1)
            elif op == "max":
                aggregated = reshaped.amax(dim=1)
            elif op == "sum":
                aggregated = reshaped.sum(dim=1)
            else:
                raise ValueError(f"propagate_rewards: unknown op {op!r}; expected 'mean', 'max', or 'sum'.")
            new_tracks[name] = _track_with_field(track, "rewards", aggregated)

        # Restore the original track-name iteration order (sorted_names is
        # leaves-first, but RolloutResp.tracks is conventionally inserted
        # parents-first / by use-case order; preserve self's order).
        ordered = {name: new_tracks[name] for name in self.tracks.keys()}
        return type(self)(tracks=ordered, reward_compute_s=self.reward_compute_s)

    # ---- lineage-aware advantage computation -------------------------------

    def compute_track_advantages(
        self,
        track_name: str,
        *,
        group_key: str = "parent",
        **adv_kwargs: Any,
    ) -> "RolloutTrack":
        """Compute one track's GRPO advantages, optionally grouping by an ancestor.

        ``group_key`` selects which lineage level defines a GRPO group:

        - ``"parent"`` (default): group by the track's own ``parent_ids`` —
          identical to calling ``track.compute_advantages(**adv_kwargs)`` directly
          (e.g. the diffusion track groups by rewrite, M images per group).
        - ``"root"``: group by the ROOT-track group id of each sample, resolved by
          walking ``parent_track`` + ``parent_ids`` up to the root via
          :func:`_root_group_per_sample`. For the PE diffusion track this groups
          all ``N*M`` images descended from one original prompt into a single
          group, so a rewrite that systematically beats the prompt-wide mean earns
          non-zero advantage (instead of being differenced out per-rewrite). The
          resolved labels stay group-by-parent contiguous because the lineage keeps
          a prompt's samples consecutive (``make_root_track`` / ``fork_track``).

        Returns the new track (advantages set); does NOT mutate ``self.tracks`` —
        the caller assigns it back, mirroring the ``track.compute_advantages``
        call sites. Lineage is never mutated — only the grouping used for the
        advantage reduction changes.
        """
        track = self.tracks[track_name]
        if group_key == "parent":
            return track.compute_advantages(**adv_kwargs)
        if group_key == "root":
            root_labels = _root_group_per_sample(self, track_name)
            return track.compute_advantages(group_ids=root_labels, **adv_kwargs)
        raise ValueError(f"compute_track_advantages: group_key must be 'parent' or 'root'; got {group_key!r}")


__all__ = ["RolloutResp", "RolloutTrack", "Decoded"]
