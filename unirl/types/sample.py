"""Sample + Part — the rollout endomorphism types (LIN-446)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

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
from unirl.types.advantages import (
    compute_gae_advantages as _compute_gae,
)
from unirl.types.advantages import (
    finite_mean_std,
    scatter_terminal_rewards,
)
from unirl.types.conditions import Condition
from unirl.types.media import MediaRefs
from unirl.types.media_preview import MediaPreview
from unirl.types.primitives import Images, PrimitiveValue, Texts, primitive_modality_key
from unirl.types.sample_id import ancestor_id, child_id, parent_id
from unirl.types.sampling import BaseSamplingParams
from unirl.types.segments import Segment, TextSegment
from unirl.utils.shard_balance import lpt_shard_permutation, shard_token_spread

logger = logging.getLogger(__name__)

Primitive = PrimitiveValue
PrimitiveMap = Dict[str, Primitive]
PrimitiveMetadata = Dict[str, Dict[str, Any]]
PRIMITIVE_MODALITY_ORDER = ("text", "image", "video", "audio", "media")

TURN_ROLES = ("system", "user", "assistant", "tool")


@dataclass
class Turn:
    """One conditioning turn: a role tag + its frontier-aligned content primitive."""

    role: str
    content: Primitive


@dataclass
class Part(Batch):
    """One rollout slice in a Sample lineage."""

    sample_ids: List[str] = concat_field(default_factory=list)

    segment: Optional[Segment] = field(kind=FieldKind.CONCAT, default=None)
    primitives: PrimitiveMap = field(kind=FieldKind.CONCAT, default_factory=dict)
    primitive_metadata: PrimitiveMetadata = shared_field(default_factory=dict)
    conditions: Dict[str, Condition] = field(kind=FieldKind.CONCAT, default_factory=dict)
    media_preview: Optional[MediaPreview] = concat_field(default=None)

    rewards: Optional[torch.Tensor] = concat_field(default=None)
    component_rewards: Optional[Dict[str, torch.Tensor]] = concat_field(default=None)
    advantages: Optional[torch.Tensor] = concat_field(default=None)
    status: Optional[torch.Tensor] = concat_field(default=None)

    metadata: List[Dict[str, Any]] = concat_field(default_factory=list)
    control: Dict[str, Any] = shared_field(default_factory=dict)
    sampling_params: Optional[BaseSamplingParams] = shared_field(default=None)
    role: Optional[str] = shared_field(default=None)
    output_version: Optional[int] = shared_field(default=None)
    harness_status: Optional[str] = shared_field(default=None)
    init_noise_group_ids: List[str] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        expected_batch = len(self.sample_ids)
        for key, primitive in self.primitives.items():
            actual_key = primitive_modality_key(primitive)
            if key != actual_key:
                raise ValueError(
                    f"Part.primitives[{key!r}] contains {type(primitive).__name__}, "
                    f"whose canonical modality key is {actual_key!r}."
                )
            if len(primitive) != expected_batch:
                raise ValueError(
                    f"Part.primitives[{key!r}] batch {len(primitive)} != sample_ids count {expected_batch}."
                )

        extra_metadata = sorted(set(self.primitive_metadata) - set(self.primitives))
        if extra_metadata:
            raise ValueError(f"Part.primitive_metadata has entries for absent primitive modalities: {extra_metadata}.")
        invalid_metadata = [key for key, value in self.primitive_metadata.items() if not isinstance(value, dict)]
        if invalid_metadata:
            raise TypeError(
                f"Part.primitive_metadata values must be dictionaries; invalid modalities: {sorted(invalid_metadata)}."
            )
        audio_meta = self.primitive_metadata.get("audio")
        if audio_meta is not None and "sample_rate" in audio_meta:
            sample_rate = audio_meta["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(
                    f"Part.primitive_metadata['audio']['sample_rate'] must be a positive int, got {sample_rate!r}."
                )
        if self.init_noise_group_ids and len(self.init_noise_group_ids) != expected_batch:
            raise ValueError(
                "Part.init_noise_group_ids must be empty or aligned with sample_ids; "
                f"got {len(self.init_noise_group_ids)} keys for {expected_batch} samples."
            )

    @classmethod
    def concat(cls, items: Sequence["Part"]) -> "Part":
        """Concat batch rows while requiring identical shared primitive metadata."""
        if items:
            expected = items[0].primitive_metadata
            mismatched = [i for i, item in enumerate(items[1:], start=1) if item.primitive_metadata != expected]
            if mismatched:
                raise ValueError(
                    "Part.concat: primitive_metadata must be identical across shards; "
                    f"mismatched item indices {mismatched}."
                )
        return super().concat(items)

    @classmethod
    def input(
        cls,
        sample_ids: List[str],
        segment: Optional[Segment] = None,
        *,
        primitives: Optional[PrimitiveMap] = None,
        control: Optional[Dict[str, Any]] = None,
        metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
        role: Optional[str] = None,
    ) -> "Part":
        """Build a turn-0 input Part (the prompt, a root; untrained)."""
        bad = [s for s in sample_ids if "/" in s]
        if bad:
            raise ValueError(
                f"Part.input: root sample_ids must not contain '/' (the lineage delimiter); offending ids: {bad[:3]!r}"
            )
        return cls(
            sample_ids=list(sample_ids),
            segment=segment,
            primitives=dict(primitives or {}),
            control=dict(control) if control else {},
            metadata=list(metadata) if metadata else [],
            role=role,
        )

    def input_child(self, primitives: PrimitiveMap, *, role: Optional[str] = None) -> "Part":
        """A branch-1 *input* child carrying additional conditioning primitives."""
        if not self.sample_ids:
            raise ValueError("Part.input_child: parent has no sample_ids")
        return Part(
            sample_ids=[child_id(pid, 0) for pid in self.sample_ids],
            primitives=dict(primitives),
            role=role,
        )

    @property
    def batch_size(self) -> int:
        if self.sample_ids:
            return len(self.sample_ids)
        return super().batch_size

    @property
    def is_root(self) -> bool:
        """Whether this is a chain head (input/root) — its sample ids carry no"""
        return bool(self.sample_ids) and not any("/" in sid for sid in self.sample_ids)

    @property
    def is_gen(self) -> bool:
        """Whether this part was generated by the policy — it carries the"""
        return self.sampling_params is not None

    def resolved_role(self) -> str:
        """This turn's conversation role, deriving when :attr:`role` is unset."""
        if self.role is not None:
            return self.role
        return "assistant" if self.is_gen else "user"

    @property
    def group_ids(self) -> List[str]:
        """Grouping labels: each sample's parent id (siblings = one group), or its"""
        return [p if (p := parent_id(sid)) is not None else sid for sid in self.sample_ids]

    def balance_shards(self, num_shards: int, *, min_spread: float = 0.05) -> "Part":
        """Reorder samples so ``num_shards`` equal contiguous shards carry ~equal"""
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

    def fork(
        self,
        branch: int,
        *,
        sampling_params: BaseSamplingParams,
        new_segment: Optional[Segment] = None,
    ) -> "Part":
        """Create a child generation shell — ``N`` self-samples → ``N*branch``."""
        if not self.sample_ids:
            raise ValueError("Part.fork: parent has no sample_ids")
        if branch < 1:
            raise ValueError(f"Part.fork: branch must be >= 1, got {branch}")

        child_sample_ids = [child_id(pid, j) for pid in self.sample_ids for j in range(branch)]

        return Part(
            sample_ids=child_sample_ids,
            segment=new_segment,
            sampling_params=sampling_params,
        )

    def fill(
        self,
        *,
        segment: Optional[Segment] = None,
        primitives: Optional[PrimitiveMap] = None,
        primitive_metadata: Optional[PrimitiveMetadata] = None,
        conditions: Optional[Dict[str, Condition]] = None,
        media_preview: Optional[MediaPreview] = None,
        status: Optional[torch.Tensor] = None,
        output_version: Optional[int] = None,
    ) -> "Part":
        """Return a copy of this gen-shell part with generation outputs written."""
        kwargs: Dict[str, Any] = {f.name: getattr(self, f.name) for f in dc_fields(self)}
        for name, value in (
            ("segment", segment),
            ("primitives", primitives),
            ("primitive_metadata", primitive_metadata),
            ("conditions", conditions),
            ("media_preview", media_preview),
            ("status", status),
            ("output_version", output_version),
        ):
            if value is not None:
                kwargs[name] = value
        return type(self)(**kwargs)

    def compute_advantages(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
        scope: str = "group",
        use_global_std: bool = False,
        group_layer: Optional[int] = None,
    ) -> "Part":
        """GRPO per-group advantage ``(reward - group_mean) / (group_std + eps)``."""
        if self.rewards is None:
            raise ValueError("Part.compute_advantages: part has no rewards")
        n = len(self.sample_ids)
        if n == 0:
            return self

        rewards_local = hydrate(self.rewards)

        if scope == "global":
            rewards_g = rewards_local.to(torch.float32)
            finite = torch.isfinite(rewards_g)
            mean, std = finite_mean_std(rewards_g)
            centered = torch.where(finite, rewards_g - mean, torch.zeros_like(rewards_g))
            if normalize:
                adv_g = centered / (std + eps)
            else:
                adv_g = centered
            return _part_with_field(self, "advantages", adv_g)

        layer = group_layer if group_layer is not None else max(self.sample_ids[0].count("/") - 1, 0)
        group_labels = [ancestor_id(sid, layer) for sid in self.sample_ids]

        unique_pids = list(dict.fromkeys(group_labels))
        n_groups = len(unique_pids)
        if n % n_groups != 0:
            raise ValueError(
                f"compute_advantages: non-uniform group sizes (n={n}, n_groups={n_groups}). "
                f"Expected uniform branching with group-by-parent ordering — use fork to build the Part."
            )
        branch = n // n_groups
        expected = [pid for pid in unique_pids for _ in range(branch)]
        if list(group_labels) != expected:
            raise ValueError(
                "compute_advantages: grouping labels not in group-by-parent contiguous order. "
                "Siblings must be consecutive (use fork), got interleaved ordering."
            )

        rewards = rewards_local.to(torch.float32)
        reshaped = rewards.view(n_groups, branch)
        # Per-row finite mean/std (same contract as :func:`finite_mean_std`, vectorized).
        finite = torch.isfinite(reshaped)
        counts = finite.sum(dim=1, keepdim=True)
        finite_values = torch.where(finite, reshaped, torch.zeros_like(reshaped))
        mean = finite_values.sum(dim=1, keepdim=True) / counts.clamp_min(1)
        centered = torch.where(finite, reshaped - mean, torch.zeros_like(reshaped))
        if normalize:
            if use_global_std:
                _, std = finite_mean_std(rewards)
            else:
                variance = (centered * centered).sum(dim=1, keepdim=True) / counts.clamp_min(1)
                std = torch.where(counts > 1, variance.sqrt(), torch.ones_like(variance))
            adv = centered / (std + eps)
        else:
            adv = centered
        return _part_with_field(self, "advantages", adv.flatten())

    def compute_gae_advantages(
        self,
        *,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> "Part":
        """Attach packed token advantages and returns for an AR PPO update."""
        if self.rewards is None:
            raise ValueError("Part.compute_gae_advantages: part has no rewards")
        if not isinstance(self.segment, TextSegment):
            raise ValueError("Part.compute_gae_advantages: requires a TextSegment")
        segment = self.segment
        if segment.values is None:
            raise ValueError("Part.compute_gae_advantages: segment.values is None")
        if segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "Part.compute_gae_advantages: segment requires framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )

        values = hydrate(segment.values).to(torch.float32)
        cu_seqlens = segment.cu_seqlens.to(device=values.device)
        total_tokens = int(cu_seqlens[-1].item())
        if values.ndim != 1 or int(values.numel()) != total_tokens:
            raise ValueError(
                "Part.compute_gae_advantages: values must be a packed 1D tensor "
                f"with {total_tokens} elements, got shape {tuple(values.shape)}"
            )

        rewards = hydrate(self.rewards).to(device=values.device, dtype=torch.float32)
        token_rewards = scatter_terminal_rewards(rewards, cu_seqlens=cu_seqlens)
        token_advantages = values.new_zeros(values.shape)
        token_returns = values.new_zeros(values.shape)
        cu = [int(offset) for offset in cu_seqlens.tolist()]
        for start, end in zip(cu, cu[1:]):
            if end <= start:
                continue
            advantages, returns = _compute_gae(
                token_rewards[start:end],
                values[start:end],
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            token_advantages[start:end] = advantages
            token_returns[start:end] = returns

        segment_fields = {f.name: getattr(segment, f.name) for f in dc_fields(segment)}
        segment_fields["token_advantages"] = token_advantages
        segment_fields["returns"] = token_returns
        updated_segment = segment._rebuild(segment_fields)

        loss_mask = None
        if segment.loss_mask is not None:
            loss_mask = hydrate(segment.loss_mask).to(device=values.device, dtype=torch.bool)
            if loss_mask.shape != values.shape:
                raise ValueError(
                    "Part.compute_gae_advantages: loss_mask shape "
                    f"{tuple(loss_mask.shape)} != values shape {tuple(values.shape)}"
                )
        sample_advantages: List[torch.Tensor] = []
        for start, end in zip(cu, cu[1:]):
            if end <= start:
                sample_advantages.append(values.new_zeros(()))
                continue
            selected = token_advantages[start:end]
            if loss_mask is not None:
                selected = selected[loss_mask[start:end]]
            sample_advantages.append(selected.mean() if selected.numel() else values.new_zeros(()))
        mean_advantages = (
            torch.stack(sample_advantages) if sample_advantages else values.new_zeros((0,), dtype=torch.float32)
        )

        updated = _part_with_field(self, "segment", updated_segment)
        return _part_with_field(updated, "advantages", mean_advantages)


def _part_with_field(part: Part, field_name: str, value: Any) -> Part:
    """Copy of ``part`` with one field replaced."""
    kwargs: Dict[str, Any] = {f.name: getattr(part, f.name) for f in dc_fields(part)}
    kwargs[field_name] = value
    return type(part)(**kwargs)


@dataclass
class Sample(Batch):
    """Rollout container — an ordered ``parts: List[Part]`` (the merged"""

    parts: List[Part] = field(kind=FieldKind.CONCAT, default_factory=list)
    reward_compute_s: float = max_field(default=0.0)

    def __post_init__(self) -> None:
        for i, p in enumerate(self.parts):
            if len(p.sample_ids) == 0:
                continue
            if p.is_root:
                if i != 0:
                    raise ValueError(
                        f"Sample.parts[{i}] is a root (ids carry no lineage segment) but is not the head; "
                        f"only the first part may be a root."
                    )
                continue
            if i == 0:
                raise ValueError("Sample.parts[0] is non-root; the head has no parent.")
            prev_ids = set(self.parts[i - 1].sample_ids)
            bad = [sid for sid in p.sample_ids if parent_id(sid) not in prev_ids]
            if bad:
                raise ValueError(
                    f"Sample.parts[{i}]: {len(bad)} ids whose parent id is not in the preceding part "
                    f"(index {i - 1}); first bad: {bad[:3]!r}"
                )

    @classmethod
    def concat(cls, items: Sequence["Sample"]) -> "Sample":
        """Concat Samples (e.g. DP gather): concat each part position-wise across"""
        if not items:
            raise ValueError("Sample.concat: cannot concat an empty sequence")
        if len(items) == 1:
            return items[0]
        n_parts = len(items[0].parts)
        merged = [Part.concat([it.parts[i] for it in items]) for i in range(n_parts)]
        return cls(parts=merged, reward_compute_s=max(it.reward_compute_s for it in items))

    @classmethod
    def request(cls, *input_parts: Part) -> "Sample":
        """A *request* — a ``Sample`` of only input Part(s), e.g."""
        return cls(parts=list(input_parts))

    @property
    def batch_size(self) -> int:
        """Size of the root Part (one prompt + its fan-out = "one sample"); max"""
        if not self.parts:
            return 0
        roots = [p for p in self.parts if p.is_root]
        if len(roots) == 1:
            return roots[0].batch_size
        return max(p.batch_size for p in self.parts)

    def split(self) -> List["Sample"]:
        """Split into one ``Sample`` per root-group, tree-complete: each shard holds"""
        if not self.parts:
            return [self]

        root_gids = self.root_group_ids(0)
        if not root_gids:
            return [self]

        per_part_buckets: List[Dict[str, List[int]]] = []
        for part in self.parts:
            buckets: Dict[str, List[int]] = {}
            for k, sid in enumerate(part.sample_ids):
                buckets.setdefault(ancestor_id(sid, 0), []).append(k)
            per_part_buckets.append(buckets)

        results: List["Sample"] = []
        for rgid in dict.fromkeys(root_gids):
            shard_parts: List[Part] = []
            for i, part in enumerate(self.parts):
                indices = per_part_buckets[i].get(rgid)
                if not indices:
                    raise RuntimeError(
                        f"Sample.split: part {i} has no samples in root group {rgid!r}; lineage tree is malformed."
                    )
                shard_parts.append(part.select(torch.tensor(indices, dtype=torch.long)))
            results.append(type(self)(parts=shard_parts))
        return results

    def root_group_ids(self, part_index: int) -> List[str]:
        """Root-prompt group label per sample of ``parts[part_index]`` — the"""
        if not self.parts:
            return []
        return [ancestor_id(sid, 0) for sid in self.parts[part_index].sample_ids]

    def root_metadata(self, part_index: int = -1) -> List[Optional[Dict[str, Any]]]:
        """Root Part's per-sample metadata aligned to ``parts[part_index]`` rows."""
        if not self.parts:
            return []
        n = self.parts[part_index].batch_size
        root = self.parts[0]
        if not root.metadata:
            return [None] * n
        by_root_id = dict(zip(root.sample_ids, root.metadata))
        return [by_root_id.get(rgid) for rgid in self.root_group_ids(part_index)]

    def gen_parts(self) -> List[Part]:
        """The generated (non-input) Parts — those with :attr:`Part.is_gen`."""
        return [p for p in self.parts if p.is_gen]

    def _gen_part_indices(self, params_type: type) -> List[int]:
        """Indices of generated Parts whose sampling params match ``params_type``."""
        return [i for i, part in enumerate(self.parts) if part.is_gen and isinstance(part.sampling_params, params_type)]

    @staticmethod
    def _require_unique_gen_part_index(indices: List[int], params_type: type, *, caller: str) -> int:
        if not indices:
            raise ValueError(f"{caller}: no Part with sampling_params of type {params_type.__name__}")
        if len(indices) > 1:
            raise ValueError(
                f"{caller}: multiple Parts with sampling_params of type {params_type.__name__} "
                f"at indices {indices}; typed lookup is ambiguous. Use frontier_gen_part(...) "
                "for a generation frontier or inspect gen_parts() explicitly."
            )
        return indices[0]

    def gen_part(self, params_type: type) -> Part:
        """The unique gen Part whose sampling params match ``params_type``."""
        index = self._require_unique_gen_part_index(
            self._gen_part_indices(params_type), params_type, caller="Sample.gen_part"
        )
        return self.parts[index]

    def gen_part_or_none(self, params_type: type) -> Optional[Part]:
        """:meth:`gen_part`, returning ``None`` only when there is no match."""
        indices = self._gen_part_indices(params_type)
        if not indices:
            return None
        index = self._require_unique_gen_part_index(indices, params_type, caller="Sample.gen_part_or_none")
        return self.parts[index]

    def gen_part_index(self, params_type: type) -> int:
        """Index of the unique :meth:`gen_part`, for non-frontier write-back."""
        return self._require_unique_gen_part_index(
            self._gen_part_indices(params_type), params_type, caller="Sample.gen_part_index"
        )

    def frontier_gen_part(self, params_type: type) -> Part:
        """Return the final Part after validating it is a gen of ``params_type``."""
        if not self.parts:
            raise ValueError("Sample.frontier_gen_part: Sample has no Parts")
        index = len(self.parts) - 1
        frontier = self.parts[index]
        if not frontier.is_gen:
            raise ValueError(
                f"Sample.frontier_gen_part: final Part at index {index} is not generated (sampling_params is None)"
            )
        if not isinstance(frontier.sampling_params, params_type):
            raise ValueError(
                f"Sample.frontier_gen_part: final generated Part at index {index} has sampling_params "
                f"of type {type(frontier.sampling_params).__name__}, expected {params_type.__name__}"
            )
        return frontier

    def with_parts(self, parts: List[Part]) -> "Sample":
        """A copy carrying replacement ``parts`` but the same ``reward_compute_s``"""
        return type(self)(parts=list(parts), reward_compute_s=self.reward_compute_s)

    def map_sample_ids(self, mapper: Callable[[str], str]) -> "Sample":
        """Rewrite every Part's ids and revalidate the complete lineage tree."""
        return self.with_parts(
            [
                _part_with_field(part, "sample_ids", [mapper(sample_id) for sample_id in part.sample_ids])
                for part in self.parts
            ]
        )

    def slice(self, start: int, end: int) -> "Sample":
        """Shard ``[start, end)`` along the batch dim (the P root prompts) by whole"""
        picked = self.split()[start:end]
        parts = Sample.concat(picked).parts if picked else []
        return type(self)(parts=parts, reward_compute_s=self.reward_compute_s)

    def select(self, indices: "torch.Tensor") -> "Sample":
        """Gather whole root prompt-trees by index (shuffle / subsample), mirroring"""
        groups = self.split()
        idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        picked = [groups[int(i)] for i in idx]
        parts = Sample.concat(picked).parts if picked else []
        return type(self)(parts=parts, reward_compute_s=self.reward_compute_s)

    def fork(
        self,
        branch: int,
        *,
        sampling_params: BaseSamplingParams,
        new_segment: Optional[Segment] = None,
    ) -> "Sample":
        """Append a generation shell forked from the frontier (the last part) — the"""
        if not self.parts:
            raise ValueError("Sample.fork: no parts to fork from (empty Sample)")
        child = self.parts[-1].fork(
            branch,
            sampling_params=sampling_params,
            new_segment=new_segment,
        )
        return type(self)(parts=[*self.parts, child], reward_compute_s=self.reward_compute_s)

    def observe(self, observation: Primitive, *, role: str = "tool") -> "Sample":
        """Append an observation as a branch-1, mask-0 *input* Part off the frontier."""
        if not self.parts:
            raise ValueError("Sample.observe: no parts to observe from (empty Sample)")
        obs_part = self.parts[-1].input_child({primitive_modality_key(observation): observation}, role=role)
        return type(self)(parts=[*self.parts, obs_part], reward_compute_s=self.reward_compute_s)

    def propagate_rewards(self, op: Literal["mean", "max", "sum"] = "mean") -> "Sample":
        """Aggregate child rewards up the chain (leaf → root) into unscored parents."""
        new_parts = list(self.parts)
        for i in range(len(new_parts) - 1, -1, -1):
            part = self.parts[i]
            if part.rewards is not None:
                continue
            if i + 1 >= len(new_parts):
                continue
            child = new_parts[i + 1]
            if child.is_root:
                continue
            if child.rewards is None:
                raise ValueError(
                    f"propagate_rewards: cannot aggregate from part {i + 1} to {i} — child.rewards is None. "
                    f"Score the leaf parts first."
                )
            n_parent = len(part.sample_ids)
            n_child = len(child.sample_ids)
            if n_parent == 0 or n_child % n_parent != 0:
                raise ValueError(
                    f"propagate_rewards: non-uniform branching from part {i + 1} ({n_child} samples) "
                    f"to {i} ({n_parent} samples). Group-by-parent ordering requires n_child % n_parent == 0."
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
            new_parts[i] = _part_with_field(part, "rewards", aggregated)

        return type(self)(parts=new_parts, reward_compute_s=self.reward_compute_s)

    def turns(self) -> List[Turn]:
        """Role-tagged, turn-ordered, frontier-aligned conditioning — the agent"""
        if not self.parts:
            return []
        frontier = self.parts[-1]
        if frontier.is_root:
            return []
        out: List[Turn] = []
        active_ids = [parent_id(sid) for sid in frontier.sample_ids]
        anc = len(self.parts) - 2
        while anc >= 0:
            part = self.parts[anc]
            sid_to_row = {sid: r for r, sid in enumerate(part.sample_ids)}
            try:
                rows = [sid_to_row[aid] for aid in active_ids]
            except KeyError as e:
                raise ValueError(
                    f"Sample.turns: ancestor id {e.args[0]!r} not found in part {anc}; lineage chain is malformed."
                ) from None
            for key in reversed(PRIMITIVE_MODALITY_ORDER):
                primitive = part.primitives.get(key)
                if primitive is None:
                    continue
                content = primitive.select(torch.tensor(rows, dtype=torch.long))
                out.append(Turn(role=part.resolved_role(), content=content))
            if part.is_root:
                break
            active_ids = [parent_id(aid) for aid in active_ids]
            anc -= 1
        out.reverse()
        return out

    def conditioning(self) -> List[Primitive]:
        """Conditioning primitives for generating the frontier (last) part: each"""
        return [t.content for t in self.turns()]

    def prompt_media_refs(self) -> Optional[MediaRefs]:
        """Return typed URI prompt-media refs from the root input Part, if any."""
        if not self.parts:
            return None
        media = self.parts[0].primitives.get("media")
        if media is None:
            return None
        if not isinstance(media, MediaRefs):
            raise TypeError(
                f"Sample.prompt_media_refs: expected MediaRefs under primitives['media'], got {type(media).__name__}."
            )
        return media

    def conditioning_at(self, index: int) -> List[Primitive]:
        """:meth:`conditioning` for generating ``parts[index]`` rather than the frontier."""
        if not self.parts:
            raise ValueError("Sample.conditioning_at: Sample has no Parts")
        if not -len(self.parts) <= index < len(self.parts):
            raise IndexError(f"Sample.conditioning_at: index {index} out of range for {len(self.parts)} Parts")
        stop = (index if index >= 0 else len(self.parts) + index) + 1
        return type(self)(parts=list(self.parts[:stop]), reward_compute_s=self.reward_compute_s).conditioning()

    def text_conditioning(self) -> List[Turn]:
        """LLM render: the trajectory as an all-text conversation. Fails loud on"""
        ts = self.turns()
        non_text = [primitive_modality_key(t.content) for t in ts if not isinstance(t.content, Texts)]
        if non_text:
            raise ValueError(f"Sample.text_conditioning: the LLM path requires all-text turns; got non-text {non_text}")
        return ts

    def vision_conditioning(self) -> tuple[List[Turn], List[Images]]:
        """VLM render: the trajectory as a text+image conversation. Returns the"""
        ts = self.turns()
        images = [t.content for t in ts if isinstance(t.content, Images)]
        extra = [primitive_modality_key(t.content) for t in ts if not isinstance(t.content, (Texts, Images))]
        if extra or not images:
            raise ValueError(
                f"Sample.vision_conditioning: requires text+image turns only; got "
                f"{[primitive_modality_key(t.content) for t in ts]}"
            )
        return ts, images

    def has_image_input(self) -> bool:
        """Whether any non-frontier Part carries an image primitive — the"""
        return any("image" in p.primitives for p in self.parts[:-1])

    def replace_frontier(self, part: Part) -> "Sample":
        """Swap the frontier (last) part, preserving the chain (``parts[:-1]``) and"""
        return self.with_parts([*self.parts[:-1], part])

    def with_filled_frontier(self, **fill_kwargs: Any) -> "Sample":
        """Fill the frontier gen shell with generation outputs, chain preserved —"""
        return self.replace_frontier(self.parts[-1].fill(**fill_kwargs))


__all__ = [
    "Sample",
    "Part",
    "Primitive",
    "PrimitiveMap",
    "PrimitiveMetadata",
    "PRIMITIVE_MODALITY_ORDER",
    "Turn",
    "TURN_ROLES",
]
