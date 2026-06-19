"""ComposedRolloutEngine — two-child engine orchestrating PE serial flow.

Holds an ``ar`` child and a ``diffusion`` child. ``generate(req)`` runs
them sequentially:

1. AR child rewrites each of P prompts into N PE candidates  → ``[P*N]`` AR track
2. Diffusion child generates M images per PE                  → ``[P*N*M]`` diffusion track
3. Returns a 2-track ``RolloutResp`` with explicit lineage
   (``parent_track="ar"`` on diffusion, ``parent_ids`` chains from
   prompt → PE → image).

Each child config carries its own ``_target_``; ``__init__`` builds the child
engine via ``config.<child>.make_engine(...)``.

Weight sync uses prefix-based tensor routing: the training side prepends
``"{track_name}."`` to tensor keys; this engine demuxes by prefix and
forwards each subset to the matching child with the prefix stripped.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, _track_with_field

logger = logging.getLogger(__name__)


def _extract_pe_text(raw_text: str, marker: str) -> str:
    """Return the substring after the LAST occurrence of ``marker``.

    Pre-strips an optional ``<think>...</think>`` reasoning preamble (Qwen3
    chat output), then takes everything after the last ``marker`` and
    removes a wrapping pair of quotes. Returns ``""`` when the marker is
    absent so the caller can fall back to the original user prompt.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    think_close = text.rfind("</think>")
    if think_close != -1:
        text = text[think_close + len("</think>") :].strip()
        if not text:
            return ""

    marker_idx = text.rfind(marker)
    if marker_idx == -1:
        return ""

    pe_text = text[marker_idx + len(marker) :].strip()
    if len(pe_text) >= 2 and pe_text[0] == pe_text[-1] and pe_text[0] in ('"', "'"):
        pe_text = pe_text[1:-1].strip()
    return pe_text


class ComposedRolloutEngine(BaseRolloutEngine):
    """Two-child rollout engine for prompt-enhancement (PE) serial flow."""

    _component_name = "composed"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: ComposedRolloutEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, ComposedRolloutEngineConfig),
            f"ComposedRolloutEngine requires ComposedRolloutEngineConfig; got {type(config).__name__}",
        )
        self.cfg = config
        self._device = device
        self._strategy = strategy
        self._rank = rank
        self._model_config = model_config

        deps = dict(device=device, rank=rank, model_config=model_config)

        self._ar: BaseRolloutEngine = config.ar.make_engine(strategy=None, **deps)
        self._diffusion: BaseRolloutEngine = config.diffusion.make_engine(strategy=strategy, **deps)

        self._child_by_name: Dict[str, BaseRolloutEngine] = {
            "ar": self._ar,
            "diffusion": self._diffusion,
        }

        if config.sleep_diffusion_on_start:
            self._diffusion.sleep()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        for name, child in self._child_by_name.items():
            try:
                child.shutdown()
            except Exception as exc:
                logger.warning("Child %r shutdown raised: %s", name, exc)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        for child in self._child_by_name.values():
            child.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        for child in self._child_by_name.values():
            child.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        for child in self._children_for_track_prefix(track_prefix):
            child.onload_weights()

    @property
    def is_offloaded(self) -> bool:
        return all(child.is_offloaded for child in self._child_by_name.values())

    def health_check(self) -> bool:
        return all(child.health_check() for child in self._child_by_name.values())

    def get_memory_info(self) -> Dict[str, float]:
        out: Dict[str, float] = {"allocated_gb": 0.0, "cached_gb": 0.0}
        for child in self._child_by_name.values():
            m = child.get_memory_info() or {}
            out["allocated_gb"] += float(m.get("allocated_gb", 0.0))
            out["cached_gb"] += float(m.get("cached_gb", 0.0))
        return out

    # ------------------------------------------------------------------
    # Generation — PE serial flow
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run PE serial flow: AR expansion → diffusion sampling → 2-track resp.

        Dispatched ``DP_SCATTER`` (like vllm-omni / trainside): the Handle shards the
        req across DP workers (each owns its own sglang + sglang_diffusion subprocess),
        every worker runs the serial flow on its prompt-shard, and ``_collect_dp_merge``
        merges the per-worker 2-track resps. (``BROADCAST`` would return a list
        of per-worker resps via ``_collect_passthrough`` and break the trainer.)
        """
        require(int(req.batch_size) > 0, "ComposedRolloutEngine.generate: empty req")
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None,
            "ComposedRolloutEngine.generate: req.primitives['text'] required",
        )
        P = int(req.batch_size)

        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        N = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        M = int(diff_params.samples_per_prompt) if diff_params is not None else 1
        require(N >= 1, f"ComposedRolloutEngine.generate: ar.n={N} must be >= 1")
        require(M >= 1, f"ComposedRolloutEngine.generate: diffusion.samples_per_prompt={M} must be >= 1")

        # ── Stage 1: AR child ────────────────────────────────────────
        self._ar.wake_up()
        self._diffusion.sleep()

        ar_shell = req.make_root_track(
            track_name="ar",
            branch=N,
            decode_to_condition=None,
        )

        # AR sub-req stage_config: forward parent's "chat" + "ar" subsets
        # and inject pe_instruction on both — ``sglang`` reads "ar" while
        # ``Qwen3Pipeline`` reads "chat".
        ar_stage_config: Dict[str, Any] = {
            key: dict(req.stage_config[key]) for key in ("chat", "ar") if key in req.stage_config
        }
        if self.cfg.pe_instruction:
            for key in ("ar", "chat"):
                ar_stage_config.setdefault(key, {})["system_instruction"] = self.cfg.pe_instruction

        ar_sub_req = RolloutReq(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            primitives={"text": text_primitive},
            request_conditions=dict(req.request_conditions),
            sampling_params={"ar": ar_params},
            stage_config=ar_stage_config,
            sigmas=None,
        )
        ar_resp = self._ar.generate(ar_sub_req)
        require(
            len(ar_resp.tracks) == 1,
            f"ComposedRolloutEngine: AR child must return single-track resp; got {sorted(ar_resp.tracks.keys())}",
        )
        ar_inner = next(iter(ar_resp.tracks.values()))
        require(
            len(ar_inner.sample_ids) == P * N,
            f"ComposedRolloutEngine: AR child returned {len(ar_inner.sample_ids)} samples; expected {P}*{N}={P * N}",
        )

        ar_track = _track_with_field(ar_shell, "segment", ar_inner.segment)
        ar_track = _track_with_field(ar_track, "decoded", ar_inner.decoded)
        ar_track = _track_with_field(ar_track, "conditions", dict(ar_inner.conditions))

        # ── Stage 1 → Stage 2 transition ────────────────────────────
        self._ar.sleep()
        self._diffusion.wake_up()

        # ── Stage 2: diffusion track shell via fork_track ───────────
        diff_shell = ar_track.fork_track(
            parent_name="ar",
            child_name="diffusion",
            branch=M,
            decode_to_condition=None,
        )

        pe_texts = ar_track.decoded
        require(
            pe_texts is not None and len(pe_texts.texts) == P * N,
            f"ComposedRolloutEngine: AR track decoded missing or wrong "
            f"length (expected {P * N}, got "
            f"{len(pe_texts.texts) if pe_texts is not None else 'None'})",
        )

        # Optional marker-based PE extraction: keep only the substring after
        # the marker so the diffusion child sees the rewritten prompt instead
        # of the LLM's reasoning preamble. Off-format outputs fall back to
        # the original user prompt to keep diffusion from collapsing to
        # blank text. ``ar_track.decoded`` is rewritten in place so wandb
        # / logging see the cleaned text.
        if self.cfg.pe_marker:
            cleaned_texts, stats = self._postprocess_pe_texts(
                pe_texts.texts,
                user_prompts=text_primitive.texts,
                samples_per_prompt=N,
            )
            if any(stats.values()):
                logger.info(
                    "ComposedRolloutEngine: PE-extract — marker=%r, %d/%d empty, %d truncated, %d fallback_to_original",
                    self.cfg.pe_marker,
                    stats["empty"],
                    len(pe_texts.texts),
                    stats["truncated"],
                    stats["fallback"],
                )
            pe_texts = Texts(texts=cleaned_texts)
            ar_track = _track_with_field(ar_track, "decoded", pe_texts)

        expanded_texts = [t for t in pe_texts.texts for _ in range(M)]
        require(
            len(expanded_texts) == P * N * M,
            f"ComposedRolloutEngine: expanded_texts wrong length ({len(expanded_texts)} vs expected {P * N * M})",
        )

        diff_sub_req = RolloutReq(
            sample_ids=list(diff_shell.sample_ids),
            group_ids=list(diff_shell.parent_ids or diff_shell.sample_ids),
            primitives={"text": Texts(texts=expanded_texts)},
            request_conditions=dict(req.request_conditions),
            sampling_params={"diffusion": diff_params},
            sigmas=req.sigmas,
        )
        diff_resp = self._diffusion.generate(diff_sub_req)
        require(
            len(diff_resp.tracks) == 1,
            f"ComposedRolloutEngine: diffusion child must return single-track "
            f"resp; got {sorted(diff_resp.tracks.keys())}",
        )
        diff_inner = next(iter(diff_resp.tracks.values()))
        require(
            len(diff_inner.sample_ids) == P * N * M,
            f"ComposedRolloutEngine: diffusion child returned "
            f"{len(diff_inner.sample_ids)} samples; expected "
            f"{P}*{N}*{M}={P * N * M}",
        )

        diff_track = _track_with_field(diff_shell, "segment", diff_inner.segment)
        diff_track = _track_with_field(diff_track, "decoded", diff_inner.decoded)
        diff_track = _track_with_field(diff_track, "conditions", dict(diff_inner.conditions))
        diff_track = _track_with_field(diff_track, "media_preview", diff_inner.media_preview)

        return RolloutResp(tracks={"ar": ar_track, "diffusion": diff_track})

    # ------------------------------------------------------------------
    # Prefix demux helper
    # ------------------------------------------------------------------

    def _demux_by_prefix(
        self,
        keys: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Split a dict by child-name prefix, stripping the prefix."""
        result: Dict[str, Dict[str, Any]] = {}
        for child_name in self._child_by_name:
            prefix = f"{child_name}."
            subset = {k[len(prefix) :]: v for k, v in keys.items() if k.startswith(prefix)}
            if subset:
                result[child_name] = subset
        return result

    def _children_for_track_prefix(self, track_prefix: str) -> List[BaseRolloutEngine]:
        """Resolve the tensor-payload track routing hint to child engines."""
        if not track_prefix:
            return list(self._child_by_name.values())
        child = self._child_by_name.get(track_prefix)
        if child is None:
            raise ValueError(
                f"ComposedRolloutEngine: unknown track_prefix {track_prefix!r}; "
                f"expected one of {sorted(self._child_by_name)}."
            )
        return [child]

    def _postprocess_pe_texts(
        self,
        raw_texts: List[str],
        *,
        user_prompts: List[str],
        samples_per_prompt: int,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Run marker extraction + truncation + empty-fallback over PE outputs.

        ``raw_texts`` is PE-major over ``[P*N]``; the user prompt for slot
        ``k`` is ``user_prompts[k // samples_per_prompt]``.
        """
        marker = self.cfg.pe_marker
        max_chars = self.cfg.pe_max_chars
        cleaned: List[str] = []
        stats = {"empty": 0, "truncated": 0, "fallback": 0}
        for k, raw in enumerate(raw_texts):
            pe = _extract_pe_text(raw, marker)
            if not pe:
                stats["empty"] += 1
            if max_chars is not None and len(pe) > int(max_chars):
                pe = pe[: int(max_chars)]
                stats["truncated"] += 1
            if not pe.strip():
                idx = k // max(1, samples_per_prompt)
                pe = user_prompts[idx] if idx < len(user_prompts) else ""
                stats["fallback"] += 1
            cleaned.append(pe)
        return cleaned, stats

    # ------------------------------------------------------------------
    # Weight sync — prefix-based demux
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
        track_prefix: str = "",
    ) -> None:
        """Route a bucketed-IPC weight push to one child via ``track_prefix``.

        Only a vLLM-Omni child implements the IPC receiver (an SGLang child
        raises). ``replica_rank`` is a vLLM-Omni-specific socket discriminator,
        not part of the base IPC contract, so it is not propagated here.
        """
        if not track_prefix:
            raise ValueError(
                "ComposedRolloutEngine.update_weights_from_ipc requires track_prefix "
                f"so the update can be routed to one child; expected one of {sorted(self._child_by_name)}."
            )
        for child in self._children_for_track_prefix(track_prefix):
            child.update_weights_from_ipc(
                peft_config=peft_config,
                base_sync_done=base_sync_done,
                use_shm=use_shm,
            )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        """Route NCCL group setup to one child via ``track_prefix``."""
        if not track_prefix:
            raise ValueError(
                "ComposedRolloutEngine.init_weights_update_group requires track_prefix "
                f"so the group can be routed to one child; expected one of {sorted(self._child_by_name)}."
            )
        for child in self._children_for_track_prefix(track_prefix):
            child.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend,
            )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Route a NCCL-broadcast weight push to one child via ``track_prefix``."""
        if not track_prefix:
            raise ValueError(
                "ComposedRolloutEngine.update_weights_from_distributed requires track_prefix "
                f"so the update can be routed to one child; expected one of {sorted(self._child_by_name)}."
            )
        for child in self._children_for_track_prefix(track_prefix):
            child.update_weights_from_distributed(
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=group_name,
                target_modules=target_modules,
                flush_cache=flush_cache,
            )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        """Route NCCL group teardown to one child via ``track_prefix``."""
        if not track_prefix:
            raise ValueError(
                "ComposedRolloutEngine.destroy_weights_update_group requires track_prefix "
                f"so the teardown can be routed to one child; expected one of {sorted(self._child_by_name)}."
            )
        for child in self._children_for_track_prefix(track_prefix):
            child.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        demuxed = self._demux_by_prefix(lora_tensors)
        if demuxed:
            for child_name, child_tensors in demuxed.items():
                child = self._child_by_name[child_name]
                if child.is_offloaded:
                    child.onload_weights()
                child.set_lora_from_tensors(
                    adapter_name,
                    child_tensors,
                    peft_config=peft_config,
                )
        else:
            raise ValueError(
                "ComposedRolloutEngine.set_lora_from_tensors requires child-prefixed tensor keys; "
                f"expected prefixes {sorted(self._child_by_name)}."
            )

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        if not track_prefix:
            raise ValueError(
                "ComposedRolloutEngine.update_weights_from_tensor requires track_prefix "
                f"so the update can be routed to one child; expected one of {sorted(self._child_by_name)}."
            )
        children = self._children_for_track_prefix(track_prefix)
        for child in children:
            child.update_weights_from_tensor(
                serialized_named_tensors=serialized_named_tensors,
                target_modules=target_modules,
                load_format=load_format,
                flush_cache=flush_cache,
            )


__all__ = ["ComposedRolloutEngine"]
