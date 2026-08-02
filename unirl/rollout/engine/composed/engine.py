"""ComposedRolloutEngine — two-child engine orchestrating PE serial flow.

Holds an ``ar`` child and a ``diffusion`` child. ``generate(sample)`` runs
them sequentially:

1. AR child rewrites each of P prompts into N PE candidates  → ``[P*N]`` ar Part
2. Diffusion child generates M images per PE                  → ``[P*N*M]`` diffusion Part
3. Returns the filled 3-part ``[input, ar, diffusion]`` ``Sample`` — lineage is
   positional (parent = preceding part) and the path ids chain
   prompt → PE → image.

The request ``Sample`` is pre-forked ``[input, ar_shell, diffusion_shell]``
(the caller forks; carrying BOTH the AR and diffusion sampling params, which a
single Part's one ``sampling_params`` slot cannot). Each child config carries
its own ``_target_``; ``__init__`` builds the child engine via
``config.<child>.make_engine(...)``.

Weight sync uses prefix-based tensor routing: the training side prepends
the child component name to tensor keys; this engine demuxes by prefix and
forwards each subset to the matching child with the prefix stripped.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.pe.instruction import postprocess_pe_texts
from unirl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from unirl.rollout.engine.synchronous import SyncRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

logger = logging.getLogger(__name__)


def _cleanup_constructed_child(name: str, child: Any) -> None:
    shutdown = getattr(child, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown()
    except Exception as exc:
        logger.warning("Child %r cleanup after construction failure raised: %s", name, exc)


class ComposedRolloutEngine(SyncRolloutEngine):
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

        ar = config.ar.make_engine(strategy=None, **deps)
        try:
            require(
                isinstance(ar, SyncRolloutEngine),
                f"ComposedRolloutEngine ar child must be a SyncRolloutEngine; got {type(ar).__name__}",
            )
        except BaseException:
            _cleanup_constructed_child("ar", ar)
            raise

        try:
            diffusion = config.diffusion.make_engine(strategy=strategy, **deps)
        except BaseException:
            _cleanup_constructed_child("ar", ar)
            raise
        try:
            require(
                isinstance(diffusion, SyncRolloutEngine),
                f"ComposedRolloutEngine diffusion child must be a SyncRolloutEngine; got {type(diffusion).__name__}",
            )
        except BaseException:
            _cleanup_constructed_child("diffusion", diffusion)
            _cleanup_constructed_child("ar", ar)
            raise

        self._ar = ar
        self._diffusion = diffusion

        self._child_by_name: Dict[str, SyncRolloutEngine] = {
            "ar": self._ar,
            "diffusion": self._diffusion,
        }

        # The whole AR→diffusion transition is one serialized operation because
        # the children share wake/sleep state. Composed owns no weights and
        # propagates each child's stamp rather than stamping its own version.
        self._weight_version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False
        try:
            if config.sleep_diffusion_on_start:
                self._diffusion.sleep()
        except BaseException:
            _cleanup_constructed_child("diffusion", diffusion)
            _cleanup_constructed_child("ar", ar)
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._generate_lock:
                self._shutdown_requested = True
            with self._generate_lock:
                for name, child in self._child_by_name.items():
                    try:
                        child.shutdown()
                    except Exception as exc:
                        logger.warning("Child %r shutdown raised: %s", name, exc)
            self._shutdown_complete = True

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
    def generate(self, sample: Sample) -> Sample:
        """Run the whole PE serial flow for one DP shard synchronously."""
        return self._generate_locked(sample)

    def _generate_locked(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("ComposedRolloutEngine.generate called after shutdown")
            return self._generate_core(sample)

    def _generate_core(self, sample: Sample) -> Sample:
        """Run the PE serial flow for a whole ``Sample`` → filled 3-part output.

        Composed's children share wake/sleep state, so the shard is processed as ONE
        atomic unit, giving a single AR→diffusion wake/sleep transition.
        ``sample`` is the pre-forked request ``[input, ar_shell, diffusion_shell]`` for
        P prompts: the AR shell carries ``ARSamplingParams`` (branch N), the diffusion
        shell ``DiffusionSamplingParams`` (branch M, σ / x_T recipe).
        """
        input_part, ar_shell, diffusion_shell = self._unpack_request(sample)

        P = len(input_part.sample_ids)
        require(P > 0, "ComposedRolloutEngine.generate: empty Sample")
        text_primitive = input_part.primitives.get("text")
        require(
            isinstance(text_primitive, Texts),
            "ComposedRolloutEngine.generate: input Part.primitives['text'] must be Texts",
        )
        require(
            len(ar_shell.sample_ids) % P == 0,
            f"ComposedRolloutEngine: ar shell {len(ar_shell.sample_ids)} not a multiple of P={P}",
        )
        N = len(ar_shell.sample_ids) // P
        require(
            N >= 1 and len(diffusion_shell.sample_ids) % len(ar_shell.sample_ids) == 0,
            f"ComposedRolloutEngine: diffusion shell {len(diffusion_shell.sample_ids)} not a multiple of P*N={P * N}",
        )
        M = len(diffusion_shell.sample_ids) // len(ar_shell.sample_ids)
        require(M >= 1, f"ComposedRolloutEngine.generate: diffusion branch M={M} must be >= 1")

        # ── Stage 1: AR child ────────────────────────────────────────
        self._ar.wake_up()
        self._diffusion.sleep()

        # AR child request: the prompts + the AR shell. The input Part's
        # ``control`` carries the AR routing (forward parent's "chat" + "ar"
        # subsets, inject pe_instruction on both — ``sglang`` reads "ar" while
        # ``Qwen3Pipeline`` reads "chat").
        ar_input = Part.input(
            sample_ids=list(input_part.sample_ids),
            primitives={"text": text_primitive},
            control=self._ar_control(input_part.control or {}),
        )
        ar_out = self._ar.generate(Sample(parts=[ar_input, ar_shell]))
        ar_part = ar_out.parts[-1]
        require(
            len(ar_part.sample_ids) == P * N,
            f"ComposedRolloutEngine: AR child returned {len(ar_part.sample_ids)} samples; expected {P}*{N}={P * N}",
        )

        # PE extraction: clean the AR text the diffusion child conditions on
        # (rewritten onto the AR Part's text primitive so logging sees it too).
        pe_texts = ar_part.primitives.get("text")
        require(
            isinstance(pe_texts, Texts) and len(pe_texts.texts) == P * N,
            f"ComposedRolloutEngine: AR Part decoded missing or wrong length "
            f"(expected {P * N}, got {len(pe_texts.texts) if isinstance(pe_texts, Texts) else 'None'})",
        )
        pe_texts = self._extract_pe(pe_texts, text_primitive, N)
        ar_part = ar_part.fill(primitives={"text": pe_texts})

        # ── Stage 1 → Stage 2 transition ────────────────────────────
        self._ar.sleep()
        self._diffusion.wake_up()

        # ── Stage 2: diffusion child ─────────────────────────────────
        # The PE ids ("p0/0") are non-root, so they cannot be a child Sample's
        # root input. Re-root the PE prompts onto fresh ids, fork the diffusion
        # shell off them (preserving the shell's sampling params + x_T segment),
        # generate, then map the per-sample outputs back onto our
        # lineage-correct diffusion_shell — row order matches (both
        # group-by-parent, branch=M, same parent order).
        pe_input = Part.input(
            sample_ids=[f"pe{k}" for k in range(P * N)],
            primitives={"text": pe_texts},
        )
        diff_child_shell = pe_input.fork(
            M,
            sampling_params=diffusion_shell.sampling_params,
            new_segment=diffusion_shell.segment,
        )
        diff_out = self._diffusion.generate(Sample(parts=[pe_input, diff_child_shell]))
        diff_child = diff_out.parts[-1]
        require(
            len(diff_child.sample_ids) == len(diffusion_shell.sample_ids),
            f"ComposedRolloutEngine: diffusion child returned {len(diff_child.sample_ids)} "
            f"samples; expected {P}*{N}*{M}={P * N * M}",
        )
        # Carry the diffusion child's weight version onto the frontier part — composed
        # owns no weights, so it propagates the child's stamp rather than its own 0.
        diffusion_part = diffusion_shell.fill(
            segment=diff_child.segment,
            primitives=dict(diff_child.primitives),
            primitive_metadata=dict(diff_child.primitive_metadata),
            conditions=dict(diff_child.conditions),
            media_preview=diff_child.media_preview,
            weight_version=diff_child.weight_version,
        )

        return Sample(parts=[input_part, ar_part, diffusion_part])

    # ------------------------------------------------------------------
    # Control plane — forwarded to both children (best-effort).
    # ------------------------------------------------------------------

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Abort in-flight generation on both children. Partials surface via the
        pending ``generate`` returns, so this returns ``[]``. Composed-level ids
        don't map onto child id-spaces (PE prompts are re-rooted), so abort-all."""
        del ids
        for child in self._child_by_name.values():
            child.abort()
        return []

    def pause(self) -> None:
        for child in self._child_by_name.values():
            child.pause()

    def resume(self) -> None:
        for child in self._child_by_name.values():
            child.resume()

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _unpack_request(self, sample: Sample) -> tuple:
        """Resolve the pre-forked ``[input, ar_shell, diffusion_shell]`` request.

        The serial PE flow currently accepts exactly one input Part. Reject extra
        inputs explicitly instead of dropping them from the returned lineage."""
        require(
            bool(sample.parts) and sample.parts[0].is_root,
            "ComposedRolloutEngine.generate: requires a root input Part at parts[0]",
        )
        require(
            len(sample.parts) == 3,
            "ComposedRolloutEngine.generate: requires exactly "
            f"[input, ar_shell, diffusion_shell]; got {len(sample.parts)} Parts.",
        )
        input_part, ar_shell, diffusion_shell = sample.parts
        require(
            isinstance(ar_shell.sampling_params, ARSamplingParams),
            "ComposedRolloutEngine.generate: requires an AR gen-shell Part (ARSamplingParams)",
        )
        require(
            isinstance(diffusion_shell.sampling_params, DiffusionSamplingParams),
            "ComposedRolloutEngine.generate: requires a diffusion gen-shell Part (DiffusionSamplingParams)",
        )
        return input_part, ar_shell, diffusion_shell

    def _ar_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        """The AR child input Part's ``control``: parent's "chat" + "ar" subsets
        with pe_instruction injected on both."""
        ar_control: Dict[str, Any] = {key: dict(control[key]) for key in ("chat", "ar") if key in control}
        if self.cfg.pe_instruction:
            for key in ("ar", "chat"):
                ar_control.setdefault(key, {})["system_instruction"] = self.cfg.pe_instruction
        return ar_control

    def _extract_pe(self, pe_texts: Texts, text_primitive: Texts, samples_per_prompt: int) -> Texts:
        """Optional marker-based PE extraction: keep only the substring after the
        marker so the diffusion child sees the rewritten prompt instead of the
        LLM's reasoning preamble. Off-format outputs fall back to the original
        user prompt to keep diffusion from collapsing to blank text."""
        if not self.cfg.pe_marker:
            return pe_texts
        cleaned_texts, stats = postprocess_pe_texts(
            pe_texts.texts,
            user_prompts=text_primitive.texts,
            samples_per_prompt=samples_per_prompt,
            marker=self.cfg.pe_marker,
            max_chars=self.cfg.pe_max_chars,
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
        return Texts(texts=cleaned_texts)

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

    def _children_for_track_prefix(self, track_prefix: str) -> List[SyncRolloutEngine]:
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
