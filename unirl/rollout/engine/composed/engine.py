"""ComposedRolloutEngine — two-child engine orchestrating PE serial flow."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.pe.instruction import ar_child_control, postprocess_pe_texts
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

logger = logging.getLogger(__name__)


class ComposedRolloutEngine(BaseRolloutEngine):
    """Two-child rollout engine for prompt-enhancement (PE) serial flow."""

    _component_name = "composed"

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
        self.child_by_name: Dict[str, BaseRolloutEngine] = {}
        self.lock = threading.Lock()
        self.closed = False

        deps = dict(device=device, rank=rank, model_config=model_config)
        try:
            self.ar = config.ar.make_engine(strategy=None, **deps)
            self.child_by_name["ar"] = self.ar
            self.diffusion = config.diffusion.make_engine(strategy=strategy, **deps)
            self.child_by_name["diffusion"] = self.diffusion
            if config.sleep_diffusion_on_start:
                self.diffusion.sleep()
        except BaseException:
            self.shutdown()
            raise

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for name, child in self.child_by_name.items():
                try:
                    child.shutdown()
                except Exception as exc:
                    logger.warning("Child %r shutdown raised: %s", name, exc)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        for child in self.child_by_name.values():
            child.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        for child in self.child_by_name.values():
            child.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        if track_prefix:
            self.child_by_name[track_prefix].onload_weights()
            return
        for child in self.child_by_name.values():
            child.onload_weights()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_version(self, train_version: int) -> None:
        for child in self.child_by_name.values():
            child.set_version(train_version)

    @property
    def is_offloaded(self) -> bool:
        return all(child.is_offloaded for child in self.child_by_name.values())

    def health_check(self) -> bool:
        return all(child.health_check() for child in self.child_by_name.values())

    def get_memory_info(self) -> Dict[str, float]:
        out: Dict[str, float] = {"allocated_gb": 0.0, "cached_gb": 0.0}
        for child in self.child_by_name.values():
            m = child.get_memory_info()
            out["allocated_gb"] += m.get("allocated_gb", 0.0)
            out["cached_gb"] += m.get("cached_gb", 0.0)
        return out

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Run the whole PE serial flow for one DP shard synchronously."""
        with self.lock:
            if self.closed:
                raise RuntimeError("ComposedRolloutEngine.generate called after shutdown")
            sample.reject_row_canvases(engine="ComposedRolloutEngine")
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

            P = len(input_part.sample_ids)
            require(P > 0, "ComposedRolloutEngine.generate: empty Sample")
            text_primitive = input_part.primitives.get("text")
            require(
                isinstance(text_primitive, Texts),
                "ComposedRolloutEngine.generate: input Part.primitives['text'] must be Texts",
            )
            require(
                len(ar_shell.sample_ids) > 0 and len(ar_shell.sample_ids) % P == 0,
                f"ComposedRolloutEngine: ar shell {len(ar_shell.sample_ids)} must be a positive multiple of P={P}",
            )
            N = len(ar_shell.sample_ids) // P
            require(
                len(diffusion_shell.sample_ids) > 0 and len(diffusion_shell.sample_ids) % len(ar_shell.sample_ids) == 0,
                f"ComposedRolloutEngine: diffusion shell {len(diffusion_shell.sample_ids)} must be a positive multiple of P*N={P * N}",
            )
            M = len(diffusion_shell.sample_ids) // len(ar_shell.sample_ids)

            self.ar.wake_up()
            self.diffusion.sleep()

            ar_input = Part.input(
                sample_ids=list(input_part.sample_ids),
                primitives={"text": text_primitive},
                control=ar_child_control(input_part.control, system_instruction=self.cfg.pe_instruction),
            )
            ar_out = self.ar.generate(Sample(parts=[ar_input, ar_shell]))
            ar_part = ar_out.parts[-1]
            require(
                len(ar_part.sample_ids) == P * N,
                f"ComposedRolloutEngine: AR child returned {len(ar_part.sample_ids)} samples; expected {P}*{N}={P * N}",
            )

            pe_texts = ar_part.primitives.get("text")
            require(
                isinstance(pe_texts, Texts) and len(pe_texts.texts) == P * N,
                f"ComposedRolloutEngine: AR Part decoded missing or wrong length "
                f"(expected {P * N}, got {len(pe_texts.texts) if isinstance(pe_texts, Texts) else 'None'})",
            )
            if self.cfg.pe_marker:
                cleaned_texts, stats = postprocess_pe_texts(
                    pe_texts.texts,
                    user_prompts=text_primitive.texts,
                    samples_per_prompt=N,
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
                pe_texts = Texts(texts=cleaned_texts)
            ar_part = ar_part.fill(primitives={"text": pe_texts})

            self.ar.sleep()
            self.diffusion.wake_up()

            # Re-root diffusion prompts because composed PE IDs are not valid child roots.
            pe_input = Part.input(
                sample_ids=[f"pe{k}" for k in range(P * N)],
                primitives={"text": pe_texts},
            )
            diff_child_shell = pe_input.fork(
                M,
                sampling_params=diffusion_shell.sampling_params,
                new_segment=diffusion_shell.segment,
            )
            diff_out = self.diffusion.generate(Sample(parts=[pe_input, diff_child_shell]))
            diff_child = diff_out.parts[-1]
            require(
                len(diff_child.sample_ids) == len(diffusion_shell.sample_ids),
                f"ComposedRolloutEngine: diffusion child returned {len(diff_child.sample_ids)} "
                f"samples; expected {P}*{N}*{M}={P * N * M}",
            )
            diffusion_part = diffusion_shell.fill(
                segment=diff_child.segment,
                primitives=dict(diff_child.primitives),
                primitive_metadata=dict(diff_child.primitive_metadata),
                conditions=dict(diff_child.conditions),
                media_preview=diff_child.media_preview,
                output_version=diff_child.output_version,
            )

            return Sample(parts=[input_part, ar_part, diffusion_part])

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Abort in-flight generation on both children. Partials surface via the"""
        del ids
        for child in self.child_by_name.values():
            child.abort()
        return []

    def pause(self) -> None:
        for child in self.child_by_name.values():
            child.pause()

    def resume(self) -> None:
        for child in self.child_by_name.values():
            child.resume()

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
        track_prefix: str = "",
    ) -> None:
        """Route a bucketed-IPC weight push to one child via ``track_prefix``."""
        self.child_by_name[track_prefix].update_weights_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
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
        self.child_by_name[track_prefix].init_weights_update_group(
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
        self.child_by_name[track_prefix].update_weights_from_distributed(
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
        self.child_by_name[track_prefix].destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        matched = False
        for child_name, child in self.child_by_name.items():
            prefix = f"{child_name}."
            subset = {k[len(prefix) :]: v for k, v in lora_tensors.items() if k.startswith(prefix)}
            if subset:
                matched = True
                if child.is_offloaded:
                    child.onload_weights()
                child.set_lora_from_tensors(
                    adapter_name,
                    subset,
                    peft_config=peft_config,
                )
        if not matched:
            raise ValueError(
                "ComposedRolloutEngine.set_lora_from_tensors requires child-prefixed tensor keys; "
                f"expected prefixes {sorted(self.child_by_name)}."
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
        self.child_by_name[track_prefix].update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )


__all__ = ["ComposedRolloutEngine"]
