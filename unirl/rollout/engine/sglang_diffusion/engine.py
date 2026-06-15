"""``sglang_diffusion`` engine core — wiring + delegation only.

A thin core over the backend seam: it names no concrete model (the adapter, picked
from the registry by ``config.model_family``, owns the ``RolloutReq``↔``RolloutResp``
conversion) and no concrete backend (the seam owns the runtime). Weight sync is a
:class:`WeightSync` component constructed over the seam; the offload lifecycle (a
single flag) lives directly on the engine. The frozen ``base.py`` surface is
implemented as thin forwards here — they must be real class attributes anyway
(``Worker.call`` dispatches by name; ``@distributed`` binds the most-derived
attribute) — which also absorbs the surface quirks (``track_prefix``) so the
component keeps clean signatures.

One-shot construction: after ``__init__`` returns, the generator is spawned and the
engine is usable. ``generate`` / ``sleep`` / ``wake_up`` re-apply ``@distributed``
(the decorator is not inherited — see ``base.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang_diffusion.adapters import get_adapter
from unirl.rollout.engine.sglang_diffusion.backends import SGLangBackend
from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)
from unirl.rollout.engine.sglang_diffusion.weight_sync import WeightSync
from unirl.sde.noise import generate_latents
from unirl.sde.runtime import ensure_req_sigmas
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

#: Memory tags released on sleep / restored on wake.
_OFFLOAD_TAGS = ("transformer", "vae", "text_encoder")
#: Tags backed up to CPU rather than dropped.
_CPU_BACKUP_TAGS = ("vae", "text_encoder")


class SGLangDiffusionRolloutEngine(BaseRolloutEngine):
    """Rollout engine backed by ``sglang.multimodal_gen.DiffGenerator`` (v2 layout)."""

    _component_name = "sglang_diffusion"

    def __init__(
        self,
        config: SGLangDiffusionEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[SGLangDiffusionPorts] = None,
    ) -> None:
        require(
            isinstance(config, SGLangDiffusionEngineConfig),
            f"SGLangDiffusionRolloutEngine requires SGLangDiffusionEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "SGLangDiffusionRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )

        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False

        # Adapter (the only read of a model knob) — owns the conversion + schedule.
        self.adapter = get_adapter(config.model_family)(config, model_config, strategy=strategy)
        pipeline_prefix, target_modules = self.adapter.lora_spec()

        logger.info(
            "Initializing sglang_diffusion engine (rank=%s, local_mode=%s, "
            "model_family=%s, target_modules=%s, populate_conditions=%s)",
            rank,
            config.local_mode,
            config.model_family,
            target_modules,
            config.populate_conditions,
        )

        # Ports — engine-reserved on this node at the last moment before the spawn.
        # Tests inject a fixed set; remote mode uses cfg host/port/scheduler_port.
        if config.local_mode and ports is None:
            ports = SGLangDiffusionPorts.reserve()

        # Backend (the seam) — booted from the config-spelled intent (ports overlaid).
        intent = config.server_intent(
            model_config=model_config,
            ports=ports,
            extra=self.adapter.boot_kwargs(),
        )
        self._backend = SGLangBackend.boot(
            intent,
            local_mode=bool(config.local_mode),
        )

        # Weight sync — owns all sync/LoRA state, over the live seam.
        self._weight_sync = WeightSync(
            self._backend,
            pipeline_prefix=pipeline_prefix,
            target_modules=target_modules,
            uses_lora=bool(model_config.use_lora),
        )

        # σ schedule policy comes from the adapter (absorbs the generic-vs-factory branch).
        self.schedule_policy = self.adapter.schedule_policy()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        require(
            int(req.batch_size) > 0,
            "SGLangDiffusionRolloutEngine.generate requires a non-empty req (batch_size > 0)",
        )
        # σ SSOT: pin once on the full batch (a shared field, so req.slice keeps it).
        ensure_req_sigmas(req, self.schedule_policy)

        fbs = self.cfg.forward_batch_size
        bs = int(req.batch_size)
        if fbs is None or bs <= fbs:
            return self._generate_batch(req)

        outputs: List[RolloutResp] = []
        for start in range(0, bs, fbs):
            end = min(start + fbs, bs)
            outputs.append(self._generate_batch(req.slice(start, end)))
            torch.cuda.empty_cache()
        return RolloutResp.concat(outputs)

    def _generate_batch(self, req: RolloutReq) -> RolloutResp:
        initial_noise = self._resolve_initial_noise(req)
        kwargs = self.adapter.build_inputs(req, initial_noise=initial_noise)
        raw = self._backend.generate(kwargs)
        return self.adapter.build_response(req, raw)

    def _resolve_initial_noise(self, req: RolloutReq) -> Optional[torch.Tensor]:
        """NoiseRecipe (driver-authoritative x_T) → init_same_noise → None. Model-agnostic."""
        xt = NoiseRecipe.from_rollout_req(req).resolve()
        if xt is not None:
            return xt
        if not bool(self.cfg.init_same_noise):
            return None

        diffusion = get_diffusion_params(req.sampling_params)
        require(
            diffusion is not None and diffusion.seed is not None,
            "init_same_noise=True requires req.sampling_params diffusion seed",
        )
        batch_size = int(req.batch_size)
        latent_shape = self._backend.prepare_latent_shape(
            height=int(diffusion.height),
            width=int(diffusion.width),
            num_frames=int(diffusion.num_frames),
            batch_size=batch_size,
        )
        dtype = parse_torch_dtype(diffusion.autocast_precision, field_name="autocast_precision")
        return generate_latents(
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=self._device,
            dtype=dtype,
            init_same_noise=True,
            samples_per_prompt=int(diffusion.samples_per_prompt),
            noise_group_ids=[str(gid) for gid in req.group_ids],
            base_seed=int(diffusion.seed),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle — the offload flag lives here; decorators re-applied (base.py footgun)
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        # Idempotent, symmetric with ``wake_up``: a second ``sleep()`` while already
        # offloaded would issue ``release_memory_occupation`` to the scheduler twice.
        if self._is_offloaded:
            return
        self._backend.release_memory(tags=_OFFLOAD_TAGS, cpu_backup_tags=_CPU_BACKUP_TAGS)
        self._is_offloaded = True
        # The released tags include the transformer weights → the loaded LoRA pool
        # is gone; the next weight sync must re-push.
        self._weight_sync.mark_weights_released()
        logger.info("sglang_diffusion engine slept (release_memory_occupation).")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        self._backend.resume_memory(tags=_OFFLOAD_TAGS)
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def onload_weights(self, *, track_prefix: str = "") -> None:
        # Diffusion release/resume is all-or-nothing on one tag set, so onloading
        # weights == waking.
        del track_prefix
        self.wake_up()

    def health_check(self) -> bool:
        return self._backend.ping()

    def shutdown(self) -> None:
        self._backend.shutdown()

    # ------------------------------------------------------------------ #
    # Weight sync — frozen base.py surface; thin forwards to the component.
    # Un-decorated: reached per worker via the raw ``Worker.call`` RPC, not
    # through ``@distributed``. ``track_prefix`` is absorbed here.
    # ------------------------------------------------------------------ #

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
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
        del track_prefix
        self._weight_sync.init_weights_update_group(
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
        del track_prefix
        self._weight_sync.update_weights_from_distributed(
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
        del track_prefix
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    def loaded_param_checksums(self, *, names: List[str]) -> Dict[int, List[Dict[str, str]]]:
        return self._weight_sync.loaded_param_checksums(names=names)

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._weight_sync.lora_dirty

    # ``update_weights_from_ipc`` is deliberately NOT defined — the base raises
    # NotImplementedError (SGLang has no bucketed-IPC receiver).


__all__ = ["SGLangDiffusionRolloutEngine"]
