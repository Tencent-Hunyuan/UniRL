"""SGLang rollout engine.

One class, one-shot construction. Implements
:class:`unirl.rollout.engine.base.BaseRolloutEngine` and speaks the
``RolloutReq`` / ``RolloutResp`` types end-to-end.

Lifecycle:
    The ctor takes config + runtime deps (``device``, ``strategy``, ``rank``,
    ``model_config``) and returns a fully-usable engine: ``ServerArgs`` built,
    ``DiffGenerator.from_pretrained`` complete, scheduler reachable. There is
    no separate ``initialize(device)`` step.

Generation:
    ``generate(req)`` either reads the pre-shipped ``initial_latents`` off
    ``req.request_conditions['initial_latents']`` or computes Gaussian noise
    via :meth:`_compute_initial_noise` (gated on ``cfg.init_same_noise`` for
    per-group sharing). The translators in :mod:`request` / :mod:`response`
    handle the kwargs build and the result → ``RolloutResp`` packing.

Weight sync:
    Five direct forwards to SGLang's scheduler request types.
    ``update_weights_from_ipc`` raises ``NotImplementedError`` —
    SGLang has no bucketed-IPC receiver today.

What this engine intentionally does NOT do (vs upstream SGLang at
``unirl/samplers/sglang/engine.py``):

- No ``initialize(device)`` step / ``_is_initialized`` flag — one-shot ctor.
- No ``update_weights(state_dict)`` / ``update_weights_from_path`` — the
  trainer-side ``TensorWeightSync`` handler packages and pushes via
  ``update_weights_from_tensor`` directly.
- No ``_infer_model_type`` substring match — ``cfg.model_family`` is an
  explicit enum.
- No instance-level ``_cached_runtime`` import hook — lazy module-level.
- No ``encode_prompt`` / ``decode_latents`` — not on BaseRolloutEngine.
- No ``supports_distributed`` / ``requires_external_service`` properties.
- No ``get_last_weight_checksum`` / ``_verify_weight_checksum`` flag — use
  :meth:`loaded_param_checksums` on demand.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang._patches import SglangDiffusionHijack
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.request import _to_sglang_kwargs
from unirl.rollout.engine.sglang.response import _to_rollout_resp
from unirl.sde.noise import generate_latents
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params
from unirl.utils.dtypes import parse_torch_dtype
from unirl.utils.peft_merge import adapt_lora_for_sglang

logger = logging.getLogger(__name__)


def _import_sglang_runtime() -> Dict[str, Any]:
    """Lazy import of SGLang scheduler types. Imported once per process.

    RL request structs that exist only in the ``celve/sglang@diffusionrl`` fork
    (weight-update groups, tensor/distributed weight sync, in-memory LoRA, memory
    occupation) are sourced from UniRL's ``_patches`` package -- the SAME
    single-definition site the in-process ``patch_scheduler`` keys its
    ``request_handlers`` dict by, so ``type(req)`` dispatch matches on stock
    upstream sglang. Only ``GetWeightsChecksumReqInput`` (present upstream) plus
    the generator/server/client stay sourced from sglang.
    """
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        GetWeightsChecksumReqInput,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from unirl.rollout.engine.sglang._patches.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from unirl.rollout.engine.sglang._patches.lora_req import (
        SetLoraFromTensorsReq,
    )

    return {
        "DiffGenerator": DiffGenerator,
        "ServerArgs": ServerArgs,
        "GetWeightsChecksumReqInput": GetWeightsChecksumReqInput,
        "InitWeightsUpdateGroupReqInput": InitWeightsUpdateGroupReqInput,
        "DestroyWeightsUpdateGroupReqInput": DestroyWeightsUpdateGroupReqInput,
        "UpdateWeightsFromDistributedReqInput": UpdateWeightsFromDistributedReqInput,
        "UpdateWeightsFromTensorReqInput": UpdateWeightsFromTensorReqInput,
        "ReleaseMemoryOccupationReqInput": ReleaseMemoryOccupationReqInput,
        "ResumeMemoryOccupationReqInput": ResumeMemoryOccupationReqInput,
        "SetLoraFromTensorsReq": SetLoraFromTensorsReq,
        "sync_scheduler_client": sync_scheduler_client,
    }


class SGLangRolloutEngine(BaseRolloutEngine):
    """Rollout engine backed by ``sglang.multimodal_gen.DiffGenerator``."""

    _component_name = "sglang"

    def __init__(
        self,
        config: SGLangEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, SGLangEngineConfig),
            f"SGLangRolloutEngine requires SGLangEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "SGLangRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )
        # Per-engine SGLang ports. In dedicated-rollout (separate) the engine is
        # built with an explicit ``rank``; in colocate the Worker injects rank via
        # setup() AFTER __init__, so ``rank`` is None here. SGLang still spawns a
        # per-engine scheduler that binds a TCP dist-init port (master_port), so
        # each colocated engine on a node needs a distinct port block — otherwise
        # all 8 fall back to the same default and collide (EADDRINUSE). Fall back
        # to the DevicePool-provided ``RANK`` env (device id, unique per node).
        port_rank = rank
        if port_rank is None:
            env_rank = os.environ.get("RANK")
            port_rank = int(env_rank) if env_rank is not None and env_rank.isdigit() else 0
        config = config.with_sglang_ports(int(port_rank))

        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sde_label = self._resolve_sde_label(strategy)
        self._target_modules: List[str] = list(self.cfg.target_modules or ("transformer",))
        # Install the in-process sglang monkey-patches BEFORE the first sglang
        # import below. `_import_sglang_runtime()` imports DiffGenerator, which at
        # module load forces `mp.set_start_method("spawn")`; the hijack's spawn
        # shim must already be live so the scheduler/worker child re-installs the
        # patches before serving. Idempotent.
        SglangDiffusionHijack.hijack()
        self._runtime = _import_sglang_runtime()
        self._is_offloaded = False
        # Pipeline prefix embedded in canonical LoRA keys, e.g. "transformer."
        # for SD3/WAN/HV15/Qwen or "model." for HunyuanImage3.  Stripped by
        # adapt_lora_for_sglang so keys match SGLang's named_modules() space.
        self._pipeline_prefix: str = str(getattr(model_config, "weight_sync_param_name_prefix", "") or "")

        server_kwargs = self.cfg.build_server_kwargs(
            self._runtime["ServerArgs"],
            model_config=model_config,
        )

        # LoRA / target-module agreement check — pre-existing safety net.
        if model_config.use_lora and not server_kwargs.get("lora_target_modules"):
            logger.warning(
                "SGLang LoRA enabled without lora_target_modules; set bundle "
                "default_lora_target_modules() or --training.lora-target-modules."
            )

        logger.info(
            "Initializing SGLang engine (rank=%s, local_mode=%s, "
            "target_modules=%s, model_family=%s, populate_conditions=%s)",
            rank,
            self.cfg.local_mode,
            self._target_modules,
            self.cfg.model_family,
            self.cfg.populate_conditions,
        )

        disable_autocast = server_kwargs.get("disable_autocast")
        server_args = self._runtime["ServerArgs"].from_kwargs(**server_kwargs)
        if disable_autocast is not None:
            server_args.disable_autocast = disable_autocast

        self._server_args = server_args
        # Each colocated rank spawns its own sglang-diffusion worker subprocess
        # that brings up a dist group via ``env://`` (reads MASTER_PORT). In v2
        # colocate every rank inherits the DevicePool training MASTER_PORT, so
        # the subprocesses collide on it (EADDRINUSE). Point the spawn at this
        # rank's dedicated port (from ``with_sglang_ports``) and restore after —
        # the training FSDP group is already initialized, so it won't re-read env.
        sglang_master_port = (self.cfg.engine_kwargs or {}).get("master_port")
        _saved_master_port = os.environ.get("MASTER_PORT")
        if sglang_master_port is not None:
            os.environ["MASTER_PORT"] = str(sglang_master_port)
        try:
            self._generator = self._runtime["DiffGenerator"].from_pretrained(
                server_args=server_args,
                local_mode=bool(self.cfg.local_mode),
            )
        finally:
            if sglang_master_port is not None:
                if _saved_master_port is None:
                    os.environ.pop("MASTER_PORT", None)
                else:
                    os.environ["MASTER_PORT"] = _saved_master_port

        # σ schedule policy — loaded once from the pretrained checkpoint
        # dir's JSONs (scheduler/transformer/vae configs). ``ensure_req_sigmas``
        # consumes it in ``generate`` to pin ``req.sigmas`` before the
        # request crosses the wire to the SGLang worker.
        #
        # ``shift`` source: ``model_config.shift`` is the single SOT.
        # ``SD3PipelineConfig`` / ``WAN21PipelineConfig`` /
        # ``WAN22PipelineConfig`` / ``HunyuanImage3PipelineConfig`` all
        # carry it (SD3=3.0, Wan=5.0, etc.).
        if not hasattr(model_config, "shift"):
            raise RuntimeError(
                f"SGLangRolloutEngine requires model_config.shift; got "
                f"{type(model_config).__name__}. Use a registered model preset "
                f"(e.g. ``sd3``, ``wan21``, ``hunyuan_image3``)."
            )
        # Per-model schedule-policy factory hook. Some models (FLUX.2-Klein)
        # require a SchedulePolicy subclass with a model-specific
        # ``compute_mu`` override that the generic
        # ``FlowMatchSchedulePolicy.from_pretrained`` path cannot
        # synthesize from ``scheduler_config.json``. When the model_config
        # exposes ``build_schedule_policy()`` we delegate to it (mirrors
        # the trainside engine's ``pipeline.build_schedule_policy()``
        # branch); otherwise fall back to the generic constructor that
        # works for SD3 / Wan / Qwen-Image / etc.
        #
        # Same use_dynamic_shifting hook as vllm_omni engine. Generic —
        # any model config that declares it (Qwen-Image, future dynamic
        # models) gets the right policy without engine-side dispatch.
        if hasattr(model_config, "build_schedule_policy") and callable(
            getattr(model_config, "build_schedule_policy", None)
        ):
            self.schedule_policy = model_config.build_schedule_policy()
        else:
            require_dynamic = bool(getattr(model_config, "use_dynamic_shifting", False))
            dynamic_overrides = getattr(model_config, "dynamic_shift_overrides", None)
            self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                model_config.pretrained_model_ckpt_path,
                shift=float(model_config.shift),
                require_dynamic=require_dynamic,
                dynamic_overrides=dynamic_overrides,
            )

    # ------------------------------------------------------------------
    # Strategy → SGLang SDE kernel label mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sde_label(strategy: Any) -> Optional[str]:
        """Resolve the SDE-strategy → SGLang kernel label once at ctor time.

        Mirrors legacy ``samplers/sglang/engine.py:_resolve_rollout_sde_type``.
        Returns ``None`` when strategy is missing — ODE-mode callers (eval,
        DiffusionNFT) won't hit the SDE branch in the request translator anyway.

        The returned string is the value of SGLang's ``rollout_sde_type``
        kwarg in the per-request kwargs (see ``request.py``). For each
        strategy below the SGLang fork must register a matching kernel
        whose update math is bit-for-bit identical to the UniRL
        kernel in :mod:`unirl.sde.kernels`; otherwise iter-0
        importance ratios drift and the trainer-side replay diverges.

        - ``flow`` → ``"sde"`` — FlowGRPO (SD3, Wan, Qwen-Image, etc.)
        - ``cps``  → ``"cps"`` — coefficient-preserving sampling
        - ``dance`` → ``"dance"`` — DanceGRPO (FLUX.2-Klein). Assumes the
          SGLang fork registers the Dance kernel under the string
          ``"dance"``; if the fork uses a different identifier
          (e.g. ``"flux2_dance"``), adjust this branch.
        """
        if strategy is None:
            return None
        canonical = type(strategy).canonical_name.strip().lower()
        if canonical == "flow":
            return "sde"
        if canonical == "cps":
            return "cps"
        if canonical == "dance":
            return "dance"
        raise ValueError(
            f"SGLang rollout currently supports only sde_type in {{'flow', 'cps', 'dance'}} "
            f"(those have a verified SGLang-side kernel that matches UniRL's math); "
            f"got canonical={canonical!r}. Either switch the SDE strategy on this engine, "
            f"or add an explicit mapping after verifying the SGLang-side kernel is "
            f"mathematically equivalent."
        )

    # ------------------------------------------------------------------
    # Scheduler request plumbing
    # ------------------------------------------------------------------

    def _send_scheduler_request(self, request: Any, *, operation: str) -> Any:
        response = self._runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, operation=operation)
        require(success, f"{operation} failed: {message}")
        return response

    @staticmethod
    def _extract_update_status(response: Any, *, operation: str) -> tuple[bool, str]:
        output = getattr(response, "output", None)
        require(isinstance(output, dict), f"Invalid SGLang response for {operation}: {response}")
        success = bool(output.get("success", False))
        message = str(output.get("message", "Unknown status"))
        return success, message

    def _call_memory_api(
        self,
        method_name: str,
        *,
        tags: Sequence[str],
        cpu_backup_tags: Optional[Sequence[str]] = None,
    ) -> Any:
        # Stock upstream DiffGenerator has no memory-occupation methods (the fork
        # added them); route through the scheduler client to the handlers that
        # ``patch_scheduler`` installs, keyed by the same ``_patches`` req types.
        if method_name == "release_memory_occupation":
            request = self._runtime["ReleaseMemoryOccupationReqInput"](
                tags=list(tags),
                cpu_backup_tags=(list(cpu_backup_tags) if cpu_backup_tags is not None else None),
            )
        elif method_name == "resume_memory_occupation":
            request = self._runtime["ResumeMemoryOccupationReqInput"](tags=list(tags))
        else:
            raise ValueError(f"SGLang engine: unknown memory API {method_name!r}")
        response = self._runtime["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, operation=method_name)
        require(success, f"{method_name} failed: {message}")
        return getattr(response, "output", None)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        require(
            int(req.batch_size) > 0,
            "SGLangRolloutEngine.generate requires non-empty req (batch_size > 0)",
        )

        # Main-repo SSOT for σ: pin once via the shared helper. Request
        # translator reads ``req.sigmas`` (no recompute) and forwards to
        # SGLang; response handler asserts SGLang echoed back what we sent.
        # ``sigmas`` is a shared field, so the ``req.slice`` below keeps it
        # intact and every chunk reuses this one schedule.
        ensure_req_sigmas(req, self.schedule_policy)

        # ``forward_batch_size`` bounds the per-forward activation: slice the
        # request into chunks, run one SGLang forward each, and concat. Noise is
        # resolved per chunk (not sliced from a full-batch tensor) so both the
        # pre-shipped ``initial_latents`` path (sliced by ``req.slice``) and the
        # ``init_same_noise`` group-keyed path stay correct; ``RolloutResp.concat``
        # is a plain per-field merge (segment rows stay 1:1 with samples), so the
        # reassembled response matches an
        # unchunked call. Determinism caveat: when neither pre-shipped latents nor
        # ``init_same_noise`` is set, SGLang draws its own initial noise and a
        # different chunk size can change its batch layout (and thus sampling).
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
        """Run one SGLang forward over an already-σ-pinned request.

        Assumes ``ensure_req_sigmas`` has populated ``req.sigmas`` (the chunking
        wrapper in :meth:`generate` does this once before slicing).
        """
        initial_noise = self._resolve_initial_noise(req)

        kwargs = _to_sglang_kwargs(
            req,
            cfg=self.cfg,
            sde_label=self._sde_label,
            initial_noise=initial_noise,
        )

        raw_results = self._generator.generate(sampling_params_kwargs=kwargs)
        require(
            raw_results is not None,
            "SGLang generator returned None — full-batch failure (see DiffGenerator.generate docstring)",
        )
        results = list(raw_results) if isinstance(raw_results, list) else [raw_results]

        diffusion = get_diffusion_params(req.sampling_params)
        num_steps = int(diffusion.num_inference_steps)
        sde_indices_raw = diffusion.sde_indices
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None
        # Best-effort emit: whenever the rollout ran SDE-gated steps, try to
        # land SGLang's native per-step log-probs on the segment. Whether they
        # are *used* (vs trainer-side replay) is decided downstream by the
        # algorithm's ``old_logp_source`` — not by the engine. An empty
        # ``sde_indices`` (DiffusionNFT / forward-process, num_sde_steps=0 resolves to [])
        # has no per-step log-probs to emit, so skip the block entirely —
        # matching the prior behavior for that path.
        emit_native_logprob = sde_indices is not None and len(sde_indices) > 0

        return _to_rollout_resp(
            req,
            results,
            cfg=self.cfg,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
        )

    def _resolve_initial_noise(self, req: RolloutReq) -> Optional[torch.Tensor]:
        """Decide where ``initial_noise`` comes from for this generate call.

        Precedence:
        1. Pre-shipped ``req.request_conditions['initial_latents'].latents`` →
           use verbatim (caller owns the tensor).
        2. ``cfg.init_same_noise=True`` → engine-internal Gaussian noise keyed
           on ``req.group_ids`` + ``sampling_params.diffusion.seed`` for
           per-group determinism.
        3. Otherwise → ``None`` (SGLang draws its own; matches legacy semantic
           when ``init_same_noise=False`` and no pre-shipped tensor).
        """
        # Path 1: driver-authoritative x_T — a pre-shipped initial_latents tensor
        # (img2img) OR the lightweight recipe (gids+shape) regenerated on CPU-fp32.
        # Both handled by NoiseRecipe.resolve() (returns None if neither present).
        xt = NoiseRecipe.from_rollout_req(req).resolve()
        if xt is not None:
            return xt

        # Path 2: engine-computed (same-group sharing)
        if not bool(self.cfg.init_same_noise):
            return None

        diffusion = get_diffusion_params(req.sampling_params)
        require(
            diffusion is not None and diffusion.seed is not None,
            "SGLangRolloutEngine: init_same_noise=True requires req.sampling_params diffusion seed",
        )

        batch_size = int(req.batch_size)
        latent_shape = self._latent_shape(
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

    def _latent_shape(
        self,
        *,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
    ) -> tuple:
        """Resolve the per-sample latent shape via SGLang's pipeline_config."""
        from types import SimpleNamespace

        pcfg = self._server_args.pipeline_config
        # SGLang populates ``arch_config.vae_scale_factor`` lazily in
        # ``vae_config.post_init()`` (its own comfyui_* pipelines call it
        # explicitly during load). Our standalone prepare_latent_shape call here
        # — only reached on the init_same_noise pre-noise path — can run before
        # that hook fired on this config instance, so ensure it's populated.
        # Guarded + idempotent: only acts when the field is absent and a
        # post_init() hook exists (e.g. FLUX.2's Flux2VAEConfig).
        vae_cfg = getattr(pcfg, "vae_config", None)
        arch = getattr(vae_cfg, "arch_config", None)
        if arch is not None and not hasattr(arch, "vae_scale_factor") and hasattr(vae_cfg, "post_init"):
            vae_cfg.post_init()

        batch_stub = SimpleNamespace(height=height, width=width, num_frames=num_frames)
        full_shape = pcfg.prepare_latent_shape(
            batch_stub,
            batch_size,
            num_frames,
        )
        return tuple(full_shape[1:])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        # single-stage engines ignore it (the parent ComposedRolloutEngine
        # handles the routing).
        self._call_memory_api(
            "release_memory_occupation",
            tags=["transformer", "vae", "text_encoder"],
            cpu_backup_tags=["vae", "text_encoder"],
        )
        self._is_offloaded = True
        logger.info("SGLang engine entered sleep state via release_memory_occupation().")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        self._call_memory_api(
            "resume_memory_occupation",
            tags=["transformer", "vae", "text_encoder"],
        )
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if self._generator is None:
            return False
        try:
            return bool(self._runtime["sync_scheduler_client"].ping())
        except Exception as exc:
            logger.warning("SGLang health_check ping failed: %s", exc)
            return False

    def shutdown(self) -> None:
        if self._generator is not None:
            try:
                self._generator.shutdown()
            except Exception as exc:
                logger.warning("SGLang shutdown failed: %s", exc)
        self._generator = None

    # ------------------------------------------------------------------
    # Weight sync — direct forwards (no stage_ids needed;
    # SGLang is single-stage).
    # ------------------------------------------------------------------

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
        require(bool(serialized_named_tensors), "serialized_named_tensors must be non-empty")
        self._send_scheduler_request(
            self._runtime["UpdateWeightsFromTensorReqInput"](
                serialized_named_tensors=serialized_named_tensors,
                target_modules=list(target_modules or self._target_modules),
                load_format=load_format,
                flush_cache=flush_cache,
            ),
            operation="update_weights_from_tensor",
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
        self._send_scheduler_request(
            self._runtime["InitWeightsUpdateGroupReqInput"](
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            ),
            operation="init_weights_update_group",
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
        require(bool(names), "names must be non-empty for distributed update")
        self._send_scheduler_request(
            self._runtime["UpdateWeightsFromDistributedReqInput"](
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(shape) for shape in shapes],
                group_name=str(group_name),
                target_modules=list(target_modules or self._target_modules),
                flush_cache=flush_cache,
            ),
            operation="update_weights_from_distributed",
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        self._send_scheduler_request(
            self._runtime["DestroyWeightsUpdateGroupReqInput"](
                group_name=str(group_name),
            ),
            operation="destroy_weights_update_group",
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        # Senders (nccl.py, tensor.py) ship keys in canonical wire format:
        #   ``<pipeline_prefix><module>.lora_A.weight``
        # SGLang's ``lora_layers`` dict is keyed by ``named_modules()`` of
        # ``self.modules["transformer"]`` — i.e. starting INSIDE the
        # transformer — so layer keys must be bare module names with no
        # pipeline prefix.  ``adapt_lora_for_sglang`` strips the prefix
        # (read from ``model_config.weight_sync_param_name_prefix``) and
        # injects ``.alpha`` keys so SGLang computes scale = alpha/rank
        # correctly (without them SGLang falls back to inferred_alpha =
        # inferred_rank → scale = 1.0, wrong for alpha ≠ rank).
        stripped = adapt_lora_for_sglang(
            lora_tensors,
            pipeline_prefix=self._pipeline_prefix,
            peft_config=peft_config,
        )

        request = self._runtime["SetLoraFromTensorsReq"](
            lora_nickname=str(adapter_name),
            lora_tensors=stripped,
            target="all",
            strength=1.0,
        )
        response = self._runtime["sync_scheduler_client"].forward(request)
        error = getattr(response, "error", None)
        require(error is None, f"set_lora_from_tensors failed: {error}")
        # Count the distinct LoRA layer names we registered (each layer ships
        # two tensors: ``lora_A`` + ``lora_B``).  Alpha keys are excluded.
        # For SD3.5-medium this is ~191.
        layer_names = set()
        for key in stripped:
            if key.endswith(".alpha"):
                continue  # injected by LoRA-SCALE FIX above; not a layer suffix
            base = key
            for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            layer_names.add(base)
        logger.info(
            "SGLang LoRA initialized from tensors (adapter=%s) — LoRA applied to %d layers",
            adapter_name,
            len(layer_names),
        )

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        track_prefix: str = "",
    ) -> None:
        """Bucketed-IPC weight sync is not implemented for SGLang.

        SGLang has no ``BucketedWeightReceiver`` today. Callers should use
        :meth:`update_weights_from_tensor` (SGLang-shape one-bag payload)
        or :meth:`update_weights_from_distributed` (NCCL broadcast) instead.
        """
        raise NotImplementedError(
            "SGLangRolloutEngine.update_weights_from_ipc: SGLang lacks a "
            "BucketedWeightReceiver. Use update_weights_from_tensor or "
            "update_weights_from_distributed instead."
        )

    # ------------------------------------------------------------------
    # Post-load value-correctness query (vllm-omni-shape return)
    # ------------------------------------------------------------------

    def loaded_param_checksums(
        self,
        *,
        names: List[str],
    ) -> Dict[int, List[Dict[str, str]]]:
        """Query SGLang for short SHA256 hashes of loaded parameter values.

        Returns ``{0: [{name: hex_short, ...}]}`` — a single-stage,
        single-rank-aggregated map to match the vllm-omni shape so that
        :mod:`unirl.distributed.weight_sync.checksum` helpers work
        identically against either engine.

        Note: SGLang's ``GetWeightsChecksumReqInput`` returns one map per
        target module (TP-flat aggregated server-side), so the per-rank list
        on the result always has length 1.
        """
        response = self._runtime["sync_scheduler_client"].forward(
            self._runtime["GetWeightsChecksumReqInput"](
                module_names=list(names),
            )
        )
        output = getattr(response, "output", None)
        require(
            isinstance(output, dict) and bool(output),
            f"SGLang checksum query returned invalid payload: {output!r}",
        )
        return {0: [{str(k): str(v) for k, v in output.items()}]}


__all__ = ["SGLangRolloutEngine"]
