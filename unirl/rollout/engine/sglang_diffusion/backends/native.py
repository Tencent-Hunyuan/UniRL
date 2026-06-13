"""The native ``Backend`` impl — ``DiffGenerator`` + the ZMQ scheduler client.

The ONLY module that imports the SGLang runtime or does I/O. Covers both local
mode (``from_pretrained`` spawns the worker in-process) and the existing remote
mode (``local_mode=False`` connects the scheduler client to an externally launched
server's ``scheduler_port``). Weight-sync ``*ReqInput`` io_struct types stay
*inside* this module.

Because the SGLang import is lazy (only in :meth:`SGLangBackend.boot` and the
verbs), the module imports on CPU — the rest of the package is exercisable
without a GPU.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence

from unirl.rollout.engine.sglang_diffusion.backends.base import RawResult

logger = logging.getLogger(__name__)


def _import_sglang_runtime() -> Dict[str, Any]:
    """Install the UniRL patch suite, then import the runtime types. Once per process.

    Stock upstream sglang (>= 0.5.12.post1) replaced the fork: the RL additions
    (weight-sync verbs, in-memory LoRA, sleep/wake, rollout IO fields) are
    re-hosted as in-process patches under ``unirl.rollout.engine.sglang_diffusion._patches``
    and MUST be installed before any scheduler/worker spawns — ``hijack()`` also
    wraps the mp process target so spawned children re-install (mirrors the v1
    engine, ``sglang/engine.py``).

    Import sourcing mirrors v1: types that exist upstream come from upstream;
    fork-only req types come from ``_patches.io_struct`` / ``_patches.lora_req``
    (``patch_scheduler`` registers handlers keyed on those exact classes).
    """
    from unirl.rollout.engine.sglang_diffusion._patches import SglangDiffusionHijack

    SglangDiffusionHijack.hijack()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        GetWeightsChecksumReqInput,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    from unirl.rollout.engine.sglang_diffusion._patches.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from unirl.rollout.engine.sglang_diffusion._patches.lora_req import SetLoraFromTensorsReq

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


class _RawResultView:
    """Flat ``RawResult`` view over upstream's ``GenerationResult``.

    Stock upstream packs the rollout trajectory + native log-probs into the
    nested ``rollout_trajectory_data`` (RolloutTrajectoryData) instead of the
    fork's flat ``trajectory_latents`` / ``trajectory_timesteps`` /
    ``trajectory_log_probs``. This view flattens that path (rtd-only, tolerant
    of missing levels — mirrors the v1 ``response.py`` accessors; GRPO uses the
    ``dit_trajectory`` latents so the trajectory stays aligned with
    ``rollout_log_probs``) and passes every other wire field through, keeping
    adapters/utils on the unchanged ``RawResult`` protocol.
    """

    __slots__ = ("_result",)

    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def trajectory_latents(self) -> Any:
        rtd = getattr(self._result, "rollout_trajectory_data", None)
        return getattr(getattr(rtd, "dit_trajectory", None), "latents", None)

    @property
    def trajectory_timesteps(self) -> Any:
        rtd = getattr(self._result, "rollout_trajectory_data", None)
        return getattr(getattr(rtd, "dit_trajectory", None), "timesteps", None)

    @property
    def trajectory_log_probs(self) -> Any:
        rtd = getattr(self._result, "rollout_trajectory_data", None)
        return getattr(rtd, "rollout_log_probs", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)


class SGLangBackend:
    """The native ``Backend`` impl over ``DiffGenerator`` + the scheduler client."""

    def __init__(self, generator: Any, runtime: Dict[str, Any], server_args: Any) -> None:
        self._gen = generator
        self._rt = runtime
        self._server_args = server_args

    # ------------------------------------------------------------------ #
    # Boot — the only place from_pretrained / the import live
    # ------------------------------------------------------------------ #

    @classmethod
    def boot(
        cls,
        server_intent: Dict[str, Any],
        *,
        local_mode: bool,
    ) -> "SGLangBackend":
        """Filter intent against ServerArgs, build the generator, return the backend.

        ``server_intent`` is the model/parallelism/port intent dict from
        ``config.server_intent`` (reserved ports already overlaid — including
        ``master_port``, the spawned workers' dist init, so no ``MASTER_PORT``
        env manipulation happens here). We filter it to real ServerArgs fields
        here (the only place that knows them), then spawn.
        """
        rt = _import_sglang_runtime()
        allowed = {f.name for f in dataclasses.fields(rt["ServerArgs"])}
        server_kwargs = {k: v for k, v in server_intent.items() if k in allowed}

        disable_autocast = server_kwargs.get("disable_autocast")
        server_args = rt["ServerArgs"].from_kwargs(**server_kwargs)
        if disable_autocast is not None:
            server_args.disable_autocast = disable_autocast

        generator = rt["DiffGenerator"].from_pretrained(
            server_args=server_args,
            local_mode=bool(local_mode),
        )
        return cls(generator, rt, server_args)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(self, sampling_kwargs: Dict[str, Any]) -> List[RawResult]:
        raw = self._gen.generate(sampling_params_kwargs=sampling_kwargs)
        if raw is None:
            raise RuntimeError(
                "SGLang generator returned None — full-batch failure (see DiffGenerator.generate docstring)."
            )
        results = list(raw) if isinstance(raw, list) else [raw]
        return [_RawResultView(result) for result in results]

    def prepare_latent_shape(self, *, height: int, width: int, num_frames: int, batch_size: int) -> tuple:
        """Per-sample latent shape via SGLang's ``pipeline_config`` (no RL types)."""
        from types import SimpleNamespace

        pcfg = self._server_args.pipeline_config
        # SGLang populates arch_config.vae_scale_factor lazily in
        # vae_config.post_init(); our standalone call here (init_same_noise path)
        # can run before that hook fired — populate it idempotently.
        vae_cfg = getattr(pcfg, "vae_config", None)
        arch = getattr(vae_cfg, "arch_config", None)
        if arch is not None and not hasattr(arch, "vae_scale_factor") and hasattr(vae_cfg, "post_init"):
            vae_cfg.post_init()

        batch_stub = SimpleNamespace(height=height, width=width, num_frames=num_frames)
        full_shape = pcfg.prepare_latent_shape(batch_stub, batch_size, num_frames)
        return tuple(full_shape[1:])

    # ------------------------------------------------------------------ #
    # Memory / lifecycle / health
    # ------------------------------------------------------------------ #

    def release_memory(self, *, tags: Sequence[str], cpu_backup_tags: Optional[Sequence[str]] = None) -> None:
        # Stock upstream DiffGenerator has no memory-occupation methods (the fork
        # added them); route through the scheduler client to the handlers that
        # ``patch_scheduler`` installs, keyed on the ``_patches`` req types
        # (mirrors the v1 engine's ``_call_memory_api``).
        self._forward(
            self._rt["ReleaseMemoryOccupationReqInput"](
                tags=list(tags),
                cpu_backup_tags=(list(cpu_backup_tags) if cpu_backup_tags is not None else None),
            ),
            op="release_memory_occupation",
        )

    def resume_memory(self, *, tags: Sequence[str]) -> None:
        self._forward(
            self._rt["ResumeMemoryOccupationReqInput"](tags=list(tags)),
            op="resume_memory_occupation",
        )

    def shutdown(self) -> None:
        if self._gen is not None:
            try:
                self._gen.shutdown()
            except Exception as exc:  # noqa: BLE001 — best-effort teardown
                logger.warning("SGLang shutdown failed: %s", exc)
            self._gen = None

    def ping(self) -> bool:
        if self._gen is None:
            return False
        try:
            return bool(self._rt["sync_scheduler_client"].ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("SGLang health_check ping failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Weight-sync verbs (io_struct types stay here; no RL types cross)
    # ------------------------------------------------------------------ #

    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None:
        self._forward(
            self._rt["UpdateWeightsFromTensorReqInput"](
                serialized_named_tensors=serialized_named_tensors,
                target_modules=list(target_modules),
                load_format=load_format,
                flush_cache=flush_cache,
            ),
            op="update_weights_from_tensor",
        )

    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        self._forward(
            self._rt["InitWeightsUpdateGroupReqInput"](
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            ),
            op="init_weights_update_group",
        )

    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: List[str],
        flush_cache: bool,
    ) -> None:
        self._forward(
            self._rt["UpdateWeightsFromDistributedReqInput"](
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(shape) for shape in shapes],
                group_name=str(group_name),
                target_modules=list(target_modules),
                flush_cache=flush_cache,
            ),
            op="update_weights_from_distributed",
        )

    def destroy_weights_group(self, *, group_name: str) -> None:
        self._forward(
            self._rt["DestroyWeightsUpdateGroupReqInput"](group_name=str(group_name)),
            op="destroy_weights_update_group",
        )

    def set_lora(
        self,
        *,
        lora_nickname: str,
        lora_tensors: Dict[str, Any],
        target: str = "all",
        strength: float = 1.0,
    ) -> None:
        request = self._rt["SetLoraFromTensorsReq"](
            lora_nickname=str(lora_nickname),
            lora_tensors=lora_tensors,
            target=target,
            strength=strength,
        )
        response = self._rt["sync_scheduler_client"].forward(request)
        error = getattr(response, "error", None)
        if error is not None:
            raise RuntimeError(f"set_lora_from_tensors failed: {error}")

    def weights_checksum(self, *, module_names: List[str]) -> dict:
        response = self._rt["sync_scheduler_client"].forward(
            self._rt["GetWeightsChecksumReqInput"](module_names=list(module_names))
        )
        output = getattr(response, "output", None)
        if not (isinstance(output, dict) and output):
            raise RuntimeError(f"SGLang checksum query returned invalid payload: {output!r}")
        return output

    # ------------------------------------------------------------------ #
    # Scheduler request plumbing
    # ------------------------------------------------------------------ #

    def _forward(self, request: Any, *, op: str) -> Any:
        response = self._rt["sync_scheduler_client"].forward(request)
        success, message = self._extract_update_status(response, op=op)
        if not success:
            raise RuntimeError(f"{op} failed: {message}")
        return response

    @staticmethod
    def _extract_update_status(response: Any, *, op: str) -> tuple:
        output = getattr(response, "output", None)
        if not isinstance(output, dict):
            raise RuntimeError(f"Invalid SGLang response for {op}: {response}")
        return bool(output.get("success", False)), str(output.get("message", "Unknown status"))


__all__ = ["SGLangBackend"]
