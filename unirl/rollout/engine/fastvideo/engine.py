"""``fastvideo`` engine core — in-process FastVideo ``VideoGenerator`` rollout.

Mirrors the ``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine`` shells:
``generate`` is ``@distributed(DP_SCATTER)``, pins σ via ``ensure_req_sigmas``,
optionally chunks by ``forward_batch_size``, and packs one ``RolloutResp`` track
with a ``LatentSegment`` (trajectory + native per-step log-probs).

FastVideo remains an exact upstream snapshot. UniRL installs its RL contracts,
transition math, worker response fields, and weight hot-swap at runtime before
FastVideo spawns workers. Model-specific schedule/shape/condition behavior lives
behind an adapter rather than in this engine shell.

Validated scope:
  * Replay and native modes use the same resolved SDE window; native mode also
    returns FastVideo's transition log-probs for ``old_logp_source=rollout``.
  * x_T SSOT: FastVideo currently regenerates its own initial noise from
    ``sp.seed`` rather than consuming the driver's NoiseRecipe x_T; wiring the
    shared x_T into FastVideo byte-for-byte is a follow-up.
  * Local-mode colocate, single model_family (wan2.1) only for now.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.fastvideo.adapters import get_adapter
from unirl.rollout.engine.fastvideo.backends import FastVideoBackend
from unirl.rollout.engine.fastvideo.config import FastVideoEngineConfig, FastVideoPorts
from unirl.sde.runtime import ensure_req_sigmas
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

logger = logging.getLogger(__name__)


class FastVideoRolloutEngine(BaseRolloutEngine):
    """Rollout engine backed by a runtime-patched upstream ``VideoGenerator``."""

    _component_name = "fastvideo"

    def __init__(
        self,
        config: FastVideoEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[FastVideoPorts] = None,
    ) -> None:
        require(
            isinstance(config, FastVideoEngineConfig),
            f"FastVideoRolloutEngine requires FastVideoEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "FastVideoRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )
        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.adapter = get_adapter(config.model_family)(config, model_config, strategy=strategy)
        self.schedule_policy = self.adapter.schedule_policy()
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False
        self._backend: FastVideoBackend | None = None
        # Last checkpoint pushed by the weight sync. ``VideoGenerator`` loads the
        # PRETRAINED weights from ``model_path`` on every (re)build, so a sleep/wake
        # would silently roll back to pretrained; we re-apply this on wake. None
        # until the first ``update_weights_from_path``.
        self._last_weights_path: Optional[str] = None

        if ports is None:
            ports = FastVideoPorts.reserve()
        self._ports = ports

        self._ensure_fastvideo_importable()
        self._build_backend()

        logger.info(
            "Initialized fastvideo engine (rank=%s, native_logprob=%s, master_port=%s)",
            rank,
            config.native_logprob,
            self._ports.master_port,
        )

    # ------------------------------------------------------------------ #
    # FastVideo import + VideoGenerator boot (ported from DiffusionRL)
    # ------------------------------------------------------------------ #
    def _ensure_fastvideo_importable(self) -> None:
        try:
            importlib.import_module("fastvideo")
            return
        except ModuleNotFoundError:
            pass
        path = self.cfg.fastvideo_path or os.getenv("FASTVIDEO_PATH", "")
        require(bool(path), "fastvideo not importable; set cfg.fastvideo_path or $FASTVIDEO_PATH")
        if path not in sys.path:
            sys.path.insert(0, str(Path(path).expanduser()))
        importlib.import_module("fastvideo")

    def _build_backend(self) -> None:
        ekw = dict(self.cfg.engine_kwargs or {})
        fv_kwargs: Dict[str, Any] = {
            "model_path": self.model_config.pretrained_model_ckpt_path,
            "num_gpus": int(self.cfg.num_gpus),
            "tp_size": int(self.cfg.tp_size),
            "sp_size": int(self.cfg.sp_size),
            "inference_mode": True,
            # Force decoded pixels as a [B, C, T, H, W] tensor (not PIL/latent)
            # so execute_forward populates batch.output for the reward path.
            "output_type": "pt",
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "master_port": int(self._ports.master_port),
        }
        fv_kwargs.update(ekw)
        self._backend = FastVideoBackend.boot(
            fv_kwargs,
            configure_args=self.adapter.align_runtime_args,
            reserve_port=self._reserve_master_port,
        )

    def _reserve_master_port(self) -> int:
        self._ports = FastVideoPorts.reserve()
        return int(self._ports.master_port)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        require(
            int(req.batch_size) > 0,
            "FastVideoRolloutEngine.generate requires a non-empty req (batch_size > 0)",
        )
        # σ SSOT: pin once on the full batch (shared field, survives req.slice).
        ensure_req_sigmas(req, self.schedule_policy)

        # ``forward_batch_size`` here is a CHUNKING cadence, NOT a GPU batch size:
        # ``_drive_fastvideo`` runs FastVideo one video at a time (per-sample seeds
        # preclude a batched forward), so peak GPU activation is fixed at one video
        # regardless of ``fbs``. What ``fbs`` bounds is how many per-sample outputs
        # (trajectory/decoded tensors, already on CPU) accumulate before a concat +
        # ``empty_cache``. Leave it None to run the whole shard in one go.
        fbs = self.cfg.forward_batch_size
        bs = int(req.batch_size)
        if fbs is None or bs <= fbs:
            return self._generate_batch(req)

        outputs: List[RolloutResp] = []
        for start in range(0, bs, fbs):
            end = min(start + fbs, bs)
            outputs.append(self._generate_batch(req.slice(start, end)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return RolloutResp.concat(outputs)

    def _generate_batch(self, req: RolloutReq) -> RolloutResp:
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None and isinstance(text_primitive, Texts),
            f"fastvideo engine requires req.primitives['text']: Texts; "
            f"got {type(text_primitive).__name__ if text_primitive is not None else 'None'}",
        )
        prompts = list(text_primitive.texts)
        require(
            len(prompts) == int(req.batch_size),
            f"fastvideo engine expects req.primitives['text'] of len batch_size; "
            f"got {len(prompts)} vs {int(req.batch_size)}",
        )
        params = req.sampling_params.get("diffusion")
        require(
            params is not None,
            "fastvideo engine requires req.sampling_params['diffusion']",
        )
        seeds = self.adapter.per_sample_seeds(req, params)
        raw = self._drive_fastvideo(prompts, params, req.sigmas, seeds)
        return self.adapter.build_response(req, params, raw)

    def _drive_fastvideo(
        self,
        prompts: List[str],
        params: Any,
        sigmas: torch.Tensor,
        seeds: List[int],
    ) -> List[Dict[str, Any]]:
        require(
            len(seeds) == len(prompts),
            f"fastvideo engine expects one seed per prompt; got {len(seeds)} vs {len(prompts)}",
        )
        require(self._backend is not None, "FastVideo backend is not initialized")
        outputs: List[Dict[str, Any]] = []
        for prompt, seed in zip(prompts, seeds):
            batch = self.adapter.build_forward_batch(
                prompt=prompt,
                seed=seed,
                params=params,
                sigmas=sigmas,
                fastvideo_args=self._backend.fastvideo_args,
            )
            output = self._backend.execute(batch)
            outputs.append(self.adapter.collect_output(output))
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        if self._is_offloaded:
            return
        if self._backend is not None:
            try:
                self._backend.sleep()
            except Exception as exc:  # noqa: BLE001
                logger.warning("fastvideo sleep/shutdown warning: %s", exc)
        self._is_offloaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        require(self._backend is not None, "FastVideo backend is not initialized")
        self._backend.wake(reserve_port=self._reserve_master_port)
        self._is_offloaded = False
        # ``from_fastvideo_args`` reloads the PRETRAINED transformer from
        # ``model_path``; without this the engine would sample under pretrained
        # weights on every wake that isn't immediately followed by a weight sync
        # (i.e. any ``weight_sync_interval > 1`` step). Re-apply the last synced
        # checkpoint so wake is weight-preserving, matching the other engines'
        # sleep/wake contract (sglang resume_memory keeps weights resident).
        if self._last_weights_path is not None:
            self._backend.update_weights_from_path(self._last_weights_path)
            logger.info("fastvideo wake_up: re-applied synced weights from %s", self._last_weights_path)

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def onload_weights(self, *, track_prefix: str = "") -> None:
        del track_prefix
        self.wake_up()

    def shutdown(self) -> None:
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None

    # ------------------------------------------------------------------ #
    # Weight sync — checkpoint_path (full-param hot-swap). Reached per worker
    # via the local sibling call from CheckpointWeightSync (not @distributed).
    # ------------------------------------------------------------------ #
    def update_weights_from_path(self, checkpoint_path: str, *, track_prefix: str = "") -> None:
        del track_prefix
        require(bool(checkpoint_path), "update_weights_from_path requires a non-empty path")
        require(self._backend is not None and not self._is_offloaded, "fastvideo engine is offloaded/not initialized")
        self._backend.update_weights_from_path(checkpoint_path)
        # Remember it so ``wake_up`` can re-apply after a rebuild (see wake_up).
        self._last_weights_path = checkpoint_path
        logger.info("fastvideo transformer weights updated from %s", checkpoint_path)


__all__ = ["FastVideoRolloutEngine"]
