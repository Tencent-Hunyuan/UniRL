"""``fastvideo`` engine core — in-process FastVideo ``VideoGenerator`` rollout.

Mirrors the ``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine`` shells:
``generate`` is ``@distributed(DP_SCATTER)``, pins σ on the frontier generation
``Part``, optionally chunks that frontier by ``forward_batch_size``, and fills it
with a ``LatentSegment`` (trajectory + native per-step log-probs) and decoded
``Videos``.

The FastVideo-driving logic (VideoGenerator boot, PR #1222 ``ForwardBatch.RLData``
native-logprob path, transformer hot-swap, sleep/wake) is ported from the proven
DiffusionRL FastVideo engine; only the typed boundary (``Sample``/``Part``/
``LatentSegment``, σ SSOT) is new.

Validated scope:
  * Replay and native modes use the same resolved SDE window; native mode also
    returns FastVideo's transition log-probs for ``old_logp_source=rollout``.
  * x_T SSOT: FastVideo currently regenerates its own initial noise from
    ``sp.seed`` rather than consuming the driver's NoiseRecipe x_T. Because that
    seed also drives SDE noise, ``init_same_noise=True`` shares the whole random
    stream, not only x_T; separate x_T/SDE streams are a follow-up.
  * Local-mode colocate, single model_family (wan2.1) only for now.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.fastvideo.config import FastVideoEngineConfig, FastVideoPorts
from unirl.rollout.engine.synchronous import SyncRolloutEngine
from unirl.sde.noise import _derive_group_seed
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.conditions import TextEmbedCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_video_segment

logger = logging.getLogger(__name__)


def _resolve_sde_window(raw_indices: Any, num_steps: int) -> tuple[Optional[List[int]], List[int]]:
    """Return the FastVideo wire value and canonical segment indices.

    ``None`` keeps FastVideo's legacy "all steps are SDE" spelling. An explicit
    empty iterable remains empty (the framework's deterministic forward-process
    contract). The segment always gets the resolved concrete list.
    """
    if raw_indices is None:
        return None, list(range(int(num_steps)))
    selected = sorted({int(i) for i in raw_indices})
    bad = [i for i in selected if i < 0 or i >= int(num_steps)]
    if bad:
        raise ValueError(f"FastVideo SDE indices out of range for num_steps={num_steps}: {bad}")
    return selected, selected


class FastVideoRolloutEngine(SyncRolloutEngine):
    """Rollout engine backed by FastVideo ``VideoGenerator`` (RL fork, PR #1222)."""

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
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False
        self._generator: Any = None
        self._fastvideo_args: Any = None
        self._last_weights_path: Optional[str] = None

        if ports is None:
            ports = FastVideoPorts.reserve()
        self._ports = ports

        self._ensure_fastvideo_importable()
        self._build_generator()

        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            model_config.pretrained_model_ckpt_path,
            shift=float(model_config.shift),
            require_dynamic=bool(getattr(model_config, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(model_config, "dynamic_shift_overrides", None),
        )
        logger.info(
            "Initialized fastvideo engine (rank=%s, native_logprob=%s, master_port=%s)",
            rank,
            config.native_logprob,
            ports.master_port,
        )

        self._weight_version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False

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

    def _build_generator(self) -> None:
        from fastvideo import VideoGenerator
        from fastvideo.fastvideo_args import FastVideoArgs

        ekw = dict(self.cfg.engine_kwargs or {})
        fv_kwargs: Dict[str, Any] = {
            "model_path": self.model_config.pretrained_model_ckpt_path,
            "num_gpus": int(self.cfg.num_gpus),
            "tp_size": int(self.cfg.tp_size),
            "sp_size": int(self.cfg.sp_size),
            "inference_mode": True,
            "output_type": "pt",
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "master_port": int(self._ports.master_port),
        }
        fv_kwargs.update(ekw)
        self._fastvideo_args = FastVideoArgs.from_kwargs(**fv_kwargs)
        target_shift = float(self.model_config.shift)
        pc = self._fastvideo_args.pipeline_config
        if getattr(pc, "flow_shift", None) != target_shift:
            logger.info(
                "fastvideo engine: pipeline_config.flow_shift %s -> %s (model_config.shift)",
                getattr(pc, "flow_shift", None),
                target_shift,
            )
            pc.flow_shift = target_shift
        max_port_attempts = 5
        for attempt in range(1, max_port_attempts + 1):
            try:
                self._generator = VideoGenerator.from_fastvideo_args(self._fastvideo_args)
                break
            except Exception as exc:  # noqa: BLE001
                port_in_use = "EADDRINUSE" in str(exc) or "address already in use" in str(exc).lower()
                if not port_in_use or attempt == max_port_attempts:
                    raise
                self._ports = FastVideoPorts.reserve()
                self._fastvideo_args.master_port = int(self._ports.master_port)
                logger.warning(
                    "fastvideo init: master port busy (attempt %d/%d); retrying with %s",
                    attempt,
                    max_port_attempts,
                    self._ports.master_port,
                )

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        return self._generate_locked(sample)

    def _generate_locked(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("FastVideoRolloutEngine.generate called after shutdown")
            return self._stamp_weight_version(self._generate_core(sample))

    def _generate_core(self, sample: Sample) -> Sample:
        """Generate and fill the frontier diffusion Part."""
        require(
            not self._is_offloaded and self._generator is not None,
            "FastVideoRolloutEngine.generate: engine is offloaded (wake_up first).",
        )
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        require(
            int(gen.batch_size) > 0,
            "FastVideoRolloutEngine.generate requires a non-empty Sample (gen batch_size > 0)",
        )
        ensure_sample_sigmas(sample, self.schedule_policy)

        fbs = self.cfg.forward_batch_size
        bs = int(gen.batch_size)
        if fbs is None or bs <= fbs:
            return self._generate_batch(sample)

        gen_chunks: List[Part] = []
        for start in range(0, bs, fbs):
            end = min(start + fbs, bs)
            chunk = self._generate_batch(sample.replace_frontier(gen.slice(start, end)))
            gen_chunks.append(chunk.frontier_gen_part(DiffusionSamplingParams))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return sample.replace_frontier(Part.concat(gen_chunks))

    def _generate_batch(self, sample: Sample) -> Sample:
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        turns = sample.text_conditioning()
        require(
            len(turns) == 1 and isinstance(turns[0].content, Texts),
            "fastvideo engine requires exactly one frontier-aligned text conditioning turn; "
            f"got {[type(turn.content).__name__ for turn in turns]}",
        )
        text_primitive = turns[0].content
        prompts = list(text_primitive.texts)
        require(
            len(prompts) == int(gen.batch_size),
            "fastvideo engine expects frontier-aligned text of len gen.batch_size; "
            f"got {len(prompts)} vs {int(gen.batch_size)}",
        )
        params = gen.sampling_params
        require(
            isinstance(params, DiffusionSamplingParams),
            "fastvideo engine requires DiffusionSamplingParams on the frontier gen Part",
        )
        require(params.sigmas is not None, "fastvideo engine requires engine-pinned diffusion.sigmas")
        seeds = self._per_sample_seeds(sample, params)
        raw = self._drive_fastvideo(prompts, params, params.sigmas, seeds)
        return self._build_response(sample, params, raw)

    def _per_sample_seeds(self, sample: Sample, params: DiffusionSamplingParams) -> List[int]:
        """Per-sample seeds so sibling samples of one prompt diverge.

        Without this every sample of a prompt shared ``params.seed`` → identical
        video → identical reward → zero GRPO advantage → zero loss/grad. We key
        the seed the same way the driver keys x_T (``_derive_group_seed``):
        per-sample ids when ``init_same_noise`` is false (siblings differ),
        per-group ids when true (siblings share). Prefer the driver's
        normalized ``NoiseRecipe`` keys (carry rollout id + same/diff policy);
        fall back to frontier sample/group ids, then to the flat seed.
        NOTE: FastVideo uses this one seed for x_T and per-step SDE noise. Thus
        ``init_same_noise=True`` shares the entire stream across siblings rather
        than only x_T. Full driver-authoritative x_T plus a sample-specific SDE
        stream is a separate follow-up.
        """
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        bs = int(gen.batch_size)
        base_seed = int(params.seed) if params.seed is not None else 0
        keys = NoiseRecipe.from_sample(sample).noise_group_ids
        if not (isinstance(keys, (list, tuple)) and len(keys) == bs):
            same = bool(getattr(params, "init_same_noise", False))
            keys = list(gen.group_ids) if same else list(gen.sample_ids)
        if not (isinstance(keys, (list, tuple)) and len(keys) == bs):
            return [base_seed] * bs
        return [_derive_group_seed(base_seed, str(k)) for k in keys]

    def _drive_fastvideo(
        self,
        prompts: List[str],
        params: Any,
        sigmas: torch.Tensor,
        seeds: List[int],
    ) -> Dict[str, Any]:
        """PR #1222 native-logprob path via executor.execute_forward + RLData.

        Returns dict(trajectory=[B,T+1,...], log_probs=[B,T], decoded=[B,...]).
        """
        from copy import deepcopy

        from fastvideo.configs.sample.base import SamplingParam
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.utils import shallow_asdict

        sp = SamplingParam()
        sp.height = int(params.height)
        sp.width = int(params.width)
        sp.num_frames = int(params.num_frames)
        sp.num_inference_steps = int(params.num_inference_steps)
        sp.guidance_scale = float(params.guidance_scale)
        sp.seed = int(params.seed) if params.seed is not None else 0
        sp.num_videos_per_prompt = 1
        sp.save_video = False
        sp.return_frames = False
        sp.return_trajectory_latents = False
        sp.return_trajectory_decoded = False
        _f = float(getattr(self._fastvideo_args.pipeline_config, "flow_shift", self.model_config.shift))
        _s = sigmas.detach().cpu().double()
        _g = _s / (_f - _s * (_f - 1.0))
        sp.sigmas = [float(x) for x in _g.tolist()[:-1]]

        sde_step_indices, _ = _resolve_sde_window(
            getattr(params, "sde_indices", None),
            int(params.num_inference_steps),
        )

        all_log_probs: List[torch.Tensor] = []
        all_traj: List[torch.Tensor] = []
        all_decoded: List[torch.Tensor] = []
        all_text_embeds: List[torch.Tensor] = []
        all_text_masks: List[Optional[torch.Tensor]] = []
        all_neg_embeds: List[torch.Tensor] = []
        all_neg_masks: List[Optional[torch.Tensor]] = []

        require(
            len(seeds) == len(prompts),
            f"fastvideo engine expects one seed per prompt; got {len(seeds)} vs {len(prompts)}",
        )
        for prompt, seed in zip(prompts, seeds):
            one = deepcopy(sp)
            one.prompt = prompt
            one.seed = int(seed)
            latents_size = [(one.num_frames - 1) // 4 + 1, one.height // 8, one.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
            sp_dict = shallow_asdict(one)
            sp_dict.pop("eta", None)
            batch = ForwardBatch(
                **sp_dict,
                eta=float(params.eta),
                n_tokens=n_tokens,
                VSA_sparsity=self._fastvideo_args.VSA_sparsity,
                rl_data=ForwardBatch.RLData(
                    enabled=True,
                    collect_log_probs=bool(self.cfg.native_logprob),
                    store_trajectory=True,
                    keep_trajectory_on_cpu=True,
                    sde_step_indices=sde_step_indices,
                    sde_type=str(getattr(self.strategy, "canonical_name", "flow")),
                ),
            )
            out = self._generator.executor.execute_forward(batch, self._fastvideo_args)
            rl = out.rl_data
            traj = rl.trajectory_latents if rl is not None else None
            if traj is None:
                traj = out.trajectory_latents
            require(torch.is_tensor(traj), "FastVideo returned no trajectory tensor")
            if traj.dim() == 5:
                traj = traj.unsqueeze(0)
            all_traj.append(traj.detach().cpu())
            dec = getattr(out, "output", None)
            require(torch.is_tensor(dec), "FastVideo returned no decoded output (batch.output)")
            if dec.dim() == 4:
                dec = dec.unsqueeze(0)
            all_decoded.append(dec.detach().cpu().float())

            pe = out.prompt_embeds
            require(
                isinstance(pe, (list, tuple)) and len(pe) > 0 and torch.is_tensor(pe[0]),
                "FastVideo returned no prompt_embeds for text conditioning",
            )
            te = pe[0]
            if te.dim() == 2:
                te = te.unsqueeze(0)
            all_text_embeds.append(te.detach().cpu().float())
            pm = out.prompt_attention_mask
            tm = pm[0] if isinstance(pm, (list, tuple)) and len(pm) > 0 and torch.is_tensor(pm[0]) else None
            all_text_masks.append(tm.detach().cpu() if tm is not None else None)

            ne = out.negative_prompt_embeds
            if isinstance(ne, (list, tuple)) and len(ne) > 0 and torch.is_tensor(ne[0]):
                nte = ne[0]
                if nte.dim() == 2:
                    nte = nte.unsqueeze(0)
                all_neg_embeds.append(nte.detach().cpu().float())
                nm = out.negative_attention_mask
                ntm = nm[0] if isinstance(nm, (list, tuple)) and len(nm) > 0 and torch.is_tensor(nm[0]) else None
                all_neg_masks.append(ntm.detach().cpu() if ntm is not None else None)

            if self.cfg.native_logprob and sde_step_indices != []:
                lp = rl.log_probs if rl is not None else None
                require(torch.is_tensor(lp), "FastVideo native rollout returned no log_probs")
                all_log_probs.append(lp.detach().cpu())

            del out, rl, traj, dec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "trajectory": torch.cat(all_traj, dim=0),
            "decoded": torch.cat(all_decoded, dim=0),
            "log_probs": torch.cat(all_log_probs, dim=0) if all_log_probs else None,
            "text_embeds": all_text_embeds,
            "text_masks": all_text_masks,
            "neg_embeds": all_neg_embeds,
            "neg_masks": all_neg_masks,
        }

    def _build_response(
        self,
        sample: Sample,
        params: DiffusionSamplingParams,
        raw: Dict[str, Any],
    ) -> Sample:
        segment = self._build_segment(params, raw)
        decoded = self._build_decoded(raw)
        conditions = self._build_conditions(raw)
        return sample.with_filled_frontier(
            segment=segment,
            primitives={"video": decoded},
            conditions=conditions,
        )

    def _build_conditions(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the WAN21 ``conditions`` dict the trainer replays against.

        Packs the captured FastVideo prompt embeddings into ``TextEmbedCondition``s
        (``text`` + optional CFG ``negative_text``), padding variable token lengths
        with zeros via ``TextEmbedCondition.concat`` (WAN's zeroed-pad convention).
        """
        text_embeds: List[torch.Tensor] = raw.get("text_embeds") or []
        require(len(text_embeds) > 0, "fastvideo engine produced no text embeddings")
        text_masks: List[Optional[torch.Tensor]] = raw.get("text_masks") or [None] * len(text_embeds)

        text = TextEmbedCondition.concat(
            [
                TextEmbedCondition(embeds=text_embeds[i], pooled=None, attn_mask=text_masks[i])
                for i in range(len(text_embeds))
            ]
        )
        conditions: Dict[str, Any] = {"text": text}

        neg_embeds: List[torch.Tensor] = raw.get("neg_embeds") or []
        if len(neg_embeds) == len(text_embeds) and len(neg_embeds) > 0:
            neg_masks: List[Optional[torch.Tensor]] = raw.get("neg_masks") or [None] * len(neg_embeds)
            conditions["negative_text"] = TextEmbedCondition.concat(
                [
                    TextEmbedCondition(embeds=neg_embeds[i], pooled=None, attn_mask=neg_masks[i])
                    for i in range(len(neg_embeds))
                ]
            )
        return conditions

    def _build_decoded(self, raw: Dict[str, Any]) -> Videos:
        """Pack FastVideo's decoded output [B, C, T, H, W] into a ``Videos``.

        Mirrors WAN21VAEDecodeStage: permute each sample (C, T, H, W) →
        (T, C, H, W) so Video.frames matches the canonical [T, C, H, W]
        contract the reward path (video_pickscore) consumes.
        """
        frames = raw["decoded"]
        require(
            torch.is_tensor(frames) and frames.dim() == 5,
            f"fastvideo decoded must be [B, C, T, H, W]; got "
            f"{tuple(frames.shape) if torch.is_tensor(frames) else type(frames).__name__}",
        )
        videos = [Video(frames=frames[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(frames.shape[0]))]
        return Videos.from_list(videos)

    def _build_segment(self, params: DiffusionSamplingParams, raw: Dict[str, Any]):
        traj = raw["trajectory"]
        device = traj.device
        T = int(traj.shape[1]) - 1
        indices = torch.arange(traj.shape[1], dtype=torch.long, device=device)

        _, sde_set = _resolve_sde_window(getattr(params, "sde_indices", None), T)
        sde_indices = torch.tensor(sde_set, dtype=torch.long, device=device) if sde_set else None

        sde_logp = None
        lp = raw.get("log_probs")
        if lp is not None:
            if lp.shape[1] == T and len(sde_set) < T:
                cols = [s for s in sde_set if 0 <= s < lp.shape[1]]
                sde_logp = lp[:, cols].contiguous()
            else:
                sde_logp = lp.contiguous()

        return make_video_segment(
            latents=traj,
            sigmas=params.sigmas,
            indices=indices,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        with self._generate_lock:
            if self._shutdown_requested or self._is_offloaded:
                return
            if self._generator is not None:
                try:
                    self._generator.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fastvideo sleep/shutdown warning: %s", exc)
                self._generator = None
            self._is_offloaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("FastVideoRolloutEngine.wake_up called after shutdown")
            self._wake_up_locked()

    def _wake_up_locked(self) -> None:
        """Rebuild and restore the generator while ``_generate_lock`` is held."""
        if not self._is_offloaded:
            return
        from fastvideo import VideoGenerator

        max_port_attempts = 5
        for attempt in range(1, max_port_attempts + 1):
            self._ports = FastVideoPorts.reserve()
            self._fastvideo_args.master_port = int(self._ports.master_port)
            try:
                self._generator = VideoGenerator.from_fastvideo_args(self._fastvideo_args)
                break
            except Exception as exc:  # noqa: BLE001
                port_in_use = "EADDRINUSE" in str(exc) or "address already in use" in str(exc).lower()
                if not port_in_use or attempt == max_port_attempts:
                    raise
                logger.warning(
                    "fastvideo wake_up: master_port=%s busy (attempt %d/%d); retrying",
                    self._ports.master_port,
                    attempt,
                    max_port_attempts,
                )
        try:
            if self._last_weights_path is not None:
                self._generator.update_transformer_weights_from_path(self._last_weights_path)
                logger.info("fastvideo wake_up: re-applied synced weights from %s", self._last_weights_path)
        except Exception:
            try:
                if self._generator is not None:
                    self._generator.shutdown()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("fastvideo wake_up cleanup after restore failure: %s", cleanup_exc)
            self._generator = None
            self._is_offloaded = True
            raise
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def onload_weights(self, *, track_prefix: str = "") -> None:
        del track_prefix
        self.wake_up()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._generate_lock:
                self._shutdown_requested = True
            with self._generate_lock:
                if self._generator is not None:
                    self._generator.shutdown()
                    self._generator = None
                self._is_offloaded = True
            self._shutdown_complete = True

    def update_weights_from_path(self, checkpoint_path: str, *, track_prefix: str = "") -> None:
        del track_prefix
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("FastVideoRolloutEngine.update_weights_from_path called after shutdown")
            require(bool(checkpoint_path), "update_weights_from_path requires a non-empty path")
            require(self._generator is not None, "fastvideo engine is offloaded/not initialized")
            self._generator.update_transformer_weights_from_path(checkpoint_path)
            self._last_weights_path = checkpoint_path
            self._weight_version += 1
            logger.info("fastvideo transformer weights updated from %s", checkpoint_path)


__all__ = ["FastVideoRolloutEngine"]
