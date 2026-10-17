"""``fastvideo`` engine core — in-process FastVideo ``VideoGenerator`` rollout."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.fastvideo._patches import FastVideoUniPCPlan, patch_fastvideo
from unirl.rollout.engine.fastvideo.config import FastVideoEngineConfig, FastVideoPorts
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.sde.noise import _derive_group_seed
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.sde.unipc import UniPCSpec
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_video_segment

logger = logging.getLogger(__name__)


def _load_checkpoint_json(ckpt_path: str, filename: str, *, context: str, target: str) -> tuple[Path, dict]:
    """Resolve and parse ``filename`` from a local checkpoint directory or Hugging Face repo."""
    checkpoint = Path(ckpt_path).expanduser()
    if checkpoint.is_dir():
        path = checkpoint / filename
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(repo_id=ckpt_path, filename=filename))
        except Exception as exc:
            raise RuntimeError(
                f"{context} cannot resolve {ckpt_path!r}/{filename} as either a local "
                "diffusers-layout checkpoint or a Hugging Face model repo."
            ) from exc
    try:
        with path.open("r", encoding="utf-8") as f:
            return path, json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} cannot verify {target} without {path}") from exc


def _verify_checkpoint_unipc_spec(ckpt_path: str, spec: UniPCSpec) -> None:
    """Fail closed unless the checkpoint's scheduler_config.json declares the model-owned UniPC solver spec."""
    path, declared_cfg = _load_checkpoint_json(
        ckpt_path,
        "scheduler/scheduler_config.json",
        context="FastVideo canonical UniPC",
        target="model_config.unipc_*",
    )
    class_name = declared_cfg.get("_class_name", "")
    if "UniPC" not in str(class_name):
        raise RuntimeError(
            f"Checkpoint scheduler {path} declares _class_name={class_name!r}; the canonical "
            "FastVideo path requires the checkpoint's native solver to be a UniPC scheduler."
        )
    defaults = UniPCSpec()
    declared = UniPCSpec(
        solver_order=declared_cfg.get("solver_order", defaults.solver_order),
        solver_type=declared_cfg.get("solver_type", defaults.solver_type),
        lower_order_final=declared_cfg.get("lower_order_final", defaults.lower_order_final),
        disable_corrector=tuple(declared_cfg.get("disable_corrector") or ()),
    )
    if declared != spec:
        hint = (
            "; the checkpoint's non-empty disable_corrector has no model-config knob"
            if declared.disable_corrector != spec.disable_corrector
            else ""
        )
        raise RuntimeError(
            f"Checkpoint scheduler {path} declares {declared}, but model_config.unipc_* spells "
            f"{spec}; align model_config.unipc_* with the checkpoint scheduler{hint}."
        )


class FastVideoRolloutEngine(BaseRolloutEngine):
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

        try:
            importlib.import_module("fastvideo")
        except ModuleNotFoundError:
            fv_path = self.cfg.fastvideo_path or os.getenv("FASTVIDEO_PATH", "")
            require(bool(fv_path), "fastvideo not importable; set cfg.fastvideo_path or $FASTVIDEO_PATH")
            if fv_path not in sys.path:
                sys.path.insert(0, str(Path(fv_path).expanduser()))
            importlib.import_module("fastvideo")
        require(
            os.getenv("FASTVIDEO_WAN_SCHEDULER", "unipc").strip().lower() == "unipc",
            "FastVideo canonical rollout requires FASTVIDEO_WAN_SCHEDULER=unipc; "
            "Euler fallback would change the deterministic trajectory.",
        )
        require(
            hasattr(model_config, "unipc_solver_order"),
            "FastVideo canonical UniPC requires the model config to own the solver spec "
            "(unipc_solver_order / unipc_solver_type / unipc_lower_order_final)",
        )
        # Model-owned solver SSOT, verified against the checkpoint scheduler config below (README: Solver SSOT).
        self._unipc_spec = UniPCSpec(
            solver_order=model_config.unipc_solver_order,
            solver_type=model_config.unipc_solver_type,
            lower_order_final=model_config.unipc_lower_order_final,
        )
        require(
            strategy is not None and strategy.canonical_name is not None,
            "FastVideoRolloutEngine requires an injected SDE strategy with a canonical_name; "
            "set the rollout node's `strategy:` in the recipe (a separate injection from pipeline.strategy)",
        )
        self._sde_type = strategy.canonical_name
        self._is_dual_expert = config.model_family in {"wan2.2", "wan22"}
        if self._is_dual_expert:
            from unirl.models.wan22.diffusion import WAN22DiffusionStep

            self._timestep_scale = WAN22DiffusionStep.TIMESTEP_SCALE
            boundary_ratio = model_config.boundary_ratio
            require(
                0.0 < boundary_ratio < 1.0,
                f"WAN 2.2 model_config.boundary_ratio must be in (0, 1); got {boundary_ratio}",
            )
            # FastVideo routes on t >= boundary_ratio * num_train_timesteps; UniRL routes on
            # sigma >= boundary_ratio. The two agree only at this scale (README: dual expert).
            require(
                model_config.num_train_timesteps == self._timestep_scale,
                "WAN 2.2 FastVideo rollout requires model_config.num_train_timesteps to equal the "
                f"step kernel's TIMESTEP_SCALE ({self._timestep_scale:g})",
            )
            self._boundary_ratio = boundary_ratio
            idx_path, payload = _load_checkpoint_json(
                model_config.pretrained_model_ckpt_path,
                "model_index.json",
                context="FastVideo WAN 2.2",
                target="the dual-expert layout",
            )
            missing = [name for name in ("transformer", "transformer_2") if name not in payload]
            if missing:
                raise RuntimeError(
                    f"Checkpoint {idx_path} lacks {missing}; the WAN 2.2 A14B rollout requires both boundary-routed experts."
                )
        else:
            from unirl.models.wan21.diffusion import WAN21DiffusionStep

            self._timestep_scale = WAN21DiffusionStep.TIMESTEP_SCALE
        # Probe plan so unsupported kernels (cps/dpm2) fail at init, not per request.
        FastVideoUniPCPlan(
            sde_type=self._sde_type,
            sde_indices=(),
            spec=self._unipc_spec,
            timestep_scale=self._timestep_scale,
        )
        _verify_checkpoint_unipc_spec(model_config.pretrained_model_ckpt_path, self._unipc_spec)
        patch_fastvideo()
        self._build_generator()

        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            model_config.pretrained_model_ckpt_path,
            shift=model_config.shift,
        )
        logger.info(
            "Initialized fastvideo engine (rank=%s, native_logprob=%s, master_port=%s)",
            rank,
            config.native_logprob,
            ports.master_port,
        )

        self._version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False

    def _build_generator(self) -> None:
        from fastvideo.fastvideo_args import FastVideoArgs

        fv_kwargs: Dict[str, Any] = {
            "model_path": self.model_config.pretrained_model_ckpt_path,
            "num_gpus": self.cfg.num_gpus,
            "tp_size": self.cfg.tp_size,
            "sp_size": self.cfg.sp_size,
            "inference_mode": True,
            "output_type": "pt",
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "master_port": self._ports.master_port,
        }
        fv_kwargs.update(self.cfg.engine_kwargs)
        self._fastvideo_args = FastVideoArgs.from_kwargs(**fv_kwargs)
        if self._is_dual_expert:
            pipeline_config = self._fastvideo_args.pipeline_config
            pipeline_config.boundary_ratio = self._boundary_ratio
            pipeline_config.dit_config.boundary_ratio = self._boundary_ratio
        backend = self._fastvideo_args.distributed_executor_backend
        require(
            backend == "mp",
            f"FastVideo canonical UniPC patches only reach 'mp' executor workers; got "
            f"distributed_executor_backend={backend!r} (Ray actors would run unpatched)",
        )
        self._spawn_generator(reserve_first=False)

    def _spawn_generator(self, *, reserve_first: bool) -> None:
        """Spawn ``VideoGenerator`` from ``self._fastvideo_args``, retrying on master-port collisions."""
        from fastvideo import VideoGenerator

        max_port_attempts = 5
        for attempt in range(1, max_port_attempts + 1):
            if reserve_first or attempt > 1:
                self._ports = FastVideoPorts.reserve()
                self._fastvideo_args.master_port = self._ports.master_port
            try:
                self._generator = VideoGenerator.from_fastvideo_args(self._fastvideo_args)
                return
            except Exception as exc:  # noqa: BLE001
                port_in_use = "EADDRINUSE" in str(exc) or "address already in use" in str(exc).lower()
                if not port_in_use or attempt == max_port_attempts:
                    raise
                logger.warning(
                    "fastvideo master_port=%s busy (attempt %d/%d); retrying",
                    self._ports.master_port,
                    attempt,
                    max_port_attempts,
                )

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("FastVideoRolloutEngine.generate called after shutdown")
            return self._stamp_output_version(self._generate_core(sample))

    def _generate_core(self, sample: Sample) -> Sample:
        """Generate and fill the frontier diffusion Part."""
        sample.reject_row_canvases(engine="FastVideoRolloutEngine")
        require(
            not self._is_offloaded and self._generator is not None,
            "FastVideoRolloutEngine.generate: engine is offloaded (wake_up first).",
        )
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        require(
            gen.batch_size > 0,
            "FastVideoRolloutEngine.generate requires a non-empty Sample (gen batch_size > 0)",
        )
        ensure_sample_sigmas(sample, self.schedule_policy)

        fbs = self.cfg.forward_batch_size
        bs = gen.batch_size
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
        prompts = turns[0].content.texts
        require(
            len(prompts) == gen.batch_size,
            "fastvideo engine expects frontier-aligned text of len gen.batch_size; "
            f"got {len(prompts)} vs {gen.batch_size}",
        )
        params = gen.sampling_params
        base_seed = params.seed if params.seed is not None else 0
        seed_keys = (
            gen.init_noise_group_ids
            if gen.init_noise_group_ids and not params.disable_driver_xt
            else (gen.group_ids if params.init_same_noise else gen.sample_ids)
        )
        seeds = [_derive_group_seed(base_seed, k) for k in seed_keys]
        num_steps = params.num_inference_steps
        if params.sde_indices is None:
            sde_indices = list(range(num_steps))
        else:
            sde_indices = sorted(set(params.sde_indices))
            bad = [i for i in sde_indices if i < 0 or i >= num_steps]
            if bad:
                raise ValueError(f"FastVideo SDE indices out of range for num_steps={num_steps}: {bad}")
        raw = self._drive_fastvideo(prompts, params, params.sigmas, seeds, sde_indices)
        frames = raw["decoded"]
        require(
            torch.is_tensor(frames) and frames.dim() == 5,
            f"fastvideo decoded must be [B, C, T, H, W]; got "
            f"{tuple(frames.shape) if torch.is_tensor(frames) else type(frames).__name__}",
        )
        decoded = Videos.from_list(
            [Video(frames=frames[i].permute(1, 0, 2, 3).contiguous()) for i in range(frames.shape[0])]
        )
        return sample.with_filled_frontier(
            segment=self._build_segment(params, raw, sde_indices),
            primitives={"video": decoded},
            conditions=self._build_conditions(raw),
        )

    def _drive_fastvideo(
        self,
        prompts: List[str],
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        seeds: List[int],
        resolved_sde_indices: List[int],
    ) -> Dict[str, Any]:
        """PR #1222 native-logprob path via executor.execute_forward + RLData."""
        from copy import deepcopy

        from fastvideo.configs.sample.base import SamplingParam
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.utils import shallow_asdict

        sp = SamplingParam()
        sp.height = params.height
        sp.width = params.width
        sp.num_frames = params.num_frames
        sp.num_inference_steps = params.num_inference_steps
        sp.guidance_scale = params.guidance_scale
        sp.seed = params.seed if params.seed is not None else 0
        sp.num_videos_per_prompt = 1
        sp.save_video = False
        sp.return_frames = False
        sp.return_trajectory_latents = False
        sp.return_trajectory_decoded = False
        # Canonical σ verbatim — already shifted; no engine-side transform (README: σ SSOT).
        sp.sigmas = sigmas.detach().cpu().to(torch.float32).tolist()[:-1]

        step_plan = FastVideoUniPCPlan(
            sde_type=self._sde_type,
            sde_indices=tuple(resolved_sde_indices),
            spec=self._unipc_spec,
            timestep_scale=self._timestep_scale,
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
        for sample_index, (prompt, seed) in enumerate(zip(prompts, seeds)):
            one = deepcopy(sp)
            one.prompt = prompt
            one.seed = seed
            latents_size = [(one.num_frames - 1) // 4 + 1, one.height // 8, one.width // 8]
            n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
            sp_dict = shallow_asdict(one)
            sp_dict.pop("eta", None)
            batch = ForwardBatch(
                **sp_dict,
                eta=params.eta,
                n_tokens=n_tokens,
                VSA_sparsity=self._fastvideo_args.VSA_sparsity,
                rl_data=ForwardBatch.RLData(
                    enabled=True,
                    collect_log_probs=self.cfg.native_logprob,
                    store_trajectory=True,
                    keep_trajectory_on_cpu=True,
                    # sde_step_indices=None routes every index through the patched helper (README: Solver SSOT).
                    sde_step_indices=None,
                    sde_type=step_plan,
                ),
            )
            if self._is_dual_expert:
                batch.boundary_ratio = self._boundary_ratio
                low_scale = (
                    params.guidance_scale_2
                    if params.guidance_scale_2 is not None
                    else self.model_config.guidance_scale_2
                )
                batch.guidance_scale_2 = low_scale if low_scale is not None else params.guidance_scale
            out = self._generator.executor.execute_forward(batch, self._fastvideo_args)
            rl = out.rl_data
            actual_sigmas = rl.trajectory_timesteps if rl is not None else None
            if actual_sigmas is not None:
                actual_t = (
                    actual_sigmas.detach().cpu() if torch.is_tensor(actual_sigmas) else torch.as_tensor(actual_sigmas)
                )
                if actual_t.ndim == 1 and actual_t.shape[0] == sigmas.shape[0] - 1:
                    actual_sigmas = torch.cat([actual_t, torch.zeros(1, dtype=actual_t.dtype)])
            verify_engine_used_sigmas(
                actual_sigmas,
                expected=sigmas,
                engine_name=f"fastvideo sample {sample_index}",
            )
            traj = rl.trajectory_latents if rl is not None else None
            if traj is None:
                traj = out.trajectory_latents
            require(torch.is_tensor(traj), "FastVideo returned no trajectory tensor")
            if traj.dim() == 5:
                traj = traj.unsqueeze(0)
            all_traj.append(traj.detach().cpu())
            dec = out.output
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

            if self.cfg.native_logprob and resolved_sde_indices:
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

    def _build_conditions(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble the WAN21 ``conditions`` dict the trainer replays against."""
        text_embeds: List[torch.Tensor] = raw["text_embeds"]
        require(len(text_embeds) > 0, "fastvideo engine produced no text embeddings")
        text_masks: List[Optional[torch.Tensor]] = raw["text_masks"]

        text = TextEmbedCondition.concat(
            [
                TextEmbedCondition(embeds=text_embeds[i], pooled=None, attn_mask=text_masks[i])
                for i in range(len(text_embeds))
            ]
        )
        conditions: Dict[str, Any] = {"text": text}

        neg_embeds: List[torch.Tensor] = raw["neg_embeds"]
        if len(neg_embeds) == len(text_embeds) and len(neg_embeds) > 0:
            neg_masks: List[Optional[torch.Tensor]] = raw["neg_masks"]
            conditions["negative_text"] = TextEmbedCondition.concat(
                [
                    TextEmbedCondition(embeds=neg_embeds[i], pooled=None, attn_mask=neg_masks[i])
                    for i in range(len(neg_embeds))
                ]
            )
        return conditions

    def _build_segment(self, params: DiffusionSamplingParams, raw: Dict[str, Any], sde_set: List[int]):
        traj = raw["trajectory"]
        device = traj.device
        T = traj.shape[1] - 1
        indices = torch.arange(traj.shape[1], dtype=torch.long, device=device)
        sde_indices = torch.tensor(sde_set, dtype=torch.long, device=device) if sde_set else None

        sde_logp = None
        lp = raw["log_probs"]
        if lp is not None:
            if lp.shape[1] == T and len(sde_set) < T:
                sde_logp = lp[:, sde_set].contiguous()
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
            if not self._is_offloaded:
                return
            self._spawn_generator(reserve_first=True)
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

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._generate_lock:
                self._shutdown_requested = True
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
            logger.info("fastvideo transformer weights updated from %s", checkpoint_path)


__all__ = ["FastVideoRolloutEngine"]
