"""``fastvideo`` engine core — in-process FastVideo ``VideoGenerator`` rollout.

Mirrors the ``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine`` shells:
``generate`` is ``@distributed(DP_SCATTER)``, pins σ via ``ensure_req_sigmas``,
optionally chunks by ``forward_batch_size``, and packs one ``RolloutResp`` track
with a ``LatentSegment`` (trajectory + native per-step log-probs).

The FastVideo-driving logic (VideoGenerator boot, PR #1222 ``ForwardBatch.RLData``
native-logprob path, transformer hot-swap, sleep/wake) is ported from the proven
DiffusionRL FastVideo engine; only the typed boundary (RolloutReq/RolloutResp/
LatentSegment, σ SSOT) is new.

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
from unirl.rollout.engine.fastvideo.config import FastVideoEngineConfig, FastVideoPorts
from unirl.sde.noise import _derive_group_seed
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
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
        # Last checkpoint pushed by the weight sync. ``VideoGenerator`` loads the
        # PRETRAINED weights from ``model_path`` on every (re)build, so a sleep/wake
        # would silently roll back to pretrained; we re-apply this on wake. None
        # until the first ``update_weights_from_path``.
        self._last_weights_path: Optional[str] = None

        if ports is None:
            ports = FastVideoPorts.reserve()
        self._ports = ports

        self._ensure_fastvideo_importable()
        self._build_generator()

        # σ SSOT: same schedule policy the trainer/replay uses, so the engine can
        # pin req.sigmas and FastVideo consumes that exact schedule.
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
        self._fastvideo_args = FastVideoArgs.from_kwargs(**fv_kwargs)
        # WanT2V480PConfig (1.3B) defaults flow_shift=3.0. UniRL may train at
        # model_config.shift=5.0 (baseline). FastVideo re-applies flow_shift inside
        # set_timesteps even for custom sigmas, so pipeline_config.flow_shift MUST
        # match model_config.shift or native old_logp and trainer replay diverge.
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
        seeds = self._per_sample_seeds(req, params)
        raw = self._drive_fastvideo(prompts, params, req.sigmas, seeds)
        return self._build_resp(req, params, raw)

    def _per_sample_seeds(self, req: RolloutReq, params: Any) -> List[int]:
        """Per-sample seeds so sibling samples of one prompt diverge.

        Without this every sample of a prompt shared ``params.seed`` → identical
        video → identical reward → zero GRPO advantage → zero loss/grad. We key
        the seed the same way the driver keys x_T (``_derive_group_seed``):
        per-sample ids when ``init_same_noise`` is false (siblings differ),
        per-group ids when true (siblings share). Prefer the driver's
        ``init_noise_group_ids`` (carries rollout id + same/diff policy); fall
        back to sample/group ids, then to the flat seed.
        NOTE: this only decorrelates siblings; full driver-authoritative x_T SSOT
        (byte-identical to other engines via ``regen_initial_noise``) is a
        separate follow-up.
        """
        bs = int(req.batch_size)
        base_seed = int(params.seed)
        keys = getattr(req, "init_noise_group_ids", None)
        if not (isinstance(keys, (list, tuple)) and len(keys) == bs):
            same = bool(getattr(params, "init_same_noise", False))
            keys = list(req.group_ids) if same else list(req.sample_ids)
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
        sp.seed = int(params.seed)  # per-sample override applied in the loop below
        sp.num_videos_per_prompt = 1
        sp.save_video = False
        sp.return_frames = False
        # RLData already stores the trajectory directly on CPU. Enabling the
        # generic trajectory output as well retains a second full copy on GPU
        # throughout denoising and then copies it to CPU again.
        sp.return_trajectory_latents = False
        sp.return_trajectory_decoded = False
        # σ SSOT — AVOID the double-shift bug. ``req.sigmas`` is ALREADY the
        # shift-applied flow-match schedule (σ = shift·t/(1+(shift-1)·t)), but
        # FastVideo's FlowMatchEulerDiscreteScheduler.set_timesteps re-applies
        # the SAME shift to whatever sigmas we hand it (no guard). Feeding
        # req.sigmas directly makes FastVideo denoise on a *doubly*-shifted grid
        # while the trainer replays on the single-shift grid — the trajectory's
        # true noise level then mismatches the σ used to score its log-prob,
        # corrupting the GRPO gradient. So hand FastVideo the shift PRE-IMAGE g,
        # for which FastVideo's own shift reproduces req.sigmas exactly:
        #     g = s / (shift - s·(shift-1))   ⇒   shift·g/(1+(shift-1)·g) == s
        # (valid because FastVideo's WAN flow_shift == model_config.shift). Drop
        # the terminal 0 — FastVideo appends its own endpoint.
        _f = float(getattr(self._fastvideo_args.pipeline_config, "flow_shift", self.model_config.shift))
        _s = sigmas.detach().cpu().double()
        _g = _s / (_f - _s * (_f - 1.0))
        sp.sigmas = [float(x) for x in _g.tolist()[:-1]]

        # SDE window handed to FastVideo's denoiser so it injects exploration
        # noise ONLY on the trainer's SDE steps and runs the rest as a
        # deterministic Euler step (clean low-sigma tail). ``params.sde_indices``
        # is stamped per rollout by the trainer (resolve_sde_indices); it matches
        # the columns the trainer replays. ``None`` keeps the legacy all-steps
        # fallback; an explicit empty list means no SDE steps.
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
            one.seed = int(seed)  # decorrelate sibling samples (see _per_sample_seeds)
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
            # Decoded pixels: the FastVideo pipeline's DecodingStage writes the
            # final video to batch.output as [B, C, T, H, W] in [0, 1] (float32,
            # CPU). The reward path needs this as track.decoded (Videos).
            dec = getattr(out, "output", None)
            require(torch.is_tensor(dec), "FastVideo returned no decoded output (batch.output)")
            if dec.dim() == 4:
                dec = dec.unsqueeze(0)
            all_decoded.append(dec.detach().cpu().float())

            # Text conditioning: reuse the *exact* prompt embeddings FastVideo fed
            # its transformer this rollout, so the trainer's replay forward yields
            # an on-policy importance ratio (no re-encode drift). prompt_embeds is
            # a per-encoder list; WAN uses a single UMT5 encoder -> index 0.
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

            # Per-video: outputs are already copied to CPU above, so drop this
            # video's GPU tensors before the next iteration. This — not
            # ``forward_batch_size`` — is what actually bounds peak GPU memory,
            # since the forward runs one video at a time.
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

    def _build_resp(self, req: RolloutReq, params: Any, raw: Dict[str, Any]) -> RolloutResp:
        segment = self._build_segment(req, params, raw)
        decoded = self._build_decoded(raw)
        conditions = self._build_conditions(raw)
        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conditions,
                    segment=segment,
                    decoded=decoded,
                ),
            }
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

    def _build_segment(self, req: RolloutReq, params: Any, raw: Dict[str, Any]):
        traj = raw["trajectory"]  # [B, T+1, C, T_lat, H, W]
        device = traj.device
        T = int(traj.shape[1]) - 1
        indices = torch.arange(traj.shape[1], dtype=torch.long, device=device)

        # Mirror the SGLang reference: ``None`` means every transition is an SDE
        # step, while an explicit empty list is a deterministic forward process
        # and leaves ``sde_indices`` absent.
        _, sde_set = _resolve_sde_window(getattr(params, "sde_indices", None), T)
        sde_indices = torch.tensor(sde_set, dtype=torch.long, device=device) if sde_set else None

        # sde_logp: native per-step log-prob [B, T] from FastVideo's RLData. Slice
        # to the SDE columns when a strict subset was requested; otherwise the
        # full [B, T] already matches the all-steps schedule.
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
            sigmas=req.sigmas,
            indices=indices,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        if self._is_offloaded:
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
        if not self._is_offloaded:
            return
        from fastvideo import VideoGenerator

        # MultiprocExecutor's TCPStore must bind a master port every time the
        # generator is rebuilt. Reusing the constructor-time port after sleep
        # can hit a lingering listener; its get_open_port() check also has a
        # close-before-child-bind TOCTOU window when all DP actors wake
        # concurrently. Refresh the hint for every attempt and self-heal the
        # rare EADDRINUSE race instead of killing the Ray actor/run.
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
        # ``from_fastvideo_args`` reloads the PRETRAINED transformer from
        # ``model_path``; without this the engine would sample under pretrained
        # weights on every wake that isn't immediately followed by a weight sync
        # (i.e. any ``weight_sync_interval > 1`` step). Re-apply the last synced
        # checkpoint so wake is weight-preserving, matching the other engines'
        # sleep/wake contract (sglang resume_memory keeps weights resident).
        try:
            if self._last_weights_path is not None:
                self._generator.update_transformer_weights_from_path(self._last_weights_path)
                logger.info("fastvideo wake_up: re-applied synced weights from %s", self._last_weights_path)
        except Exception:
            # Fail closed: a rebuilt generator contains pretrained weights until
            # the cached checkpoint is restored. Keep the engine offloaded so a
            # retry cannot silently skip restoration and serve stale weights.
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
        if self._generator is not None:
            self._generator.shutdown()
            self._generator = None

    # ------------------------------------------------------------------ #
    # Weight sync — checkpoint_path (full-param hot-swap). Reached per worker
    # via the local sibling call from CheckpointWeightSync (not @distributed).
    # ------------------------------------------------------------------ #
    def update_weights_from_path(self, checkpoint_path: str, *, track_prefix: str = "") -> None:
        del track_prefix
        require(bool(checkpoint_path), "update_weights_from_path requires a non-empty path")
        require(self._generator is not None, "fastvideo engine is offloaded/not initialized")
        self._generator.update_transformer_weights_from_path(checkpoint_path)
        # Remember it so ``wake_up`` can re-apply after a rebuild (see wake_up).
        self._last_weights_path = checkpoint_path
        logger.info("fastvideo transformer weights updated from %s", checkpoint_path)


__all__ = ["FastVideoRolloutEngine"]
