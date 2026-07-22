"""UniRL v2 PE (Prompt Enhancement) joint trainer.

Two :class:`~unirl.train.stack.TrainStack` siblings — one for the
diffusion model, one for the AR LLM — colocated on the whole pool, sharing the
composed :class:`~unirl.models.pe.pipeline.PEPipeline` as a *trainside*
rollout (the rollout reads the live FSDP modules, so no weight sync).

One ``train_step``::

    rollout.generate(req)           → 2-track RolloutResp {"ar", "diffusion"}
    reward.score_and_attach(image)  → score the "diffusion" (image) track only
    resp.propagate_rewards("mean")  → credit-assign image reward up to "ar"
    track.compute_advantages()      → per-track GRPO (ar by prompt, diff by rewrite)
    {name}.stack.train_track(track) → route each track to its own model

Mirrors :class:`~unirl.trainer.diffusion.DiffusionTrainer` but wires two
of everything and a composed rollout. Deferred (same as the reference trainer):
multi-epoch replay, checkpoint / eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.models.pe.pipeline import PEPipeline
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.trainer.eval_suites import build_eval_suites
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names match PEPipeline's output and the per-side attributes on the
# trainer (``self.ar`` / ``self.diffusion``); also the algorithms' stage_attr.
TRACK_NAMES: Tuple[str, ...] = ("ar", "diffusion")


@dataclass
class _Side:
    """The sibling Remotes that make up one track.

    ``bundle`` + ``pipeline`` are always built (the pipeline is the rollout
    sampler). The training trio (``backend`` / ``algorithm`` / ``stack``) is
    ``None`` for a frozen, rollout-only side (``freeze_llm=True`` skips them on
    the AR side — see :meth:`PETrainer._wire_rollout_only_side`); a trained side
    populates all five.
    """

    bundle: Any
    pipeline: Any
    backend: Any = None
    algorithm: Any = None
    stack: Any = None


class PETrainer(BaseTrainer):
    """PE joint trainer: two TrainStack siblings + composed trainside rollout.

    ``freeze_llm=True`` switches the AR side to a frozen, rollout-only rewriter:
    the LLM still generates the N prompt rewrites each rollout (the composed
    :class:`PEPipeline` samples its live module under ``torch.no_grad``), but it
    has no backend / algorithm / stack and never trains — only the diffusion
    track updates. Use it to learn diffusion against a fixed prompt-enhancer.

    ``diffusion_group_scope`` selects the diffusion track's GRPO grouping (the
    advantage baseline): ``"rewrite"`` (default) compares the M images of one
    rewrite; ``"prompt"`` compares all N*M images of one original prompt across
    every rewrite, so diffusion learns to render well for the original intent
    regardless of how the (frozen) rewriter phrased it. The objective recipe
    pairs ``freeze_llm=True`` with ``diffusion_group_scope="prompt"``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        diffusion_cfg: DictConfig,
        ar_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = False,
        pe_cfg: Optional[DictConfig] = None,
        freeze_llm: bool = False,
        diffusion_group_scope: str = "rewrite",
        eval_interval: int = 0,
        eval_num_prompts: int = 8,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # Offload both tracks' FSDP train state to CPU during generate so the
        # awake sglang engines have room; onload before the train backward.
        # Never runs for trainside (it samples the live FSDP modules) — see train_step.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._rollout_is_trainside = False
        # Frozen LLM: the AR side is a rollout-only rewriter — built bundle +
        # pipeline (so the composed PEPipeline can sample it under no_grad), but
        # NO backend / algorithm / stack, so it never trains. Only the diffusion
        # track updates. ``_train_tracks`` is the set ``train_step`` routes to
        # ``stack.train_track``; with a frozen LLM that's diffusion alone.
        self._freeze_llm = bool(freeze_llm)
        self._train_tracks: Tuple[str, ...] = ("diffusion",) if self._freeze_llm else TRACK_NAMES
        # Diffusion-track GRPO grouping level (the advantage baseline):
        #   "rewrite" (default): group = the M images of one rewrite (group by
        #       the rewrite's sample id) — images compared only to siblings with
        #       identical conditioning text. Byte-identical to the prior behavior.
        #   "prompt": group = all N*M images descended from one original prompt
        #       (group by the ROOT prompt id) — a rewrite that systematically
        #       beats the prompt-wide mean earns non-zero advantage, so diffusion
        #       learns to render well for the original intent across the rewriter's
        #       rephrasings. Pairs with ``freeze_llm`` (fixed rewriter) for the
        #       "same semantics, different wordings → better images" objective.
        self._diffusion_group_scope = str(diffusion_group_scope)
        if self._diffusion_group_scope not in ("rewrite", "prompt"):
            raise ValueError(
                f"PETrainer.diffusion_group_scope must be 'rewrite' or 'prompt'; got {diffusion_group_scope!r}."
            )

        # Periodic eval on the eval set (run.eval_data_path), logged under eval/*;
        # eval_interval=0 disables it. Scores only the image ("diffusion") track,
        # generated at the deterministic best-quality setting (CFG=
        # eval_cfg_text_scale, eta=eval_eta) — same knobs/semantics as
        # DiffusionTrainer; extra eval-only rewards: unirl.trainer.eval_suites.
        # No eval_samples_per_prompt knob: the P->P*N*M two-level fan-out makes
        # a single per-prompt count ambiguous (N*M is already a dense eval).
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)

        # PE prompt-rewrite knobs forwarded to the composed PEPipeline (trainside
        # only — they shape the LLM rewrite + the text the diffusion child sees,
        # mirroring the sglang ComposedRolloutEngine's pe_instruction / pe_marker).
        # ``None`` everywhere preserves the prior bare-prompt behavior.
        pe = pe_cfg if pe_cfg is not None else {}
        self._pe_instruction = pe.get("pe_instruction", None)
        self._pe_marker = pe.get("pe_marker", None)
        self._pe_max_chars = pe.get("pe_max_chars", None)

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        # {"ar": ARSamplingParams(N), "diffusion": DiffusionSamplingParams(M)} —
        # the modality-keyed sampling dict driving PEPipeline's two-level fan-out.
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Per-track weight-sync bridges; None trainside (shares the modules).
        self.diffusion_sync = None
        self.ar_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.diffusion = self._wire_side(diffusion_cfg)
            # Frozen LLM → rollout-only AR (bundle + pipeline, no training trio).
            self.ar = self._wire_rollout_only_side(ar_cfg) if self._freeze_llm else self._wire_side(ar_cfg)

            # Pass the (composed) pipeline only to engines whose role_cls
            # declares it (trainside). For a separate-process engine
            # (``composed_pe``: sglang + sglang_diffusion) there is no shared
            # pipeline — trained weights reach the engine via the sync bridges.
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            takes_pipeline = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
            # Trainside samples the live FSDP modules → must not FSDP-offload.
            self._rollout_is_trainside = bool(takes_pipeline)
            if takes_pipeline:
                # Trainside: the composed PE pipeline shares both trained child
                # pipelines in-process, so the rollout samples the live FSDP
                # modules — no weight sync. ``stage_attrs: [diffusion, ar]``
                # eval-scopes both trained models.
                self.pe_pipeline = remote(
                    PEPipeline,
                    diffusion_pipeline=self.diffusion.pipeline,
                    llm_pipeline=self.ar.pipeline,
                    pe_instruction=self._pe_instruction,
                    pe_marker=self._pe_marker,
                    pe_max_chars=self._pe_max_chars,
                )
                self.rollout = remote(**rollout_parsed, pipeline=self.pe_pipeline)
            else:
                self.pe_pipeline = None
                self.rollout = remote(**rollout_parsed)

            self.reward = remote_hydra(reward_cfg)
            # Extra eval-only rewards (eval_rewards) — siblings of the training
            # reward on this slab; see unirl.trainer.eval_suites.
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )

            # Non-trainside: one bridge per track, each routed to its child of
            # the composed engine by ``track_prefix`` (set in the sync block).
            # A frozen LLM has no AR backend (and never trains), so it needs no
            # AR sync bridge — only the diffusion adapter is pushed to the engine.
            if sync_cfg is not None:
                self.diffusion_sync = remote_hydra(
                    sync_cfg.diffusion, backend=self.diffusion.backend, rollout=self.rollout
                )
                if not self._freeze_llm:
                    self.ar_sync = remote_hydra(sync_cfg.ar, backend=self.ar.backend, rollout=self.rollout)

    def _wire_side(self, cfg: DictConfig) -> _Side:
        """Build one track's bundle → pipeline → backend → algorithm → stack.

        Identical to ``DiffusionTrainer``'s single-side chain; called twice
        (diffusion + ar) inside the shared placement block.
        """
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        backend = remote_hydra(cfg.backend, bundle=bundle)
        algorithm = remote_hydra(cfg.algorithm, pipeline=pipeline)
        stack = remote_hydra(cfg.stack, fsdp_backend=backend, algorithm=algorithm)
        return _Side(bundle=bundle, pipeline=pipeline, backend=backend, algorithm=algorithm, stack=stack)

    def _wire_rollout_only_side(self, cfg: DictConfig) -> _Side:
        """Build a frozen, rollout-only side: bundle + pipeline, NO training trio.

        Used for the AR side under ``freeze_llm=True``. The bundle materializes
        the model on its device at load time (e.g. ``Qwen3Bundle.from_config``
        does ``.to(device)``), and the composed :class:`PEPipeline` samples this
        pipeline's stage under ``torch.no_grad`` via the trainside engine's
        eval-scope — so the LLM rewrites prompts but never trains. ``backend`` /
        ``algorithm`` / ``stack`` stay ``None``; the recipe's AR training blocks
        (if present) are intentionally ignored. The model is frozen by absence
        of an optimizer — no LoRA/FSDP train state is built for it.
        """
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        return _Side(bundle=bundle, pipeline=pipeline)

    def _build_req(
        self, inputs: RolloutInputs, rollout_id: int, *, base_sampling: Optional[Dict[str, BaseSamplingParams]] = None
    ) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        No pre-expansion: ``PEPipeline`` fans out ``P → P*N → P*N*M`` internally
        from the sampling dict (``ar.samples_per_prompt`` rewrites,
        ``diffusion.samples_per_prompt`` images each). The single-track trainer
        pre-expands here; PE must not, or it would double-count.

        ``rollout_id`` keys the diffusion SDE-step schedule: the indices are
        resolved off the diffusion sub-block (``resolve_sde_indices``), stamped
        onto a per-request copy, and the ``scheduler`` is nulled so only the
        concrete ``sde_indices`` ride to the engine (mirrors
        :meth:`DiffusionTrainer._build_req` / :meth:`UnifiedModelTrainer._build_req`).
        The AR sub-block has no SDE machinery and is left untouched.

        ``base_sampling`` overrides the modality-keyed sampling dict (``evaluate``
        passes its own deterministic params); ``None`` uses ``self.sampling_params``.
        """
        base = base_sampling if base_sampling is not None else self.sampling_params
        diff_params = base.get("diffusion")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        sampling_params = {**base, "diffusion": diffusion}
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)``. ``mean_reward`` is the
        mean unnormalized image reward (for the log line).

        ``sync_weights`` pushes each track's freshly-trained adapter into the
        engine between ``wake_up`` and ``generate`` — no-op trainside (the
        rollout shares the live FSDP modules, so the bridges are ``None``).
        ``rollout_id`` only keys the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.diffusion_sync is not None:
            self.diffusion_sync.sync()
            if self.ar_sync is not None:
                self.ar_sync.sync()
        # Free both tracks' train state during the separate-engine generate.
        # Sync above reads the FSDP weights, so offload only after it. A frozen
        # LLM has no AR backend, so only the diffusion train state is offloaded.
        do_fsdp_offload = self._enable_fsdp_offload and not self._rollout_is_trainside
        if do_fsdp_offload:
            self.diffusion.backend.offload()
            if self.ar.backend is not None:
                self.ar.backend.offload()
        resp = self.rollout.generate(req)
        self.rollout.sleep()
        if do_fsdp_offload:
            self.diffusion.backend.onload()
            if self.ar.backend is not None:
                self.ar.backend.onload()

        # 1. Score the IMAGE track only — the AR track's TextSegment is not
        #    directly scorable; its reward is credit-assigned below.
        #    ``score_and_attach`` is DP_SCATTER: it shards the diffusion track
        #    (P*N*M) across workers, but the P-prompt ``req`` would broadcast
        #    whole, so each worker would see a (track-shard) vs P size
        #    mismatch. Expand the req prompt-major to the track size first
        #    (mirrors DiffusionTrainer's pre-expanded single-track req) so the
        #    req and track shard identically across DP workers.
        diff_track = resp.tracks["diffusion"]
        n_track, p = len(diff_track.sample_ids), max(1, req.batch_size)
        reward_req = req.repeat_interleave(n_track // p) if n_track > p and n_track % p == 0 else req
        scored = self.reward.score_and_attach(req=reward_req, track=diff_track)
        # propagate_rewards reshapes child.rewards directly (no hydration), so
        # turn the worker-returned TensorRef into a real tensor first.
        if scored.rewards is not None:
            scored.rewards = hydrate(scored.rewards)
        resp.tracks["diffusion"] = scored

        # 2. Credit-assign image reward up the lineage → fills the "ar" track
        #    (mean over the M images of each rewrite). Kept even with a frozen
        #    LLM: it is cheap and gives the AR track a logged reward for parity,
        #    though the resulting AR advantage is unused when the LLM is frozen.
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks["diffusion"].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        # 4. Per-track GRPO advantages. "ar" groups by prompt (its N rewrites).
        #    "diffusion" groups by rewrite (M images) by default, or — when
        #    ``diffusion_group_scope="prompt"`` — by the ROOT prompt (all N*M
        #    images of a prompt) so cross-rewrite quality becomes signal. Only
        #    the trained tracks need advantages; a frozen LLM skips the AR one.
        for name in self._train_tracks:
            if name == "diffusion" and self._diffusion_group_scope == "prompt":
                resp.tracks[name] = resp.compute_track_advantages(name, group_key="root", normalize=True)
            else:
                resp.tracks[name] = resp.tracks[name].compute_advantages(normalize=True)

        # ``reward_req`` text is repeat_interleaved to the diffusion track size
        # (one prompt per sample), so it captions the image previews correctly —
        # unlike ``req`` (one prompt per group).
        self._drop_decoded(
            req,
            resp,
            rollout_id=rollout_id,
            media_prompts={"diffusion": list(reward_req.primitives["text"].texts)},
        )
        # 5. Route each TRAINED track to its own stack (each DP_SCATTER-sharded
        #    on dispatch). A frozen LLM trains the diffusion track only.
        results: Dict[str, TrainStepResult] = {
            name: getattr(self, name).stack.train_track(resp.tracks[name], training_progress=float(training_progress))
            for name in self._train_tracks
        }
        self.wandb_logger.log_rollout_step(rollout_id, results, resp, step_time_s=time.perf_counter() - t0)
        return results, mean_reward

    def evaluate(self, step: int) -> float:
        """Periodic eval on the eval set (no training); returns the mean image reward.

        Mirrors :meth:`train_step`'s rollout+reward path but skips
        credit-assign/advantage/backward: pull ``eval_num_prompts`` eval prompts
        (``run.eval_data_path``), generate the composed ``P→P*N*M`` fan-out with
        the diffusion sub-block forced onto the deterministic best-quality setting
        (CFG at ``eval_cfg_text_scale``, ``eta=eval_eta``), and score ONLY the
        image ("diffusion") track. The training reward plus every shared-set
        ``eval_rewards`` suite scores the SAME generated images; each own-set
        suite then gets its own generation pass over its own prompts. All means
        land in one ``eval/*`` row (``eval/reward`` + ``eval/<suite>``); returns
        ``eval/reward``.
        """
        # Override only the "diffusion" entry of the modality-keyed sampling dict.
        # CFG strength lives in ``cfg_text_scale`` on Bagel-style sampling params
        # and in ``guidance_scale`` on the standard DiffusionSamplingParams (SD3,
        # ...) — same fallback as :meth:`DiffusionTrainer.evaluate`.
        base_diffusion = self.sampling_params.get("diffusion")
        replace_kwargs = dict(eta=self.eval_eta)
        if "cfg_text_scale" in {f.name for f in dataclasses.fields(base_diffusion)}:
            replace_kwargs["cfg_text_scale"] = self.eval_cfg_text_scale
        else:
            replace_kwargs["guidance_scale"] = self.eval_cfg_text_scale
        eval_diffusion = dataclasses.replace(base_diffusion, **replace_kwargs)
        eval_sp = {**self.sampling_params, "diffusion": eval_diffusion}
        self.rollout.wake_up()
        if self.diffusion_sync is not None:  # no-op trainside (bridges are None)
            self.diffusion_sync.sync()
            if self.ar_sync is not None:
                self.ar_sync.sync()
        # Default pass: training reward + shared-set suites score the SAME images.
        scorers = [("reward", self.reward)] + [(s.name, s.reward) for s in self._eval_suites if s.data_source is None]
        metrics = self._eval_pass(self.data_source, self.eval_num_prompts, scorers, eval_sp, step)
        for suite in self._eval_suites:
            if suite.data_source is not None:
                n = suite.num_prompts or self.eval_num_prompts
                metrics.update(self._eval_pass(suite.data_source, n, [(suite.name, suite.reward)], eval_sp, step))
        self.rollout.sleep()
        logger.info(
            "EVAL step %d  (cfg=%.1f eta=%.1f)  %s",
            step,
            self.eval_cfg_text_scale,
            self.eval_eta,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        self.wandb_logger.log_eval(step, metrics)
        return metrics["reward"]

    def _eval_pass(
        self,
        data_source: Any,
        num_prompts: int,
        scorers: List[Tuple[str, Any]],
        eval_sp: Dict[str, BaseSamplingParams],
        step: int,
    ) -> Dict[str, float]:
        """One generate→score sweep over one eval set; returns each scorer's mean.

        Chunked by ``self.batch_size`` — the rollout DP-splits the un-expanded
        P-prompt req, so the chunk must be divisible by the engine dp; ``batch_size``
        is what training already runs, so it is divisible. A ragged tail
        (``num_prompts`` not a multiple of ``batch_size``) is floored off.
        """
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = len(all_inputs.sample_ids)
        chunk = max(1, self.batch_size)
        usable = n_prompts - n_prompts % chunk or n_prompts
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, usable, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            req = self._build_req(sub, step, base_sampling=eval_sp)
            resp = self.rollout.generate(req)
            # Score the image track only; align the P-prompt req to the P*N*M
            # image track so req and track shard identically across DP workers
            # (same expansion as train_step).
            diff_track = resp.tracks["diffusion"]
            n_track, p = len(diff_track.sample_ids), max(1, req.batch_size)
            reward_req = req.repeat_interleave(n_track // p) if n_track > p and n_track % p == 0 else req
            for name, reward in scorers:
                scored = reward.score_and_attach(req=reward_req, track=diff_track)
                if scored.rewards is not None:
                    r = hydrate(scored.rewards).to(torch.float32)
                    sums[name] += float(r.sum().item())
                    counts[name] += int(r.numel())
        return {name: sums[name] / max(1, counts[name]) for name, _ in scorers}

    # ---- checkpointing (PE trains two sides → one subdir per trained side) --

    def _ckpt_sides(self):
        """The trained sides to checkpoint: diffusion always; ar only when it trains.

        A frozen LLM (``freeze_llm=True``) has no AR backend and never trains, so
        only the diffusion adapter is persisted. Otherwise BOTH LoRA adapters are
        saved — eval reward depends on the AR rewriter AND the diffusion renderer,
        so a resumed checkpoint must restore both for the A/B consistency check.
        """
        sides = [("diffusion", self.diffusion)]
        if not self._freeze_llm and self.ar.backend is not None:
            sides.append(("ar", self.ar))
        return sides

    def _wait_for_checkpoints(self) -> None:
        """Flush both side backends before another save or worker teardown."""
        for _, side in self._ckpt_sides():
            side.backend.wait_for_checkpoint()

    def maybe_save_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str = "auto",
    ) -> None:
        """Save every ``save_interval`` rollouts (and on the last one), one subdir
        per trained side. ``save_interval <= 0`` disables saving.

        PE has no single ``self.backend`` (it owns ``self.diffusion.backend`` +
        ``self.ar.backend``), so this overrides the BaseTrainer single-backend
        version: each side writes ``<save_dir>/checkpoint-<step>/<side>/`` and the
        driver-owned ``trainer_state.json`` (wandb run id + step axis) sits beside
        them, mirroring the base method's semantics.
        """
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}")
        logger.info("Saving checkpoint at rollout %d/%d -> %s", step, num_rollouts, path)
        for name, side in self._ckpt_sides():
            side.backend.save(os.path.join(path, name), step=step, mode=save_mode)
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state_tmp = f"{trainer_state_path}.tmp"
        with open(trainer_state_tmp, "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)
        os.replace(trainer_state_tmp, trainer_state_path)
        if step >= num_rollouts:
            self._wait_for_checkpoints()

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts: Optional[int] = None) -> int:
        """Restore both trained sides from ``load_dir``; return the resume step.

        Returns 0 for a fresh run (``load_dir`` empty). Each trained side loads
        from its subdir; both advance in lockstep so either side's returned step
        is the resume point. Restores the driver-side ``_resume_state`` (wandb run
        id / step axis) so a resume appends to the same wandb run.
        """
        if not load_dir:
            return 0
        load_dir = os.path.abspath(load_dir)
        logger.info("Loading checkpoint from %s", load_dir)
        start = 0
        for name, side in self._ckpt_sides():
            result = side.backend.load(os.path.join(load_dir, name))
            if isinstance(result, list):  # BROADCAST dispatch collects one result per worker
                result = result[0]
            start = int(result or 0)
        state_path = os.path.join(load_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                self._resume_state = json.load(f)
        logger.info("Checkpoint restored; resuming at rollout %d", start)
        if num_rollouts is not None and start >= num_rollouts:
            logger.warning(
                "Checkpoint step %d >= num_rollouts %d — nothing left to train (num_rollouts is the TOTAL budget).",
                start,
                num_rollouts,
            )
        return start

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: push each track's adapter into the engine
        every N rollouts (fused into ``train_step``'s generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one), one subdir per trained side; ``0`` disables it. ``save_dir`` is the
        output folder (defaults to ``./checkpoints``); ``save_mode="auto"`` writes
        LoRA-only checkpoints when LoRA is active. ``load_dir``: restore from a
        checkpoint directory and RESUME from its saved step — ``num_rollouts`` is
        the TOTAL budget.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)  # baseline eval before any training
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                results, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                # eval(k) BEFORE save(checkpoint-k) at the same step, so a
                # resumed checkpoint re-runs the same eval (A/B consistency).
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
