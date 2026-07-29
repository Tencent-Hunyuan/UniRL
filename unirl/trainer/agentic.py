"""AgenticTrainer — multi-turn agentic RL over the AgenticRolloutEngine (LIN-519).

A sibling of :class:`~unirl.trainer.ar.ARTrainer` for the AGENTIC path. The rollout is
the rank-0-coordinated
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`, whose ``generate``
returns a FLAT ``List[Sample]`` of variable-depth, independently terminated
trajectories (one per GRPO sibling) — not a single batched Sample. So this trainer
overrides only:

- ``__init__`` — wire the rank-0 coordinator (``set_workers``);
- ``_build_request_sample`` — emit JUST the prompts (no ``fork``: the engine fans the
  ``n`` GRPO siblings internally) with the per-turn ``stop`` on the root control bag;
- ``train_step`` — consume the trajectory list: judge each trajectory's ``<answer>``
  with the reward backend, compute GROUP-relative GRPO advantages over the ``n``
  siblings of each prompt, assign each trajectory's scalar advantage to ALL its
  assistant turns, concatenate every turn into ONE training Part (padded to a DP
  multiple), and run ONE optimizer step.

Everything else — worker construction, weight sync, checkpointing, the ``train`` loop
— is inherited from ``ARTrainer`` / ``BaseTrainer`` unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from unirl.algorithms.normalizers import build_group_index_map
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import prepare_input_sample
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: Optional[str]) -> str:
    """The last ``<answer>…</answer>`` span, else the whole text (the verifier is
    tolerant of an unwrapped / ``\\boxed{}`` answer)."""
    if not text:
        return ""
    matches = list(_ANSWER_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else text.strip()


def _is_failed(traj: Sample) -> bool:
    """Whether the engine marked this trajectory an infrastructure failure.

    :meth:`AgenticRolloutEngine._run_one` isolates faults by returning ``done=True``
    with a NaN reward on the terminal Part, or — when the fault preceded the first
    turn — with no gen Parts at all. Either shape must stay out of GRPO's group
    statistics; grading a dead backend's empty output as a genuine miss would bias
    every sibling in the group.
    """
    gens = traj.gen_parts()
    if not gens:
        return True
    rewards = gens[-1].rewards
    return rewards is not None and bool(torch.isnan(hydrate(rewards)).any())


def _validate_agentic_cfg(kw: dict) -> None:
    """Fail fast on cross-config invariants only jointly visible at the trainer.

    The recipes assert these in prose comments but nothing enforced them, so a
    mismatch silently broke the on-policy ratio or DP scatter. Each check is
    skipped when either side is absent, so it never rejects a config that merely
    omits a key. (The ``env.max_turns == config.max_turns`` check lives in
    ``AgenticRolloutEngine.__init__``, where the built env is in scope.)
    """
    rollout_cfg = kw.get("rollout_cfg")
    ep = OmegaConf.select(rollout_cfg, "config.episode_sampling") if rollout_cfg is not None else None
    if ep is None:
        return
    n_ep = int(OmegaConf.select(ep, "samples_per_prompt") or 1)
    t_ep = OmegaConf.select(ep, "temperature")

    sampling_cfg = kw.get("sampling_cfg")
    if sampling_cfg is not None:
        n_s = int(OmegaConf.select(sampling_cfg, "samples_per_prompt") or 1)
        if n_s != n_ep:
            raise ValueError(
                f"sampling.samples_per_prompt ({n_s}) must equal "
                f"rollout.config.episode_sampling.samples_per_prompt ({n_ep})"
            )
        t_s = OmegaConf.select(sampling_cfg, "temperature")
        if t_ep is not None and t_s is not None and abs(float(t_s) - float(t_ep)) > 1e-9:
            raise ValueError(f"sampling.temperature ({t_s}) must equal episode_sampling.temperature ({t_ep})")

    algorithm_cfg = kw.get("algorithm_cfg")
    t_a = OmegaConf.select(algorithm_cfg, "sampling_temperature") if algorithm_cfg is not None else None
    if t_ep is not None and t_a is not None and abs(float(t_a) - float(t_ep)) > 1e-9:
        raise ValueError(
            f"algorithm.sampling_temperature ({t_a}) must equal episode_sampling.temperature "
            f"({t_ep}) — else replay's tempered log-softmax diverges from the sampler (ratio != 1)"
        )

    cfg = kw.get("cfg")
    batch_size = kw.get("batch_size")
    nd = OmegaConf.select(cfg, "num_devices") if cfg is not None else None
    if nd is not None and batch_size is not None and (int(batch_size) * n_ep) % int(nd) != 0:
        raise ValueError(
            f"batch_size*samples_per_prompt ({int(batch_size) * n_ep}) must be divisible "
            f"by num_devices ({int(nd)}) for the DP scatter"
        )


class AgenticTrainer(ARTrainer):
    """Agentic (multi-turn tool-use) RL trainer over the ``AgenticRolloutEngine``."""

    def __init__(self, *, stop: Optional[List[str]] = None, **kwargs) -> None:
        _validate_agentic_cfg(kwargs)
        super().__init__(**kwargs)
        # Per-turn stop: a tool-call turn ends at ``</tool_call>`` and yields to the
        # tool; a final-answer turn runs to EOS. Rides the request root's control bag
        # (``resolve_sampling`` reads ``control["ar"]``).
        self._stop = list(stop) if stop else ["</tool_call>"]
        # Wire the rank-0 coordinator (``AgenticRolloutEngine.set_workers`` — the
        # ``NCCLWeightSync.set_rollout_targets`` shape). ``.workers`` / ``.role_name``
        # are ``Handle`` attributes.
        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """The ``P`` prompts as an input-only ``Sample`` tree — NO ``fork`` (the
        agentic engine fans the ``n`` GRPO siblings itself) — with the per-turn
        ``stop`` on the root control bag and root ``metadata`` (the ground-truth
        answer) carried for the reward judge."""
        del sampling  # the engine's ``episode_sampling`` owns per-turn params + ``n``
        return prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="AgenticTrainer._build_request_sample",
            root_control={"ar": {"stop": list(self._stop)}},
        )

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One agentic ``rollout → judge → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` for the progress line.
        """
        t0 = time.perf_counter()

        # 1) Rollout — the barrier multi-turn generate. On-policy: sync first.
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        trajs: List[Sample] = self.rollout.generate(sample)[0]  # BROADCAST+RANK_ZERO -> [List[Sample]]
        self.rollout.sleep()

        # 2) Per-trajectory scalar reward + GRPO group id (root id). Overridable:
        #    answer-graded here (``<answer>`` -> reward backend), env-sourced in
        #    ``AgenticEnvTrainer`` (ALFWorld etc.).
        rewards, group_ids = self._rewards_and_groups(sample, trajs, rollout_id)

        # 3-6) GROUP-relative advantage -> assign to every turn -> ONE step -> log.
        return self._advantage_train_and_log(
            trajs, rewards, group_ids, rollout_id=rollout_id, training_progress=training_progress, t0=t0
        )

    def _advantage_train_and_log(
        self,
        trajs: List[Sample],
        rewards: torch.Tensor,
        group_ids: List[str],
        *,
        rollout_id: int,
        training_progress: float,
        t0: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TrainStepResult, float]:
        """GROUP-relative GRPO advantage → assign each trajectory's scalar advantage to ALL
        its assistant turns → ONE padded ``train_track`` step → log. Shared by the barrier
        ``train_step`` and the colocate partial-rollout trainer
        (:class:`~unirl.trainer.agentic_partial.AgenticPartialTrainer`); the reward SOURCE
        (answer vs env) is already resolved into ``rewards``/``group_ids`` by the caller.
        ``extra_metrics`` are merged into the logged ``agent/*`` metrics (e.g. the partial
        trainer's committed/carried/dropped counts)."""
        # A NaN reward marks a crashed trajectory (env bug, not a policy outcome) —
        # excluded from the reported mean and from GRPO (see _group_advantages).
        finite = torch.isfinite(rewards)
        mean_reward = float(rewards[finite].mean().item()) if bool(finite.any()) else 0.0

        # GROUP-relative GRPO advantage over the ``n`` siblings per prompt.
        advantages = self._group_advantages(rewards, group_ids)

        # Assign each trajectory's scalar advantage to ALL its assistant turns; gather.
        train_parts: List[Part] = []
        for i, tr in enumerate(trajs):
            adv_i = float(advantages[i].item())
            for gp in tr.gen_parts():
                gp = _part_with_field(gp, "advantages", torch.full((gp.batch_size,), adv_i, dtype=torch.float32))
                gp = _part_with_field(gp, "primitives", {})  # free decoded text before train
                # Drop any per-trajectory env reward: it rides only the TERMINAL turn (a
                # TensorRef on 1 of the trajectory's k gen parts), so concatenating turns
                # would leave a short, misaligned rewards field that breaks DP scatter.
                # The reward was already read into ``advantages`` (row-aligned across turns).
                gp = _part_with_field(gp, "rewards", None)
                train_parts.append(gp)

        depths = [len(tr.gen_parts()) for tr in trajs]
        # Per-trajectory turn distribution — the workload's depth VARIANCE (a straggler-cut only
        # pays when this is wide; ~uniform means over-sample-and-drop is pure waste). LIN-531.
        logger.info(
            "rollout %d trajectory turns: n=%d mean=%.2f min=%d max=%d hist=%s",
            rollout_id,
            len(depths),
            (sum(depths) / len(depths)) if depths else 0.0,
            min(depths, default=0),
            max(depths, default=0),
            dict(sorted(Counter(depths).items())),
        )
        if not train_parts:  # pathological: every sampled trajectory failed to generate
            logger.warning("AgenticTrainer rollout %d produced no trainable turns.", rollout_id)
            return TrainStepResult(0.0, 0.0, 0.0, False, [], {}), mean_reward

        # ONE training Part -> pad to a DP multiple (zero-advantage rows) -> ONE step.
        train_part = Part.concat(train_parts)
        train_part = self._pad_to_dp_multiple(train_part)
        result = self.stack.train_track(train_part, training_progress=float(training_progress))

        # Logging sample: one row per trajectory whose gen frontier carries the reward +
        # advantage (compute_rollout_sample_metrics reads gen_parts). Built from the
        # computed tensors so it is independent of the reward SOURCE (answer vs env).
        log_sample = self._build_log_sample(trajs, rewards, advantages, rollout_id)
        metrics: Dict[str, Any] = {
            "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
            "agent/max_turns": max(depths) if depths else 0,
            "agent/genless_trajectories": sum(1 for d in depths if d == 0),
            "agent/train_rows": int(train_part.batch_size),
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics=metrics,
        )
        return result, mean_reward

    def _rewards_and_groups(
        self, sample: Sample, trajs: List[Sample], rollout_id: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Per-trajectory scalar reward + GRPO group id (root id) — the overridable
        reward step. Base path grades each trajectory's ``<answer>`` against the
        ground truth via the reward backend (MathVerify / LLM judge); the ``n``
        siblings of a prompt share its root id (group-by-root). Subclasses swap the
        reward SOURCE (e.g. :class:`~unirl.trainer.agentic_env.AgenticEnvTrainer`
        reads the environment's per-trajectory return) while keeping the GRPO tail.

        A FRESH flat scoring Sample is required because ``score_and_attach`` rejects
        precomputed frontier rewards; its frontier carries no ``segment`` so the
        truncation-shaping branch is skipped.
        """
        root = sample.parts[0]
        root_meta = root.metadata or [None] * len(root.sample_ids)
        gt_by_root = {sid: (md or {}).get("answer") for sid, md in zip(root.sample_ids, root_meta)}
        questions: List[str] = []
        predictions: List[str] = []
        answers: List[Optional[str]] = []
        group_ids: List[str] = []
        for tr in trajs:
            root_id = tr.parts[0].sample_ids[0]
            q_prim = tr.parts[0].primitives.get("text")
            questions.append(q_prim.texts[0] if (q_prim is not None and q_prim.texts) else "")
            gens = tr.gen_parts()
            term = ""
            terminal_text = gens[-1].primitives.get("text") if gens else None
            if isinstance(terminal_text, Texts) and terminal_text.texts:
                term = terminal_text.texts[0]
            predictions.append(_extract_answer(term))
            answers.append(gt_by_root.get(root_id))
            group_ids.append(root_id)

        m = len(trajs)
        ar_sp = self.sampling_params.get("ar")
        score_in = Part.input(
            [f"score{rollout_id}:{i}" for i in range(m)],
            primitives={"text": Texts(texts=list(questions))},
            metadata=[{"answer": a} for a in answers],
        )
        scoring = (
            Sample.request(score_in)
            .fork(1, sampling_params=ar_sp)  # frontier is a gen Part (reward/adv panels read gen_parts)
            .with_filled_frontier(primitives={"text": Texts(texts=list(predictions))})
        )
        scoring = self.reward.score_and_attach(scoring)
        rewards = hydrate(scoring.parts[-1].rewards).to(torch.float32)
        # A failed trajectory was still graded above (its empty answer scores as a
        # miss); overwrite with NaN so _group_advantages excludes it from the group's
        # mean/std and gives it zero advantage instead of a real negative signal.
        failed = torch.tensor([_is_failed(tr) for tr in trajs], dtype=torch.bool)
        if bool(failed.any()):
            logger.warning(
                "AgenticTrainer rollout %d: %d/%d trajectories failed; excluded from GRPO statistics.",
                rollout_id,
                int(failed.sum()),
                len(trajs),
            )
            rewards = torch.where(failed, torch.full_like(rewards, float("nan")), rewards)
        return rewards, group_ids

    def _build_log_sample(
        self, trajs: List[Sample], rewards: torch.Tensor, advantages: torch.Tensor, rollout_id: int
    ) -> Sample:
        """A flat one-row-per-trajectory Sample whose gen frontier carries the
        per-trajectory reward + advantage, for ``compute_rollout_sample_metrics``
        (reward/advantage distributions). Reward-source agnostic — no reward backend,
        no scoring sample — so it works for both the answer-graded and env-sourced paths."""
        m = len(trajs)
        ar_sp = self.sampling_params.get("ar")
        log_in = Part.input([f"log{rollout_id}:{i}" for i in range(m)], primitives={"text": Texts(texts=[""] * m)})
        log_sample = (
            Sample.request(log_in)
            .fork(1, sampling_params=ar_sp)
            .with_filled_frontier(primitives={"text": Texts(texts=[""] * m)})
        )
        frontier = _part_with_field(log_sample.parts[-1], "rewards", rewards.to(torch.float32))
        frontier = _part_with_field(frontier, "advantages", advantages.to(torch.float32))
        return log_sample.with_parts([*log_sample.parts[:-1], frontier])

    def _group_advantages(self, rewards: torch.Tensor, group_ids: List[str]) -> torch.Tensor:
        """Group-relative GRPO advantages, ``ARTrainer.compute_advantages`` parity
        (population std), over the ``n`` siblings of each prompt (grouped by root id;
        completion order is fine). ``adv_normalization_scope='global'`` z-scores the
        whole batch; ``normalize_adv_by_std=False`` mean-centers only."""
        r = rewards.to(torch.float32)
        # NaN reward = crashed trajectory: excluded from the group's mean/std and given
        # ZERO advantage (neutral), so an env crash neither rewards nor penalizes its
        # actions. All-finite (the answer-graded path) is byte-identical to before.
        finite = torch.isfinite(r)
        if self.adv_normalization_scope == "global":
            rf = r[finite]
            mean = rf.mean() if rf.numel() else r.new_zeros(())
            centered = torch.where(finite, r - mean, torch.zeros_like(r))
            if self.normalize_adv_by_std:
                std = rf.std(unbiased=False) if rf.numel() > 1 else r.new_ones(())
                centered = centered / (std + 1e-8)
            return torch.where(finite, centered, torch.zeros_like(centered))
        adv = torch.zeros_like(r)
        for idxs in build_group_index_map(group_ids).values():
            idx = torch.tensor(idxs, dtype=torch.long)
            fin = finite[idx]
            g = r[idx]
            gf = g[fin]
            if gf.numel() == 0:
                continue  # whole group crashed -> zero advantage
            centered = g - gf.mean()
            if self.normalize_adv_by_std:
                std = gf.std(unbiased=False) if gf.numel() > 1 else g.new_ones(())
                centered = centered / (std + 1e-8)
            adv[idx] = torch.where(fin, centered, torch.zeros_like(centered))
        return adv

    def _pad_to_dp_multiple(self, part: Part) -> Part:
        """Pad ``part`` up to a multiple of the train DP size by replicating the
        shortest row with advantage 0 (zero gradient for GRPO/DRPO/CPPO), so the ragged
        Σ-turns batch satisfies ``pytree_chunk``'s divisibility check."""
        dp = int(getattr(self.stack, "dp_size", self.num_devices))
        pad = (-int(part.batch_size)) % dp
        if pad == 0:
            return part
        lengths = part.segment.lengths if part.segment is not None else None
        src = int(torch.argmin(lengths).item()) if (lengths is not None and lengths.numel()) else 0
        pad_block = part.select(torch.full((pad,), src, dtype=torch.long))
        pad_block = _part_with_field(pad_block, "advantages", torch.zeros(pad, dtype=torch.float32))
        return Part.concat([part, pad_block])

    def evaluate(self, rollout_id: int) -> float:
        raise NotImplementedError(
            "AgenticTrainer.evaluate is not implemented: the agentic engine returns "
            "List[Sample], not a Sample. Set eval_interval=0 (agentic eval is a follow-up)."
        )
