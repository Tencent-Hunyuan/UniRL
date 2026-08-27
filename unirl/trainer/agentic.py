"""Service-scored multi-turn RL over the driver-side rollout manager."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.rollout.manager import RolloutManager, required_worker_concurrency, validate_worker_inflight
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import (
    BaseTrainer,
    build_sampling_dict,
    prepare_input_sample,
    unwrap_replicated_int,
)
from unirl.trainer.hydra import parse_hydra_cfg, remote_hydra
from unirl.types.advantages import finite_mean_std
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.graceful_shutdown import run_with_timeout

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ROLLOUT_SHUTDOWN_TIMEOUT_S = 60.0


def _extract_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    matches = list(_ANSWER_RE.finditer(text))
    return matches[-1].group(1).strip() if matches else text.strip()


def _is_failed(trajectory: Sample) -> bool:
    return not trajectory.gen_parts() or not trajectory.parts or trajectory.parts[-1].harness_status == "failed"


class AgenticTrainer(BaseTrainer):
    """One barrier agentic trainer with terminal-answer reward-service scoring."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
        stop: Optional[List[str]] = None,
        per_worker_inflight: int = 8,
    ) -> None:
        per_worker_inflight = int(per_worker_inflight)
        configured_concurrency = cfg.get("worker_max_concurrency")
        worker_max_concurrency = (
            required_worker_concurrency(per_worker_inflight)
            if configured_concurrency is None
            else int(configured_concurrency)
        )
        self._validate_config(
            cfg=cfg,
            batch_size=batch_size,
            rollout_cfg=rollout_cfg,
            sampling_cfg=sampling_cfg,
            algorithm_cfg=algorithm_cfg,
            sync_cfg=sync_cfg,
            per_worker_inflight=per_worker_inflight,
            worker_max_concurrency=worker_max_concurrency,
        )
        super().__init__(
            cfg=cfg,
            logging_cfg=logging_cfg,
            worker_max_concurrency=worker_max_concurrency,
        )

        try:
            self.batch_size = int(batch_size)
            self.data_source = instantiate(data_source_cfg)
            self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
            self._group_size = total_samples_per_prompt(self.sampling_params)
            self._stop = list(stop) if stop else ["</tool_call>"]
            self._per_worker_inflight = per_worker_inflight

            with placement(self.pool, fraction=1.0, shared_workers=True):
                self.bundle = remote_hydra(bundle_cfg)
                self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
                self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
                self.reward = remote_hydra(reward_cfg)
                self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
                self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
                self._build_colocated_rollout(rollout_cfg, sync_cfg)

            slots = self.rollout.engine_slots
            launchers = [lambda sample, slot=slot: slot.launch("generate", sample) for slot in slots]
            self._rollout_manager = RolloutManager(
                self.rollout,
                launchers=launchers,
                capacities=[self._per_worker_inflight] * len(launchers),
                group_size=self._group_size,
            )
            self._train_version = unwrap_replicated_int(
                self.backend.get_optimizer_step_count(),
                name="backend optimizer step count",
            )
        except BaseException:
            self._shutdown_runtime()
            raise

    @staticmethod
    def _validate_config(
        *,
        cfg: DictConfig,
        batch_size: int,
        rollout_cfg: DictConfig,
        sampling_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        sync_cfg: Optional[DictConfig],
        per_worker_inflight: int,
        worker_max_concurrency: int,
    ) -> None:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        validate_worker_inflight(
            per_worker_inflight,
            worker_max_concurrency=worker_max_concurrency,
            engine_concurrency=None,
        )
        if sync_cfg is None:
            raise ValueError("AgenticTrainer requires colocated TensorWeightSync")
        sync_target = str(sync_cfg.get("_target_", ""))
        if not sync_target.endswith("TensorWeightSync"):
            raise ValueError(f"AgenticTrainer requires colocated TensorWeightSync; got {sync_target!r}")

        episode = rollout_cfg.get("config", {}).get("episode_sampling")
        if episode is None:
            raise ValueError("rollout.config.episode_sampling is required")
        group_size = int(sampling_cfg.get("samples_per_prompt", 1))
        episode_group_size = int(episode.get("samples_per_prompt", 1))
        if group_size != episode_group_size:
            raise ValueError(
                "sampling.samples_per_prompt must equal rollout.config.episode_sampling.samples_per_prompt"
            )
        sampling_temperature = float(sampling_cfg.get("temperature", 1.0))
        episode_temperature = float(episode.get("temperature", 1.0))
        algorithm_temperature = float(algorithm_cfg.get("sampling_temperature", 1.0))
        if abs(sampling_temperature - episode_temperature) > 1e-9:
            raise ValueError("sampling.temperature must equal rollout episode temperature")
        if abs(sampling_temperature - algorithm_temperature) > 1e-9:
            raise ValueError("sampling.temperature must equal algorithm.sampling_temperature")
        if (int(batch_size) * group_size) % int(cfg.num_devices) != 0:
            raise ValueError("batch_size*samples_per_prompt must be divisible by num_devices")

    def _build_colocated_rollout(self, rollout_cfg: DictConfig, sync_cfg: DictConfig) -> None:
        self.backend.offload()
        self.rollout = remote(**parse_hydra_cfg(rollout_cfg))
        self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)
        self.rollout.sleep()
        self.backend.onload()

    def _build_request_sample(self, inputs: Sample, rollout_id: int) -> Sample:
        return prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="AgenticTrainer._build_request_sample",
            root_control={"ar": {"stop": list(self._stop)}},
        )

    def _collect_groups(self, requests: Sample) -> List[List[Sample]]:
        self.rollout.wake_up()
        self._rollout_manager.sync_weights(self.weight_sync, output_version=self._train_version)
        self.backend.offload()

        tasks = [prompt for prompt in requests.split() for _ in range(self._group_size)]
        self._rollout_manager.submit(tasks)
        groups = self._rollout_manager.collect(self.batch_size, current_version=self._train_version)
        if not self._rollout_manager.empty:
            raise RuntimeError("agentic barrier rollout must leave RolloutManager empty")

        self.rollout.sleep()
        self.backend.onload()
        return groups

    def train_step(
        self,
        inputs: Sample,
        *,
        training_progress: float = 0.0,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        t0 = time.perf_counter()
        requests = self._build_request_sample(inputs, rollout_id)
        groups = self._collect_groups(requests)
        trajectories = [trajectory for group in groups for trajectory in group]
        rewards = self._score_trajectories(trajectories, rollout_id)
        advantages = self._group_advantages(groups, rewards)
        result, mean_reward = self._train_and_log(
            trajectories,
            rewards,
            advantages,
            rollout_id=rollout_id,
            training_progress=training_progress,
            t0=t0,
        )
        self._train_version += result.optimizer_updates
        return result, mean_reward

    def _score_trajectories(self, trajectories: List[Sample], rollout_id: int) -> torch.Tensor:
        rewards = torch.full((len(trajectories),), float("nan"), dtype=torch.float32)
        valid_indices = [i for i, trajectory in enumerate(trajectories) if not _is_failed(trajectory)]
        if not valid_indices:
            logger.warning("Agentic rollout %d: every trajectory failed", rollout_id)
            return rewards

        questions: List[str] = []
        predictions: List[str] = []
        answers: List[Optional[str]] = []
        for i in valid_indices:
            trajectory = trajectories[i]
            root = trajectory.parts[0]
            metadata = root.metadata[0] if root.metadata else {}
            prompt = root.primitives.get("text")
            questions.append(prompt.texts[0] if isinstance(prompt, Texts) and prompt.texts else "")
            terminal = trajectory.gen_parts()[-1].primitives.get("text")
            terminal_text = terminal.texts[0] if isinstance(terminal, Texts) and terminal.texts else ""
            predictions.append(_extract_answer(terminal_text))
            answers.append((metadata or {}).get("answer"))

        score_count = len(valid_indices)
        reward_dp_size = max(1, int(self.reward.dp_size))
        score_padding = (-score_count) % reward_dp_size
        if score_padding:
            questions.extend([questions[0]] * score_padding)
            predictions.extend([predictions[0]] * score_padding)
            answers.extend([answers[0]] * score_padding)

        ar_sampling = self.sampling_params.get("ar")
        score_input = Part.input(
            [f"score{rollout_id}:{i}" for i in valid_indices]
            + [f"score{rollout_id}:padding{i}" for i in range(score_padding)],
            primitives={"text": Texts(texts=questions)},
            metadata=[{"answer": answer} for answer in answers],
        )
        scoring = (
            Sample.request(score_input)
            .fork(1, sampling_params=ar_sampling)
            .with_filled_frontier(primitives={"text": Texts(texts=predictions)})
        )
        scored = self.reward.score_and_attach(scoring)
        scored_rewards = scored.parts[-1].rewards
        if scored_rewards is None:
            raise RuntimeError("reward service returned no rewards")
        values = hydrate(scored_rewards).to(torch.float32).flatten()
        if values.numel() != score_count + score_padding:
            raise RuntimeError(
                f"reward service returned {values.numel()} rewards for {score_count + score_padding} scoring rows"
            )
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("reward service returned non-finite rewards")
        rewards[torch.tensor(valid_indices, dtype=torch.long)] = values[:score_count]

        failed = len(trajectories) - len(valid_indices)
        if failed:
            logger.warning(
                "Agentic rollout %d: %d/%d trajectories failed and were excluded",
                rollout_id,
                failed,
                len(trajectories),
            )
        return rewards

    def _group_advantages(self, groups: List[List[Sample]], rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        offset = 0
        for group in groups:
            end = offset + len(group)
            group_rewards = rewards[offset:end]
            finite = torch.isfinite(group_rewards)
            mean, std = finite_mean_std(group_rewards)
            normalized = (group_rewards - mean) / (std + 1e-8)
            advantages[offset:end] = torch.where(finite, normalized, torch.zeros_like(normalized))
            offset = end
        if offset != rewards.numel():
            raise RuntimeError("rollout group/reward cardinality mismatch")
        return advantages

    def _train_and_log(
        self,
        trajectories: List[Sample],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        *,
        rollout_id: int,
        training_progress: float,
        t0: float,
    ) -> Tuple[TrainStepResult, float]:
        finite = torch.isfinite(rewards)
        mean_reward = float(rewards[finite].mean().item()) if bool(finite.any()) else 0.0

        train_parts: List[Part] = []
        for i, trajectory in enumerate(trajectories):
            if not bool(finite[i]):
                continue
            advantage = float(advantages[i].item())
            for generated in trajectory.gen_parts():
                generated = _part_with_field(
                    generated,
                    "advantages",
                    torch.full((generated.batch_size,), advantage, dtype=torch.float32),
                )
                generated = _part_with_field(generated, "primitive_metadata", {})
                generated = _part_with_field(generated, "media_preview", None)
                generated = _part_with_field(generated, "primitives", {})
                generated = _part_with_field(generated, "rewards", None)
                generated = _part_with_field(generated, "component_rewards", None)
                train_parts.append(generated)

        depths = [len(trajectory.gen_parts()) for trajectory in trajectories]
        logger.info(
            "rollout %d trajectory turns: n=%d mean=%.2f min=%d max=%d hist=%s",
            rollout_id,
            len(depths),
            (sum(depths) / len(depths)) if depths else 0.0,
            min(depths, default=0),
            max(depths, default=0),
            dict(sorted(Counter(depths).items())),
        )
        if train_parts:
            train_part = self._pad_to_dp_multiple(Part.concat(train_parts))
            result = self.stack.train_track(train_part, training_progress=float(training_progress))
            train_rows = int(train_part.batch_size)
        else:
            result = TrainStepResult(0.0, 0.0, 0.0, False, [], {}, optimizer_updates=0)
            train_rows = 0

        log_sample = self._build_log_sample(trajectories, rewards, advantages, rollout_id)
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={
                "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
                "agent/max_turns": max(depths) if depths else 0,
                "agent/failed_trajectories": int((~finite).sum().item()),
                "agent/train_rows": train_rows,
            },
        )
        return result, mean_reward

    def _build_log_sample(
        self,
        trajectories: List[Sample],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        rollout_id: int,
    ) -> Sample:
        if len(trajectories) != rewards.numel() or rewards.numel() != advantages.numel():
            raise RuntimeError("trajectory/reward/advantage cardinality mismatch")
        finite = torch.isfinite(rewards)
        count = int(finite.sum().item())
        if count == 0:
            return Sample(parts=[])

        ar_sampling = self.sampling_params.get("ar")
        root = Part.input(
            [f"log{rollout_id}:{i}" for i in range(count)],
            primitives={"text": Texts(texts=[""] * count)},
        )
        sample = (
            Sample.request(root)
            .fork(1, sampling_params=ar_sampling)
            .with_filled_frontier(primitives={"text": Texts(texts=[""] * count)})
        )
        frontier = _part_with_field(sample.parts[-1], "rewards", rewards[finite].to(torch.float32))
        frontier = _part_with_field(frontier, "advantages", advantages[finite].to(torch.float32))
        return sample.with_parts([*sample.parts[:-1], frontier])

    def _pad_to_dp_multiple(self, part: Part) -> Part:
        dp_size = int(getattr(self.stack, "dp_size", self.num_devices))
        pad = (-int(part.batch_size)) % dp_size
        if pad == 0:
            return part
        lengths = part.segment.lengths if part.segment is not None else None
        source = int(torch.argmin(lengths).item()) if lengths is not None and lengths.numel() else 0
        padding = part.select(torch.full((pad,), source, dtype=torch.long))
        padding = _part_with_field(padding, "advantages", torch.zeros(pad, dtype=torch.float32))
        return Part.concat([part, padding])

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        try:
            start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
            self._train_version = unwrap_replicated_int(
                self.backend.get_optimizer_step_count(),
                name="backend optimizer step count",
            )
            for _ in range(start_rollout):
                self.data_source.get_samples(self.batch_size)
            self._init_wandb(num_rollouts=num_rollouts, extra={"agentic_execution": "barrier"})

            for rollout_id in range(start_rollout, num_rollouts):
                inputs = self.data_source.get_samples(self.batch_size)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self.train_step(
                    inputs,
                    training_progress=training_progress,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                self.maybe_save_checkpoint(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            try:
                self._finish_wandb()
            finally:
                self._shutdown_runtime()

    def shutdown(self) -> None:
        self._shutdown_runtime()

    def _shutdown_runtime(self) -> None:
        if getattr(self, "_runtime_shutdown_done", False):
            return
        self._runtime_shutdown_done = True

        manager = getattr(self, "_rollout_manager", None)
        rollout = getattr(self, "rollout", None)
        shutdown = getattr(rollout, "shutdown", None)
        pool = getattr(self, "pool", None)
        try:
            if manager is not None:
                manager.close()
        finally:
            try:
                if callable(shutdown):
                    run_with_timeout(
                        shutdown,
                        timeout=_ROLLOUT_SHUTDOWN_TIMEOUT_S,
                        what="agentic rollout engine shutdown",
                    )
            finally:
                if pool is not None:
                    pool.shutdown()


__all__ = ["AgenticTrainer"]
