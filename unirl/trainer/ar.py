import inspect
import logging
import math
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.prompts import RolloutInputs
from unirl.types.sample import Part, Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class ARTrainer(BaseTrainer):
    """Autoregressive (VLM / LLM) RL trainer: rollout + train colocated.

    Sibling of :class:`~unirl.trainer.diffusion.DiffusionTrainer` for the
    AR path. Structurally identical except ``_build_request_sample`` carries **no SDE step
    scheduling** — that is diffusion-only (``DiffusionSamplingParams`` owns
    ``scheduler`` / ``sde_indices`` / ``resolve_sde_indices``), and
    ``ARSamplingParams`` has none of it. Keeping the AR trainer separate means
    the AR path never touches diffusion code (no ``hasattr`` guard, no
    ``dataclasses.replace`` of SDE fields).

    Trainside colocate (the qwen_vl recipe): the training pipeline IS the
    sampler, so ``sync_cfg`` is absent and ``weight_sync`` stays ``None``.
    """

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
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # "group" (textbook GRPO, default) or "global" (v1 baseline parity).
        self.adv_normalization_scope = adv_normalization_scope
        # True (default) = standard GRPO: divide the group-relative advantage by the
        # group std. False = mean-center only (reward - group_mean), NO std division —
        # removes the difficulty bias that over-amplifies low-std (hard) prompts.
        self.normalize_adv_by_std = normalize_adv_by_std
        # verl trainer.balance_batch parity: driver-side reorder of the rollout
        # batch so each DP shard receives a similar total-token workload. FSDP
        # collectives sync all ranks every micro, so a step runs at the SLOWEST
        # rank's pace — without balancing, the rank that drew the longest
        # sequences straggles (~+/-11%% rank-total variance at heavy lengths).
        self.balance_shards = bool(balance_shards)  # overrides the BaseTrainer default (False)
        # AIME-style periodic eval — avg@k accuracy on the eval prompt set
        # (run.eval_data_path), logged under eval/*. eval_interval=0 disables it.
        # ``eval_num_prompts`` sentinel:
        #   -1 (default, or any negative)  → full eval set
        #    0                             → yield nothing (explicit skip)
        #    N > 0                         → cap: score first N prompts
        # ``eval_batch_size`` (default 8) is the iteration batch size, decoupled
        # from the eval-set size (mirrors verl's ``data.val_batch_size``). Bounds
        # peak GPU memory during eval-time rollout.
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)  # for direct sampling
            else:
                self.rollout = remote(**rollout_parsed)  # for vllm / sglang

            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

    def _build_request_sample(
        self,
        inputs: RolloutInputs,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data source batch into a request :class:`Sample`.

        The ``P`` prompts become a root text input Part — ids rollout-keyed
        (``r{rollout_id}:…``) so each rollout is its own provenance — and
        ``Part.fork`` fans out the AR gen shell to the ``N``-sample GRPO group
        (replacing the old ``inputs.expand``; siblings stay consecutive). A VLM
        recipe chains the image input off the prompt via ``Part.input_child``.
        AR params ride on the gen Part — no SDE schedule to resolve (that is the
        diffusion trainer's job). ``sampling`` overrides the dict (``evaluate``
        passes its own); ``None`` uses ``self.sampling_params``.
        """
        sp = sampling if sampling is not None else self.sampling_params
        unsupported = set(inputs.primitives) - {"text", "image", "video"}
        if unsupported:
            raise ValueError(
                f"ARTrainer._build_request_sample: unsupported input primitive keys: {sorted(unsupported)}"
            )
        root_ids = [f"r{rollout_id}:{sid}" for sid in inputs.sample_ids]
        text = Part.input(
            root_ids,
            primitives={"text": inputs.primitives["text"]},
            metadata=list(inputs.metadata) if inputs.metadata else None,
        )
        input_parts = [text]
        parent = text
        for key in ("image", "video"):
            primitive = inputs.primitives.get(key)
            if primitive is not None:
                parent = parent.input_child(primitives={key: primitive})
                input_parts.append(parent)
        return Sample.request(*input_parts).fork(total_samples_per_prompt(sp), sampling_params=sp.get("ar"))

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the frontier gen Part (0.0 if none), for the log
        line. ``rollout_id`` only keys the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        sample = self.rollout.generate(sample)
        self.rollout.sleep()

        # Score the frontier gen Part (Sample -> Sample; the reward service is
        # migrated alongside on its own branch — see the LIN-480 plan).
        sample = self.reward.score_and_attach(sample)

        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorRef from the worker.
            part.rewards = hydrate(part.rewards)
            if isinstance(part.component_rewards, dict):
                part.component_rewards = {name: hydrate(value) for name, value in part.component_rewards.items()}
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
            part = part.compute_advantages(normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope)
            sample = sample.with_parts([*sample.parts[:-1], part])

        self._dump_rollout_samples(sample, rollout_id)
        self._drop_decoded(sample, rollout_id=rollout_id)
        train_part = sample.parts[-1]
        # verl balance_batch parity: reorder so each DP shard gets a near-equal
        # token load before DP_SCATTER (no-op when already balanced).
        if self.balance_shards:
            train_part = train_part.balance_shards(int(self.num_devices))
        result = self.stack.train_track(train_part, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            sample,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        return result, mean_reward

    def evaluate(self, rollout_id: int) -> float:
        """Periodic eval — ``avg@k`` accuracy on the eval prompt set.

        Mirrors :meth:`train_step`'s rollout+reward path but skips
        advantage/backward: iterate up to ``eval_num_prompts`` prompts from
        ``run.eval_data_path`` in ``eval_batch_size``-sized batches, expand
        each prompt to ``eval_samples_per_prompt`` siblings, generate at
        ``eval_temperature``, score, and log the mean reward under both
        ``eval/acc`` (= avg@k accuracy, since the MC reward is 0/1) and
        ``eval/reward`` (shares the eval axis with the other trainers).
        Returns it.

        ``eval_num_prompts=-1`` (default) evaluates the full eval set;
        ``eval_num_prompts=0`` yields no batches (explicit skip). See the
        sentinel table on :meth:`~unirl.data.data_source.MultimodalRLDataSource.iter_eval_batches`.
        """
        import dataclasses

        eval_ar = dataclasses.replace(
            self.sampling_params.get("ar"),
            samples_per_prompt=self.eval_samples_per_prompt,
            temperature=self.eval_temperature,
        )
        eval_sp = {**self.sampling_params, "ar": eval_ar}
        eval_batches = self.data_source.iter_eval_batches(
            self.eval_batch_size,
            eval_num_prompts=self.eval_num_prompts,
        )
        reward_sum, reward_n, prompt_n, batch_n = 0.0, 0, 0, 0

        self.rollout.wake_up()
        try:
            if self.weight_sync is not None:
                self.weight_sync.sync()
            for eval_inputs in eval_batches:
                real_prompt_n = len(eval_inputs.sample_ids)
                batch_n += 1
                prompt_n += real_prompt_n
                dispatch_inputs = self._pad_eval_inputs(eval_inputs)
                sample = self._build_request_sample(dispatch_inputs, rollout_id, sampling=eval_sp)
                generated = self.rollout.generate(sample)
                scored = self.reward.score_and_attach(generated)
                rewards = scored.parts[-1].rewards
                if rewards is not None:
                    rewards = hydrate(rewards).to(torch.float32)
                    fanout = total_samples_per_prompt(eval_sp)
                    expected_total = len(dispatch_inputs.sample_ids) * fanout
                    if int(rewards.numel()) != expected_total:
                        raise RuntimeError(
                            f"ARTrainer.evaluate: reward count {int(rewards.numel())} != "
                            f"dispatch prompts {len(dispatch_inputs.sample_ids)} * fanout {fanout} "
                            f"({expected_total})."
                        )
                    rewards = rewards[: real_prompt_n * fanout]
                    reward_sum += float(rewards.sum().item())
                    reward_n += int(rewards.numel())
        finally:
            self.rollout.sleep()

        acc = reward_sum / max(1, reward_n)
        logger.info(
            "EVAL rollout %d  eval_acc(avg@%d over %d prompts, %d batches of <=%d)=%.4f",
            rollout_id + 1,
            self.eval_samples_per_prompt,
            prompt_n,
            batch_n,
            self.eval_batch_size,
            acc,
        )
        # MC reward is 0/1 so mean reward == accuracy; also emit it as `reward`
        # so this run shares the eval/reward axis with the other trainers.
        self.wandb_logger.log_eval(rollout_id + 1, {"acc": acc, "reward": acc})
        return acc

    def _pad_eval_inputs(self, inputs: RolloutInputs) -> RolloutInputs:
        """Append replicated prompt rows until rollout and reward DP can shard.

        Evaluation still reports only the original rows; the replicas exist solely
        to satisfy ``DP_SCATTER``. Their ids are rewritten because Sample lineage
        requires distinct root ids within one request.
        """
        n = len(inputs.sample_ids)
        if n == 0:
            return inputs
        rollout_dp = max(1, int(getattr(self.rollout, "dp_size", 1)))
        reward_dp = max(1, int(getattr(self.reward, "dp_size", 1)))
        multiple = math.lcm(rollout_dp, reward_dp)
        pad_n = (-n) % multiple
        if pad_n == 0:
            return inputs

        pad = inputs.select(torch.full((pad_n,), n - 1, dtype=torch.long))
        used_sample_ids = set(inputs.sample_ids)
        padded_sample_ids = []
        for i in range(pad_n):
            candidate = f"{inputs.sample_ids[-1]}:eval-pad:{i}"
            while candidate in used_sample_ids:
                candidate += ":pad"
            used_sample_ids.add(candidate)
            padded_sample_ids.append(candidate)
        pad.sample_ids = padded_sample_ids

        if inputs.group_ids:
            used_group_ids = set(inputs.group_ids)
            padded_group_ids = []
            for i in range(pad_n):
                candidate = f"{inputs.group_ids[-1]}:eval-pad:{i}"
                while candidate in used_group_ids:
                    candidate += ":pad"
                used_group_ids.add(candidate)
                padded_group_ids.append(candidate)
            pad.group_ids = padded_group_ids
        return inputs.concat_with(pad)

    def _dump_rollout_samples(self, sample, rollout_id: int) -> None:
        """Debug dump of the first N (prompt, output, reward) triples per rollout.

        Off unless ``ROLLOUT_DUMP_DIR`` is set (driver-side env). Writes one
        ``rollout_<id>.jsonl`` per rollout (``ROLLOUT_DUMP_N`` samples, default
        4) so rollout-engine quality can be eyeballed without keeping the full
        decoded batch alive. Must run BEFORE ``_drop_decoded``. Never raises.
        (Ported from the b182a511 LIN-371 lineage — lost in the rebase.)
        """
        import json
        import os

        out_dir = os.environ.get("ROLLOUT_DUMP_DIR", "")
        if not out_dir:
            return
        try:
            from unirl.types.primitives import Texts

            n = int(os.environ.get("ROLLOUT_DUMP_N", "4"))
            # Prompts row-aligned to the frontier samples (the lineage walk
            # expands the P prompts to the P*N gen samples).
            cond = sample.conditioning()
            prompts = next((list(c.texts) for c in cond if isinstance(c, Texts)), [])
            part = sample.parts[-1]
            outputs = getattr(part.primitives.get("text"), "texts", None) or []
            rewards = part.rewards.to(torch.float32).tolist() if part.rewards is not None else []
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"rollout_{int(rollout_id):04d}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(min(n, len(outputs))):
                    f.write(
                        json.dumps(
                            {
                                "rollout": int(rollout_id),
                                "sample": i,
                                "prompt": prompts[i] if i < len(prompts) else None,
                                "output": outputs[i],
                                "output_chars": len(outputs[i] or ""),
                                "reward": rewards[i] if i < len(rewards) else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as exc:  # debug path — never let it kill training
            logger.warning("rollout sample dump failed: %s", exc)

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

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` is the output folder (defaults to
        ``./checkpoints``); ``save_mode="auto"`` writes LoRA-only checkpoints
        when LoRA is active and full checkpoints otherwise.
        ``load_dir``: restore from a checkpoint directory and RESUME from its
        saved step — ``num_rollouts`` is the TOTAL budget.

        Deferred: ``num_updates_per_batch`` multi-epoch replay, eval cadence.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope},
        )
        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)  # baseline AIME accuracy, logged at eval step 0
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                result, mean_reward = self.train_step(
                    sample,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id=rollout_id)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
