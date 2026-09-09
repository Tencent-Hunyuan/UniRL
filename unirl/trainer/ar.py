import importlib
import importlib.util
import inspect
import logging
import math
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, Iterator, Optional, Set, Tuple

import torch
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.hydra import parse_hydra_cfg, remote_hydra
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.graceful_shutdown import run_with_timeout

logger = logging.getLogger(__name__)

_ROLLOUT_SHUTDOWN_TIMEOUT_S = 60.0


def ar_preflight(
    *,
    pipeline_cfg: DictConfig,
    backend_cfg: DictConfig,
    rollout_cfg: DictConfig,
    stack_cfg: DictConfig,
) -> Set[str]:
    """Model-blind pre-flight shared by :class:`ARTrainer` and ``AsyncARTrainer``."""
    target = str(pipeline_cfg.get("_target_", "") or "")
    parts = target.split(".")
    if target.startswith("unirl.models.") and len(parts) > 3:
        validation_module = f"unirl.models.{parts[2]}.validation"
        if importlib.util.find_spec(validation_module) is not None:
            validate = getattr(importlib.import_module(validation_module), "validate_training_contract", None)
            if validate is not None:
                validate(
                    pipeline_cfg=pipeline_cfg,
                    backend_cfg=backend_cfg,
                    rollout_cfg=rollout_cfg,
                    stack_cfg=stack_cfg,
                )
    allowed: Set[str] = {"text", "image", "video"}
    if target:
        resolved = get_object(target)
        pipeline_cls = resolved if isinstance(resolved, type) else getattr(resolved, "__self__", None)
        allowed.update(getattr(pipeline_cls, "extra_input_primitives", ()))
    return allowed


class ARTrainer(BaseTrainer):
    """Autoregressive (VLM / LLM) RL trainer: rollout + train colocated."""

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
        advantage_mode: str = "grpo",
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        rollout_anchor_device: Optional[int] = None,
        enable_fsdp_offload: bool = True,
    ) -> None:
        self._allowed_input_primitives = ar_preflight(
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            rollout_cfg=rollout_cfg,
            stack_cfg=stack_cfg,
        )
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.advantage_mode = str(advantage_mode).strip().lower()
        if self.advantage_mode not in ("grpo", "gae"):
            raise ValueError(f"ARTrainer: advantage_mode must be 'grpo' or 'gae', got {advantage_mode!r}")
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)

        self._rollout_anchor_device: Optional[int] = (
            int(rollout_anchor_device) if rollout_anchor_device is not None else None
        )
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._anchored_backend_offloaded: Optional[bool] = False
        self._anchored_rollout_awake: Optional[bool] = None

        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        self.weight_sync = None
        self._supports_staged_wake = False

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)

            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if self._rollout_anchor_device is None:
                self._supports_staged_wake = "tags" in inspect.signature(rollout_parsed["role_cls"].wake_up).parameters
                bootstrap_offload = sync_cfg is not None and self._enable_fsdp_offload
                bootstrap_offloaded = False
                rollout_boot_started = False
                rollout_constructed = False
                rollout_sleep_attempted = False
                rollout_memory_released = False
                try:
                    if bootstrap_offload:
                        try:
                            self.backend.offload()
                            bootstrap_offloaded = True
                        except BaseException:
                            self.backend.onload()
                            raise

                    rollout_boot_started = True
                    if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                        self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)
                    else:
                        self.rollout = remote(**rollout_parsed)
                    rollout_constructed = True
                    if sync_cfg is not None:
                        self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

                    if self.weight_sync is not None:
                        rollout_sleep_attempted = True
                        self.rollout.sleep()
                        rollout_memory_released = True
                    if bootstrap_offloaded:
                        self.backend.onload()
                except BaseException:
                    if bootstrap_offloaded and not rollout_boot_started:
                        self.backend.onload()
                    elif bootstrap_offloaded and rollout_constructed and not rollout_sleep_attempted:
                        try:
                            self.rollout.sleep()
                        except BaseException:
                            logger.exception("Failed to release rollout memory after bootstrap failure")
                        else:
                            self.backend.onload()
                    elif bootstrap_offloaded and rollout_memory_released:
                        pass
                    raise
            else:
                if sync_cfg is not None and self._rollout_anchor_device == 0:
                    raise ValueError(
                        "rollout_anchor_device=0 would colocate the TP engine with "
                        "the rank-0 RemoteLoraWeightSync sender and self-deadlock; "
                        "use a nonzero anchor device."
                    )
                if self._enable_fsdp_offload:
                    self._anchored_backend_offloaded = None
                    self.backend.offload()
                    self._anchored_backend_offloaded = True

                role_cls = rollout_parsed.pop("role_cls")
                self.rollout = self.pool.create_remote(
                    role_cls,
                    device_ids=[self._rollout_anchor_device],
                    init_kwargs=rollout_parsed,
                )
                self._anchored_rollout_awake = None
                if self._enable_fsdp_offload:
                    self.rollout.sleep()
                    self._anchored_rollout_awake = False
                else:
                    self._anchored_rollout_awake = True

                if sync_cfg is not None:
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
                    self.weight_sync.set_rollout_targets([(self.rollout.role_name, self.rollout.workers)])

    def _ensure_anchored_backend_loaded(self) -> None:
        if not self._enable_fsdp_offload or self._anchored_backend_offloaded is False:
            return
        self._anchored_backend_offloaded = None
        self.backend.onload()
        self._anchored_backend_offloaded = False

    def _ensure_anchored_backend_offloaded(self) -> None:
        if not self._enable_fsdp_offload or self._anchored_backend_offloaded is True:
            return
        self._anchored_backend_offloaded = None
        self.backend.offload()
        self._anchored_backend_offloaded = True

    def _ensure_anchored_rollout_awake(self) -> None:
        if not self._enable_fsdp_offload:
            return
        if self._anchored_rollout_awake is True:
            return
        self._anchored_rollout_awake = None
        self.rollout.wake_up()
        self._anchored_rollout_awake = True

    def _ensure_anchored_rollout_asleep(self) -> None:
        if not self._enable_fsdp_offload:
            return  # resident colocate: never sleep the engine
        if self._anchored_rollout_awake is False:
            return
        self._anchored_rollout_awake = None
        self.rollout.sleep()
        self._anchored_rollout_awake = False

    @contextmanager
    def _anchored_rollout_session(
        self,
        *,
        sync_weights: bool,
        restore_backend: bool = True,
    ) -> Iterator[None]:
        """Run one anchored rollout phase and restore the requested steady state."""
        original_error: Optional[BaseException] = None
        try:
            if sync_weights and self.weight_sync is not None:
                self._ensure_anchored_backend_loaded()
                self.weight_sync.extract()
            self._ensure_anchored_backend_offloaded()
            self._ensure_anchored_rollout_awake()
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.push()
            yield
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            cleanup_errors: list[tuple[str, BaseException]] = []
            cleanup_ops = [("sleep rollout", self._ensure_anchored_rollout_asleep)]
            if restore_backend:
                cleanup_ops.append(("onload backend", self._ensure_anchored_backend_loaded))
            for operation, cleanup in cleanup_ops:
                try:
                    cleanup()
                except BaseException as exc:
                    cleanup_errors.append((operation, exc))

            if cleanup_errors:
                if original_error is not None:
                    for operation, cleanup_error in cleanup_errors:
                        original_error.add_note(
                            f"anchored cleanup failed while trying to {operation}: {cleanup_error!r}"
                        )
                        logger.error(
                            "Anchored cleanup failed while trying to %s",
                            operation,
                            exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
                        )
                else:
                    operation, cleanup_error = cleanup_errors[0]
                    for later_operation, later_error in cleanup_errors[1:]:
                        cleanup_error.add_note(
                            f"additional anchored cleanup failure while trying to {later_operation}: {later_error!r}"
                        )
                    cleanup_error.add_note(f"anchored cleanup operation: {operation}")
                    raise cleanup_error

    def _prepare_rollout(self, *, sync_weights: bool) -> bool:
        """Wake/sync the SPMD rollout and optionally offload train state."""
        do_offload = self._enable_fsdp_offload and self.weight_sync is not None
        do_sync = sync_weights and self.weight_sync is not None
        train_state_maybe_offloaded = False
        full_wake_after_train_offload_in_progress = False

        try:
            if do_sync and do_offload and self._supports_staged_wake:
                self.rollout.wake_up(tags=["weights"])
                self.weight_sync.sync()
                train_state_maybe_offloaded = True
                self.backend.offload()
                full_wake_after_train_offload_in_progress = True
                self.rollout.wake_up()
                full_wake_after_train_offload_in_progress = False
            elif do_sync:
                self.rollout.wake_up()
                self.weight_sync.sync()
                if do_offload:
                    train_state_maybe_offloaded = True
                    self.backend.offload()
            elif do_offload:
                train_state_maybe_offloaded = True
                self.backend.offload()
                full_wake_after_train_offload_in_progress = True
                self.rollout.wake_up()
                full_wake_after_train_offload_in_progress = False
            else:
                self.rollout.wake_up()
        except BaseException:
            if full_wake_after_train_offload_in_progress:
                raise
            self._finish_rollout(train_state_offloaded=train_state_maybe_offloaded)
            raise

        return train_state_maybe_offloaded

    def _finish_rollout(self, *, train_state_offloaded: bool) -> None:
        """Sleep rollout before restoring the colocated training state."""
        # Keep training state offloaded if engine sleep fails.
        self.rollout.sleep()
        if train_state_offloaded:
            self.backend.onload()

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data source batch into a request :class:`Sample`."""
        sp = sampling if sampling is not None else self.sampling_params
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives=self._allowed_input_primitives,
            caller="ARTrainer._build_request_sample",
        )
        return request.fork(total_samples_per_prompt(sp), sampling_params=sp.get("ar"))

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass."""
        t0 = time.perf_counter()
        anchored = self._rollout_anchor_device is not None
        if not anchored:
            train_state_offloaded = self._prepare_rollout(sync_weights=sync_weights)
            try:
                sample = self.rollout.generate(sample)
            finally:
                self._finish_rollout(train_state_offloaded=train_state_offloaded)
        else:
            with self._anchored_rollout_session(sync_weights=sync_weights, restore_backend=False):
                sample = self.rollout.generate(sample)
                from unirl.trainer.unified_model import deep_hydrate

                sample = deep_hydrate(sample)

        sample = self.reward.score_and_attach(sample)

        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            if isinstance(part.component_rewards, dict):
                part.component_rewards = {name: hydrate(value) for name, value in part.component_rewards.items()}
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
            if self.advantage_mode == "grpo":
                part = part.compute_advantages(
                    normalize=self.normalize_adv_by_std,
                    scope=self.adv_normalization_scope,
                )
            sample = sample.with_parts([*sample.parts[:-1], part])

        self._dump_rollout_samples(sample, rollout_id)
        self._drop_decoded(sample, rollout_id=rollout_id)
        train_part = sample.parts[-1]
        if self.balance_shards:
            train_part = train_part.balance_shards(int(self.num_devices))
        if anchored:
            self._ensure_anchored_backend_loaded()
        try:
            result = self.stack.train_track(train_part, training_progress=float(training_progress))
        finally:
            if anchored:
                self._ensure_anchored_backend_offloaded()
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            sample,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        return result, mean_reward

    def evaluate(self, rollout_id: int) -> float:
        """Periodic eval — ``avg@k`` accuracy on the eval prompt set."""
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

        anchored = self._rollout_anchor_device is not None

        logger.info(
            "EVAL rollout %d starting: max_prompts=%s batch_size=%d samples_per_prompt=%d temperature=%s anchored=%s",
            rollout_id + 1,
            "all" if self.eval_num_prompts < 0 else self.eval_num_prompts,
            self.eval_batch_size,
            self.eval_samples_per_prompt,
            self.eval_temperature,
            anchored,
        )
        train_state_offloaded = False
        if not anchored:
            train_state_offloaded = self._prepare_rollout(sync_weights=self.weight_sync is not None)
        eval_session = (
            self._anchored_rollout_session(
                sync_weights=self.weight_sync is not None,
                restore_backend=False,
            )
            if anchored
            else nullcontext()
        )
        try:
            with eval_session:
                for eval_inputs in eval_batches:
                    batch_n += 1
                    real_prompt_n = eval_inputs.batch_size
                    prompt_n += real_prompt_n
                    dispatch_inputs = self._pad_eval_inputs(eval_inputs)
                    sample = self._build_request_sample(dispatch_inputs, rollout_id, sampling=eval_sp)
                    if anchored:
                        generated = self.rollout.generate(sample)
                        from unirl.trainer.unified_model import deep_hydrate

                        generated = deep_hydrate(generated)
                    else:
                        generated = self.rollout.generate(sample)
                    scored = self.reward.score_and_attach(generated)
                    rewards = scored.parts[-1].rewards
                    if rewards is not None:
                        rewards = hydrate(rewards).to(torch.float32)
                        fanout = total_samples_per_prompt(eval_sp)
                        expected_total = dispatch_inputs.batch_size * fanout
                        if int(rewards.numel()) != expected_total:
                            raise RuntimeError(
                                f"ARTrainer.evaluate: reward count {int(rewards.numel())} != "
                                f"dispatch prompts {dispatch_inputs.batch_size} * fanout {fanout} "
                                f"({expected_total})."
                            )
                        rewards = rewards[: real_prompt_n * fanout]
                        reward_sum += float(rewards.sum().item())
                        reward_n += int(rewards.numel())
                    logger.info(
                        "EVAL rollout %d progress: batch=%d prompts=%d samples=%d",
                        rollout_id + 1,
                        batch_n,
                        prompt_n,
                        reward_n,
                    )
        finally:
            if not anchored:
                self._finish_rollout(train_state_offloaded=train_state_offloaded)

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
        self.wandb_logger.log_eval(rollout_id + 1, {"acc": acc, "reward": acc})
        return acc

    def _pad_eval_inputs(self, inputs: Sample) -> Sample:
        """Append replicated prompt rows until rollout and reward DP can shard."""
        n = inputs.batch_size
        if n == 0:
            return inputs
        rollout_dp = max(1, int(getattr(self.rollout, "dp_size", 1)))
        reward_dp = max(1, int(getattr(self.reward, "dp_size", 1)))
        multiple = math.lcm(rollout_dp, reward_dp)
        pad_n = (-n) % multiple
        if pad_n == 0:
            return inputs

        source = inputs.slice(n - 1, n)
        source_root_id = source.parts[0].sample_ids[0]
        used_root_ids = set(inputs.parts[0].sample_ids)
        padded: list[Sample] = []
        for i in range(pad_n):
            candidate = f"{source_root_id}:eval-pad:{i}"
            while candidate in used_root_ids:
                candidate += ":pad"
            used_root_ids.add(candidate)

            def replace_root(sample_id: str, *, new_root: str = candidate) -> str:
                root, separator, suffix = sample_id.partition("/")
                if root != source_root_id:
                    raise ValueError(
                        "ARTrainer._pad_eval_inputs: selected pad tree contains "
                        f"unexpected root {root!r}; expected {source_root_id!r}."
                    )
                return new_root + (f"/{suffix}" if separator else "")

            padded.append(source.map_sample_ids(replace_root))
        return Sample.concat([inputs, *padded])

    def _dump_rollout_samples(self, sample, rollout_id: int) -> None:
        """Debug dump of the first N (prompt, output, reward) triples per rollout."""
        import json
        import os

        out_dir = os.environ.get("ROLLOUT_DUMP_DIR", "")
        if not out_dir:
            return
        try:
            from unirl.types.primitives import Texts

            n = int(os.environ.get("ROLLOUT_DUMP_N", "4"))
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
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``."""
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope},
        )
        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
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
            try:
                self._finish_wandb()
            finally:
                self._shutdown_runtime()

    def shutdown(self) -> None:
        """Release every runtime resource this trainer owns. Idempotent."""
        self._shutdown_runtime()

    def _shutdown_runtime(self) -> None:
        """Best-effort ordered teardown for rollout children and Ray actors."""
        if getattr(self, "_runtime_shutdown_done", False):
            return
        self._runtime_shutdown_done = True

        rollout = getattr(self, "rollout", None)
        shutdown = getattr(rollout, "shutdown", None)
        if callable(shutdown):
            run_with_timeout(shutdown, timeout=_ROLLOUT_SHUTDOWN_TIMEOUT_S, what="AR rollout engine shutdown")

        pool = getattr(self, "pool", None)
        if pool is not None:
            try:
                pool.shutdown()
            except Exception:
                logger.exception("Failed to shut down AR trainer device pool")
