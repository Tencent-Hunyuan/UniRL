"""UniRL v2 HunyuanImage3 unified-backbone trainer."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import TensorRef, hydrate
from unirl.distributed.tensor.batch import Batch
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import build_eval_suites
from unirl.trainer.hydra import parse_hydra_cfg, remote_hydra
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, DiffusionSamplingParams

logger = logging.getLogger(__name__)


def deep_hydrate(obj: Any) -> Any:
    """Materialize every ``TensorRef`` leaf in ``obj`` to a real tensor, in place."""
    if isinstance(obj, TensorRef):
        return hydrate(obj)
    if isinstance(obj, Batch):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is not None:
                new = deep_hydrate(v)
                if new is not v:
                    setattr(obj, f.name, new)
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = deep_hydrate(obj[k])
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = deep_hydrate(obj[i])
        return obj
    if isinstance(obj, tuple):
        return tuple(deep_hydrate(x) for x in obj)
    return obj


class UnifiedModelTrainer(BaseTrainer):
    """HunyuanImage3 unified-backbone joint trainer (AR + DiT, one LoRA)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        control: Optional[Dict[str, Any]] = None,
        ar_rollout_cfg: Optional[DictConfig] = None,
        dit_rollout_cfg: Optional[DictConfig] = None,
        rollout_cfg: Optional[DictConfig] = None,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
        rollout_pipeline_chunks: int,
        eval_interval: int = 0,
        eval_num_prompts: int = 32,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
    ) -> None:
        if batch_size < 1 or batch_size % cfg.num_devices != 0:
            raise ValueError(
                f"UnifiedModelTrainer: batch_size ({batch_size}) must be a positive multiple of "
                f"num_devices ({cfg.num_devices}); reward scoring and unified training scatter whole prompt trees."
            )
        if rollout_pipeline_chunks < 1:
            raise ValueError(
                f"UnifiedModelTrainer.rollout_pipeline_chunks must be >= 1; got {rollout_pipeline_chunks}."
            )
        self._rollout_pipeline_chunks = rollout_pipeline_chunks
        single_engine = rollout_cfg is not None
        if single_engine:
            if ar_rollout_cfg is not None or dit_rollout_cfg is not None:
                raise ValueError(
                    "UnifiedModelTrainer: pass either rollout_cfg (single-engine mode) or "
                    "ar_rollout_cfg + dit_rollout_cfg (two-engine mode), not both."
                )
            if rollout_pipeline_chunks > 1:
                raise ValueError(
                    "UnifiedModelTrainer.rollout_pipeline_chunks > 1 requires the two-engine "
                    "ar_rollout_cfg/dit_rollout_cfg rollout; the single-engine rollout_cfg path does not pipeline."
                )
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" not in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError(
                    "UnifiedModelTrainer: single-engine mode supports only the trainside direct-sampling engine; "
                    "use ar_rollout + dit_rollout for dedicated rollout engines."
                )
            if sync_cfg is not None:
                raise ValueError(
                    "UnifiedModelTrainer: single-engine mode samples on the train actors and cannot use a `sync:` block."
                )
        else:
            rollout_parsed = None
            if ar_rollout_cfg is None or dit_rollout_cfg is None:
                raise ValueError(
                    "UnifiedModelTrainer: two-engine mode needs ar_rollout_cfg + dit_rollout_cfg; "
                    "pass a single rollout_cfg for single-engine (M=1 / UniGRPO) mode."
                )
            if sync_cfg is None:
                raise ValueError("UnifiedModelTrainer: two-engine dedicated rollout requires a `sync:` block.")
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._enable_fsdp_offload = enable_fsdp_offload and not single_engine

        self.eval_interval = eval_interval
        self.eval_num_prompts = eval_num_prompts
        self.eval_cfg_text_scale = eval_cfg_text_scale
        self.eval_eta = eval_eta

        self.dump_dir = dump_dir or None
        self._dump_rollout_id = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self._control: Dict[str, Any] = dict(control) if control else {}

        self.weight_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )

            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)

            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )

            diffusion_params = self.sampling_params["diffusion"]
            self._shared_advantage = diffusion_params.samples_per_prompt == 1
            if single_engine:
                self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)
                return

            # Offload the frozen base before engine boot to avoid colocated OOMs.
            if self._enable_fsdp_offload:
                self.backend.offload()

            # Anchor AR and DiT engines on distinct workers to prevent GPU overlap.
            per_node = self.pool.devices_per_node
            # Require at least eight GPUs per node so engine pairs do not split across nodes.
            if per_node < 8:
                raise ValueError(
                    "UnifiedModelTrainer: HI3 needs >= 8 devices/node for one "
                    "(AR 0-3, DiT 4-7) engine pair per node; got "
                    f"devices_per_node={per_node}."
                )
            self.dp = max(1, self.pool.num_devices // per_node)
            self.ar_rollouts = []
            self.dit_rollouts = []
            for r in range(self.dp):
                base = r * per_node
                # Boot engines serially and off rank 0 to avoid warmup and LoRA-sync deadlocks.
                ar = self._wire_engine(ar_rollout_cfg, anchor_device=base + 1)
                ar.sleep()
                self.ar_rollouts.append(ar)
                dit = self._wire_engine(dit_rollout_cfg, anchor_device=base + 4)
                dit.sleep()
                self.dit_rollouts.append(dit)

            self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            self.weight_sync.set_rollout_targets(
                [(eng.role_name, eng.workers) for eng in self.ar_rollouts + self.dit_rollouts]
            )

    def _wire_engine(self, cfg: DictConfig, *, anchor_device: int) -> Any:
        """Build ONE multi-GPU vLLM-Omni engine actor anchored on one worker."""
        parsed = parse_hydra_cfg(cfg)
        role_cls = parsed.pop("role_cls")
        return self.pool.create_remote(role_cls, device_ids=[anchor_device], init_kwargs=parsed)

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data-source batch of ``P`` prompts into the unified request ``Sample``."""
        base = sampling if sampling is not None else self.sampling_params
        diff_params = base["diffusion"]
        ar_params = base["ar"]
        sde_indices = diff_params.get_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="UnifiedModelTrainer._build_request_sample",
            control=self._control,
            require_single_input_part=True,
        )
        return request.fork(ar_params.samples_per_prompt, sampling_params=ar_params).fork(
            diffusion.samples_per_prompt, sampling_params=diffusion
        )

    def run_rollout(self, sample: Sample) -> Sample:
        """DP rollout: scatter the ``P`` prompt-trees of the request ``Sample``"""
        valid_layout = (
            len(sample.parts) == 3
            and sample.parts[0].is_root
            and isinstance(sample.parts[1].sampling_params, ARSamplingParams)
            and isinstance(sample.parts[2].sampling_params, DiffusionSamplingParams)
        )
        if not valid_layout:
            raise ValueError(
                "UnifiedModelTrainer.run_rollout requires exactly "
                f"[input, ar_shell, image_shell]; got {len(sample.parts)} Parts with sampling types "
                f"{[type(part.sampling_params).__name__ for part in sample.parts]}."
            )
        if self.weight_sync is None:
            return self.rollout.generate(sample)

        n = sample.parts[0].batch_size
        if self.dp <= 1 or n <= 1:
            return self._run_rollout_one(self.ar_rollouts[0], self.dit_rollouts[0], sample)

        groups = sample.split()
        bounds = [(n * r) // self.dp for r in range(self.dp + 1)]
        shards: list[Sample] = []
        for r in range(self.dp):
            lo, hi = bounds[r], bounds[r + 1]
            if lo >= hi:
                continue
            sub = Sample.concat(groups[lo:hi])
            shards.append(self._run_rollout_one(self.ar_rollouts[r], self.dit_rollouts[r], sub))
        return Sample.concat(shards)

    def _run_rollout_one(self, ar_engine: Any, dit_engine: Any, sample: Sample) -> Sample:
        """One (AR, DiT) engine pair: fill the unified ``[input, ar, image]`` lineage."""
        root_part, ar_shell, image_shell = sample.parts
        if not isinstance(root_part.primitives.get("text"), Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: input Part must contain a 'text' Texts primitive.")
        n_rec = ar_shell.sampling_params.samples_per_prompt
        n_img = image_shell.sampling_params.samples_per_prompt
        rid = self._dump_rollout_id
        n = root_part.batch_size
        n_chunks = min(self._rollout_pipeline_chunks, n)
        if n_chunks <= 1:
            chunks = [sample]
        else:
            groups = sample.split()
            bounds = [(n * i) // n_chunks for i in range(n_chunks + 1)]
            chunks = [Sample.concat(groups[lo:hi]) for lo, hi in zip(bounds, bounds[1:])]

        ar_requests: List[Sample] = []
        for chunk in chunks:
            input_part = chunk.parts[0]
            prompts = input_part.primitives["text"]
            ar_texts = Texts(texts=[t for t in prompts.texts for _ in range(n_rec)])
            ar_input = Part.input(
                [f"r{rid}:a{k}" for k in range(len(ar_texts.texts))],
                primitives={"text": ar_texts},
                control=input_part.control,
            )
            ar_requests.append(
                Sample.request(ar_input)
                .fork(1, sampling_params=image_shell.sampling_params)
                .fork(1, sampling_params=ar_shell.sampling_params)
            )

        pending_ar = None
        pending_dit = None
        ar_parts: List[Part] = []
        dit_outs: List[Sample] = []
        try:
            pending_ar = ar_engine.launch_nowait("generate", ar_requests[0])
            for i, chunk in enumerate(chunks):
                ar_out = pending_ar.result()
                pending_ar = None
                if i + 1 < len(chunks):
                    pending_ar = ar_engine.launch_nowait("generate", ar_requests[i + 1])

                input_part, chunk_ar_shell, chunk_image_shell = chunk.parts
                prompts = input_part.primitives["text"]
                n_ar = len(prompts.texts) * n_rec
                ar_gen = ar_out.parts[-1]
                recaptions = ar_gen.primitives.get("text")
                if not isinstance(recaptions, Texts) or len(recaptions.texts) != n_ar:
                    got = len(recaptions.texts) if isinstance(recaptions, Texts) else type(recaptions).__name__
                    raise RuntimeError(
                        f"UnifiedModelTrainer.run_rollout: AR engine must return {n_ar} decoded Texts (= P*N); got {got}."
                    )
                ar_parts.append(
                    chunk_ar_shell.fill(
                        segment=ar_gen.segment,
                        primitives={"text": recaptions},
                        conditions=dict(ar_gen.conditions),
                        output_version=ar_gen.output_version,
                    )
                )
                dit_prompts = Texts(texts=[prompts.texts[j // n_rec] for j in range(n_ar) for _ in range(n_img)])
                dit_cot = Texts(texts=[recaptions.texts[j] for j in range(n_ar) for _ in range(n_img)])
                dit_input = Part.input(
                    [sid.replace("/", "_") for sid in chunk_image_shell.sample_ids],
                    primitives={"text": dit_prompts},
                    control=input_part.control,
                )
                cot_input = dit_input.input_child(primitives={"text": dit_cot})
                dit_request = Sample.request(dit_input, cot_input).fork(
                    1, sampling_params=chunk_image_shell.sampling_params
                )
                if pending_dit is not None:
                    dit_outs.append(pending_dit.result())
                    pending_dit = None
                pending_dit = dit_engine.launch_nowait("generate", dit_request)

            dit_outs.append(pending_dit.result())
            pending_dit = None
        except BaseException:
            for pending in (pending_ar, pending_dit):
                if pending is not None:
                    try:
                        pending.wait()
                    except Exception:
                        logger.debug("UnifiedModelTrainer: pending rollout call failed while draining", exc_info=True)
            raise

        filled: List[Sample] = []
        for chunk, ar_part, dit_out in zip(chunks, ar_parts, dit_outs):
            input_part, _, chunk_image_shell = chunk.parts
            img_gen = dit_out.parts[-1]
            if len(img_gen.sample_ids) != len(chunk_image_shell.sample_ids):
                raise RuntimeError(
                    f"UnifiedModelTrainer.run_rollout: DiT engine returned {len(img_gen.sample_ids)} image(s) "
                    f"but the image shell expects {len(chunk_image_shell.sample_ids)} (= P*N*M). "
                    "The DiT engine must be 1:1."
                )
            image_part = chunk_image_shell.fill(
                segment=img_gen.segment,
                primitives=dict(img_gen.primitives),
                primitive_metadata=dict(img_gen.primitive_metadata),
                conditions=dict(img_gen.conditions),
                media_preview=img_gen.media_preview,
                output_version=img_gen.output_version,
            )
            filled.append(Sample(parts=[input_part, ar_part, image_part]))
        return Sample.concat(filled)

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass."""
        t0 = time.perf_counter()
        if self.weight_sync is None:
            sample = self.run_rollout(sample)
        else:
            if sync_weights:
                if self._enable_fsdp_offload:
                    self.backend.onload()
                self.weight_sync.extract()
                if self._enable_fsdp_offload:
                    self.backend.offload()
            for _eng in self.ar_rollouts + self.dit_rollouts:
                _eng.wake_up()
            try:
                if sync_weights:
                    self.weight_sync.push()
                sample = self.run_rollout(sample)
            finally:
                for _eng in self.ar_rollouts + self.dit_rollouts:
                    _eng.sleep()
                if self._enable_fsdp_offload:
                    self.backend.onload()

        ar_idx = sample.gen_part_index(ARSamplingParams)
        img_idx = sample.gen_part_index(DiffusionSamplingParams)

        sample = self.reward.score_and_attach(sample)
        img_part = sample.parts[img_idx]
        if img_part.rewards is not None:
            img_part.rewards = hydrate(img_part.rewards)
        if isinstance(img_part.component_rewards, dict):
            img_part.component_rewards = {name: hydrate(value) for name, value in img_part.component_rewards.items()}

        sample = sample.propagate_rewards(op="mean")

        mean_reward = 0.0
        di_rewards = sample.parts[img_idx].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        if self.dump_dir:
            self._dump_rollout(self._dump_rollout_id, sample)

        new_parts = list(sample.parts)
        new_parts[ar_idx] = new_parts[ar_idx].compute_advantages(normalize=True)
        if self._shared_advantage:
            if new_parts[img_idx].batch_size != new_parts[ar_idx].batch_size:
                raise ValueError(
                    "UnifiedModelTrainer: shared M=1 advantage requires equal AR/image batch sizes; "
                    f"got {new_parts[ar_idx].batch_size} and {new_parts[img_idx].batch_size}."
                )
            new_parts[img_idx] = dataclasses.replace(new_parts[img_idx], advantages=new_parts[ar_idx].advantages)
        else:
            new_parts[img_idx] = new_parts[img_idx].compute_advantages(normalize=True)
        sample = sample.with_parts(new_parts)

        self._drop_decoded(sample, rollout_id=rollout_id)
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            sample,
            training_progress=float(training_progress),
            rollout_id=rollout_id,
        )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            results,
            sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={"sync_weights": float(bool(sync_weights))},
        )

        if self._enable_fsdp_offload:
            self.backend.offload()
        return results, mean_reward

    def _dump_rollout(self, rollout_id: int, sample: Any) -> None:
        """Best-effort intrusive dump of one rollout to ``self.dump_dir``."""
        try:
            out_dir = os.path.join(self.dump_dir, f"rollout_{rollout_id}")
            os.makedirs(out_dir, exist_ok=True)

            prompts_obj = sample.parts[0].primitives.get("text")
            prompts = list(prompts_obj.texts) if isinstance(prompts_obj, Texts) else []

            ar_part = next((p for p in sample.parts[1:] if isinstance(p.sampling_params, ARSamplingParams)), None)
            ar_decoded = ar_part.primitives.get("text") if ar_part is not None else None
            ar_texts = list(ar_decoded.texts) if isinstance(ar_decoded, Texts) else []

            image_part = next(
                (p for p in sample.parts[1:] if isinstance(p.sampling_params, DiffusionSamplingParams)), None
            )
            img_decoded = image_part.primitives.get("image") if image_part is not None else None
            sample_ids = list(image_part.sample_ids) if image_part is not None else []
            parent_ids = list(image_part.group_ids) if image_part is not None else []

            rewards = None
            if image_part is not None and image_part.rewards is not None:
                rewards = hydrate(image_part.rewards).to(torch.float32).tolist()

            n_imgs = 0
            if isinstance(img_decoded, Images):
                from torchvision.utils import save_image

                img_decoded = deep_hydrate(img_decoded)
                images = img_decoded.to_list()
                n_imgs = len(images)
                for k, image in enumerate(images):
                    pixels = image.pixels.detach().to(torch.float32).clamp(0, 1).cpu()
                    save_image(pixels, os.path.join(out_dir, f"img_{k}.png"))

            ar_params = self.sampling_params["ar"]
            diff_params = self.sampling_params["diffusion"]
            n_rec = ar_params.samples_per_prompt
            n_img = max(1, diff_params.samples_per_prompt)
            n = max(len(sample_ids), n_imgs)
            with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
                for k in range(n):
                    p_idx = k // (n_rec * n_img)
                    a_idx = k // n_img
                    f.write(
                        json.dumps(
                            {
                                "sample_id": sample_ids[k] if k < len(sample_ids) else None,
                                "parent_id": parent_ids[k] if k < len(parent_ids) else None,
                                "prompt": prompts[p_idx] if p_idx < len(prompts) else None,
                                "ar_text_fed_to_dit": ar_texts[a_idx] if a_idx < len(ar_texts) else None,
                                "image_reward": rewards[k] if (rewards is not None and k < len(rewards)) else None,
                                "image_file": f"img_{k}.png" if k < n_imgs else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("[HI3-DUMP] rollout %d → %s (%d samples, %d images)", rollout_id, out_dir, n, n_imgs)
        except Exception as exc:  # noqa: BLE001 — dump must never break training
            logger.warning("[HI3-DUMP] rollout %d dump failed (non-fatal): %s", rollout_id, exc)

    def evaluate(self, step: int) -> float:
        """Periodic eval on the eval set (no training); returns the mean image reward."""
        base_diffusion = self.sampling_params["diffusion"]
        eval_diffusion = dataclasses.replace(
            base_diffusion,
            eta=self.eval_eta,
            guidance_scale=self.eval_cfg_text_scale,
        )
        eval_sp = {**self.sampling_params, "diffusion": eval_diffusion}
        if self.weight_sync is not None:
            if self._enable_fsdp_offload:
                self.backend.onload()
            self.weight_sync.extract()
            if self._enable_fsdp_offload:
                self.backend.offload()
            for eng in self.ar_rollouts + self.dit_rollouts:
                eng.wake_up()
            try:
                self.weight_sync.push()
            finally:
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.sleep()
        scorers = [("reward", self.reward)] + [(s.name, s.reward) for s in self._eval_suites if s.data_source is None]
        metrics = self._eval_pass(self.data_source, self.eval_num_prompts, scorers, eval_sp, step)
        for suite in self._eval_suites:
            if suite.data_source is not None:
                n = suite.num_prompts or self.eval_num_prompts
                metrics.update(self._eval_pass(suite.data_source, n, [(suite.name, suite.reward)], eval_sp, step))
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
        """One generate→score sweep over one eval set; returns each scorer's mean."""
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = all_inputs.batch_size
        chunk = max(1, self.batch_size)
        usable = n_prompts - n_prompts % chunk or n_prompts
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, usable, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            request = self._build_request_sample(sub, step, sampling=eval_sp)
            if self.weight_sync is None:
                generated = self.run_rollout(request)
            else:
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.wake_up()
                try:
                    generated = self.run_rollout(request)
                finally:
                    for eng in self.ar_rollouts + self.dit_rollouts:
                        eng.sleep()
            for name, reward in scorers:
                scored = reward.score_and_attach(generated)
                rewards = scored.parts[-1].rewards
                if rewards is not None:
                    r = hydrate(rewards).to(torch.float32)
                    sums[name] += float(r.sum().item())
                    counts[name] += int(r.numel())
        return {name: sums[name] / max(1, counts[name]) for name, _ in scorers}

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
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                self._dump_rollout_id = rollout_id
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                force_sync = (resumed and rollout_id == start_rollout) or (
                    rollout_id == 0 and bool(os.environ.get("HI3_SYNC_FIRST"))
                )
                sync_weights = force_sync or (rollout_id > 0 and rollout_id % interval == 0)
                results, mean_reward = self.train_step(
                    sample,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        except Exception:
            logger.exception("Training loop aborted at rollout %s", locals().get("rollout_id", "?"))
            raise
        finally:
            self._finish_wandb()


__all__ = ["UnifiedModelTrainer"]
