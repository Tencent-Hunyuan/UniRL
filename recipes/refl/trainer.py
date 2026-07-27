"""REFLTrainer — recipe-local trainer for WAN22 REFL/BPTT."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict

import numpy as np

from recipes.common.trainer import Trainer
from unirl.distributed.tensor.grad_context import enable_grad
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import total_samples_per_prompt


class REFLTrainer(Trainer):
    """REFL / BPTT recipe trainer: role-driven 3-RPC train step."""

    def build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Build one RolloutReq from data-source samples."""
        inputs = inputs.expand(total_samples_per_prompt(self.sampling_params))
        primitives: Dict[str, Any] = dict(inputs.primitives)

        diff_params = self.sampling_params.get("diffusion")
        guidance_scale = float(getattr(diff_params, "guidance_scale", 1.0)) if diff_params is not None else 1.0
        sampler_kwargs = getattr(diff_params, "sampler_kwargs", {}) if diff_params is not None else {}
        negative_prompt = sampler_kwargs.get("negative_prompt") if isinstance(sampler_kwargs, Mapping) else None
        if negative_prompt is None and diff_params is not None:
            negative_prompt = getattr(diff_params, "negative_prompt", None)
        if negative_prompt is not None and guidance_scale > 1.0 and "negative_text" not in primitives:
            texts = primitives.get("text")
            if not hasattr(texts, "texts"):
                raise TypeError(
                    "REFLTrainer.build_req: sampling negative_prompt requires "
                    "req.primitives['text'] to expose a `texts` field."
                )
            text_list = getattr(texts, "texts")
            text_cls: Any = type(texts)
            primitives["negative_text"] = text_cls(texts=[str(negative_prompt)] * len(text_list))

        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=primitives,
            request_conditions={},
            task_config={},
            sampling_params=dict(self.sampling_params),
            metadata=list(inputs.metadata) if inputs.metadata else [],
            init_noise_group_ids=[],
            init_noise_latent_shape=None,
        )

    def train_step(self, req: RolloutReq, *, training_progress: float = 0.0, rollout_id: int = 0) -> Dict[str, Any]:
        """One REFL step: actor generate → reward score → actor backward → actor step."""
        t0 = time.perf_counter()
        prompts = list(req.primitives["text"].texts)
        records = list(req.metadata) if req.metadata else None
        with enable_grad():
            gen = self.actor.generate_samples(req)
            rewards = self.reward.score_differentiable(gen.decoded, prompts, records)
            loss_metrics = self.actor.forward_backward_loss(
                rewards=rewards,
                kl_loss=gen.kl_loss,
            )
        step_result = self.actor.step()

        metrics: Dict[str, Any] = {
            "loss": np.mean(loss_metrics.loss),
            "reward_loss": np.mean(loss_metrics.reward_loss),
            "kl_loss": np.mean(loss_metrics.kl_loss),
            "reward_mean": np.mean(loss_metrics.reward_mean),
            "grad_norm": np.mean(step_result.metrics.get("grad_norm")) if step_result.metrics else 0.0,
            "lr": np.mean(step_result.metrics.get("lr")) if step_result.metrics else 0.0,
            "step_time_s": time.perf_counter() - t0,
            "training_progress": float(training_progress),
        }
        return metrics


__all__ = ["REFLTrainer"]
