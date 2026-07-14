"""REFLTrainer — recipe-local trainer for WAN22 REFL/BPTT."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict

from unirl.distributed.tensor.grad_context import enable_grad
from unirl.trainer.trainer import Trainer
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import total_samples_per_prompt


class REFLTrainer(Trainer):
    """REFL / BPTT recipe trainer: role-driven 3-RPC train step."""

    def validate_config(self) -> None:
        super().validate_config()
        if "actor" not in self.roles:
            raise ValueError("REFLTrainer requires a role named 'actor'.")
        if "reward" not in self.roles:
            raise ValueError("REFLTrainer requires a role named 'reward'.")

        actor_spec = self._role_specs_by_name["actor"]
        if actor_spec.raw_cfg.get("algorithm") is None:
            raise ValueError("REFLTrainer requires roles[name=actor].algorithm: ${algorithm}.")
        algo_cfg = self.cfg.get("algorithm")
        if algo_cfg is None:
            raise ValueError("REFLTrainer requires top-level cfg.algorithm.")
        if hasattr(algo_cfg, "get") and algo_cfg.get("sampling_params") is None:
            raise ValueError("REFLTrainer requires cfg.algorithm.sampling_params: ${sampling}.")

        reward_spec = self._role_specs_by_name["reward"]
        backend_cfg = reward_spec.raw_cfg.get("backend")
        backend_target = str(backend_cfg.get("_target_") or "") if backend_cfg is not None and hasattr(backend_cfg, "get") else ""
        if backend_target.endswith("RemoteRewardBackend"):
            raise ValueError(
                "REFLTrainer requires a local differentiable reward backend; "
                f"got roles[name=reward].backend._target_={backend_target!r}."
            )

        rollout_section = self.cfg.get("rollout", None)
        if rollout_section is not None:
            rollout_target = str(rollout_section.get("_target_", "")) if hasattr(rollout_section, "get") else ""
            if rollout_target and not rollout_target.endswith("TrainsideRolloutEngine"):
                raise ValueError(
                    "REFLTrainer: a rollout section is present but is not trainside. "
                    f"Got rollout._target_={rollout_target!r}."
                )

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
            stage_config={},
            sampling_params=dict(self.sampling_params),
            metadata=list(inputs.metadata) if inputs.metadata else [],
            init_noise_group_ids=[],
            init_noise_latent_shape=None,
        )

    def train_step(self, req: RolloutReq, *, training_progress: float = 0.0, rollout_id: int = 0) -> Dict[str, Any]:
        """One REFL step: actor generate → reward score → actor backward → actor step."""
        t0 = time.perf_counter()
        with enable_grad():
            gen = self.actor.generate_samples(req)
            rewards = self.reward.score_differentiable(req=req, generated=gen)
            loss_metrics = self.actor.forward_backward_loss(
                rewards=rewards,
                kl_loss=gen.kl_loss,
            )
        step_result = self.actor.step()

        metrics: Dict[str, Any] = {
            "loss": self._mean(getattr(loss_metrics, "loss", None)),
            "reward_loss": self._mean(getattr(loss_metrics, "reward_loss", None)),
            "kl_loss": self._mean(getattr(loss_metrics, "kl_loss", None)),
            "reward_mean": self._mean(getattr(loss_metrics, "reward_mean", None)),
            "grad_norm": self._mean(step_result.metrics.get("grad_norm")) if step_result.metrics else 0.0,
            "lr": self._mean(step_result.metrics.get("lr")) if step_result.metrics else 0.0,
            "step_time_s": time.perf_counter() - t0,
            "training_progress": float(training_progress),
        }
        return metrics

    @staticmethod
    def _mean(field: Any) -> float:
        try:
            if hasattr(field, "tolist"):
                field = field.tolist()
            if isinstance(field, (list, tuple)) and field:
                return float(sum(float(x) for x in field) / len(field))
            if isinstance(field, (int, float)):
                return float(field)
            return 0.0
        except Exception:
            return 0.0


__all__ = ["REFLTrainer"]
