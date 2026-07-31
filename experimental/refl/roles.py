"""ReflActorRole — family-agnostic REFL/BPTT actor Remote for the refl recipe.

Mirrors :class:`unirl.train.refl.policy.ReFLPolicy` (the SD3 image-ReFL actor):
a **config-chosen** ``Pipeline`` (``pipeline_target`` + ``model_config``, no
per-family imports), FSDP-wrapped in place via ``FSDPBackend``, driven by three
driver RPCs per step under the distributed ``enable_grad()`` context::

    gen = actor.generate_samples(texts=…, images=…, params=…)      # grad BPTT sample + VAE decode
    rew = reward.score_differentiable(gen.decoded, prompts, records)
    actor.forward_backward_loss(rewards=rew, kl_loss=gen.kl_loss)  # loss seed → local backward
    actor.step(max_grad_norm=…)                                    # ctx exit routed grads → optimizer

The pipeline named by ``pipeline_target`` must expose the recipe contract
(see ``experimental.refl.models``): ``build_refl_conditions(texts, images=…,
params=…)`` plus a ``diffusion`` stage with ``diffuse_with_grad`` and a
``vae_decode`` stage with ``decode_with_grad``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from hydra.utils import get_class

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.sde.runtime import get_sigma_schedule
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.configs import FSDPConfig, LoraConfig
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams

logger = logging.getLogger(__name__)


@dataclass
class REFLGenerated(Batch):
    """Generated BPTT payload: live-grad decoded pixels + per-sample KL.

    Both fields are batch-aligned concat fields (``decoded`` ``[B, …]``,
    ``kl_loss`` ``[B]``) so DP_SCATTER merge/re-shard keeps each DP shard's
    own values — the KL fed back into ``forward_backward_loss`` is the KL
    that shard actually accumulated.
    """

    decoded: torch.Tensor = concat_field(default_factory=lambda: torch.empty(0))
    kl_loss: torch.Tensor = concat_field(default_factory=lambda: torch.empty(0))


@dataclass
class REFLLossMetrics(Batch):
    """Per-DP-shard REFL scalar metrics."""

    loss: List[float] = concat_field(default_factory=list)
    reward_loss: List[float] = concat_field(default_factory=list)
    kl_loss: List[float] = concat_field(default_factory=list)
    reward_mean: List[float] = concat_field(default_factory=list)


class ReflActorRole(Remote):
    """Family-agnostic REFL actor: config-chosen pipeline + FSDP + BPTT loss."""

    def __init__(
        self,
        *,
        pipeline_target: str,
        model_config: Any,
        fsdp_cfg: FSDPConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        lora_cfg: Optional[LoraConfig] = None,
        strategy: Optional[Any] = None,
        block_class_names: Tuple[str, ...] = ("WanTransformerBlock",),
        reward_weight: float = 1.0,
        reward_baseline: float = 0.0,
        reward_scale: float = 1.0,
        kl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self._pipeline_target = str(pipeline_target)
        self._model_config = model_config
        self._fsdp_cfg = fsdp_cfg
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self._lora_cfg = lora_cfg
        self._strategy = strategy
        self._block_class_names = tuple(block_class_names)
        self.reward_weight = float(reward_weight)
        self.reward_baseline = float(reward_baseline)
        self.reward_scale = float(reward_scale)
        self.kl_weight = float(kl_weight)

    def initialize(self) -> None:
        torch.cuda.set_device(self.device)
        # Default PG over the actor role's workers (env:// from Remote.setup's
        # dist_env); FSDP2 fully_shard wraps over it. Same order as ReFLPolicy.
        if self.rank_info is not None and int(self.rank_info.world_size) > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        try:
            self._model_config.device = self.device  # runtime device injection
        except Exception:
            pass

        pipeline_cls = get_class(self._pipeline_target)
        self.pipeline = pipeline_cls.from_config(self._model_config, strategy=self._strategy)

        for stage_attr, method in (("diffusion", "diffuse_with_grad"), ("vae_decode", "decode_with_grad")):
            stage = getattr(self.pipeline, stage_attr, None)
            if not hasattr(stage, method):
                raise TypeError(
                    f"ReflActorRole: pipeline {self._pipeline_target} .{stage_attr} lacks {method}(...); "
                    f"use a experimental.refl.models pipeline (or implement the REFL contract)."
                )
        if not hasattr(self.pipeline, "build_refl_conditions"):
            raise TypeError(
                f"ReflActorRole: pipeline {self._pipeline_target} lacks build_refl_conditions(...); "
                f"use a experimental.refl.models pipeline."
            )

        # FSDP-wrap pipeline.bundle.transformer in place + LoRA + optimizer.
        # The pipeline's stages reference the same bundle, so sampling runs
        # through the wrapped trainable transformer.
        self.backend = FSDPBackend(
            bundle=self.pipeline.bundle,
            block_class_names=self._block_class_names,
            trainable_attr="transformer",
            fsdp_cfg=self._fsdp_cfg,
            optimizer_cfg=self._optimizer_cfg,
            scheduler_cfg=self._scheduler_cfg,
            device=self.device,
            rank=int(self.rank_info.rank) if self.rank_info is not None else 0,
            lora_cfg=self._lora_cfg,
        )
        logger.info(
            "ReflActorRole initialized: pipeline=%s reward_weight=%.3f kl_weight=%.3f",
            self._pipeline_target,
            self.reward_weight,
            self.kl_weight,
        )

    # ------------------------------------------------------------------
    # Grad chain (run under the driver's enable_grad() context)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate_samples(
        self,
        *,
        texts: Texts,
        images: Optional[Images] = None,
        params: DiffusionSamplingParams,
    ) -> REFLGenerated:
        """Grad-enabled BPTT sampling + in-graph VAE decode.

        ``texts`` / ``images`` are the data-source conditioning primitives
        (``images`` is the I2V first frame; ``None`` for T2V). Negative
        prompts ride ``params.sampler_kwargs['negative_prompt']`` and are
        expanded by the pipeline's ``build_refl_conditions``.
        """
        self.backend.model.train()
        self.backend.zero_grad()

        # Single KL knob: the actor owns kl_weight (loss-side), the sampling
        # config owns only sampling-shape knobs (mid/final window). A stale
        # sampler_kwargs.kl_weight is an error, not a silent override.
        sampler_kwargs = dict(params.sampler_kwargs or {})
        if "kl_weight" in sampler_kwargs:
            raise ValueError(
                "ReflActorRole: sampling.sampler_kwargs.kl_weight is no longer read; "
                "configure actor.kl_weight instead (single source of truth)."
            )
        sampler_kwargs["kl_weight"] = self.kl_weight

        # Fixed init noise across rollouts and ranks — the contributor's
        # verified DRaFT regime (params.seed used verbatim by the stage).
        # Varying noise per rollout is a training-semantics change; do not
        # introduce it here without its own evidence.
        params = dataclasses.replace(params, sampler_kwargs=sampler_kwargs)

        conditions = self.pipeline.build_refl_conditions(texts, images=images, params=params)
        schedule = get_sigma_schedule(
            int(params.num_inference_steps),
            shift=float(getattr(self.pipeline, "shift", 5.0)),
            device=self.pipeline.bundle.device,
        )
        result = self.pipeline.diffusion.diffuse_with_grad(conditions, schedule=schedule, params=params)
        pixels = self.pipeline.vae_decode.decode_with_grad(result.z_final)
        return REFLGenerated(decoded=pixels, kl_loss=result.kl_loss)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def forward_backward_loss(
        self,
        *,
        rewards: torch.Tensor,
        kl_loss: Optional[torch.Tensor] = None,
    ) -> REFLLossMetrics:
        """Assemble the REFL loss and backward on this shard's actor graph.

        ``rewards`` and ``kl_loss`` are RPC inputs marked as grad leaves by the
        framework; on ``enable_grad()`` exit their grads route back through the
        reward / generate calls into the sampling graph in ONE backward pass.
        """
        reward = rewards.to(dtype=torch.bfloat16)
        reward_loss = (-(reward - self.reward_baseline) / self.reward_scale * self.reward_weight).mean()
        if kl_loss is not None and self.kl_weight != 0.0:
            kl_term = self.kl_weight * kl_loss.float().mean()
        else:
            kl_term = torch.zeros((), device=reward.device, dtype=reward_loss.dtype)
        loss = reward_loss + kl_term
        loss.backward()
        return REFLLossMetrics(
            loss=[float(loss.detach().item())],
            reward_loss=[float(reward_loss.detach().item())],
            kl_loss=[float(kl_term.detach().item())],
            reward_mean=[float(reward.detach().float().mean().item())],
        )

    # ------------------------------------------------------------------
    # Optimizer / checkpoint (delegate to the composed FSDPBackend)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def step(self, *, max_grad_norm: float) -> Dict[str, float]:
        """Clip + one optimizer step; returns grad_norm and current lr."""
        grad_norm = float(self.backend.optimizer_step(max_grad_norm=float(max_grad_norm)))
        lr = 0.0
        sched = getattr(self.backend, "scheduler", None)
        if sched is not None:
            last = sched.get_last_lr()
            lr = float(last[0]) if last else 0.0
        return {"grad_norm": grad_norm, "lr": lr}

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def save(self, path: str, step: Optional[int] = None, mode: str = "adapter") -> None:
        self.backend.save(path, step=step, mode=mode)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def load(self, path: str) -> int:
        return self.backend.load(path)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def wait_for_checkpoint(self) -> None:
        self.backend.wait_for_checkpoint()


__all__ = ["ReflActorRole", "REFLGenerated", "REFLLossMetrics"]
