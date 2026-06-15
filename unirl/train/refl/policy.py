"""ReFLPolicy — self-contained ReFL policy Remote.

Composes an :class:`FSDPBackend` (FSDP-wrap of ``bundle.transformer`` + LoRA +
optimizer + checkpoint) with the SD3 sampling/decode stages over the SAME bundle,
so:

  - ``sample_and_decode`` runs grad-enabled DRaFT-K sampling + VAE decode on the
    FSDP-wrapped *trainable* transformer (returns the only cross-role tensor, the
    image);
  - ``loss_backward`` is the ``-reward.mean()`` seed node (local backward to
    populate ``rewards.grad``; the distributed ``enable_grad()`` context chains it
    back through the reward role → image → transformer params);
  - ``optimizer_step`` consumes those grads on the same params.

Construction follows the Phase-0-validated pattern: the FSDP process group is
initialized in ``initialize()`` (after ``Remote.setup`` populated the dist env),
then ``FSDPBackend`` (which calls ``fully_shard``) is built over it.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.sd3.bundle import SD3Bundle
from unirl.models.sd3.conditions import SD3Conditions
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.diffusion import SD3DiffusionStage, SD3DiffusionStep
from unirl.models.sd3.text_embed import SD3TextEmbedStage
from unirl.models.sd3.vae import SD3VAEDecodeStage
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.configs import FSDPConfig, LoraConfig
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams

logger = logging.getLogger(__name__)


class ReFLPolicy(Remote):
    """SD3 ReFL policy: FSDP transformer + grad DRaFT-K sampling + optimizer."""

    def __init__(
        self,
        *,
        model_config: SD3PipelineConfig,
        fsdp_cfg: FSDPConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        lora_cfg: Optional[LoraConfig] = None,
        block_class_names: Tuple[str, ...] = ("JointTransformerBlock",),
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        draft_num_steps: int = 1,
        reward_loss_scale: float = 1.0,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        height: int = 512,
        width: int = 512,
        seed: int = 42,
        activation_checkpoint_vae: bool = True,
    ) -> None:
        super().__init__()
        self._model_config = model_config
        self._fsdp_cfg = fsdp_cfg
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self._lora_cfg = lora_cfg
        self._block_class_names = tuple(block_class_names)
        self._strategy = strategy if strategy is not None else FlowSDEStrategy()
        self._shift = float(shift)
        self.draft_num_steps = int(draft_num_steps)
        self.reward_loss_scale = float(reward_loss_scale)
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self.height = int(height)
        self.width = int(width)
        self.base_seed = int(seed)
        self.activation_checkpoint_vae = bool(activation_checkpoint_vae)

    def initialize(self) -> None:
        torch.cuda.set_device(self.device)
        # Default PG over the policy role's workers (env:// from Remote.setup's
        # dist_env); FSDP2 fully_shard (mode=full) wraps over it. Phase-0-validated.
        if (
            self.rank_info is not None
            and int(self.rank_info.world_size) > 1
            and not dist.is_initialized()
        ):
            dist.init_process_group(backend="nccl")

        self._model_config.device = self.device
        bundle = SD3Bundle.from_config(self._model_config)
        self.bundle = bundle

        # FSDP-wrap bundle.transformer in place + inject LoRA + build optimizer.
        self.backend = FSDPBackend(
            bundle=bundle,
            block_class_names=self._block_class_names,
            trainable_attr="transformer",
            fsdp_cfg=self._fsdp_cfg,
            optimizer_cfg=self._optimizer_cfg,
            scheduler_cfg=self._scheduler_cfg,
            device=self.device,
            rank=int(self.rank_info.rank) if self.rank_info is not None else 0,
            lora_cfg=self._lora_cfg,
        )

        # Sampling/decode stages over the SAME (now FSDP-wrapped) bundle.
        self.diffusion = SD3DiffusionStage(
            model=bundle,
            step=SD3DiffusionStep(),
            strategy=self._strategy,
            autocast_precision=self._model_config.autocast_precision,
            trajectory_precision=self._model_config.trajectory_precision,
            logprob_precision=self._model_config.logprob_precision,
        )
        self.text_embed = SD3TextEmbedStage(bundle)
        self.vae_decode = SD3VAEDecodeStage(bundle)
        logger.info(
            "ReFLPolicy initialized: draft_num_steps=%d nfe=%d guidance=%.2f res=%dx%d",
            self.draft_num_steps, self.num_inference_steps, self.guidance_scale, self.height, self.width,
        )

    # ------------------------------------------------------------------
    # Grad chain (run under the driver's enable_grad() context)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def sample_and_decode(self, *, prompts: Texts, rollout_id: int = 0) -> Images:
        """Grad-enabled DRaFT-K sample + in-graph VAE decode. Returns ``Images``
        whose pixels carry grad_fn into the FSDP transformer params — the single
        tensor that crosses to the reward role."""
        self.backend.model.train()
        device = torch.device(self.device)
        # Text encoders are frozen — keep their (large, e.g. T5-XXL) forward graph
        # out of the DRaFT backward; the transformer still gets grad via the latents.
        with torch.no_grad():
            cond = self.text_embed.embed(prompts)
        conditions = SD3Conditions(text=cond)
        schedule = get_sigma_schedule(self.num_inference_steps, shift=self._shift, device=device)
        dp_rank = int(self.rank_info.dp_rank) if self.rank_info is not None else 0
        params = DiffusionSamplingParams(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
            eta=0.0,  # deterministic ODE for clean DRaFT gradients
            samples_per_prompt=1,
            seed=self.base_seed + 1000 * int(rollout_id) + dp_rank,
            init_same_noise=False,
        )
        clean = self.diffusion.diffuse_draft_k(
            conditions, schedule=schedule, params=params, draft_num_steps=self.draft_num_steps
        )
        return self.vae_decode.decode_grad(clean, activation_checkpoint=self.activation_checkpoint_vae)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def loss_backward(self, *, rewards: torch.Tensor) -> None:
        """``-reward.mean()`` seed node: local backward populates ``rewards.grad``;
        the empty return makes this an always-run backward node so GradContext
        chains the grad up through score → sample → transformer params."""
        loss = -self.reward_loss_scale * rewards.to(self.device).float().mean()
        loss.backward()
        return None

    # ------------------------------------------------------------------
    # Optimizer / checkpoint (delegate to the composed FSDPBackend)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def optimizer_step(self, *, max_grad_norm: float) -> float:
        return self.backend.optimizer_step(max_grad_norm=max_grad_norm)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def zero_grad(self) -> None:
        self.backend.zero_grad()

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def param_checksum(self) -> float:
        """L1 sum of local trainable-param shards — a cheap weight-change probe
        (changes after a real optimizer step)."""
        total = 0.0
        for p in self.backend.model.parameters():
            if not p.requires_grad:
                continue
            t = p.detach()
            if hasattr(t, "to_local"):  # FSDP2 sharded DTensor
                t = t.to_local()
            total += float(t.float().abs().sum().item())
        return total

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def save(self, path: str, step: Optional[int] = None, mode: str = "adapter") -> None:
        self.backend.save(path, step=step, mode=mode)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def load(self, path: str) -> int:
        return self.backend.load(path)


__all__ = ["ReFLPolicy"]
