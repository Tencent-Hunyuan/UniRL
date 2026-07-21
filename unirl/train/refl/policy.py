"""ReFLPolicy — family-agnostic ReFL policy Remote.

Builds a **config-chosen** ``Pipeline`` (no per-family imports), FSDP-wraps its
bundle's transformer via ``FSDPBackend``, and drives grad DRaFT-K sampling + grad
VAE decode through the shared :func:`unirl.models.draft.draft_generate`. The family
is selected entirely by ``pipeline_target`` + ``model_config``; the
``loss_backward`` seed and ``optimizer_step`` are family-agnostic.

Construction follows the Phase-0/e2e-validated order: the FSDP process group is
initialized in ``initialize()`` (after ``Remote.setup`` populated the dist env),
then the pipeline + ``FSDPBackend`` (which calls ``fully_shard``) are built over it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from hydra.utils import get_class

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.draft import draft_generate
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.configs import FSDPConfig, LoraConfig
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams

logger = logging.getLogger(__name__)


class ReFLPolicy(Remote):
    """Family-agnostic ReFL policy: config-chosen Pipeline + FSDP + grad DRaFT-K."""

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
        block_class_names: Tuple[str, ...] = ("JointTransformerBlock",),
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
        self._pipeline_target = str(pipeline_target)
        self._model_config = model_config
        self._fsdp_cfg = fsdp_cfg
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self._lora_cfg = lora_cfg
        self._strategy = strategy
        self._block_class_names = tuple(block_class_names)
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
        if self.rank_info is not None and int(self.rank_info.world_size) > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        try:
            self._model_config.device = self.device  # runtime device injection
        except Exception:
            pass

        pipeline_cls = get_class(self._pipeline_target)
        self.pipeline = pipeline_cls.from_config(self._model_config, strategy=self._strategy)

        # FSDP-wrap pipeline.bundle.transformer in place + inject LoRA + optimizer.
        # The pipeline's stages reference the same bundle, so sampling uses the
        # wrapped trainable transformer.
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
            "ReFLPolicy initialized: pipeline=%s draft_num_steps=%d nfe=%d guidance=%.2f res=%dx%d",
            self._pipeline_target,
            self.draft_num_steps,
            self.num_inference_steps,
            self.guidance_scale,
            self.height,
            self.width,
        )

    # ------------------------------------------------------------------
    # Grad chain (run under the driver's enable_grad() context)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def sample_and_decode(self, *, prompts: Texts, rollout_id: int = 0) -> Images:
        """Grad-enabled DRaFT-K sample + in-graph VAE decode via the shared,
        family-agnostic ``draft_generate``. Returns ``Images`` whose pixels carry
        grad_fn into the FSDP transformer params — the single cross-role tensor."""
        self.backend.model.train()
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
        return draft_generate(
            self.pipeline,
            model_config=self._model_config,
            texts=prompts,
            params=params,
            draft_num_steps=self.draft_num_steps,
            activation_checkpoint=self.activation_checkpoint_vae,
        )

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def eval_sample(self, *, prompts: Texts, rollout_id: int = 0, guidance_scale: Optional[float] = None) -> Images:
        """Eval sampling: ``model.eval()`` + ``no_grad`` DRaFT-K, no autograd graph.

        The eval sibling of :meth:`sample_and_decode`. Reused outside the driver's
        ``enable_grad()`` context, ``sample_and_decode`` would still run in
        ``train()`` mode and build the DRaFT-K activation graph (``draft_generate``
        has no ``no_grad`` guard) only to discard it. This method fixes both: eval
        mode + ``torch.no_grad()`` so the returned ``Images`` carry no grad_fn and
        no graph is retained. The next ``train_step`` re-asserts ``model.train()``
        via ``sample_and_decode``, so no explicit mode restore is needed.

        ``guidance_scale`` overrides the training CFG strength for eval (the
        trainer passes ``eval_cfg_text_scale``; recipes train at 1.0 = no CFG);
        ``None`` falls back to the training value.

        The seed is offset from ``sample_and_decode``'s so eval does not reuse the
        exact training-step params at the same ``rollout_id``. NOTE: the init latent
        is currently drawn unseeded (``generate_latents`` ignores the seed for
        ``init_same_noise=False`` with no ``noise_group_ids``), so eval is NOT
        bit-exact reproducible — it agrees within sampling noise (~σ). Follow-up to
        make it exact: thread ``noise_group_ids`` into the DRaFT noise path."""
        self.backend.model.eval()
        dp_rank = int(self.rank_info.dp_rank) if self.rank_info is not None else 0
        params = DiffusionSamplingParams(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale if guidance_scale is None else float(guidance_scale),
            height=self.height,
            width=self.width,
            eta=0.0,  # deterministic ODE eval
            samples_per_prompt=1,
            seed=self.base_seed + 500_000 + 1000 * int(rollout_id) + dp_rank,
            init_same_noise=False,
        )
        with torch.no_grad():
            return draft_generate(
                self.pipeline,
                model_config=self._model_config,
                texts=prompts,
                params=params,
                draft_num_steps=self.draft_num_steps,
                activation_checkpoint=False,
            )

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
        """L1 sum of local trainable-param shards — a cheap weight-change probe."""
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

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def wait_for_checkpoint(self) -> None:
        self.backend.wait_for_checkpoint()


__all__ = ["ReFLPolicy"]
