"""SFTPolicy — task-agnostic supervised-training Remote.

Mirrors :class:`unirl.train.refl.policy.ReFLPolicy` structurally: a
config-chosen **task adapter** (which owns the model bundle, data unpacking,
and the loss — e.g. ``unirl.models.qwen3.sft_task.Qwen3SFTTask`` for AR
cross-entropy or ``unirl.models.sd3.sft_task.SD3SFTTask`` for diffusion
flow-matching MSE) is FSDP-wrapped through :class:`FSDPBackend`, and the
per-batch gradient work runs worker-side. The driver only ships record dicts
(paths + metadata) and reads back scalar metrics.

Batch semantics: ``train_batch`` receives the FULL global batch on every worker
(BROADCAST) and takes the ``records[dp_rank::dp_size]`` shard locally — the
trainer enforces ``batch_size % dp_size == 0`` so every rank runs the same
number of backward passes (FSDP collectives stay aligned). Per-sample losses
are scaled by ``1/len(shard)``; FSDP's mean gradient reduction then yields the
global batch mean.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from hydra.utils import get_class

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.configs import FSDPConfig, LoraConfig

logger = logging.getLogger(__name__)


class SFTPolicy(Remote):
    """Config-chosen SFT task adapter + FSDPBackend + worker-side loss/backward."""

    def __init__(
        self,
        *,
        task_target: str,
        model_config: Any,
        fsdp_cfg: FSDPConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        lora_cfg: Optional[LoraConfig] = None,
        block_class_names: Optional[Tuple[str, ...]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._task_target = str(task_target)
        self._model_config = model_config
        self._fsdp_cfg = fsdp_cfg
        self._optimizer_cfg = optimizer_cfg
        self._scheduler_cfg = scheduler_cfg
        self._lora_cfg = lora_cfg
        self._block_class_names = tuple(block_class_names) if block_class_names else None
        self.base_seed = int(seed)

    def initialize(self) -> None:
        torch.cuda.set_device(self.device)
        if self.rank_info is not None and int(self.rank_info.world_size) > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        try:
            self._model_config.device = self.device  # runtime device injection
        except Exception:
            pass

        task_cls = get_class(self._task_target)
        self.task = task_cls.from_config(self._model_config)

        self.backend = FSDPBackend(
            bundle=self.task.bundle,
            block_class_names=self._block_class_names or tuple(self.task.block_class_names),
            trainable_attr="transformer",
            fsdp_cfg=self._fsdp_cfg,
            optimizer_cfg=self._optimizer_cfg,
            scheduler_cfg=self._scheduler_cfg,
            device=self.device,
            rank=int(self.rank_info.rank) if self.rank_info is not None else 0,
            lora_cfg=self._lora_cfg,
        )
        logger.info("SFTPolicy initialized: task=%s", self._task_target)

    def _dp_rank(self) -> int:
        if self.rank_info is None:
            return 0
        return int(getattr(self.rank_info, "dp_rank", self.rank_info.rank))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def train_batch(self, *, records: List[Dict[str, Any]], step: int, dp_size: int) -> Dict[str, float]:
        """Backward over this rank's shard of the global batch; returns rank-mean metrics."""
        shard = records[self._dp_rank() :: max(1, int(dp_size))]
        if not shard:
            raise ValueError(f"empty shard: batch={len(records)} dp_size={dp_size} dp_rank={self._dp_rank()}")
        self.backend.model.train()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.base_seed + 100_003 * int(step) + self._dp_rank())

        totals: Dict[str, float] = {}
        for record in shard:
            loaded = self.task.load_record(record)
            loss, metrics = self.task.compute_loss(loaded, generator=generator)
            (loss / len(shard)).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) / len(shard)
        return totals

    # ------------------------------------------------------------------
    # Eval sampling — every FSDP rank must participate (weight all-gathers),
    # but only dp rank 0 returns the media.
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def sample_media(self, *, records: List[Dict[str, Any]], step: int) -> Optional[List[Dict[str, Any]]]:
        self.backend.model.eval()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.base_seed)  # fixed noise -> comparable evals
        outputs = []
        for record in records:
            loaded = self.task.load_record(record)
            out = self.task.sample(loaded, generator=generator)
            out["sample_id"] = record.get("sample_id")
            out["instruction"] = record.get("instruction")
            outputs.append(out)
        self.backend.model.train()
        return outputs if self._dp_rank() == 0 else None

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
    def save(self, path: str, step: Optional[int] = None, mode: str = "auto") -> None:
        self.backend.save(path, step=step, mode=mode)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def load(self, path: str) -> int:
        return self.backend.load(path)


__all__ = ["SFTPolicy"]
