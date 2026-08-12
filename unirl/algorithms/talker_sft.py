"""Explicit dual-loss SFT for Qwen3-Omni Talker + MTP."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments import TextSegment

from .base import AlgorithmStepResult, StageAlgorithm, typed_conditions


class TalkerSFT(StageAlgorithm):
    """Optimize layer-0 CE plus the mean CE of all 15 Mimi residual layers.

    The reduction is per sample, then averaged across valid samples:

    ``layer0_ce + lambda_sft * mean(mtp_ce_layer_1..15)``.

    This keeps the two independently masked timelines exact: layer-0 supervises
    the appended codec EOS, while MTP supervises only real Mimi frames.
    """

    supports_multi_update = True
    requires_advantages = False
    loss_weighting = "sample"

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        lambda_sft: float = 2.0,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("TalkerSFT: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if not callable(getattr(stage, "replay_sft", None)):
            raise TypeError("TalkerSFT requires a stage exposing replay_sft(conditions, segment)")
        if not math.isfinite(float(lambda_sft)) or float(lambda_sft) < 0.0:
            raise ValueError(f"TalkerSFT.lambda_sft must be finite and >= 0, got {lambda_sft!r}")
        self.stage = stage
        self.lambda_sft = float(lambda_sft)
        self.conditions_cls = conditions_cls

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        loss, stats = self._dual_ce(conditions, segment)
        if loss is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        (loss * loss_scale).backward()
        metrics: Dict[str, Any] = {
            "talker_sft_loss": float(loss.detach().item()),
            "layer0_ce": stats["layer0_ce"],
            "layer0_ppl": math.exp(min(stats["layer0_ce"], 20.0)),
            "mtp_ce": stats["mtp_ce"],
            "mtp_ppl": math.exp(min(stats["mtp_ce"], 20.0)),
            "lambda_sft": self.lambda_sft,
            "talker_sft_samples": stats["samples"],
            "layer0_tokens": stats["layer0_tokens"],
            "mtp_frames": stats["mtp_frames"],
        }
        for index, value in enumerate(stats["mtp_layer_ce"], 1):
            metrics[f"mtp_ce_layer_{index:02d}"] = value
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=round(stats["samples"]),
            has_backward=True,
        )

    @torch.no_grad()
    def evaluate_loss(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[float, float]:
        del sample_ids
        loss, stats = self._dual_ce(conditions, segment)
        if loss is None:
            return 0.0, 0.0
        return float(loss.item()) * stats["samples"], stats["samples"]

    def _dual_ce(
        self,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if segment is None or segment.tokens is None or segment.lengths is None:
            raise ValueError("TalkerSFT requires a packed TextSegment with layer-0 targets")
        if segment.tokens.numel() == 0:
            return None, {}
        typed = typed_conditions(conditions, self.conditions_cls)
        mtp_masks = getattr(typed, "mtp_loss_masks", None)
        if mtp_masks is None:
            raise ValueError("TalkerSFT requires typed conditions.mtp_loss_masks")

        layer0_logp, mtp_logp = self.stage.replay_sft(typed, segment=segment)
        if layer0_logp.ndim != 1 or layer0_logp.shape[0] != segment.tokens.shape[0]:
            raise ValueError(
                f"TalkerSFT layer0 log-probs must be [{segment.tokens.shape[0]}], got {tuple(layer0_logp.shape)}"
            )
        if mtp_logp.ndim != 2 or mtp_logp.shape != (segment.tokens.shape[0], 15):
            raise ValueError(
                f"TalkerSFT MTP log-probs must be [{segment.tokens.shape[0]}, 15], got {tuple(mtp_logp.shape)}"
            )
        lengths = [int(value) for value in segment.lengths.tolist()]
        if len(mtp_masks) != len(lengths):
            raise ValueError(f"TalkerSFT mtp_loss_masks len {len(mtp_masks)} != batch {len(lengths)}")
        layer0_parts = torch.split(layer0_logp, lengths)
        mtp_parts = torch.split(mtp_logp, lengths)
        if segment.loss_mask is None:
            layer0_masks = [torch.ones_like(part) for part in layer0_parts]
        else:
            layer0_masks = torch.split(segment.loss_mask.to(layer0_logp), lengths)

        objectives = []
        layer0_ces = []
        mtp_layer_ces = []
        layer0_token_count = 0.0
        mtp_frame_count = 0.0
        for index, (logp0, logp_mtp, mask0, raw_mtp_mask) in enumerate(
            zip(layer0_parts, mtp_parts, layer0_masks, mtp_masks)
        ):
            mask0 = mask0.to(device=logp0.device, dtype=logp0.dtype).flatten()
            mtp_mask = torch.as_tensor(raw_mtp_mask, device=logp_mtp.device, dtype=logp_mtp.dtype).flatten()
            if mask0.shape[0] != logp0.shape[0] or mtp_mask.shape[0] != logp_mtp.shape[0]:
                raise ValueError(
                    f"TalkerSFT sample {index}: mask timelines layer0={mask0.shape[0]}, "
                    f"mtp={mtp_mask.shape[0]}, expected={logp0.shape[0]}"
                )
            layer0_weight = mask0.sum()
            mtp_weight = mtp_mask.sum()
            if float(layer0_weight.item()) <= 0.0:
                if float(mtp_weight.item()) != 0.0:
                    raise ValueError(f"TalkerSFT sample {index}: padded layer-0 row has nonzero MTP weight")
                continue
            if float(mtp_weight.item()) <= 0.0:
                raise ValueError(f"TalkerSFT sample {index}: no valid MTP frames")
            layer0_ce = -(logp0 * mask0).sum() / layer0_weight
            per_layer_mtp_ce = -(logp_mtp * mtp_mask[:, None]).sum(dim=0) / mtp_weight
            mtp_ce = per_layer_mtp_ce.mean()
            objectives.append(layer0_ce + self.lambda_sft * mtp_ce)
            layer0_ces.append(layer0_ce)
            mtp_layer_ces.append(per_layer_mtp_ce)
            layer0_token_count += float(layer0_weight.item())
            mtp_frame_count += float(mtp_weight.item())

        if not objectives:
            return None, {}
        loss = torch.stack(objectives).mean()
        layer0_ce_mean = torch.stack(layer0_ces).mean()
        mtp_layer_ce_mean = torch.stack(mtp_layer_ces).mean(dim=0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"TalkerSFT produced non-finite loss {float(loss.detach().item())!r}")
        return loss, {
            "samples": float(len(objectives)),
            "layer0_ce": float(layer0_ce_mean.detach().item()),
            "mtp_ce": float(mtp_layer_ce_mean.mean().detach().item()),
            "mtp_layer_ce": [float(value) for value in mtp_layer_ce_mean.detach().tolist()],
            "layer0_tokens": layer0_token_count,
            "mtp_frames": mtp_frame_count,
        }


__all__ = ["TalkerSFT"]
