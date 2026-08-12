"""Typed conditions for Qwen3-Omni Talker TTS AR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition

from .talker_contract import NUM_CODE_GROUPS
from .talker_sampling import TalkerSamplingConfig


@dataclass
class Qwen3OmniTalkerConditions(Batch):
    """Compact, replayable direct-TTS conditions.

    Conditions intentionally carry IDs/codes only: prompt IDs+mask, discrete
    speaker IDs, Talker prefix IDs, MTP residual codes, and the exact
    behavior sampling configuration. Waveforms and full hidden states never
    enter the trajectory.
    """

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    speaker_ids: Optional[List[int]] = field(kind=FieldKind.CONCAT, default=None)
    prefix_ids: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    # Legacy input accepted for old rollout adapters. New direct-TTS paths store
    # speaker_ids instead and never emit this field.
    speakers: Optional[List[str]] = field(kind=FieldKind.CONCAT, default=None)
    # Per-sample residual codec codes ``[15, T]`` aligned with frontier layer-0 lengths.
    # Filled by Talker generate so GSPO ``stage.replay(conditions, segment=...)`` can
    # recompute MTP logπ without a custom algorithm fork.
    residual_codes: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    # Per-sample ``[T]`` SFT weights for residual targets. The appended layer-0
    # EOS has no Mimi residual target and therefore carries weight zero.
    mtp_loss_masks: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    # Exact layer-0 behavior distribution used by rollout. Replay must not
    # silently fall back to unfiltered logits.
    behavior_sampling: Optional[Dict[str, Any]] = field(kind=FieldKind.SHARED, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3OmniTalkerConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3OmniTalkerConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got {type(prompt).__name__ if prompt is not None else 'None'}"
            )
        speakers = d.get("speakers")
        if speakers is not None and not isinstance(speakers, list):
            raise TypeError(
                f"Qwen3OmniTalkerConditions.from_dict: expected d['speakers'] to be a list, "
                f"got {type(speakers).__name__}"
            )
        speaker_ids = d.get("speaker_ids")
        if speaker_ids is not None and not isinstance(speaker_ids, list):
            raise TypeError(
                "Qwen3OmniTalkerConditions.from_dict: expected d['speaker_ids'] "
                f"to be a list, got {type(speaker_ids).__name__}"
            )
        prefix_ids = d.get("prefix_ids")
        if prefix_ids is not None and not isinstance(prefix_ids, list):
            raise TypeError(
                "Qwen3OmniTalkerConditions.from_dict: expected d['prefix_ids'] "
                f"to be a list, got {type(prefix_ids).__name__}"
            )
        mtp_loss_masks = d.get("mtp_loss_masks")
        if mtp_loss_masks is not None and not isinstance(mtp_loss_masks, list):
            raise TypeError(
                "Qwen3OmniTalkerConditions.from_dict: expected d['mtp_loss_masks'] "
                f"to be a list, got {type(mtp_loss_masks).__name__}"
            )
        return cls(
            prompt=prompt,
            speaker_ids=[int(v) for v in speaker_ids] if speaker_ids is not None else None,
            prefix_ids=prefix_ids,
            speakers=speakers,
            residual_codes=d.get("residual_codes"),
            mtp_loss_masks=mtp_loss_masks,
            behavior_sampling=d.get("behavior_sampling"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("Qwen3OmniTalkerConditions.to_dict: prompt field is None")
        out: Dict[str, Any] = {"prompt": self.prompt}
        if self.speaker_ids is not None:
            out["speaker_ids"] = [int(v) for v in self.speaker_ids]
        elif self.speakers is not None:
            # Compatibility for trajectories produced by the old vLLM adapter.
            out["speakers"] = list(self.speakers)
        if self.prefix_ids is not None:
            out["prefix_ids"] = list(self.prefix_ids)
        if self.residual_codes is not None:
            out["residual_codes"] = list(self.residual_codes)
        if self.mtp_loss_masks is not None:
            out["mtp_loss_masks"] = list(self.mtp_loss_masks)
        if self.behavior_sampling is not None:
            out["behavior_sampling"] = dict(self.behavior_sampling)
        return out

    def validate_rollout_replay(self, *, lengths: List[int]) -> TalkerSamplingConfig:
        """Validate the complete, lossless layer-0 trajectory contract."""
        if self.prompt is None or self.prompt.input_ids is None or self.prompt.attention_mask is None:
            raise ValueError("Talker rollout replay requires prompt input_ids and attention_mask")
        batch_size = int(self.prompt.input_ids.shape[0])
        if len(lengths) != batch_size:
            raise ValueError(f"Talker replay lengths batch {len(lengths)} != prompt batch {batch_size}")
        if self.speaker_ids is None:
            raise ValueError("Talker rollout replay requires discrete speaker_ids")
        if len(self.speaker_ids) != batch_size:
            raise ValueError(f"Talker replay speaker_ids len {len(self.speaker_ids)} != batch {batch_size}")
        if self.prefix_ids is None:
            raise ValueError("Talker rollout replay requires the exact rollout prefix_ids")
        if len(self.prefix_ids) != batch_size:
            raise ValueError(f"Talker replay prefix_ids len {len(self.prefix_ids)} != batch {batch_size}")
        if self.residual_codes is None:
            raise ValueError("Talker rollout replay requires the exact rollout residual_codes")
        if len(self.residual_codes) != batch_size:
            raise ValueError(f"Talker replay residual_codes len {len(self.residual_codes)} != batch {batch_size}")
        for index, (raw_prefix, raw_residual, length) in enumerate(
            zip(self.prefix_ids, self.residual_codes, lengths)
        ):
            import torch

            prefix = torch.as_tensor(raw_prefix)
            residual = torch.as_tensor(raw_residual)
            if prefix.numel() == 0 or prefix.dim() not in (1, 2):
                raise ValueError(f"prefix_ids[{index}] must be a non-empty rank-1/2 tensor")
            if residual.dim() != 2 or residual.shape[0] != NUM_CODE_GROUPS - 1:
                raise ValueError(
                    f"residual_codes[{index}] expected [{NUM_CODE_GROUPS - 1}, T], "
                    f"got {tuple(residual.shape)}"
                )
            if int(residual.shape[1]) != int(length):
                raise ValueError(
                    f"residual_codes[{index}] timeline {residual.shape[1]} "
                    f"!= layer-0 timeline {length}"
                )
        if self.behavior_sampling is None:
            raise ValueError("Talker rollout replay requires the exact behavior_sampling payload")
        return TalkerSamplingConfig.from_dict(dict(self.behavior_sampling))


__all__ = ["Qwen3OmniTalkerConditions"]
