"""Worker-side Cosmos3 video/action supervised track builder.

Lives in the model package (not ``unirl/train/sft/``) because its record →
(conditions, segment) mapping is inherently Cosmos3-specific: it packs
:class:`Cosmos3SFTCondition` through the joint stage's own tokenize/encode
helpers. The generic builders in ``unirl/train/sft/track_builder.py`` stay
model-agnostic; per-model logic stays here, per that package's contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.cosmos3.conditions import Cosmos3SFTCondition
from unirl.train.sft.track_builder import SupervisedTrackBuilder
from unirl.types.sample import Part
from unirl.types.segments.latent import make_video_segment

Record = Dict[str, Any]


def _media_uri(record: Record, *, modality: str, role: str) -> str:
    matches: List[str] = []
    for ref in record.get("media_refs", []) or []:
        ref_modality = getattr(ref, "modality", None) if not isinstance(ref, dict) else ref.get("modality")
        ref_role = getattr(ref, "role", None) if not isinstance(ref, dict) else ref.get("role")
        ref_uri = getattr(ref, "uri", None) if not isinstance(ref, dict) else ref.get("uri")
        if ref_modality == modality and ref_role == role and ref_uri:
            matches.append(str(ref_uri))
    if len(matches) != 1:
        raise ValueError(
            f"Cosmos3SupervisedTrackBuilder: sample {record.get('sample_id')!r} needs exactly one "
            f"{modality!r}/{role!r} media ref; found {len(matches)}."
        )
    uri = matches[0]
    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(f"Cosmos3 SFT requires a local/shared tensor path, got {uri!r}.")
    return uri


class Cosmos3SupervisedTrackBuilder(SupervisedTrackBuilder):
    """Normalized records → packed Cosmos3 conditions + clean video latents."""

    def __init__(self, *, pipeline: Any, train_action: bool = False) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.stage = pipeline.joint
        self.train_action = bool(train_action)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        if not records:
            raise ValueError("Cosmos3SupervisedTrackBuilder.build: empty record shard.")

        x0_rows: List[torch.Tensor] = []
        token_rows: List[torch.Tensor] = []
        action_rows: List[torch.Tensor] = []
        fps_rows: List[float] = []
        shift_rows: List[float] = []
        pad_rows: List[float] = []

        with torch.no_grad():
            for record in records:
                frames = torch.load(
                    _media_uri(record, modality="video", role="target"),
                    map_location="cpu",
                    weights_only=True,
                )
                if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[1] != 3:
                    shape = tuple(frames.shape) if isinstance(frames, torch.Tensor) else type(frames).__name__
                    raise ValueError(f"Cosmos3 frames must be a tensor [T,3,H,W], got {shape}.")
                if frames.dtype != torch.uint8:
                    raise ValueError(f"Cosmos3 frames must be uint8, got {frames.dtype}.")

                num_frames, _, height, width = map(int, frames.shape)
                metadata = record.get("metadata") or {}
                fps = float(metadata.get("fps", self.stage.config.fps))
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"Cosmos3 sample {record.get('sample_id')!r} has no non-empty prompt.")

                x0_rows.append(self.stage.encode_video(frames))
                ids = self.stage.tokenize_prompt(
                    prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    fps=fps,
                    action_mode="policy" if self.train_action else None,
                )
                token_rows.append(torch.tensor(ids, dtype=torch.long, device=self.stage.device))
                fps_rows.append(fps)
                shift_rows.append(self.stage.flow_shift(height, width))
                pad_rows.append(0.0 if record.get("_eval_pad", False) else 1.0)

                if self.train_action:
                    actions = torch.load(
                        _media_uri(record, modality="action", role="target"),
                        map_location="cpu",
                        weights_only=True,
                    )
                    expected = (self.stage.config.action_chunk_size, self.stage.config.raw_action_dim)
                    if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
                        shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
                        raise ValueError(f"Cosmos3 actions must have shape {expected}, got {shape}.")
                    if num_frames != self.stage.config.action_chunk_size + 1:
                        raise ValueError(
                            "Cosmos3 action BC needs action_chunk_size + 1 frames; "
                            f"got {num_frames} frames for chunk {self.stage.config.action_chunk_size}."
                        )
                    action_rows.append(actions.to(device=self.stage.device, dtype=torch.float32))

        latent_shapes = {tuple(row.shape[1:]) for row in x0_rows}
        if len(latent_shapes) != 1:
            raise ValueError(
                "Cosmos3SupervisedTrackBuilder requires one latent shape per batch; "
                f"got {sorted(latent_shapes)}. Bucket records by resolution/frame count."
            )
        x0 = torch.cat(x0_rows, dim=0)
        actions_batch: Optional[torch.Tensor] = torch.stack(action_rows) if self.train_action else None
        condition = Cosmos3SFTCondition.pack(
            input_ids=token_rows,
            fps=torch.tensor(fps_rows, dtype=torch.float32, device=self.stage.device),
            flow_shifts=torch.tensor(shift_rows, dtype=torch.float32, device=self.stage.device),
            actions=actions_batch,
        )
        segment = make_video_segment(
            latents=x0.unsqueeze(1),
            loss_mask=torch.tensor(pad_rows, dtype=torch.float32, device=x0.device),
        )
        part = Part(
            sample_ids=[str(r.get("sample_id", f"cosmos3:{i}")) for i, r in enumerate(records)],
            conditions={"cosmos3": condition},
            segment=segment,
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )
        if condition.batch_size != len(records) or part.batch_size != len(records):
            raise RuntimeError(
                "Cosmos3SupervisedTrackBuilder: record/condition/part batch sizes diverged: "
                f"{len(records)}/{condition.batch_size}/{part.batch_size}."
            )
        return part


__all__ = ["Cosmos3SupervisedTrackBuilder"]
