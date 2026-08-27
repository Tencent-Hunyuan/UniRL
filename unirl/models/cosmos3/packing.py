"""Joint-sequence packing and flow-matching noising for Cosmos3 SFT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def resolution_tier(height: int, width: int) -> str:
    """Map a sample to the official 256, 480, or 720 short-edge resolution tier."""
    short_edge = min(int(height), int(width))
    if short_edge <= 256:
        return "256"
    if short_edge <= 640:
        return "480"
    return "720"


def noise_vision_latents(
    x0: torch.Tensor,
    sigma: torch.Tensor,
    condition_frame_indexes: Sequence[int],
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate vision latents ``[1,C,T,H,W]`` while preserving conditioned frames."""
    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    for frame_idx in condition_frame_indexes:
        x_t[:, :, frame_idx] = x0[:, :, frame_idx]
    return x_t, noise - x0


def pad_action_chunk(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Zero-pad a raw action chunk ``[T,D_raw]`` to the model action width."""
    if actions.ndim != 2:
        raise ValueError(f"action chunk must be [T, D], got {tuple(actions.shape)}")
    if actions.shape[1] > action_dim:
        raise ValueError(f"action width {actions.shape[1]} exceeds model action_dim={action_dim}")
    if actions.shape[1] == action_dim:
        return actions
    pad = torch.zeros(actions.shape[0], action_dim - actions.shape[1], dtype=actions.dtype, device=actions.device)
    return torch.cat([actions, pad], dim=-1)


def noise_action_latents(
    x0_padded: torch.Tensor,
    sigma: torch.Tensor,
    raw_action_dim: int,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate a fully noisy action chunk ``[T,D]`` while keeping padded channels zero."""
    noise = torch.randn(x0_padded.shape, generator=generator, device=x0_padded.device, dtype=x0_padded.dtype)
    noise[:, raw_action_dim:] = 0
    x_t = (1.0 - sigma) * x0_padded + sigma * noise
    x_t[:, raw_action_dim:] = 0
    return x_t, noise - x0_padded


def pack_joint_sequence(
    pipe: Any,
    *,
    input_ids: Sequence[int],
    vision_tokens: torch.Tensor,
    condition_frame_indexes: Sequence[int],
    vision_fps: float,
    device: torch.device,
    action_tokens: Optional[torch.Tensor] = None,
    action_condition_frame_indexes: Sequence[int] = (),
    action_domain_id: Optional[torch.Tensor] = None,
    action_fps: Optional[float] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assemble one sample's packed ``Cosmos3OmniTransformer.forward`` arguments."""
    text_seg = pipe._prepare_text_segment(list(input_ids), device=device)
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=vision_tokens,
        has_image_condition=bool(condition_frame_indexes),
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=vision_fps,
        curr=text_seg["und_len"],
        device=device,
        condition_frame_indexes=list(condition_frame_indexes),
    )
    mrope_segments = [text_seg["text_mrope_ids"], vision_seg["vision_mrope_ids"]]
    sequence_length = text_seg["und_len"] + vision_seg["num_vision_tokens"]

    action_seg: Dict[str, Any] = {}
    if action_tokens is not None:
        if action_domain_id is None:
            raise ValueError("action_tokens require an action_domain_id")
        action_seg = pipe._prepare_action_segment(
            input_action_tokens=action_tokens,
            condition_frame_indexes=list(action_condition_frame_indexes),
            mrope_offset=text_seg["vision_start_temporal_offset"],
            action_fps=action_fps if action_fps is not None else vision_fps,
            curr=sequence_length,
            device=device,
        )
        mrope_segments.append(action_seg["action_mrope_ids"])
        sequence_length += action_seg["action_len"]

    model_dtype = compute_dtype if compute_dtype is not None else pipe.transformer.dtype
    kwargs: Dict[str, Any] = {
        "input_ids": text_seg["input_ids"],
        "text_indexes": text_seg["text_indexes"],
        "position_ids": torch.cat(mrope_segments, dim=1).contiguous(),
        "und_len": text_seg["und_len"],
        "sequence_length": sequence_length,
        "vision_tokens": [vision_tokens.to(dtype=model_dtype)],
        "vision_token_shapes": vision_seg["vision_token_shapes"],
        "vision_sequence_indexes": vision_seg["vision_sequence_indexes"],
        "vision_mse_loss_indexes": vision_seg["vision_mse_loss_indexes"],
        "vision_noisy_frame_indexes": vision_seg["vision_noisy_frame_indexes"],
    }
    if action_seg:
        kwargs.update(
            {
                "action_tokens": [action_tokens.to(dtype=model_dtype)],
                "action_token_shapes": action_seg["action_token_shapes"],
                "action_sequence_indexes": action_seg["action_sequence_indexes"],
                "action_mse_loss_indexes": action_seg["action_mse_loss_indexes"],
                "action_timesteps": None,  # filled by the caller
                "action_noisy_frame_indexes": action_seg["action_noisy_frame_indexes"],
                "action_domain_ids": [action_domain_id],
            }
        )

    meta = {
        "num_noisy_vision_tokens": vision_seg["num_noisy_vision_tokens"],
        "vision_noisy_frames": vision_seg["vision_noisy_frame_indexes"][0],
        "num_noisy_action_tokens": action_seg.get("num_noisy_action_tokens", 0),
        "action_noisy_frames": (action_seg["action_noisy_frame_indexes"][0] if action_seg else None),
    }
    return kwargs, meta


__all__ = [
    "noise_action_latents",
    "noise_vision_latents",
    "pack_joint_sequence",
    "pad_action_chunk",
    "resolution_tier",
]
