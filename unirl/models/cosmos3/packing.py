"""Joint-sequence packing + flow-matching noising for Cosmos3 SFT.

Training mirrors ``Cosmos3OmniPipeline.__call__`` steps 2-5 bit-for-bit by
calling the pipeline's own helpers (``tokenize_prompt``,
``_prepare_text_segment``, ``_prepare_vision_segment``,
``_prepare_action_segment``, ``_encode_video``), then replaces the denoising
loop with a single noised forward:

    sigma ~ p(t), flow-shift-warped exactly like ``set_timesteps``
    x_t   = (1 - sigma) * x0 + sigma * eps      # noisy frames only; condition
                                                # frames stay clean x0 and get
                                                # no timestep embedding
    v*    = eps - x0                            # UniPC ``flow_prediction``
                                                # convention: x0_pred =
                                                # sample - sigma * v
    loss  = MSE over noisy tokens

The transformer consumes ONE packed sequence per call (no batch dim); a
training micro-batch is one sample, with gradient accumulation across samples.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def sample_train_sigma(
    *,
    time_dist: str,
    logitnormal_mean: float,
    logitnormal_std: float,
    shift: float,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Draw one training sigma in (0, 1), then apply the same shift warp
    ``set_timesteps`` uses: ``sigma' = shift*s / (1 + (shift-1)*s)``."""
    if time_dist == "uniform":
        base = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    elif time_dist == "logitnormal":
        z = torch.randn((), generator=generator, device=device, dtype=torch.float32)
        base = torch.sigmoid(z * logitnormal_std + logitnormal_mean)
    else:
        raise ValueError(f"Unknown time_dist={time_dist!r}; expected 'logitnormal' or 'uniform'.")
    sigma = shift * base / (1.0 + (shift - 1.0) * base)
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def noise_vision_latents(
    x0: torch.Tensor,
    sigma: torch.Tensor,
    condition_frame_indexes: Sequence[int],
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flow-matching interpolation for vision latents ``[1, C, T_lat, H, W]``.

    Returns ``(x_t, velocity_target)``; conditioned latent frames carry clean
    ``x0`` in ``x_t`` (their velocity-target slots are never read by the loss).
    """
    noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    for frame_idx in condition_frame_indexes:
        x_t[:, :, frame_idx] = x0[:, :, frame_idx]
    return x_t, noise - x0


def pad_action_chunk(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Zero-pad a raw action chunk ``[T, D_raw]`` up to the model's
    ``action_dim`` (the checkpoint convention: padded dims are exactly zero)."""
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
    """Flow-matching interpolation for a fully-noisy action chunk ``[T, action_dim]``
    (policy-mode BC: no clean action conditioning). Channels >= ``raw_action_dim``
    are pinned to zero in both the sample and the target, matching inference."""
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assemble ``Cosmos3OmniTransformer.forward`` kwargs for ONE sample.

    Mirrors the conditional-pass assembly in ``Cosmos3OmniPipeline.__call__``
    (text segment -> vision segment -> optional action segment -> position_ids
    concat). ``pipe`` only needs the pipeline's segment helpers plus
    ``transformer.config`` / ``vae.config`` — a duck-typed stand-in works in
    tests.

    Returns ``(kwargs, meta)``: ``kwargs`` lacks only the per-call
    ``vision_timesteps`` / ``action_timesteps`` tensors (sized via ``meta``).
    """
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

    model_dtype = pipe.transformer.dtype
    kwargs: Dict[str, Any] = {
        "input_ids": text_seg["input_ids"],
        "text_indexes": text_seg["text_indexes"],
        "position_ids": torch.cat(mrope_segments, dim=1),
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
    "sample_train_sigma",
]
