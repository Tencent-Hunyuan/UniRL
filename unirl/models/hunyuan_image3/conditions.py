"""HunyuanImage3 typed conditions containers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field
from unirl.distributed.tensor.ref import hydrate
from unirl.types.conditions import (
    Condition,
    FusedMultimodalCondition,
    ImageEmbedCondition,
    Modality,
)


def _pad_seq(t: Any, length: int, dim: int = -1) -> Optional[torch.Tensor]:
    """Zero-pad negative ``dim`` up to ``length``, hydrating a ``TensorRef``; longer tensors pass through."""
    t = hydrate(t)
    if t is None or t.shape[dim] >= length:
        return t
    return torch.nn.functional.pad(t, (0, 0) * (-dim - 1) + (0, length - t.shape[dim]))


def _pad_positions(t: Any, length: int) -> Optional[torch.Tensor]:
    """Pad ``position_ids`` with continuing indices, since they are KV-cache write slots — README ``## Gotchas``."""
    t = hydrate(t)
    if t is None or t.shape[-1] >= length:
        return t
    tail = torch.arange(t.shape[-1], length, dtype=t.dtype, device=t.device)
    return torch.cat([t, tail.expand(*t.shape[:-1], -1)], dim=-1)


@dataclass
class HunyuanImage3FusedMultimodalCondition(FusedMultimodalCondition):
    """Hunyuan's fused-sequence layout."""

    # Store RoPE as a CONCAT tensor so DP transport preserves per-sample rows.
    rope_cache: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, 2, L, D]

    gen_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    gen_timestep_scatter_index: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, K] long
    cond_vae_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    cond_vit_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    cond_timestep_scatter_index: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, K] long
    prompt_lengths: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B] long

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HunyuanImage3FusedMultimodalCondition":
        """Build from a flat dict shape (the on-the-wire form)."""
        kwargs: Dict[str, Any] = {}
        for name in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "rope_cache",
            "gen_image_mask",
            "gen_timestep_scatter_index",
            "cond_vae_image_mask",
            "cond_vit_image_mask",
            "cond_timestep_scatter_index",
            "prompt_lengths",
        ):
            if name in d and d[name] is not None:
                kwargs[name] = d[name]
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is not None and not isinstance(rope_cache, torch.Tensor):
            # Reject legacy RoPE tuples at the transport boundary.
            raise TypeError(
                "HunyuanImage3FusedMultimodalCondition.from_dict: rope_cache "
                f"must be a stacked [B, 2, L, D] tensor; got {type(rope_cache).__name__}. "
                "Producers must stack (cos, sin) pairs via torch.stack(pair, dim=1)."
            )
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a flat dict shape (only set fields emitted)."""
        out: Dict[str, Any] = {}
        for name in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "rope_cache",
            "gen_image_mask",
            "gen_timestep_scatter_index",
            "cond_vae_image_mask",
            "cond_vit_image_mask",
            "cond_timestep_scatter_index",
            "prompt_lengths",
        ):
            v = getattr(self, name)
            if v is not None:
                out[name] = v
        return out

    @classmethod
    def concat(cls, items: list) -> "HunyuanImage3FusedMultimodalCondition":
        """Override ``Batch.concat`` to pad ragged L dims before cat."""
        seq_lens = {item.input_ids.shape[-1] for item in items}
        if len(seq_lens) <= 1:
            return super().concat(items)

        max_L = max(seq_lens)
        padded_items = [
            replace(
                item,
                input_ids=_pad_seq(item.input_ids, max_L),
                attention_mask=_pad_seq(_pad_seq(item.attention_mask, max_L), max_L, dim=-2),
                position_ids=_pad_positions(item.position_ids, max_L),
                rope_cache=_pad_seq(item.rope_cache, max_L, dim=-2),
                gen_image_mask=_pad_seq(item.gen_image_mask, max_L),
                cond_vae_image_mask=_pad_seq(item.cond_vae_image_mask, max_L),
                cond_vit_image_mask=_pad_seq(item.cond_vit_image_mask, max_L),
            )
            for item in items
        ]
        return super().concat(padded_items)


@dataclass
class HunyuanImage3VAECondition(Condition):
    """Per-sample VAE payloads emitted by HI3's private image encoder."""

    modality: ClassVar[Modality] = Modality.IMAGE
    latents: list[torch.Tensor] = concat_field(default_factory=list)


@dataclass
class HunyuanImage3DiffusionConditions(Batch):
    """Typed conditions container for HunyuanImage3 DiT-mode diffusion."""

    fused: Optional[HunyuanImage3FusedMultimodalCondition] = field(kind=FieldKind.SHARED, default=None)
    # Store CFG's unconditional branch separately so B-sample transport preserves it.
    fused_uncond: Optional[HunyuanImage3FusedMultimodalCondition] = field(kind=FieldKind.SHARED, default=None)
    cond_vae: Optional[HunyuanImage3VAECondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_vit: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_timestep: Optional[torch.Tensor | list[torch.Tensor]] = field(kind=FieldKind.CONCAT, default=None)
    tokenizer_output: Optional[Any] = field(kind=FieldKind.SHARED, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HunyuanImage3DiffusionConditions":
        """Build from the generic ``Conditions`` dict shape."""
        fused = d.get("fused")
        if fused is not None and not isinstance(fused, HunyuanImage3FusedMultimodalCondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['fused'] "
                f"to be a HunyuanImage3FusedMultimodalCondition or absent, "
                f"got {type(fused).__name__}"
            )
        if fused is None or fused.input_ids is None:
            raise TypeError(
                "HunyuanImage3DiffusionConditions.from_dict: 'fused.input_ids' "
                "is required for the diffusion stage to consume."
            )
        cond_vae = d.get("cond_vae")
        if cond_vae is not None and not isinstance(cond_vae, HunyuanImage3VAECondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['cond_vae'] "
                f"to be a HunyuanImage3VAECondition or absent, "
                f"got {type(cond_vae).__name__}"
            )
        cond_vit = d.get("cond_vit")
        if cond_vit is not None and not isinstance(cond_vit, ImageEmbedCondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['cond_vit'] "
                f"to be an ImageEmbedCondition or absent, "
                f"got {type(cond_vit).__name__}"
            )
        fused_uncond = d.get("fused_uncond")
        if fused_uncond is not None and not isinstance(fused_uncond, HunyuanImage3FusedMultimodalCondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['fused_uncond'] "
                f"to be a HunyuanImage3FusedMultimodalCondition or absent, "
                f"got {type(fused_uncond).__name__}"
            )
        return cls(
            fused=fused,
            fused_uncond=fused_uncond,
            cond_vae=cond_vae,
            cond_vit=cond_vit,
            cond_timestep=d.get("cond_timestep"),
            tokenizer_output=d.get("tokenizer_output"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to the generic ``Conditions`` dict shape."""
        if self.fused is None or self.fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3DiffusionConditions.to_dict: `fused.input_ids` is "
                "None — required for the diffusion stage to consume."
            )
        out: Dict[str, Any] = {"fused": self.fused}
        if self.fused_uncond is not None:
            out["fused_uncond"] = self.fused_uncond
        if self.cond_vae is not None:
            out["cond_vae"] = self.cond_vae
        if self.cond_vit is not None:
            out["cond_vit"] = self.cond_vit
        if self.cond_timestep is not None:
            out["cond_timestep"] = self.cond_timestep
        if self.tokenizer_output is not None:
            out["tokenizer_output"] = self.tokenizer_output
        return out


@dataclass
class HunyuanImage3ARConditions(Batch):
    """Typed conditions container for HunyuanImage3 AR-mode autoregress."""

    fused: Optional[HunyuanImage3FusedMultimodalCondition] = field(kind=FieldKind.SHARED, default=None)
    cond_vae: Optional[HunyuanImage3VAECondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_vit: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_timestep: Optional[torch.Tensor | list[torch.Tensor]] = field(kind=FieldKind.CONCAT, default=None)
    tokenizer_output: Optional[Any] = field(kind=FieldKind.SHARED, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HunyuanImage3ARConditions":
        fused = d.get("fused")
        if fused is not None and not isinstance(fused, HunyuanImage3FusedMultimodalCondition):
            raise TypeError(
                f"HunyuanImage3ARConditions.from_dict: expected d['fused'] to be "
                f"a HunyuanImage3FusedMultimodalCondition or absent, "
                f"got {type(fused).__name__}"
            )
        if fused is None or fused.input_ids is None:
            raise TypeError(
                "HunyuanImage3ARConditions.from_dict: 'fused.input_ids' is required for the AR stage to consume."
            )
        cond_vae = d.get("cond_vae")
        if cond_vae is not None and not isinstance(cond_vae, HunyuanImage3VAECondition):
            raise TypeError(
                f"HunyuanImage3ARConditions.from_dict: expected d['cond_vae'] "
                f"to be a HunyuanImage3VAECondition or absent, "
                f"got {type(cond_vae).__name__}"
            )
        cond_vit = d.get("cond_vit")
        if cond_vit is not None and not isinstance(cond_vit, ImageEmbedCondition):
            raise TypeError(
                f"HunyuanImage3ARConditions.from_dict: expected d['cond_vit'] "
                f"to be an ImageEmbedCondition or absent, "
                f"got {type(cond_vit).__name__}"
            )
        return cls(
            fused=fused,
            cond_vae=cond_vae,
            cond_vit=cond_vit,
            cond_timestep=d.get("cond_timestep"),
            tokenizer_output=d.get("tokenizer_output"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.fused is None or self.fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3ARConditions.to_dict: `fused.input_ids` is None — required for the AR stage to consume."
            )
        out: Dict[str, Any] = {"fused": self.fused}
        if self.cond_vae is not None:
            out["cond_vae"] = self.cond_vae
        if self.cond_vit is not None:
            out["cond_vit"] = self.cond_vit
        if self.cond_timestep is not None:
            out["cond_timestep"] = self.cond_timestep
        if self.tokenizer_output is not None:
            out["tokenizer_output"] = self.tokenizer_output
        return out


__all__ = [
    "HunyuanImage3ARConditions",
    "HunyuanImage3DiffusionConditions",
    "HunyuanImage3FusedMultimodalCondition",
    "HunyuanImage3VAECondition",
]
