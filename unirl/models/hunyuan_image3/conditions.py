"""HunyuanImage3 typed conditions containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field
from unirl.types.conditions import (
    Condition,
    FusedMultimodalCondition,
    ImageEmbedCondition,
    Modality,
)


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
        """Override ``Batch.concat`` to pad variable-length L dims before cat."""
        if not items or len(items) <= 1:
            from unirl.distributed.tensor.batch import Batch

            return Batch.concat.__func__(cls, items)

        seq_lens = []
        for item in items:
            if item.input_ids is not None:
                seq_lens.append(item.input_ids.shape[-1])
        if not seq_lens or len(set(seq_lens)) <= 1:
            from unirl.distributed.tensor.batch import Batch

            return Batch.concat.__func__(cls, items)

        max_L = max(seq_lens)

        def _materialize(t):
            # Materialize lazy CONCAT fields before padding and merging.
            if t is not None and not isinstance(t, torch.Tensor) and hasattr(t, "materialize"):
                return t.materialize()
            return t

        def _pad_seq(t, dim=-1, value=0):
            t = _materialize(t)
            if t is None:
                return None
            cur = t.shape[dim]
            if cur >= max_L:
                return t
            pad_size = max_L - cur
            ndim = len(t.shape)
            pad_spec = [0] * (2 * ndim)
            actual_dim = dim if dim >= 0 else ndim + dim
            pad_idx = (ndim - 1 - actual_dim) * 2
            pad_spec[pad_idx + 1] = pad_size
            return torch.nn.functional.pad(t, pad_spec, value=value)

        def _pad_attn(mask):
            mask = _materialize(mask)
            if mask is None:
                return None
            if mask.shape[-1] >= max_L:
                return mask
            N, H, L, _ = mask.shape
            padded = torch.zeros(N, H, max_L, max_L, dtype=mask.dtype, device=mask.device)
            padded[:, :, :L, :L] = mask
            return padded

        # Materialize every shard so concatenation never mixes tensors and TensorRefs.
        padded_items = [
            cls(
                input_ids=_pad_seq(item.input_ids, dim=-1, value=0),
                attention_mask=_pad_attn(item.attention_mask),
                position_ids=_pad_seq(item.position_ids, dim=-1, value=0),
                rope_cache=_pad_seq(item.rope_cache, dim=-2, value=0.0),
                gen_image_mask=_pad_seq(item.gen_image_mask, dim=-1, value=False),
                gen_timestep_scatter_index=item.gen_timestep_scatter_index,
                cond_vae_image_mask=_pad_seq(item.cond_vae_image_mask, dim=-1, value=False),
                cond_vit_image_mask=_pad_seq(item.cond_vit_image_mask, dim=-1, value=False),
                cond_timestep_scatter_index=item.cond_timestep_scatter_index,
                prompt_lengths=item.prompt_lengths,  # [B] — not L-padded
            )
            for item in items
        ]

        from unirl.distributed.tensor.batch import Batch

        return Batch.concat.__func__(cls, padded_items)


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
