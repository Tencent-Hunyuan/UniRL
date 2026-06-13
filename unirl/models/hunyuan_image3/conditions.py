"""HunyuanImage3 typed conditions containers.

Two containers, one per Stage. Both compose generic primitives
(``FusedMultimodalCondition`` subclass + ``ImageLatentCondition`` /
``ImageEmbedCondition``) plus a small number of Hunyuan-specific flat
fields.

- ``HunyuanImage3FusedMultimodalCondition`` — subclass of
  ``FusedMultimodalCondition``. Inherits the 4 sequence-level fields
  (``input_ids`` / ``attention_mask`` / ``position_ids`` / ``rope_cache``)
  and adds 5 scatter-layout fields that pin Hunyuan's roles into the
  fused sequence (``gen_image_mask``, ``gen_timestep_scatter_index``,
  ``cond_vae_image_mask``, ``cond_vit_image_mask``,
  ``cond_timestep_scatter_index``). Used by both diffusion (uses all 5)
  and AR (uses only ``cond_vit_image_mask``).

- ``HunyuanImage3DiffusionConditions`` — what the DiT-mode forward
  consumes. Composes the fused condition + ``cond_vae`` (it2i) +
  ``cond_vit`` (it2i) + per-cond ``cond_timestep`` values + opaque
  ``tokenizer_output`` for the KV-cache gather on step 0.

- ``HunyuanImage3ARConditions`` — what the AR-mode forward consumes
  (t2t / i2t / and the prefix passes inside t2i / it2i). Composes the
  fused condition + ``cond_vit`` (i2t / it2i) + opaque
  ``tokenizer_output`` for the AR ``real_pos`` derivation on step 0.

Pairs ``from_dict`` / ``to_dict`` for round-tripping between the typed
form (used inside the pipeline at stage call sites) and the generic
``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import (
    FusedMultimodalCondition,
    ImageEmbedCondition,
    ImageLatentCondition,
)


@dataclass
class HunyuanImage3FusedMultimodalCondition(FusedMultimodalCondition):
    """Hunyuan's fused-sequence layout.

    Inherits the 4 sequence-level fields from ``FusedMultimodalCondition``
    and adds 5 scatter-layout fields that describe where each Hunyuan-
    specific role's encoded content lands in the fused sequence.

    Used by both diffusion (``mode="gen_image"``, uses all 5 scatter
    fields) and AR (``mode="gen_text"``, uses only
    ``cond_vit_image_mask``). Unused scatter fields stay ``None`` per
    mode/path.

    Field shapes (let ``B = batch``, ``L = fused sequence length``, ``K``
    typically 1):

        gen_image_mask              : [B, L] bool — generated-image patch slots
        gen_timestep_scatter_index  : [B, K] long — gen timestep token slot
        cond_vae_image_mask         : [B, L] bool — cond VAE latent slots (it2i)
        cond_vit_image_mask         : [B, L] bool — cond ViT patch-embed slots (i2t/it2i)
        cond_timestep_scatter_index : [B, K] long — cond timestep token slot (it2i)

    ``gen_image_mask`` is dual-use: it both scatters the noisy latent
    into ``inputs_embeds`` on every diffusion step *and* identifies the
    positions where the diffusion head reads predicted noise on the way
    out.
    """

    gen_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    gen_timestep_scatter_index: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, K] long
    cond_vae_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    cond_vit_image_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, L] bool
    cond_timestep_scatter_index: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [B, K] long
    # Per-sample TRUE prompt length [B] long — set only on the two-engine AR
    # path (``response._build_ar_fused_condition``), where ``input_ids`` is
    # right-padded across variable-length per-request prompts. ``ARStage.replay``
    # uses it to slice each sample's real prompt (no padding corruption). A plain
    # 1D CONCAT field (cat, not _pad_attn) so it survives Batch merges.
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
        """Override ``Batch.concat`` to pad variable-length L dims before cat.

        In think_recaption mode, different prompts produce different AR token
        counts → different fused sequence lengths L. The base ``Batch.concat``
        does a plain ``torch.cat(dim=0)`` on CONCAT fields, which fails when L
        differs across items (e.g. cross-actor merge).

        This override pads all items to ``max_L`` on the L dimension before
        delegating to the base concat.
        """
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

        def _pad_seq(t, dim=-1, value=0):
            if t is None:
                return None
            cur = t.shape[dim]
            if cur >= max_L:
                return t
            pad_size = max_L - cur
            ndim = t.ndim
            pad_spec = [0] * (2 * ndim)
            actual_dim = dim if dim >= 0 else ndim + dim
            pad_idx = (ndim - 1 - actual_dim) * 2
            pad_spec[pad_idx + 1] = pad_size
            return torch.nn.functional.pad(t, pad_spec, value=value)

        def _pad_attn(mask):
            if mask is None:
                return None
            if mask.shape[-1] >= max_L:
                return mask
            N, H, L, _ = mask.shape
            padded = torch.zeros(N, H, max_L, max_L, dtype=mask.dtype, device=mask.device)
            padded[:, :, :L, :L] = mask
            return padded

        padded_items = []
        for item in items:
            if item.input_ids is not None and item.input_ids.shape[-1] < max_L:
                rope = item.rope_cache
                if rope is not None and isinstance(rope, tuple) and len(rope) == 2:
                    rope = (
                        _pad_seq(rope[0], dim=-2, value=0.0),
                        _pad_seq(rope[1], dim=-2, value=0.0),
                    )
                padded_items.append(
                    cls(
                        input_ids=_pad_seq(item.input_ids, dim=-1, value=0),
                        attention_mask=_pad_attn(item.attention_mask),
                        position_ids=_pad_seq(item.position_ids, dim=-1, value=0),
                        rope_cache=rope,
                        gen_image_mask=_pad_seq(item.gen_image_mask, dim=-1, value=False),
                        gen_timestep_scatter_index=item.gen_timestep_scatter_index,
                        cond_vae_image_mask=_pad_seq(item.cond_vae_image_mask, dim=-1, value=False)
                        if item.cond_vae_image_mask is not None
                        else None,
                        cond_vit_image_mask=_pad_seq(item.cond_vit_image_mask, dim=-1, value=False)
                        if item.cond_vit_image_mask is not None
                        else None,
                        cond_timestep_scatter_index=item.cond_timestep_scatter_index,
                        prompt_lengths=item.prompt_lengths,  # [B] — not L-padded
                    )
                )
            else:
                padded_items.append(item)

        from unirl.distributed.tensor.batch import Batch

        return Batch.concat.__func__(cls, padded_items)


@dataclass
class HunyuanImage3DiffusionConditions(Batch):
    """Typed conditions container for HunyuanImage3 DiT-mode diffusion.

    Composes the fused condition + cond-image primitives (it2i) + a few
    Hunyuan-specific flat fields.

        fused           : HunyuanImage3FusedMultimodalCondition
                          carries input_ids, 4D attention_mask, position_ids,
                          rope_cache, plus all 5 scatter masks/indices
        cond_vae        : ImageLatentCondition — cond VAE latents (it2i)
        cond_vit        : ImageEmbedCondition — cond ViT patch embeds + attn
                          mask + spatial_shapes (it2i)
        cond_timestep   : Tensor — per-cond t values being scattered (it2i;
                          data, not a destination index)
        tokenizer_output: opaque upstream apply_chat_template output, used
                          on step 0 to drive the KV-cache gather-down
    """

    fused: Optional[HunyuanImage3FusedMultimodalCondition] = field(kind=FieldKind.SHARED, default=None)
    cond_vae: Optional[ImageLatentCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_vit: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_timestep: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    # Opaque ``apply_chat_template`` output — used by the KV-cache path's
    # first ``_update_model_kwargs_for_generation`` call to drive the
    # gather-down from the full L sequence to the L' changed slice. Carries
    # ``joint_image_slices`` / ``gen_image_slices`` / etc. internally; we
    # treat it as opaque. Non-transportable; lives only for one diffuse()
    # call. ``None`` means the kernel falls back to the stateless path.
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
        if cond_vae is not None and not isinstance(cond_vae, ImageLatentCondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['cond_vae'] "
                f"to be an ImageLatentCondition or absent, "
                f"got {type(cond_vae).__name__}"
            )
        cond_vit = d.get("cond_vit")
        if cond_vit is not None and not isinstance(cond_vit, ImageEmbedCondition):
            raise TypeError(
                f"HunyuanImage3DiffusionConditions.from_dict: expected d['cond_vit'] "
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
        """Convert back to the generic ``Conditions`` dict shape."""
        if self.fused is None or self.fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3DiffusionConditions.to_dict: `fused.input_ids` is "
                "None — required for the diffusion stage to consume."
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


@dataclass
class HunyuanImage3ARConditions(Batch):
    """Typed conditions container for HunyuanImage3 AR-mode autoregress.

    Used by t2t / i2t / and the prefix passes inside t2i / it2i.

        fused           : HunyuanImage3FusedMultimodalCondition
                          carries input_ids, 4D attention_mask, position_ids,
                          rope_cache, plus cond_vae_image_mask /
                          cond_vit_image_mask / cond_timestep_scatter_index (i2t/it2i)
        cond_vae        : ImageLatentCondition — cond VAE latents (i2t/it2i).
                          HI3-Instruct represents a cond image as VAE + ViT
                          tokens, so comprehension needs BOTH halves scattered
                          (ViT-only leaves the VAE <img> slots as bare
                          embeddings → garbage comprehension).
        cond_vit        : ImageEmbedCondition — cond ViT patch embeds + attn
                          mask + spatial_shapes (i2t/it2i)
        cond_timestep   : Tensor — per-cond t values for the VAE-latent scatter
                          (clean cond image → t≈0)
        tokenizer_output: opaque upstream apply_chat_template output, used
                          on step 0 to derive position_ids from real_pos
                          for right-padded batches
    """

    fused: Optional[HunyuanImage3FusedMultimodalCondition] = field(kind=FieldKind.SHARED, default=None)
    cond_vae: Optional[ImageLatentCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_vit: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    cond_timestep: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
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
        if cond_vae is not None and not isinstance(cond_vae, ImageLatentCondition):
            raise TypeError(
                f"HunyuanImage3ARConditions.from_dict: expected d['cond_vae'] "
                f"to be an ImageLatentCondition or absent, "
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
]
