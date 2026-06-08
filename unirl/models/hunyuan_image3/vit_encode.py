"""HunyuanImage3VitEncodeStage — Images → vision-tower features.

Two complementary paths:

- :meth:`encode` — bare ViT pass. Returns
  :class:`ImageEmbedCondition` ``(embeds=[B, num_patches, hidden],
  attn_mask=[B, num_patches])``. Used by reward heads and any caller
  that wants the raw SigLIP2 patch grid.

- :meth:`encode_for_cond_vit` — canonical i2t / it2i input prep. Wraps
  upstream ``image_processor.preprocess`` to produce ``JointImageInfo``
  per sample, then assembles the cond_vit tensors and ``vit_kwargs``
  in the shape the unified-MM forward expects (mirrors
  ``HunyuanImage3ForCausalMM._encode_cond_image`` at
  ``hunyuan.py:2145``). Returns the joint info objects plus the cond
  tensors so callers can also assemble ``batch_message_list`` entries
  for the chat-template wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.primitives import Images

from .bundle import HunyuanImage3Bundle


class HunyuanImage3VitEncodeStage(EncodeStage[Images, ImageEmbedCondition]):
    """SigLIP2-based image → ImageEmbedCondition stage."""

    def __init__(self, bundle: HunyuanImage3Bundle) -> None:
        self.bundle = bundle

    def encode(self, p: Images) -> ImageEmbedCondition:
        """Encode pixel images into ViT patch embeddings.

        Pixel input convention: ``[B, C, H, W]`` in ``[0, 1]``. SigLIP2
        normalizes internally; we pass through after a ``[0,1] → [-1, 1]``
        rescale to match the upstream ``image_processor`` convention
        (``HunyuanImage-3.0/hunyuan_image_3/image_processor.py``).
        """
        if p.pixels is None:
            raise ValueError("HunyuanImage3VitEncodeStage.encode: pixels is None")

        x = p.pixels.to(self.bundle.device).to(self.bundle.dtype)
        # [0, 1] → [-1, 1] mirroring upstream image_processor.
        x = x * 2.0 - 1.0

        with torch.no_grad():
            out = self.bundle.vit(x)
        # SigLIP2 forward returns either a tensor (last_hidden_state) or an
        # object with a ``last_hidden_state`` attribute; tolerate both.
        embeds = getattr(out, "last_hidden_state", out)
        if not isinstance(embeds, torch.Tensor):
            raise TypeError(
                f"HunyuanImage3VitEncodeStage.encode: ViT returned non-tensor output of type {type(embeds).__name__}"
            )

        attn_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        return ImageEmbedCondition(embeds=embeds, attn_mask=attn_mask)

    # ------------------------------------------------------------------
    # Chat-template-driven input prep -- canonical i2t / it2i entry point.
    # ------------------------------------------------------------------

    def encode_for_cond_vit(self, p: Images) -> Dict[str, Any]:
        """Prep cond-image features for the unified MM forward.

        Mirrors ``HunyuanImage3ForCausalMM._encode_cond_image``
        (``hunyuan.py:2145``) for the *vision-only* path used by i2t and
        the comprehension half of it2i. Runs the upstream
        ``image_processor.preprocess`` per-sample (it expects PIL
        ``RGB``), unpacks the resulting ``JointImageInfo`` into the
        cond_vit tensors and the ``vit_kwargs`` dict shape SigLIP2 needs.

        Args:
            p: ``Images`` primitive carrying ``pixels: [B, 3, H, W]``
                float in ``[0, 1]``.

        Returns:
            Dict with the following keys (let ``B = p.pixels.shape[0]``,
            ``S_b`` = SigLIP2 patch count for sample b, ``D`` = ViT
            hidden width):

                joint_image_info  : list[list[JointImageInfo]] of length B,
                                    each inner list of length 1 -- one cond
                                    image per sample. Forwarded by
                                    ``modes/i2t.generate`` /
                                    ``modes/it2i.generate`` as
                                    ``batch_cond_image_info`` to the chat
                                    template.
                cond_vit_images   : list[Tensor [S_b, D]] of length B
                                    (per-sample ViT pixel tensors -- the
                                    upstream forward instantiates patch
                                    embeddings at ``<img>`` positions via
                                    ``instantiate_vit_image_tokens``).
                vit_kwargs        : {"spatial_shapes": list[Tensor [2]],
                                     "attention_mask": list[Tensor [S_b]]}
                                    -- per-sample SigLIP2 sizing info.
        """
        if p.pixels is None:
            raise ValueError("HunyuanImage3VitEncodeStage.encode_for_cond_vit: pixels is None")
        from torchvision.transforms.functional import to_pil_image

        transformer = self.bundle.transformer
        image_processor = getattr(transformer, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(
                "HunyuanImage3VitEncodeStage.encode_for_cond_vit: bundle's "
                "transformer has no .image_processor (unloaded checkpoint?)."
            )

        pixels = p.pixels
        if pixels.dim() != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"HunyuanImage3VitEncodeStage.encode_for_cond_vit: pixels must "
                f"be [B, 3, H, W], got {tuple(pixels.shape)}"
            )

        # Convert each sample to PIL RGB for upstream's image_processor.
        # Each sample's tensors are stacked across its (potentially multiple)
        # cond images so the per-sample shape is ``[n_cond, ...]`` -- the
        # convention the unified-MM forward iterates with at
        # ``hunyuan.py:1903-1904``.
        joint_image_info: List[List[Any]] = []
        cond_vit_images: List[torch.Tensor] = []
        spatial_shapes_list: List[torch.Tensor] = []
        attn_mask_list: List[torch.Tensor] = []
        for b in range(int(pixels.shape[0])):
            pil_image = to_pil_image(pixels[b].clamp(0.0, 1.0).float().cpu())
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            info = image_processor.preprocess(pil_image)
            joint_image_info.append([info])

            # vision_image_info.image_tensor: [1, S, D] -- keep the leading
            # 1-dim so the per-sample tensor is [n_cond=1, S, D].
            cond_vit_images.append(info.vision_image_info.image_tensor)

            ve_kwargs = info.vision_encoder_kwargs
            # Stack across the per-sample cond-image list (length 1 here)
            # to produce [n_cond, ...] tensors per sample.
            spatial_shapes_list.append(torch.stack([ve_kwargs["spatial_shapes"]], dim=0))  # [1, 2]
            attn_mask_list.append(torch.stack([ve_kwargs["pixel_attention_mask"]], dim=0))  # [1, num_patches]

        return {
            "joint_image_info": joint_image_info,
            "cond_vit_images": cond_vit_images,
            "vit_kwargs": {
                "spatial_shapes": spatial_shapes_list,
                "attention_mask": attn_mask_list,
            },
        }


__all__ = ["HunyuanImage3VitEncodeStage"]
