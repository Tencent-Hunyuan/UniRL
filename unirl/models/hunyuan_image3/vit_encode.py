"""HunyuanImage3VitEncodeStage — Images → vision-tower features."""

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
        """Encode pixel images into ViT patch embeddings."""
        try:
            x = p.to_dense()
        except ValueError as exc:
            raise ValueError(
                "HunyuanImage3VitEncodeStage.encode requires uniform image shapes; "
                "use encode_for_cond_vit for native mixed-resolution inputs"
            ) from exc
        x = x.to(self.bundle.device).to(self.bundle.dtype)
        # [0, 1] → [-1, 1] mirroring upstream image_processor.
        x = x * 2.0 - 1.0

        with torch.no_grad():
            out = self.bundle.vit(x)
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
        """Prep cond-image features for the unified MM forward."""
        transformer = self.bundle.transformer
        image_processor = getattr(transformer, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(
                "HunyuanImage3VitEncodeStage.encode_for_cond_vit: bundle's "
                "transformer has no .image_processor (unloaded checkpoint?)."
            )

        joint_image_info: List[List[Any]] = []
        cond_vit_images: List[torch.Tensor] = []
        spatial_shapes_list: List[torch.Tensor] = []
        attn_mask_list: List[torch.Tensor] = []
        for pil_image in p.to_pils():
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            if hasattr(image_processor, "preprocess"):
                info = image_processor.preprocess(pil_image)
                cond_item = info
                vit_tensor = info.vision_image_info.image_tensor
                ve_kwargs = info.vision_encoder_kwargs
            else:
                cond_image = image_processor.get_image_with_size(
                    pil_image, return_type=image_processor.cond_image_type
                )[0]
                cond_item = cond_image
                vit_t = cond_image.vit_image
                vit_tensor = vit_t.unsqueeze(0) if vit_t.dim() == 2 else vit_t
                ve_kwargs = vit_t.vision_encoder_kwargs
            joint_image_info.append([cond_item])

            cond_vit_images.append(vit_tensor)

            spatial_shapes_list.append(torch.stack([ve_kwargs["spatial_shapes"]], dim=0))
            attn_mask_list.append(torch.stack([ve_kwargs["pixel_attention_mask"]], dim=0))

        return {
            "joint_image_info": joint_image_info,
            "cond_vit_images": cond_vit_images,
            "vit_kwargs": {
                "spatial_shapes": spatial_shapes_list,
                "attention_mask": attn_mask_list,
            },
        }


__all__ = ["HunyuanImage3VitEncodeStage"]
