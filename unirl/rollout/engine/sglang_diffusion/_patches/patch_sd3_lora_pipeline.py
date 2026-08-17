"""Make SD3's rollout pipeline LoRA-capable on stock upstream sglang."""

from __future__ import annotations


def patch_sd3_lora_pipeline() -> None:
    from sglang.multimodal_gen.runtime.pipelines.stable_diffusion_3 import (
        StableDiffusion3Pipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
        LoRAPipeline,
    )

    if LoRAPipeline not in StableDiffusion3Pipeline.__bases__:
        StableDiffusion3Pipeline.__bases__ = (LoRAPipeline,) + StableDiffusion3Pipeline.__bases__

    LoRAPipeline.register(StableDiffusion3Pipeline)


__all__ = ["patch_sd3_lora_pipeline"]
