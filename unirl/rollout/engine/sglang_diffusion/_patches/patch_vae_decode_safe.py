"""Force VAE decode to use non-cuDNN convolutions (opt-in diagnostic patch)."""

from __future__ import annotations

import os

import torch


def patch_vae_decode_safe() -> None:
    if os.environ.get("UNIRL_DISABLE_CUDNN") != "1" and os.environ.get("DIFFRL_DISABLE_CUDNN") != "1":
        return

    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )

    orig = DecodingStage.forward
    if getattr(orig, "_unirl_disable_cudnn", False):
        return

    def forward(self, batch, server_args):
        torch.backends.cudnn.enabled = False
        return orig(self, batch, server_args)

    forward._unirl_disable_cudnn = True  # type: ignore[attr-defined]
    DecodingStage.forward = forward
