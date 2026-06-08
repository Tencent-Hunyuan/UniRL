"""Force VAE decode to use non-cuDNN convolutions (LIN-365 diagnostic patch).

Symptom: on cuda-compat-13 + driver 535, sglang's AutoencoderKLFlux2._decode
crashes with ``munmap_chunk(): invalid pointer`` inside a Conv2d._conv_forward.
The crash is one-worker-at-a-time and non-deterministic in worker ID, which
points to a cuDNN/forward-compat layer interaction rather than bad input data.

This patch disables cuDNN globally inside the rollout subprocess (and resets it
back to True if anybody re-enables it). It is opt-in via the env var
``UNIRL_DISABLE_CUDNN=1`` so we can A/B it cleanly. When the env var is set,
``torch.backends.cudnn.enabled = False`` is asserted before EVERY
``DecodingStage.forward`` call (cheap; just a flag toggle). Conv kernels then
fall back to PyTorch's native CUDA implementation, which doesn't go through
cuDNN's plan cache / allocator paths.

When the env var is unset (default), this patch is a no-op.
"""

from __future__ import annotations

import os

import torch


def patch_vae_decode_safe() -> None:
    # DIFFRL_DISABLE_CUDNN is the legacy (pre-rename) name; the fleet venv
    # sitecustomize.py paired with PR #285 still sets it -- accept both.
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
