"""GPU integration: real-seam smoke for the ``sglang_diffusion`` engine.

GATED — skipped unless a CUDA device and SGLang are both present, so CI skips it.
Run on the target box (with ``PRETRAINED_MODEL`` set) to validate the engine
end-to-end:

    PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \
        pytest tests/rollout/engine/sglang_diffusion/test_parity_gpu.py -v

What it checks:
  - smoke: build the v2 engine, assert ServerArgs keeps the reserved ports,
    generate one tiny batch, and sleep/wake — proves the seam wiring is live.

Per-family numeric parity is gated against the **trainside oracle** (the
ground-truth replay), not the now-removed legacy ``sglang`` engine — see the
qwen trainside-oracle gate for the pattern.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("sglang", reason="SGLang fork not installed")
if not torch.cuda.is_available():
    pytest.skip("parity gate needs a CUDA device", allow_module_level=True)

from unirl.models.sd3.config import SD3PipelineConfig  # noqa: E402
from unirl.rollout.engine.sglang_diffusion.config import (  # noqa: E402
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)
from unirl.rollout.engine.sglang_diffusion.engine import (  # noqa: E402
    SGLangDiffusionRolloutEngine,
)
from unirl.sde.kernels import FlowSDEStrategy  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.rollout_req import RolloutReq  # noqa: E402
from unirl.types.sampling import DiffusionSamplingParams  # noqa: E402

_CKPT = os.environ.get("PRETRAINED_MODEL", "stabilityai/stable-diffusion-3.5-medium")
_PROMPTS = ["a red cube on grass", "a blue sphere in snow"]


def _model_config():
    return SD3PipelineConfig(pretrained_model_ckpt_path=_CKPT, shift=3.0, use_lora=False)


def _req(*, seed=42, steps=4):
    sp = DiffusionSamplingParams(
        num_inference_steps=steps,
        height=512,
        width=512,
        guidance_scale=1.0,
        eta=0.7,
        seed=seed,
        samples_per_prompt=1,
        sde_indices=list(range(steps)),
    )
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(_PROMPTS))],
        group_ids=[f"g{i}" for i in range(len(_PROMPTS))],
        primitives={"text": Texts(texts=list(_PROMPTS))},
        sampling_params=sp,
    )


def _build_v2(ports: SGLangDiffusionPorts | None = None):
    cfg = SGLangDiffusionEngineConfig(
        sampling=None,
        model_family="sd3",
        populate_conditions=True,
        local_mode=True,
    )
    return SGLangDiffusionRolloutEngine(
        cfg,
        device=torch.device("cuda"),
        strategy=FlowSDEStrategy(),
        model_config=_model_config(),
        ports=ports,
    )


def test_smoke_generate_sleep_wake():
    ports = SGLangDiffusionPorts.reserve()
    engine = _build_v2(ports)
    try:
        # Bind-mapping gate: ServerArgs must keep the reserved ports verbatim
        # (``settle_port`` leaves a free port alone). Guards fork bumps that move
        # which ServerArgs fields the runtime actually consumes — the regression
        # cover for dropping the MASTER_PORT env-scope.
        sa = engine._backend._server_args
        assert (sa.port, sa.scheduler_port, sa.master_port) == (
            ports.server_port,
            ports.scheduler_port,
            ports.master_port,
        )
        resp = engine.generate(_req())
        seg = resp.tracks["image"].segment
        assert seg is not None and seg.latents.shape[0] == len(_PROMPTS)
        engine.sleep()
        assert engine.is_offloaded
        engine.wake_up()
        assert not engine.is_offloaded
    finally:
        engine.shutdown()
