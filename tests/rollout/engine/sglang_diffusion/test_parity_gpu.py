"""GPU integration: real-seam smoke + per-family parity gate vs the legacy engine.

GATED — skipped unless a CUDA device and the SGLang fork are both present, so CI
skips it. Run on the target box (with ``PRETRAINED_MODEL`` set) to validate the
``sglang_diffusion`` rewrite end-to-end and to gate it against the legacy ``sglang``
engine before retiring the latter:

    PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium \
        pytest tests/rollout/engine/sglang_diffusion/test_parity_gpu.py -v

What it checks (per the testing convention):
  - smoke: build the v2 engine, generate one tiny batch, run one tensor weight
    sync, and sleep/wake — proves the seam wiring is live.
  - parity: build the legacy ``sglang`` engine and the v2 ``sglang_diffusion``
    engine with an identical config / model_config / fixed seed, generate the same
    ``RolloutReq``, and assert the ``LatentSegment`` latents + σ schedule match
    (tolerance per impl-time decision D4 — start with allclose, tighten to
    bit-identity once the SDE-noise kernels are confirmed equal).

The driver/Handle/Ray placement path is exercised separately; here we construct the
engines directly so the gate is a single-process, two-engine comparison.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("sglang", reason="SGLang fork not installed")
if not torch.cuda.is_available():
    pytest.skip("parity gate needs a CUDA device", allow_module_level=True)

from unirl.rollout.engine.sglang.config import SGLangEngineConfig  # noqa: E402
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine  # noqa: E402
from unirl.rollout.engine.sglang_diffusion.config import (  # noqa: E402
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)
from unirl.rollout.engine.sglang_diffusion.engine import (  # noqa: E402
    SGLangDiffusionRolloutEngine,
)
from unirl.sde.kernels import FlowSDEStrategy  # noqa: E402
from unirl.models.sd3.config import SD3PipelineConfig  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.rollout_req import RolloutReq  # noqa: E402
from unirl.types.sampling import DiffusionSamplingParams  # noqa: E402

_CKPT = os.environ.get("PRETRAINED_MODEL", "stabilityai/stable-diffusion-3.5-medium")
_PROMPTS = ["a red cube on grass", "a blue sphere in snow"]


def _model_config():
    return SD3PipelineConfig(pretrained_model_ckpt_path=_CKPT, shift=3.0, use_lora=False)


def _req(*, seed=42, steps=4):
    sp = DiffusionSamplingParams(
        num_inference_steps=steps, height=512, width=512, guidance_scale=1.0,
        eta=0.7, seed=seed, samples_per_prompt=1, sde_indices=list(range(steps)),
    )
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(_PROMPTS))],
        group_ids=[f"g{i}" for i in range(len(_PROMPTS))],
        primitives={"text": Texts(texts=list(_PROMPTS))},
        sampling_params=sp,
    )


def _build_v2(ports: SGLangDiffusionPorts | None = None):
    cfg = SGLangDiffusionEngineConfig(
        sampling=None, model_family="sd3",
        populate_conditions=True, local_mode=True,
    )
    return SGLangDiffusionRolloutEngine(
        cfg, device=torch.device("cuda"), strategy=FlowSDEStrategy(),
        model_config=_model_config(), ports=ports,
    )


def _build_v1():
    cfg = SGLangEngineConfig(
        sampling=None, model_family="sd3",
        populate_conditions=True, local_mode=True,
    )
    return SGLangRolloutEngine(
        cfg, device=torch.device("cuda"), strategy=FlowSDEStrategy(),
        model_config=_model_config(), rank=0,
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
            ports.server_port, ports.scheduler_port, ports.master_port,
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


def test_parity_sd3_latents_match_legacy():
    new, old = _build_v2(), _build_v1()
    try:
        resp_new = new.generate(_req(seed=42))
        resp_old = old.generate(_req(seed=42))
        seg_new = resp_new.tracks["image"].segment
        seg_old = resp_old.tracks["image"].segment
        assert torch.equal(seg_new.sigmas, seg_old.sigmas), "σ schedule diverged"
        # Tolerance per D4; tighten to torch.equal once SDE kernels are confirmed equal.
        assert torch.allclose(
            seg_new.latents.float(), seg_old.latents.float(), atol=1e-3, rtol=1e-3
        ), "v2 latents diverged from legacy sglang for a fixed seed"
    finally:
        new.shutdown()
        old.shutdown()
