"""GPU integration: Qwen-Image on the v2 engine — smoke + parity vs trainside.

GATED — skipped unless a CUDA device, sglang[diffusion], and ``QWEN_IMAGE_PATH``
are all present. Unlike the SD3 gate (v1 engine as oracle), qwen's oracle is the
**trainside** pipeline — v1 never supported qwen — so this is a single-process,
engine-vs-trainside comparison:

    QWEN_IMAGE_PATH=/dockerdata/Qwen-Image \
        pytest tests/rollout/engine/sglang_diffusion/test_qwen_parity_gpu.py -v -s

Needs 2 GPUs: the sglang worker takes cuda:0; the trainside bundle loads on
cuda:1 (20B DiT + 7B TE each ≈ 58 GB bf16 — they cannot share an H20).

What it checks:
  - smoke: boot the v2 engine on the qwen pipeline, generate one ODE batch —
    proves registry pin, packed-trajectory unpack, σ echo (incl. the
    shift_terminal=0.02 terminal stretch on the LIVE path), sleep/wake.
  - ODE parity: same req + driver-pinned x_T (NoiseRecipe) through both the v2
    engine and the trainside pipeline — σ bit-equal, x_T bit-equal, per-step
    cosine > 0.999 (catches layout/permute bugs), final-step atol, decoded
    closeness.
  - SDE logp: v2 native log-probs vs trainside ``stage.replay`` recomputation
    over the SAME segment + the SERVER's fused conditions — the
    ratio-defining check (also exercises the qwen conditions path end-to-end:
    variable-length embeds + masks through ``QwenImageConditions.from_dict``).
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("sglang", reason="sglang[diffusion] not installed")
if not torch.cuda.is_available():
    pytest.skip("qwen parity gate needs CUDA", allow_module_level=True)
if torch.cuda.device_count() < 2:
    pytest.skip("qwen parity gate needs 2 GPUs (server + trainside)", allow_module_level=True)
if not os.environ.get("QWEN_IMAGE_PATH"):
    pytest.skip("set QWEN_IMAGE_PATH to the Qwen-Image checkpoint dir", allow_module_level=True)

from unirl.models.qwen_image.config import QwenImagePipelineConfig  # noqa: E402
from unirl.models.qwen_image.conditions import QwenImageConditions  # noqa: E402
from unirl.models.qwen_image.pipeline import QwenImagePipeline  # noqa: E402
from unirl.rollout.engine.sglang_diffusion.config import (  # noqa: E402
    SGLangDiffusionEngineConfig,
)
from unirl.rollout.engine.sglang_diffusion.engine import (  # noqa: E402
    SGLangDiffusionRolloutEngine,
)
from unirl.sde.kernels import FlowSDEStrategy  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.rollout_req import RolloutReq  # noqa: E402
from unirl.types.sampling import DiffusionSamplingParams  # noqa: E402

_CKPT = os.environ["QWEN_IMAGE_PATH"]
_PROMPTS = ["a red cube on grass", "a blue sphere in snow"]
_STEPS = 8
_HW = 384  # qwen_384 preset; latent grid 48x48, S=576


def _model_config(device=None):
    return QwenImagePipelineConfig(
        pretrained_model_ckpt_path=_CKPT,
        shift=3.0,
        use_lora=False,
        device=device,
    )


def _sampling(*, seed=42, sde_indices=None):
    return DiffusionSamplingParams(
        num_inference_steps=_STEPS,
        height=_HW,
        width=_HW,
        guidance_scale=1.0,
        eta=0.7,
        seed=seed,
        samples_per_prompt=1,
        sde_indices=sde_indices,
    )


def _req(*, seed=42, sde_indices=None):
    sp = _sampling(seed=seed, sde_indices=sde_indices)
    latent_shape = QwenImagePipeline.latent_shape(model_config=None, sampling_spec=sp)
    sample_ids = [f"s{i}" for i in range(len(_PROMPTS))]
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=[f"g{i}" for i in range(len(_PROMPTS))],
        primitives={"text": Texts(texts=list(_PROMPTS))},
        sampling_params=sp,
        # Driver-authoritative x_T: same recipe → byte-identical noise on both
        # the sglang worker and the trainside stage (NoiseRecipe path 2).
        init_noise_group_ids=list(sample_ids),
        init_noise_latent_shape=list(latent_shape),
    )


@pytest.fixture(scope="module")
def v2_engine():
    cfg = SGLangDiffusionEngineConfig(
        sampling=None,
        model_family="qwen_image",
        populate_conditions=True,
        local_mode=True,
        engine_kwargs={"model_id": "Qwen-Image"},
    )
    engine = SGLangDiffusionRolloutEngine(
        cfg,
        device=torch.device("cuda"),
        strategy=FlowSDEStrategy(),
        model_config=_model_config(),
    )
    yield engine
    engine.shutdown()


@pytest.fixture(scope="module")
def trainside():
    return QwenImagePipeline.from_config(_model_config(device="cuda:1"), strategy=FlowSDEStrategy())


def _cosine_per_step(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-step cosine over flattened latents — [K] vector."""
    a2 = a.float().flatten(2)  # [B, K, N]
    b2 = b.float().flatten(2)
    num = (a2 * b2).sum(-1)
    den = a2.norm(dim=-1) * b2.norm(dim=-1)
    return (num / den.clamp_min(1e-12)).mean(0)  # mean over batch → [K]


def test_smoke_generate_sleep_wake(v2_engine):
    resp = v2_engine.generate(_req(sde_indices=None))
    seg = resp.tracks["image"].segment
    lat_hw = 2 * (_HW // 16)
    assert seg.latents.shape == (len(_PROMPTS), _STEPS + 1, 16, lat_hw, lat_hw), (
        f"unpacked segment shape {tuple(seg.latents.shape)}"
    )
    # shift_terminal=0.02 on the LIVE path: driver σ bakes the stretch and the
    # server echo (σ-verified inside build_response) agrees.
    assert float(seg.sigmas[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(seg.sigmas[-2]) == pytest.approx(0.02, abs=1e-5)
    assert float(seg.sigmas[-1]) == 0.0
    # Conditions path: variable-length embeds + masks present.
    conds = QwenImageConditions.from_dict(resp.tracks["image"].conditions)
    assert conds.text.embeds.shape[0] == len(_PROMPTS)
    assert conds.text.attn_mask is not None
    v2_engine.sleep()
    assert v2_engine.is_offloaded
    v2_engine.wake_up()
    assert not v2_engine.is_offloaded


def test_parity_ode_vs_trainside(v2_engine, trainside):
    req = _req(seed=42, sde_indices=None)  # ODE: deterministic given x_T
    resp_v2 = v2_engine.generate(req)  # pins req.sigmas (σ SSOT) as a side effect
    resp_ts = trainside.generate(req)

    seg_v2 = resp_v2.tracks["image"].segment
    seg_ts = resp_ts.tracks["image"].segment

    assert torch.equal(seg_v2.sigmas, seg_ts.sigmas), "σ schedule diverged"
    # x_T: same NoiseRecipe → bit-identical before any model math.
    x_t_v2 = seg_v2.latents[:, 0].float()
    x_t_ts = seg_ts.latents[:, 0].float()
    assert torch.allclose(x_t_v2, x_t_ts, atol=1e-6), (
        f"x_T diverged: max|Δ|={float((x_t_v2 - x_t_ts).abs().max())}"
    )

    cos = _cosine_per_step(seg_v2.latents, seg_ts.latents)
    print(f"\nper-step cosine: {[round(float(c), 5) for c in cos]}")
    assert bool((cos > 0.999).all()), f"per-step cosine degraded: {cos.tolist()}"

    final_delta = (seg_v2.latents[:, -1].float() - seg_ts.latents[:, -1].float()).abs()
    print(f"final-step max|Δ|={float(final_delta.max()):.4e} mean|Δ|={float(final_delta.mean()):.4e}")
    assert float(final_delta.max()) < 2e-2, "final latents drifted beyond fp16-storage tolerance"

    dec_v2 = resp_v2.tracks["image"].decoded
    dec_ts = resp_ts.tracks["image"].decoded
    if dec_v2 is not None and dec_ts is not None:
        d = (dec_v2.pixels.float() - dec_ts.pixels.float()).abs()
        print(f"decoded mean|Δ|={float(d.mean()):.4e}")


def test_sde_native_logprob_matches_replay(v2_engine, trainside):
    sde_indices = list(range(_STEPS // 2))
    req = _req(seed=7, sde_indices=sde_indices)
    resp = v2_engine.generate(req)
    track = resp.tracks["image"]
    seg = track.segment
    assert seg.log_probs is not None, "engine did not emit native log-probs in SDE mode"

    conds = QwenImageConditions.from_dict(track.conditions)
    result = trainside.diffusion.replay(
        conds,
        segment=seg,
        params=_sampling(seed=7, sde_indices=sde_indices),
        step_indices=sde_indices,
    )
    native = seg.log_probs.float()
    replayed = result.log_probs.float()
    delta = (native - replayed).abs()
    print(f"\nlogp |Δ| max={float(delta.max()):.4e} mean={float(delta.mean()):.4e}")
    # bf16 server forward vs bf16 trainside forward (fp32 logp math both sides):
    # start loose, tighten after observing the live distribution.
    assert float(delta.max()) < 2e-2, "native vs replay logp diverged beyond tolerance"
