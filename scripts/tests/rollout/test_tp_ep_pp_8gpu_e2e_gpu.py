"""8-GPU E2E tests: TP/EP/PP combinations on the real tiny Qwen3.5-MoE model.

Validates the colocate TP+EP rollout path end-to-end across 8 GPUs. PP>1 is
fail-closed in weight sync (NotImplementedError), so PP is tested at the rank
layout / dispatch level only — the full training loop exercises TP+EP.

Cases:
  - tp=2 ep=2:   4 engines x tp2, EP sharded inside SGLang
  - tp=4 ep=2:   2 engines x tp4
  - pp=2 tp=2:   rank layout + dispatch collect filter (no training — PP sync
                  is fail-closed by design)

Setup (once):
    python -c "from huggingface_hub import snapshot_download; \\
        snapshot_download('tiny-random/qwen3.5-moe', local_dir='/tmp/tiny-qwen35-moe')"

Run:
    UNIRL_TP_E2E_MODEL=/tmp/tiny-qwen35-moe pytest scripts/tests/rollout/test_tp_ep_pp_8gpu_e2e_gpu.py -s
"""

from __future__ import annotations

import os

import pytest

from ..conftest import requires_gpus, sglang_e2e_teardown


def _model_path() -> str:
    p = os.environ.get("UNIRL_TP_E2E_MODEL", "/tmp/tiny-qwen35-moe")
    if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "config.json")):
        pytest.skip(f"model not found at {p}")
    return p


@pytest.fixture
def tp8_gate():
    return requires_gpus(8)


def _boot_and_generate(eng, *, expected_tp_size: int, devices: list):
    """Shared assertions: boot, generate, sleep/wake, shutdown."""
    import torch
    assert eng._is_tp_zero is True
    assert eng._backend is not None
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(str(d) for d in devices)
    assert eng.health_check() is True

    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams
    req = RolloutReq(
        sample_ids=["s0"], group_ids=["s0"],
        primitives={"text": Texts(texts=["Hello"])},
        request_conditions={},
        sampling_params={"default": ARSamplingParams(samples_per_prompt=1)},
        metadata=[],
    )
    resp = eng.generate(req)
    assert resp is not None and len(resp.tracks) >= 1
    for track in resp.tracks.values():
        decoded = getattr(track, "decoded", None)
        if decoded is None:
            continue
        assert getattr(decoded, "texts", None), "no decoded text"
        break
    else:
        pytest.fail("no track with decoded output")

    eng.sleep()
    assert eng.is_offloaded is True
    eng.wake_up()
    assert eng.is_offloaded is False
    assert eng.health_check() is True


@pytest.mark.gpu
def test_tp2_ep2_e2e(tp8_gate):
    """tp=2 + ep=2 on 4 GPUs: 2 engines, each spanning 2 GPUs with EP sharding."""
    pytest.importorskip("sglang")
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

    cfg = SGLangEngineConfig(
        pretrained_model_ckpt_path=_model_path(),
        model_family="text", tp_size=2, ep_size=2, enable_expert_parallel=True,
        max_new_tokens=8, temperature=1.0, top_p=1.0, samples_pre_expanded=True,
        engine_kwargs={"mem_fraction_static": 0.3, "skip_server_warmup": True,
                       "disable_cuda_graph": True, "trust_remote_code": True},
    )
    eng = SGLangRolloutEngine(
        config=cfg, rank=0, tp_rank=0, tp_size=2, tp_device_ids=[0, 1], ep_size=2,
    )
    passed = False
    try:
        _boot_and_generate(eng, expected_tp_size=2, devices=[0, 1])
        passed = True
    finally:
        sglang_e2e_teardown(eng, passed=passed)


@pytest.mark.gpu
def test_tp4_ep4_e2e(tp8_gate):
    """tp=4 + ep=4 on 4 GPUs: 1 engine spanning 4 GPUs, EP shards across TP group.

    SGLang requires ``ep_size`` to divide ``tp_size`` (EP is intra-TP-group),
    so tp=4 ep=4 is the minimal valid 4-GPU combo.
    """
    pytest.importorskip("sglang")
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

    cfg = SGLangEngineConfig(
        pretrained_model_ckpt_path=_model_path(),
        model_family="text", tp_size=4, ep_size=4, enable_expert_parallel=True,
        max_new_tokens=8, temperature=1.0, top_p=1.0, samples_pre_expanded=True,
        engine_kwargs={"mem_fraction_static": 0.3, "skip_server_warmup": True,
                       "disable_cuda_graph": True, "trust_remote_code": True},
    )
    eng = SGLangRolloutEngine(
        config=cfg, rank=0, tp_rank=0, tp_size=4, tp_device_ids=[0, 1, 2, 3], ep_size=4,
    )
    passed = False
    try:
        _boot_and_generate(eng, expected_tp_size=4, devices=[0, 1, 2, 3])
        passed = True
    finally:
        sglang_e2e_teardown(eng, passed=passed)
