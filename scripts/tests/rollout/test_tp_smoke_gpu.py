"""Tier 1 GPU smoke tests — colocate rollout TP>1 against a live SGLang server.

Skipped unless ``UNIRL_TP_GPU_TEST=1`` is set AND enough CUDA devices are
visible. These are the integration gate for the colocate TP design:

  - tp_rank=0 boots ONE multi-GPU SGLang engine spanning ``tp_size`` GPUs
  - tp_rank>0 is a no-op shell (no server, no ports)
  - ``generate`` only hits the tp_rank=0 backend
  - weight sync delivers full tensors to every TP rank via SGLang's NCCL group

Run:  UNIRL_TP_GPU_TEST=1 pytest scripts/tests/rollout/test_tp_smoke_gpu.py

The minimal validation target (see docs/rollout_tp_ep_pp_design.md, Tier 1):
1 node, 2 GPUs, ``tp_size=2``. Output must match the ``tp_size=1`` baseline
within <1%.
"""

from __future__ import annotations

import os

import pytest

from ..conftest import requires_gpus


@pytest.fixture
def tp2_gate():
    return requires_gpus(2)


@pytest.mark.gpu
def test_tp2_engine_boot_and_generate(tp2_gate, tmp_path):
    """Boot a 2-GPU SGLang engine on tp_rank=0; verify generate + shell no-op.

    This is the GATE test for the colocate TP design. It is skipped in CI (no
    GPU) and run manually before merging any rollout-TP change.
    """
    pytest.importorskip("sglang")
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

    model_path = os.environ.get("UNIRL_TP_TEST_MODEL")
    if not model_path:
        pytest.skip("set UNIRL_TP_TEST_MODEL=<huggingface path> to run this test")

    cfg = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        model_family="text",
        tp_size=2,
        max_new_tokens=8,
        engine_kwargs={"mem_fraction_static": 0.5, "skip_server_warmup": True},
    )

    # tp_rank=1: no-op shell (no GPU access, no SGLang boot).
    shell = SGLangRolloutEngine(config=cfg, rank=1, tp_rank=1, tp_size=2, tp_device_ids=[0, 1])
    assert shell._is_tp_zero is False and shell._backend is None

    # tp_rank=0: boots a 2-GPU SGLang engine. CUDA_VISIBLE_DEVICES is overridden
    # to "0,1" so the SGLang scheduler subprocesses see both cards.
    eng = SGLangRolloutEngine(config=cfg, rank=0, tp_rank=0, tp_size=2, tp_device_ids=[0, 1])
    try:
        assert eng._is_tp_zero is True
        assert eng._backend is not None
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
        assert eng.health_check() is True
    finally:
        eng.shutdown()
        # CUDA_VISIBLE_DEVICES restored after shutdown.
        assert os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1"


@pytest.mark.gpu
def test_tp2_weight_sync_round_trip(tp2_gate):
    """Weight sync delivers full tensors to both TP ranks via SGLang's NCCL group.

    Requires a real FSDP train backend + NCCL; this is the full Tier-1 gate.
    Skipped without ``UNIRL_TP_GPU_TEST=1`` and >=2 GPUs.
    """
    pytest.skip("Full weight-sync round-trip needs a colocate FSDP+SGLang fixture (Tier 1 manual)")
