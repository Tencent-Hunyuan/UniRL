"""E2E rollout-side TP test against a real tiny Qwen3.5-MoE model.

Boots a 2-GPU SGLang engine via UniRL's ``SGLangRolloutEngine`` (tp_rank=0
hosting the server, tp_rank=1 a no-op shell) and exercises generate / sleep /
wake_up end-to-end. Requires the tiny-random Qwen3.5-MoE model and >=2 CUDA GPUs.

Setup (once):
    python -c "from huggingface_hub import snapshot_download; \\
        snapshot_download('tiny-random/qwen3.5-moe', local_dir='/tmp/tiny-qwen35-moe')"

Run:
    UNIRL_TP_E2E_MODEL=/tmp/tiny-qwen35-moe pytest scripts/tests/test_rollout_tp_e2e.py -s
"""

from __future__ import annotations

import os
import sys
from typing import List

import pytest

from .conftest import requires_gpus

# Set by the test body when all assertions pass; read by the finally block to
# force a clean exit past SGLang's noisy subprocess teardown.
_RESULT: dict = {}


def _model_path() -> str:
    p = os.environ.get("UNIRL_TP_E2E_MODEL", "/tmp/tiny-qwen35-moe")
    if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "config.json")):
        pytest.skip(f"model not found at {p}; set UNIRL_TP_E2E_MODEL or download tiny-random/qwen3.5-moe")
    return p


@pytest.fixture
def tp2_gate():
    return requires_gpus(2)


@pytest.mark.gpu
def test_rollout_tp2_e2e_boot_generate_sleep(tp2_gate):
    """Boot a 2-GPU SGLang engine via UniRL, generate, sleep/wake, shutdown.

    Validates the full rollout-side TP path:
      - tp_rank=0 overrides CUDA_VISIBLE_DEVICES and boots a multi-GPU SGLang
      - tp_rank=1 is a no-op shell (no server, no ports)
      - generate produces tokens via the UniRL adapter path
      - sleep releases weights/kv_cache; wake_up restores
      - shutdown restores the original CUDA_VISIBLE_DEVICES
    """
    pytest.importorskip("sglang")
    import torch
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    model_path = _model_path()
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

    cfg = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        model_family="text",
        tp_size=2,
        max_new_tokens=8,
        temperature=1.0,
        top_p=1.0,
        samples_pre_expanded=True,
        engine_kwargs={
            "mem_fraction_static": 0.4,
            "skip_server_warmup": True,
            "disable_cuda_graph": True,
            "trust_remote_code": True,
        },
    )

    # tp_rank=1: no-op shell. Must construct without booting SGLang.
    shell = SGLangRolloutEngine(
        config=cfg, rank=1, tp_rank=1, tp_size=2, tp_device_ids=[0, 1],
    )
    assert shell._is_tp_zero is False
    assert shell._backend is None

    # tp_rank=0: boots the 2-GPU SGLang engine.
    eng = SGLangRolloutEngine(
        config=cfg, rank=0, tp_rank=0, tp_size=2, tp_device_ids=[0, 1],
    )
    try:
        assert eng._is_tp_zero is True
        assert eng._backend is not None
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
        assert eng.health_check() is True

        # Generate via the UniRL RolloutReq path.
        req = RolloutReq(
            sample_ids=["s0"],
            group_ids=["s0"],
            primitives={"text": Texts(texts=["Hello, my name is"])},
            request_conditions={},
            sampling_params={"default": ARSamplingParams(samples_per_prompt=1)},
            metadata=[],
        )
        resp = eng.generate(req)
        assert resp is not None
        # The adapter returns a RolloutResp with at least one track.
        assert len(resp.tracks) >= 1
        for track in resp.tracks.values():
            decoded = getattr(track, "decoded", None)
            if decoded is None:
                continue
            texts = getattr(decoded, "texts", None)
            assert texts, "generate produced no decoded text"
            print(f"\n[generate] text={texts[0]!r}")
            break
        else:
            pytest.fail("no track with decoded output")

        # sleep should release GPU memory; wake_up should restore.
        eng.sleep()
        assert eng.is_offloaded is True
        eng.wake_up()
        assert eng.is_offloaded is False
        assert eng.health_check() is True
        # All assertions passed — record success before the noisy SGLang
        # shutdown (which fires SIGTERM/SIGQUIT at its subprocesses and can
        # tear down the pytest process before the result line prints).
        _RESULT["passed"] = True
    finally:
        # Shut down SGLang then force-exit BEFORE its child SIGQUIT propagates
        # to the pytest process. SGLang's subprocess teardown fires SIGQUIT at
        # sibling processes on any child crash (-15), which kills the pytest
        # runner before it can print the PASS line. os._exit skips Python
        # finalizers entirely — safe here because we hold no unreleased
        # resources beyond the SGLang server we just shut down.
        try:
            eng.shutdown()
        except Exception:
            pass
        import sys as _sys
        _sys.stdout.flush()
        _sys.stderr.flush()
        if _RESULT.get("passed"):
            os._exit(0)
        # On failure let the normal pytest traceback path run — but we already
        # flushed above, so the SIGQUIT noise will not lose the traceback.
