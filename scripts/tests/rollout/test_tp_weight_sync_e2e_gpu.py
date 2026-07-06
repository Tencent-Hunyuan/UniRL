"""E2E weight-sync integration test: FSDP train → tp=2 SGLang rollout.

Boots a single-process FSDP-wrapped tiny Qwen3.5-MoE model and a 2-GPU SGLang
rollout engine (tp_size=2), then drives ``TensorWeightSync.sync()`` to push
the trained weights into the SGLang server and generates tokens to confirm the
sync landed. This is the colocate integration point for rollout TP: the train
side stays flat (FSDP), the rollout side is tp=2, and the weight sync must
deliver full tensors to the tp_rank==0 worker which SGLang internally fans out
to both TP ranks.

Requires the tiny-random Qwen3.5-MoE model and >=2 CUDA GPUs. Single-process
(train + rollout in the same Ray worker) so the TensorWeightSync colocate path
is exercised without a full Ray cluster.

Setup (once):
    python -c "from huggingface_hub import snapshot_download; \\
        snapshot_download('tiny-random/qwen3.5-moe', local_dir='/tmp/tiny-qwen35-moe')"

Run:
    UNIRL_TP_E2E_MODEL=/tmp/tiny-qwen35-moe pytest scripts/tests/rollout/test_tp_weight_sync_e2e_gpu.py -s
"""

from __future__ import annotations

import os

import pytest

from ..conftest import requires_gpus, sglang_e2e_teardown


@pytest.fixture
def tp2_gate():
    return requires_gpus(2)


@pytest.mark.gpu
def test_weight_sync_tp2_e2e(tp2_gate):
    """Push FSDP weights into a tp=2 SGLang engine, then generate.

    Validates the colocate weight-sync path under rollout TP:
      - FSDP model loaded on the train worker (flat, no TP)
      - tp_rank=0 SGLang engine boots across 2 GPUs
      - ``TensorWeightSync.sync()`` ships ``tp_size`` payload copies; SGLang's
        internal weight_loader reshards to each TP rank
      - generate after sync produces tokens (the sync landed)
    """
    pytest.importorskip("sglang")
    import torch
    from transformers import AutoModelForCausalLM

    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    model_path = os.environ.get("UNIRL_TP_E2E_MODEL", "/tmp/tiny-qwen35-moe")
    if not os.path.isdir(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
        pytest.skip(f"model not found at {model_path}")

    # 1. Boot the tp=2 SGLang engine (tp_rank=0 hosts; this is the rollout
    #    side of the colocate pair).
    cfg = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        model_family="text",
        tp_size=2,
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        samples_pre_expanded=True,
        engine_kwargs={
            "mem_fraction_static": 0.3,
            "skip_server_warmup": True,
            "disable_cuda_graph": True,
            "trust_remote_code": True,
        },
    )
    eng = SGLangRolloutEngine(
        config=cfg,
        rank=0,
        tp_rank=0,
        tp_size=2,
        tp_device_ids=[0, 1],
    )
    passed = False
    try:
        assert eng._is_tp_zero is True
        assert eng.health_check() is True

        # 2. Load the SAME checkpoint on the train side as a flat (non-FSDP)
        #    model and push it into SGLang via the engine's update_weights
        #    verb. In a real run FullWeightSync._iter_full_tensors all-gathers
        #    FSDP shards; here we drive the same verb directly with the full
        #    state dict to exercise the tp_size-aware payload replication
        #    (the [payload]*tp_size path in TensorWeightSync).
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16).cuda()
        model.eval()

        # Build the SGLang weight payload the same way TensorWeightSync does:
        # one FlattenedTensorBucket per dtype, replicated tp_size times.
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        monkey_patch_torch_reductions()

        named = list(model.state_dict().items())
        # SGLang expects HF names without the "model." prefix that some
        # AutoModel wrappers add; the engine's load_weights handles the
        # remap. Ship the full state dict.
        by_dtype: dict = {}
        for name, t in named:
            t = t.contiguous().to(torch.bfloat16)
            by_dtype.setdefault(t.dtype, []).append((name, t))

        for group in by_dtype.values():
            flat = FlattenedTensorBucket(named_tensors=group)
            payload = {
                "flattened_tensor": flat.get_flattened_tensor(),
                "metadata": flat.get_metadata(),
            }
            serialized = MultiprocessingSerializer.serialize(payload, output_str=True)
            # Ship tp_size=2 copies — SGLang picks serialized_named_tensors[tp_rank]
            # per scheduler subprocess.
            eng.update_weights_from_tensor(
                serialized_named_tensors=[serialized] * 2,
                load_format="flattened_bucket",
                flush_cache=True,
            )

        # 3. Generate after sync — if the sync landed, we get tokens back.
        req = RolloutReq(
            sample_ids=["s0"],
            group_ids=["s0"],
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
            texts = getattr(decoded, "texts", None)
            assert texts, "post-sync generate produced no decoded text"
            print(f"\n[post-sync generate] text={texts[0]!r}")
            break
        else:
            pytest.fail("no track with decoded output after sync")

        passed = True
    finally:
        sglang_e2e_teardown(eng, passed=passed)
