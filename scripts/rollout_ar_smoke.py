#!/usr/bin/env python3
"""Rollout-only e2e smoke for the AUTOREGRESSIVE engine (sglang text, LIN-454).

Boots the REAL ``SGLangRolloutEngine`` (model_family ``text``, Qwen3-4B) in a
single process, builds a request ``Sample`` by hand, runs ``generate``, and
asserts the frontier gen Part is filled with a ``TextSegment`` (tokens +
logprobs) + decoded ``Texts`` + replay conditions — the AR analogue of the
sd3 smoke. No training / reward / weight sync.

Run on a GPU pod (1 free GPU), in the sglang venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_ar_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment


def _log(msg: str) -> None:
    print(f"[ar-smoke] {msg}", flush=True)


def build_request_sample(n: int) -> Sample:
    """2 prompts, ``n`` completions each: ``[input, ar gen-shell]`` (P*n samples)."""
    prompts = ["The capital of France is", "Two plus two equals"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)}, control={}
    )
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part).fork(n, sampling_params=ar_params)


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3-4B dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",  # in-process sglang Engine (no separate server)
        tp_size=1,
        max_new_tokens=48,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
    )
    # model_family auto-derives to "text" (image_token is None).

    n = 2  # completions per prompt → exercises the sibling fan-out (n>1)
    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3) ...")
        engine = SGLangRolloutEngine(config, rank=0)
        sample = build_request_sample(n)
        gen_in = sample.parts[-1]
        _log(f"request: {len(sample.parts)} parts; gen ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled Sample ...")

        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        n_expect = 2 * n
        assert list(gen.sample_ids) == list(gen_in.sample_ids), (
            f"gen ids changed: {list(gen.sample_ids)} != {list(gen_in.sample_ids)}"
        )
        assert len(gen.sample_ids) == n_expect, f"expected {n_expect} samples; got {len(gen.sample_ids)}"
        assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
        assert gen.segment.lengths is not None and int(gen.segment.lengths.numel()) == n_expect, (
            "TextSegment.lengths missing or wrong count"
        )
        assert isinstance(gen.primitives.get("text"), Texts), (
            f"decoded text primitive must be Texts; got {type(gen.primitives.get('text'))}"
        )
        assert len(gen.primitives["text"].texts) == n_expect, (
            f"expected {n_expect} decoded texts; got {len(gen.primitives['text'].texts)}"
        )
        assert gen.conditions, "replay conditions empty (expected 'prompt')"

        toks = [int(x) for x in gen.segment.lengths.tolist()]
        _log(f"PASS: {n_expect} completions; token counts={toks}; conditions={sorted(gen.conditions.keys())}")
        for i, t in enumerate(gen.primitives["text"].texts):
            _log(f"  sample[{i}] id={gen.sample_ids[i]} text={t[:80]!r}")
        _log("AR ROLLOUT SMOKE PASSED ✅  (TextSegment + decoded Texts + conditions; path ids preserved)")
        return 0
    except Exception:
        _log("AR ROLLOUT SMOKE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


if __name__ == "__main__":
    sys.exit(main())
