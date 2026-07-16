#!/usr/bin/env python3
"""sglang-engine MULTI-TURN encode smoke (LIN-503 gap C, on GPU).

Boots the real ``SGLangRolloutEngine`` (native, in-process), builds a
``[user, assistant, tool, gen]`` request ``Sample``, runs ``generate``, and proves
the engine now CONDITIONS ON THE FULL TRAJECTORY: the captured
``conditions["prompt"]`` — the ids the SRT server tokenized via the adapter's chat
template — decodes to user → assistant → tool, not just the root prompt (the pre-fix
behavior read ``parts[0]`` and emitted one ``user`` message).

Coexists with a running GPU burn via a low ``mem_fraction_static``. Run on the pod
with the sglang venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/sglang_ar_multiturn_smoke.py

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

_USER = "What is 19 times 23?"
_ASSISTANT = "Let me calculate that for you."
_TOOL = "437"


def _log(msg: str) -> None:
    print(f"[sglang-mt] {msg}", flush=True)


def build_multiturn_sample() -> Sample:
    """A 3-turn trajectory ``[user, assistant, tool]`` + a frontier gen shell."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=[_USER])}, role="user", control={})
    asst = inp.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=[_ASSISTANT])})
    tool = asst.input_child({"text": Texts(texts=[_TOOL])}, role="tool")
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
    return Sample(parts=[inp, asst, tool, tool.fork(1, sampling_params=ar_params)])


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3 dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")
    import unirl.rollout.engine.sglang.adapters.text as _t

    _log(f"sglang text adapter from: {_t.__file__}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        chat_template_kwargs={"enable_thinking": False},
        # Coexist with the GPU burn: mem_fraction_static is a fraction of TOTAL VRAM
        # and sglang subtracts already-used memory (the burn's ~68 GB), so a HIGH
        # fraction is needed to leave a positive KV budget in the ~27 GB free; it
        # only allocates the delta above current usage. Disable CUDA-graph capture
        # (default captures dozens of batch sizes — wasteful for a batch-1 smoke).
        engine_kwargs={"mem_fraction_static": 0.9, "disable_cuda_graph": True},
    )
    engine = None
    try:
        _log("booting SGLangRolloutEngine (native, mem_fraction_static=0.2) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        sample = build_multiturn_sample()
        _log(f"request: {len(sample.parts)} parts; trajectory turns={[t.role for t in sample.text_conditioning()]}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)
        gen = out.parts[-1]
        assert gen.conditions and "prompt" in gen.conditions, "no stored prompt conditions"

        # ---- THE encode proof: decode the prompt the server tokenized ----
        tok = engine.adapter._tokenizer
        prompt_ids = gen.conditions["prompt"].input_ids[0]
        prompt_text = tok.decode([int(t) for t in prompt_ids.tolist()], skip_special_tokens=False)
        _log("--- decoded conditions['prompt'] (what SRT tokenized) ---")
        _log(prompt_text)
        _log("---------------------------------------------------------")

        assert _USER in prompt_text, "USER turn missing from the encoded prompt"
        assert _ASSISTANT in prompt_text, (
            "ASSISTANT turn DROPPED — the sglang encode flattened to the root prompt (gap C not applied)"
        )
        assert _TOOL in prompt_text, (
            "TOOL turn DROPPED — the sglang encode flattened to the root prompt (gap C not applied)"
        )
        assert prompt_text.index(_USER) < prompt_text.index(_ASSISTANT) < prompt_text.index(_TOOL), (
            "turns out of lineage order in the encoded prompt"
        )
        _log("MULTI-TURN ENCODE PASS: sglang prompt carries user → assistant → tool ✓")
        _log(f"completion: {gen.primitives['text'].texts[0]!r}")
        _log("SGLANG MULTI-TURN SMOKE PASSED ✅  (the engine conditions on the full trajectory)")
        return 0
    except Exception:
        _log("SGLANG MULTI-TURN SMOKE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
