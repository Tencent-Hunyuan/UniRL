#!/usr/bin/env python3
"""GPU trace: drive the env-driven ``AgentLoop`` over the REAL Qwen3 SGLang engine and dump the
``Sample``/``Part`` chain with ACTUAL model tokens in the GEN Parts (LIN-492).

Same tracer as the CPU ``rollout_loop_trace`` (the model-free oracle), but **fake #1 (the engine)
is replaced by the real ``SGLangRolloutEngine``** — so the GEN cells show real Qwen3-4B output, not
stand-in strings. **Fake #2 (the environment) stays a stub** (``FixedTurnsEnv``): the real
``ToolEnvironment`` isn't built yet, and ``Qwen3-4B-Base`` is not tool-tuned, so it cannot drive
tool-call termination — the env drives a fixed turn count instead.

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_loop_trace_ar.py

Trace lines are prefixed with ``@@T`` so they survive extraction from SGLang's boot logs:
    ... 2>&1 | tee /tmp/trace.log ; grep '^@@T' /tmp/trace.log | sed 's/^@@T //'
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional, Tuple

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import AgentLoop
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Primitive, Sample
from unirl.types.sampling import ARSamplingParams

W = 100
SENT = "@@T"  # sentinel prefix so the trace survives extraction from SGLang's logs


def tprint(s: str = "") -> None:
    print(f"{SENT} {s}", flush=True)


def _log(msg: str) -> None:
    print(f"[trace-ar] {msg}", flush=True)


def _oneline(t: str, n: int = 60) -> str:
    t = t.replace("\n", "↵").replace("\r", " ")
    return t if len(t) <= n else t[: n - 3] + "..."


def _prim(p: Optional[Primitive]) -> str:
    if p is None:
        return "<empty shell — engine will fill this>"
    if isinstance(p, Texts):
        return "Texts[" + ", ".join(repr(_oneline(t)) for t in p.texts) + "]"
    return type(p).__name__ + "(...)"


def _kind(part: Part) -> str:
    if part.sampling_params is not None:
        return "GEN"  # carries sampling_params -> trainable, loss-mask 1
    if part.is_root:
        return "INPUT"  # the root prompt
    return "OBS"  # observation re-entry (Sample.observe) -> mask 0


def dump(sample: Sample, title: str) -> None:
    tprint("")
    tprint(f"  {title}")
    tprint(f"  {'#':>2}  {'kind':<5} {'mask':<4} {'ids':<26} primitive (content)")
    tprint(f"  {'-' * 2}  {'-' * 5} {'-' * 4} {'-' * 26} {'-' * 44}")
    for i, part in enumerate(sample.parts):
        mask = "1" if part.sampling_params is not None else "0"
        ids = str(list(part.sample_ids))
        if len(ids) > 26:
            ids = ids[:23] + "..."
        content = ", ".join(f"{modality}={_prim(primitive)}" for modality, primitive in part.primitives.items())
        tprint(f"  {i:>2}  {_kind(part):<5} {mask:<4} {ids:<26} {content}")


def show_real_output(out: Sample, scenario: str) -> None:
    tprint("")
    tprint(f"  REAL Qwen3-4B output — full (untruncated) text of every GEN Part [{scenario}]:")
    for i, part in enumerate(out.parts):
        if part.sampling_params is None:
            continue
        for j, sid in enumerate(part.sample_ids):
            txt = part.primitives["text"].texts[j] if isinstance(part.primitives.get("text"), Texts) else "<non-text>"
            tprint(f"    part[{i}] id={sid}")
            tprint(f"        {txt!r}")


def summarize(out: Sample) -> None:
    gen = out.gen_parts()
    cond = out.conditioning()
    tprint("")
    tprint("  SUMMARY")
    tprint(f"    total Parts          : {len(out.parts)}")
    tprint(f"    trainable GEN Parts  : {len(gen)}  (gen_parts(); receive advantages + loss)")
    tprint(f"    mask-0 Parts (in+obs): {len(out.parts) - len(gen)}  (prompt + observations; never trained)")
    tprint(f"    conditioning() into the last gen: {len(cond)} ancestor primitives")


class FixedTurnsEnv:
    """Stub environment (fake #2): drives exactly ``turns`` generations, then ``done``.
    Stands in for the real (not-yet-built) ToolEnvironment so the loop runs a fixed number of turns."""

    def __init__(self, turns: int) -> None:
        self._remaining = turns

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        self._remaining -= 1
        if self._remaining <= 0:
            return None, True, {}
        ids = sample.parts[-1].sample_ids
        return Texts(texts=[f"<obs for {sid}>" for sid in ids]), False, {}


class TracingEngine:
    """Wraps the REAL engine; prints the frontier shell before generate and after fill."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.turn = 0

    def generate(self, sample: Sample) -> Sample:
        self.turn += 1
        dump(sample, f"turn {self.turn}: AFTER fork(), BEFORE generate()  (last Part = empty gen shell)")
        out = self.inner.generate(sample)
        dump(out, f"turn {self.turn}: AFTER generate()  (REAL Qwen3 filled the shell)")
        return out


class TracingEnv:
    """Wraps the env; prints what step() decided."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def reset(self, request: Sample) -> Sample:
        return self.inner.reset(request)

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        obs, done, info = self.inner.step(sample)
        verdict = "DONE — trajectory ends" if done else f"CONTINUE — observe {_prim(obs)}"
        tprint("")
        tprint(f"      >> env.step() -> done={done}  ::  {verdict}")
        return obs, done, info


def build_request(prompts, n: int) -> Tuple[Sample, ARSamplingParams]:
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
        control={},
    )
    ar = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part), ar


def scenario(engine, title: str, prompts, n: int, turns: int) -> None:
    tprint("=" * W)
    tprint(title)
    tprint("=" * W)
    req, ar = build_request(prompts, n)
    dump(req, "initial request (root INPUT Part)")
    loop = AgentLoop(environment=TracingEnv(FixedTurnsEnv(turns)), sampling_params=ar, max_turns=8)
    out = loop.run(TracingEngine(engine), req)
    dump(out, "=== FINAL trajectory ===")
    show_real_output(out, title.split("—")[0].strip())
    summarize(out)


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
    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3) ...")
        engine = SGLangRolloutEngine(config, rank=0)
        tprint("")
        tprint("#" * W)
        tprint("# REAL-ENGINE AgentLoop TRACE — GEN Parts contain actual Qwen3-4B output.")
        tprint("# fake #1 (engine) REMOVED -> real SGLang.  fake #2 (env) = FixedTurnsEnv stub, still present.")
        tprint("#" * W)

        scenario(
            engine,
            "SCENARIO A — GRPO fan-out n=2 + 2 turns (gen -> obs -> gen), 1 prompt",
            ["The capital of France is"],
            n=2,
            turns=2,
        )
        scenario(
            engine,
            "SCENARIO B — single trajectory (n=1), 3 turns (gen -> obs -> gen -> obs -> gen)",
            ["Two plus two equals"],
            n=1,
            turns=3,
        )

        _log("TRACE COMPLETE")
        return 0
    except Exception:
        _log("TRACE FAILED")
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
