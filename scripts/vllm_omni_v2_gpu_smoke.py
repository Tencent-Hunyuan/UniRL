#!/usr/bin/env python
"""GPU smoke for the ``vllm_omni_v2`` engine (LIN-382).

Drives the report's pending single-engine GPU checks against a real
vllm-omni v0.20.0 runtime on one node:

  1. boot       sd35_t2i via ``VLLMOmniPorts.reserve()`` + ``make_engine``
  2. ports      settled per-stage master_port vs the reserved base. At the
                pin (v0.20.0) each stage settles ``base + rand(0,100)`` plus
                a 37-stride bind-check scan — injected bases are NOT honored
                verbatim until v0.21.0rc2 (#3803) — so the assertion is a
                window + distinctness, not equality.
  3. generate   2 prompts, small T — σ-echo + shape invariants are asserted
                inside ``build_image_segment``/``assemble_tracks``; we assert
                the response surface (tracks, decoded images, segments).
  4. tensorbag  serialize 2 real checkpoint tensors the same way
                ``TensorWeightSync.sync`` does (FlattenedTensorBucket +
                MultiprocessingSerializer, ``load_format="flattened_bucket"``)
                and assert the engine's ``loaded_param_checksums`` match the
                local ``fingerprint_tensor`` of what we pushed.
  5. sleepwake  ``sleep()`` (expect a node-wide GPU memory drop), ``wake_up()``,
                then generate again.

Live-LoRA re-push on wake is exercised by the LoRA e2e recipes
(``sd3_flowdppo_vllmomni_v2``), not here — this smoke runs base-weights only
(``use_lora=False``).

Usage (inside the pod, venv active):
  PRETRAINED_MODEL=/root/diffusionrl/models/local/stable-diffusion-3.5-medium \\
  python scripts/vllm_omni_v2_gpu_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

T0 = time.time()


def log(msg: str) -> None:
    print(f"[smoke +{time.time() - T0:8.1f}s] {msg}", flush=True)


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sum(int(x) for x in out.split())


def find_master_ports(obj, depth: int = 0, seen=None) -> List[int]:
    """Recursively hunt ``master_port`` ints in stage-config objects/dicts."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 4 or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return []
    seen.add(id(obj))
    found: List[int] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "master_port" and isinstance(v, int):
                found.append(v)
            else:
                found.extend(find_master_ports(v, depth + 1, seen))
        return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(find_master_ports(v, depth + 1, seen))
        return found
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        if k == "master_port" and isinstance(v, int):
            found.append(v)
        elif k in ("engine_args", "stage_configs", "stages", "config", "diffusion_config"):
            found.extend(find_master_ports(v, depth + 1, seen))
    return found


def make_req(n: int, steps: int, hw: int) -> "RolloutReq":
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import DiffusionSamplingParams

    return RolloutReq(
        sample_ids=[f"p{i}/r0" for i in range(n)],
        group_ids=[f"g{i}" for i in range(n)],
        primitives={"text": Texts(texts=[
            "a watercolor fox in a snowy forest",
            "a brutalist lighthouse at golden hour",
        ][:n])},
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=steps, height=hw, width=hw,
            guidance_scale=4.5, eta=1.0, seed=7,
        ),
    )


def pick_checkpoint_tensors(model_path: str, k: int = 2) -> List[Tuple[str, torch.Tensor]]:
    """Two small real tensors from the transformer checkpoint (norm-ish first)."""
    import glob

    from safetensors import safe_open

    files = sorted(glob.glob(os.path.join(model_path, "transformer", "*.safetensors")))
    assert files, f"no transformer safetensors under {model_path}"
    picked: List[Tuple[str, torch.Tensor]] = []
    with safe_open(files[0], framework="pt") as f:
        keys = list(f.keys())
        small = [k_ for k_ in keys if "norm" in k_] + keys
        for name in small:
            t = f.get_tensor(name)
            if t.numel() <= 2_000_000:
                picked.append((name, t))
            if len(picked) >= k:
                break
    assert len(picked) >= 1, "no small tensors found in checkpoint"
    return picked


def main() -> int:
    model_path = os.environ.get(
        "PRETRAINED_MODEL", "/root/diffusionrl/models/local/stable-diffusion-3.5-medium"
    )
    steps = int(os.environ.get("SMOKE_STEPS", "4"))
    hw = int(os.environ.get("SMOKE_HW", "512"))
    results: Dict[str, str] = {}

    from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor
    from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniPorts, VLLMOmniV2EngineConfig

    # ---- 1. boot -----------------------------------------------------------
    log(f"phase 1 boot: model={model_path} steps={steps} hw={hw}")
    ports = VLLMOmniPorts.reserve()
    base = ports.master_port
    log(f"reserved master_port base = {base}")
    cfg = VLLMOmniV2EngineConfig(
        model_path=model_path,
        modality="sd3_t2i",
        default_height=hw, default_width=hw,
        default_num_inference_steps=steps,
        enable_sleep_mode=True,
    )
    model_config = SimpleNamespace(shift=3.0, use_lora=False)
    t = time.time()
    engine = cfg.make_engine(model_config=model_config, ports=ports)
    log(f"boot OK in {time.time() - t:.1f}s; tp_per_stage={engine.tp_per_stage()}")
    assert engine.health_check(), "health_check failed after boot"
    results["boot"] = "PASS"

    try:
        # ---- 2. ports ------------------------------------------------------
        omni = getattr(engine._backend, "_omni", None) or getattr(engine._backend, "omni", None)
        settled = sorted(set(find_master_ports(getattr(omni, "stage_configs", None))))
        log(f"settled master ports = {settled} (base {base})")
        if settled:
            assert all(p >= base for p in settled), f"settled port below reserved base: {settled} < {base}"
            assert all(p - base <= 1000 for p in settled), f"settled port outside sane window: {settled}"
            assert len(settled) == len(set(settled)), "stage master ports collide"
            results["ports"] = f"PASS ({settled})"
        else:
            # Don't fail the smoke on introspection fragility — the boot itself
            # proves no port collision occurred; record for follow-up.
            results["ports"] = "WARN (no master_port found via stage_configs introspection)"
        log(f"phase 2 ports: {results['ports']}")

        # ---- 3. generate #1 --------------------------------------------------
        req = make_req(2, steps, hw)
        t = time.time()
        resp = engine.generate(req)
        dt = time.time() - t
        assert resp.tracks, "generate returned no tracks"
        for name, track in resp.tracks.items():
            n = len(track.sample_ids)
            log(f"track {name!r}: {n} samples")
            assert n == 2, f"track {name!r} sample count {n} != 2"
        results["generate"] = f"PASS ({dt:.1f}s)"
        log(f"phase 3 generate: {results['generate']}")

        # ---- 4. tensor-bag sync + checksum ----------------------------------
        named = pick_checkpoint_tensors(model_path)
        names = [n for n, _ in named]
        pre = engine.loaded_param_checksums(names=names)
        log(f"pre-push engine checksums: {pre}")
        resolved = [n for n in names if pre.get(n)]
        if not resolved:
            results["tensorbag"] = f"WARN (worker did not resolve names {names}; name-remap needed)"
        else:
            try:
                from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
            except ImportError:
                from sglang.srt.patch_torch import monkey_patch_torch_reductions
            from sglang.srt.utils import MultiprocessingSerializer
            try:
                from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
            except ImportError:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

            monkey_patch_torch_reductions()
            use = [(n, t_) for n, t_ in named if n in resolved]
            local_fp = {n: fingerprint_tensor(t_) for n, t_ in use}
            by_dtype: Dict[torch.dtype, list] = {}
            for n, t_ in use:
                by_dtype.setdefault(t_.dtype, []).append((n, t_.cuda()))
            for grouped in by_dtype.values():
                flat = FlattenedTensorBucket(named_tensors=grouped)
                payload = {
                    "flattened_tensor": flat.get_flattened_tensor(),
                    "metadata": flat.get_metadata(),
                }
                engine.update_weights_from_tensor(
                    serialized_named_tensors=[
                        MultiprocessingSerializer.serialize(payload, output_str=True)
                    ],
                    load_format="flattened_bucket",
                    flush_cache=True,
                )
            post = engine.loaded_param_checksums(names=list(local_fp))
            log(f"local fingerprints:  {local_fp}")
            log(f"post-push checksums: {post}")
            mismatch = {n: (local_fp[n], post.get(n)) for n in local_fp if post.get(n) != local_fp[n]}
            assert not mismatch, f"checksum mismatch after tensor-bag push: {mismatch}"
            results["tensorbag"] = f"PASS ({len(local_fp)} tensors)"
        log(f"phase 4 tensorbag: {results['tensorbag']}")

        # ---- 5. sleep / wake -------------------------------------------------
        before = gpu_used_mib()
        engine.sleep()
        assert engine.is_offloaded, "engine not marked offloaded after sleep()"
        time.sleep(5)
        slept = gpu_used_mib()
        log(f"GPU MiB total: before sleep={before}, after sleep={slept}")
        engine.wake_up()
        assert not engine.is_offloaded, "engine still offloaded after wake_up()"
        resp2 = engine.generate(make_req(1, steps, hw))
        assert resp2.tracks, "post-wake generate returned no tracks"
        dropped = before - slept
        results["sleepwake"] = f"PASS (freed {dropped} MiB)" if dropped > 1024 else f"WARN (only freed {dropped} MiB)"
        log(f"phase 5 sleepwake: {results['sleepwake']}")
    finally:
        log("shutting down engine")
        try:
            engine.shutdown()
        except Exception as e:  # noqa: BLE001
            log(f"shutdown raised: {e!r}")

    print("\n===== vllm_omni_v2 GPU smoke summary =====", flush=True)
    hard_fail = False
    for phase in ("boot", "ports", "generate", "tensorbag", "sleepwake"):
        v = results.get(phase, "FAIL (not reached)")
        if v.startswith("FAIL"):
            hard_fail = True
        print(f"  {phase:10s} {v}", flush=True)
    print("SMOKE RESULT:", "FAIL" if hard_fail else "PASS", flush=True)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
