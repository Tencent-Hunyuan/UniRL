"""Generation drivers: t2i via diffusers, text via an OpenAI-compatible endpoint.

Deterministic image naming is the contract between stages — no manifest file:
``images/p{prompt_idx:05d}_s{sample_idx}.png`` with seed ``base + 1000*p + s``.
Both drivers resume by skipping outputs that already exist.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from .checkpoints import ResolvedCkpt

T2I_KWARGS = {"steps": "num_inference_steps", "guidance": "guidance_scale", "height": "height", "width": "width"}


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def image_path(images_dir: Path, prompt_idx: int, sample_idx: int) -> Path:
    return images_dir / f"p{prompt_idx:05d}_s{sample_idx}.png"


def t2i_jobs(
    prompts: List[str], images_dir: Path, samples_per_prompt: int, shard: Tuple[int, int] = (0, 1)
) -> List[Tuple[int, int]]:
    """Missing (prompt, sample) jobs for this shard. The full deterministic grid is
    partitioned BEFORE the exists() filter, so each job is owned by exactly one shard
    no matter when each shard scans the directory."""
    grid = [(p, s) for p in range(len(prompts)) for s in range(samples_per_prompt)]
    return [(p, s) for p, s in grid[shard[0] :: shard[1]] if not image_path(images_dir, p, s).exists()]


def _load_pipe(ckpt: ResolvedCkpt):
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(ckpt.base, torch_dtype=torch.bfloat16).to("cuda")
    if ckpt.adapter:
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(pipe.transformer, ckpt.adapter)
        lora_b = [p for name, p in peft_model.named_parameters() if ".lora_B." in name]
        if not lora_b or all(not p.abs().sum().item() for p in lora_b):
            raise SystemExit(f"adapter {ckpt.adapter} loaded no trained LoRA weights (all-zero lora_B)")
        pipe.transformer = peft_model.merge_and_unload().to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_t2i(
    prompts: List[str],
    images_dir: Path,
    ckpt: ResolvedCkpt,
    *,
    samples_per_prompt: int,
    batch_size: int,
    seed: int,
    gen_kwargs: Dict,
    shard: Tuple[int, int] = (0, 1),
    linspace_sigmas: bool = False,
    prompt_seed: bool = False,
) -> None:
    jobs = t2i_jobs(prompts, images_dir, samples_per_prompt, shard)
    if not jobs:
        print("[t2i] all images present — nothing to generate")
        return
    import hashlib

    import numpy as np
    import torch

    pipe = _load_pipe(ckpt)
    call_kwargs = dict(gen_kwargs)
    if linspace_sigmas:
        steps = int(gen_kwargs.get("num_inference_steps") or 0)
        if steps <= 0:
            raise ValueError("linspace_sigmas requires num_inference_steps in gen defaults/CLI")
        call_kwargs["sigmas"] = list(np.linspace(1.0, 1.0 / steps, steps))

    def _seed(p: int, s: int) -> int:
        if prompt_seed:
            h = int.from_bytes(hashlib.sha256(prompts[p].encode()).digest()[:4], "big")
            return (seed + h + s) % (2**31)
        return seed + 1000 * p + s

    images_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        gen_dev = "cpu" if prompt_seed else "cuda"
        generators = [torch.Generator(gen_dev).manual_seed(_seed(p, s)) for p, s in batch]
        images = pipe(prompt=[prompts[p] for p, _ in batch], generator=generators, **call_kwargs).images
        for (p, s), img in zip(batch, images):
            target = image_path(images_dir, p, s)
            tmp = target.with_suffix(".tmp.png")
            img.save(tmp, format="PNG")
            os.replace(tmp, target)
        done = i + len(batch)
        print(f"[t2i] {done}/{len(jobs)} images  ({(time.time() - t0) / done:.1f}s/img)", flush=True)


def read_completions(path: Path) -> List[Dict]:
    """Rows from a completions jsonl, skipping a torn trailing line from a killed run."""
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[text] skipping malformed line in {path} (interrupted write?)")
    return rows


def server_model(endpoint: str) -> str:
    resp = _session().get(f"{endpoint.rstrip('/')}/v1/models", timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


def run_text(
    items: List[Dict],
    out_file: Path,
    endpoint: str,
    *,
    samples_per_prompt: int,
    gen: Dict,
    concurrency: int = 8,
) -> None:
    """Sample ``samples_per_prompt`` completions per item from an OpenAI-compatible
    server (``sglang serve`` / ``vllm serve``), appending jsonl rows
    ``{id, sample, response}`` to ``out_file``."""
    done = {(row["id"], row["sample"]) for row in read_completions(out_file)} if out_file.exists() else set()
    jobs = [(item, s) for item in items for s in range(samples_per_prompt) if (item["id"], s) not in done]
    if not jobs:
        print("[text] all completions present — nothing to generate")
        return

    model = server_model(endpoint)
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    suffix = gen.get("prompt_suffix", "")
    base_payload = {
        "model": model,
        "temperature": gen.get("temperature", 0.6),
        "top_p": gen.get("top_p", 0.95),
        "max_tokens": gen.get("max_tokens", 16384),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    session = _session()
    lock = threading.Lock()
    progress = {"done": 0}

    def one(job: Tuple[Dict, int]) -> None:
        item, s = job
        payload = dict(base_payload, messages=[{"role": "user", "content": item["problem"] + suffix}])
        last_exc = None
        for attempt in range(3):
            try:
                resp = session.post(url, json=payload, timeout=3600)
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"] or ""
                with lock:
                    out.write(json.dumps({"id": item["id"], "sample": s, "response": text}) + "\n")
                    out.flush()
                    progress["done"] += 1
                    if progress["done"] % 20 == 0:
                        print(f"[text] {progress['done']}/{len(jobs)}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 — retry any transport/server hiccup
                last_exc = exc
                time.sleep(5 * (attempt + 1))
        print(f"[text] FAILED {item['id']}#{s}: {last_exc}", flush=True)

    with open(out_file, "a") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(one, jobs))
    print(f"[text] {progress['done']}/{len(jobs)} completions written to {out_file}")
