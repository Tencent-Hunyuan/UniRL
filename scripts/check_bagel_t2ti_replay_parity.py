#!/usr/bin/env python3
"""Real captured-trace gate for BAGEL T2TI exact versus collapsed replay."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from unirl.models.bagel.bundle import BagelBundle
from unirl.models.bagel.conditions import BagelT2TIDiffusionConditions, BagelThinkKVReplaySpec
from unirl.models.bagel.config import BagelPipelineConfig
from unirl.models.bagel.diffusion import BagelDiffusionParams, BagelDiffusionStage


@dataclass(frozen=True)
class TensorMetrics:
    max_abs: float
    rel_l2: float
    cosine: float


@dataclass
class ModeResult:
    cache: list[torch.Tensor]
    velocity: torch.Tensor
    transition_mean: torch.Tensor
    log_prob: torch.Tensor
    gradients: list[torch.Tensor]
    input_sample: torch.Tensor
    prev_sample: torch.Tensor
    seconds: float
    peak_gib: float


def _metrics(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> TensorMetrics:
    if len(left) != len(right):
        raise RuntimeError(f"tensor list length mismatch: {len(left)} != {len(right)}")
    max_abs = 0.0
    diff_sq = 0.0
    left_sq = 0.0
    right_sq = 0.0
    dot = 0.0
    for lhs, rhs in zip(left, right):
        if lhs.shape != rhs.shape:
            raise RuntimeError(f"tensor shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
        lhs = lhs.float()
        rhs = rhs.float()
        delta = lhs - rhs
        max_abs = max(max_abs, float(delta.abs().max().item()))
        diff_sq += float((delta * delta).sum().item())
        left_sq += float((lhs * lhs).sum().item())
        right_sq += float((rhs * rhs).sum().item())
        dot += float((lhs * rhs).sum().item())
    return TensorMetrics(
        max_abs=max_abs,
        rel_l2=math.sqrt(diff_sq / max(left_sq, 1.0e-30)),
        cosine=dot / max(math.sqrt(left_sq * right_sq), 1.0e-30),
    )


def _load_spec(path: Path, sample_index: int) -> BagelThinkKVReplaySpec:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not 0 <= sample_index < len(lines):
        raise ValueError(f"sample index {sample_index} is outside [0, {len(lines)})")
    row = json.loads(lines[sample_index])
    payload = row.get("t2ti_replay")
    if not isinstance(payload, Mapping):
        raise ValueError(f"sample {sample_index} has no captured t2ti_replay mapping")
    return BagelThinkKVReplaySpec.from_custom_output(payload)


def _cache_tensors(context: dict) -> list[torch.Tensor]:
    cache = context["past_key_values"]
    values: list[torch.Tensor] = []
    for layer in range(cache.num_layers):
        values.extend(
            (
                cache.key_cache[layer].detach().float().cpu(),
                cache.value_cache[layer].detach().float().cpu(),
            )
        )
    return values


def _select_gradient_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    suffixes = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj_moe_gen.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp_moe_gen.down_proj.weight",
    )
    selected = [(name, param) for name, param in model.language_model.named_parameters() if name.endswith(suffixes)]
    found = {next(suffix for suffix in suffixes if name.endswith(suffix)) for name, _ in selected}
    missing = sorted(set(suffixes) - found)
    if missing:
        raise RuntimeError(f"checkpoint is missing representative decoder parameters: {missing}")
    for _, param in selected:
        param.requires_grad_(True)
    return selected


def _run_mode(
    bundle: BagelBundle,
    spec: BagelThinkKVReplaySpec,
    selected: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    mode: str,
    fixed_sample: torch.Tensor | None,
    fixed_prev_sample: torch.Tensor | None,
    device: torch.device,
) -> ModeResult:
    for _, param in selected:
        param.grad = None
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    stage = BagelDiffusionStage(model=bundle, t2ti_replay_chunk_mode=mode)
    params = BagelDiffusionParams(
        samples_per_prompt=1,
        num_inference_steps=2,
        height=spec.image_shape[0],
        width=spec.image_shape[1],
        eta=0.8,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
    )
    conditions = BagelT2TIDiffusionConditions.for_sample(spec)
    gen, cfg_text, cfg_img, image_shape = stage._resolve_single(conditions)
    cache_values = _cache_tensors(gen)
    gi, gi_cfg_text, gi_cfg_img = stage._build_generation_inputs(
        gen,
        cfg_text,
        cfg_img,
        image_shape,
        device=device,
    )
    forward_kwargs = stage._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)
    generated_sample = gi["packed_init_noises"].to(device=device, dtype=torch.float32)
    sample = generated_sample if fixed_sample is None else fixed_sample.to(device)
    if sample.shape != generated_sample.shape:
        raise RuntimeError(
            f"fixed latent shape {tuple(sample.shape)} does not match generated shape {tuple(generated_sample.shape)}"
        )
    sigma = torch.tensor(0.8, device=device)
    sigma_next = torch.tensor(0.6, device=device)
    velocity = stage.predict_velocity_at(forward_kwargs, sample=sample, sigma=sigma, params=params)

    if fixed_prev_sample is None:
        torch.manual_seed(4321)
        with torch.no_grad():
            fixed_prev_sample, _, _ = stage.step.denoise(
                stage.strategy,
                v_t=velocity.detach(),
                x_t=sample,
                sigma=sigma,
                sigma_next=sigma_next,
                sigma_max=sigma,
                eta=params.eta,
            )
    else:
        fixed_prev_sample = fixed_prev_sample.to(device)
    _, log_prob, transition_mean = stage.step.denoise(
        stage.strategy,
        v_t=velocity,
        x_t=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        sigma_max=sigma,
        eta=params.eta,
        prev_sample=fixed_prev_sample,
    )
    if log_prob is None or transition_mean is None:
        raise RuntimeError("stochastic replay did not produce log_prob/transition_mean")

    # This follows the train path through both UND cache construction and GEN
    # velocity prediction. The tiny velocity term keeps a direct gradient signal
    # even if a future SDE kernel detaches its replay log-prob.
    objective = -log_prob + 1.0e-3 * velocity.float().square().mean()
    objective.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients = []
    for name, param in selected:
        if param.grad is None:
            raise RuntimeError(f"representative decoder gradient is missing: {name}")
        gradients.append(param.grad.detach().float().cpu())
    peak_gib = float(torch.cuda.max_memory_allocated(device)) / 1024**3

    result = ModeResult(
        cache=cache_values,
        velocity=velocity.detach().float().cpu(),
        transition_mean=transition_mean.detach().float().cpu(),
        log_prob=log_prob.detach().float().cpu(),
        gradients=gradients,
        input_sample=sample.detach().float().cpu(),
        prev_sample=fixed_prev_sample.detach().float().cpu(),
        seconds=elapsed,
        peak_gib=peak_gib,
    )
    del (
        objective,
        velocity,
        transition_mean,
        log_prob,
        forward_kwargs,
        gi,
        gi_cfg_text,
        gi_cfg_img,
        gen,
        cfg_text,
        cfg_img,
    )
    for _, param in selected:
        param.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-cache-rel-l2", type=float, default=5.0e-3)
    parser.add_argument("--min-cache-cosine", type=float, default=0.9999)
    parser.add_argument("--max-velocity-rel-l2", type=float, default=5.0e-3)
    parser.add_argument("--min-velocity-cosine", type=float, default=0.9999)
    parser.add_argument("--max-grad-rel-l2", type=float, default=1.0e-2)
    parser.add_argument("--min-grad-cosine", type=float, default=0.999)
    parser.add_argument("--max-log-prob-abs", type=float, default=1.0e-3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this parity gate requires CUDA")

    spec = _load_spec(args.samples_jsonl, args.sample_index)
    device = torch.device("cuda:0")
    bundle = BagelBundle.from_config(
        BagelPipelineConfig(
            pretrained_model_ckpt_path=args.model_path,
            device=device,
            model_precision="bf16",
            use_lora=True,
        )
    )
    bundle.model.requires_grad_(False)
    selected = _select_gradient_parameters(bundle.model)
    exact = _run_mode(
        bundle,
        spec,
        selected,
        mode="exact",
        fixed_sample=None,
        fixed_prev_sample=None,
        device=device,
    )
    collapsed = _run_mode(
        bundle,
        spec,
        selected,
        mode="collapsed",
        fixed_sample=exact.input_sample,
        fixed_prev_sample=exact.prev_sample,
        device=device,
    )

    cache_metrics = _metrics(exact.cache, collapsed.cache)
    velocity_metrics = _metrics([exact.velocity], [collapsed.velocity])
    mean_metrics = _metrics([exact.transition_mean], [collapsed.transition_mean])
    gradient_metrics = _metrics(exact.gradients, collapsed.gradients)
    log_prob_abs = float((exact.log_prob - collapsed.log_prob).abs().item())
    passed = (
        cache_metrics.rel_l2 <= args.max_cache_rel_l2
        and cache_metrics.cosine >= args.min_cache_cosine
        and velocity_metrics.rel_l2 <= args.max_velocity_rel_l2
        and velocity_metrics.cosine >= args.min_velocity_cosine
        and gradient_metrics.rel_l2 <= args.max_grad_rel_l2
        and gradient_metrics.cosine >= args.min_grad_cosine
        and log_prob_abs <= args.max_log_prob_abs
    )
    result = {
        "passed": passed,
        "captured_tokens": spec.kv_length,
        "exact_calls": len(spec.chunks()),
        "collapsed_calls": 1,
        "exact_seconds": exact.seconds,
        "collapsed_seconds": collapsed.seconds,
        "speedup": exact.seconds / max(collapsed.seconds, 1.0e-12),
        "exact_peak_gib": exact.peak_gib,
        "collapsed_peak_gib": collapsed.peak_gib,
        "cache": asdict(cache_metrics),
        "velocity": asdict(velocity_metrics),
        "transition_mean": asdict(mean_metrics),
        "log_prob_abs": log_prob_abs,
        "decoder_gradients": asdict(gradient_metrics),
        "gradient_parameters": [name for name, _ in selected],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
