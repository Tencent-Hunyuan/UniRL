#!/usr/bin/env python3
"""Checkpoint-backed alignment gates for the SGLang 0.5.19 migration."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _assert_tensor_exact(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
        max_abs = None
        if left.shape == right.shape and left.numel() and left.is_floating_point() and right.is_floating_point():
            max_abs = float((left.float().cpu() - right.float().cpu()).abs().max().item())
        raise AssertionError(
            f"{name} is not bitwise identical: "
            f"left={tuple(left.shape)}/{left.dtype}, right={tuple(right.shape)}/{right.dtype}, "
            f"max_abs={max_abs}"
        )


def _assert_nested_exact(name: str, left: Any, right: Any) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError(f"{name}: tensor/non-tensor mismatch")
        _assert_tensor_exact(name, left, right)
        return
    if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
        if type(left) is not type(right):
            raise AssertionError(f"{name}: dataclass type mismatch {type(left)} != {type(right)}")
        for field in dataclasses.fields(left):
            _assert_nested_exact(f"{name}.{field.name}", getattr(left, field.name), getattr(right, field.name))
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)) or set(left) != set(right):
            raise AssertionError(f"{name}: mapping keys differ")
        for key in sorted(left):
            _assert_nested_exact(f"{name}[{key!r}]", left[key], right[key])
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{name}: sequence shape/type differs")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_exact(f"{name}[{index}]", left_item, right_item)
        return
    if left != right:
        raise AssertionError(f"{name}: {left!r} != {right!r}")


def _drift(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise AssertionError(f"log-prob shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    delta = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    if not torch.isfinite(delta).all():
        raise AssertionError("log-prob drift contains NaN/Inf")
    return {
        "mean": float(delta.mean().item()) if delta.numel() else 0.0,
        "max": float(delta.max().item()) if delta.numel() else 0.0,
    }


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not torch.isfinite(grad).all():
            raise AssertionError("backward produced NaN/Inf gradients")
        total += float(grad.square().sum().item())
        found = True
    value = math.sqrt(total)
    if not found or not math.isfinite(value) or value <= 0.0:
        raise AssertionError(f"expected finite nonzero gradients, got {value}")
    return value


def _signed_backward(log_probs: torch.Tensor, module: torch.nn.Module) -> tuple[float, float]:
    module.zero_grad(set_to_none=True)
    flat = log_probs.reshape(-1)
    signs = torch.where(
        torch.arange(flat.numel(), device=flat.device) % 2 == 0,
        torch.ones((), device=flat.device),
        -torch.ones((), device=flat.device),
    )
    objective = -(flat * signs).mean()
    if not objective.requires_grad:
        raise AssertionError("replay objective has no gradient")
    objective.backward()
    return float(objective.detach().float().item()), _gradient_norm(module.parameters())


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload, flush=True)
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")


def _ar_request(args: argparse.Namespace) -> Sample:
    root = Part.input(
        ["qwen3-alignment"],
        primitives={"text": Texts(texts=[args.prompt])},
        control={"ar": {"return_logprob": True}},
    )
    sampling = ARSamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    return Sample.request(root).fork(1, sampling_params=sampling)


def _ar_engine_kwargs(mem_fraction_static: float) -> dict[str, Any]:
    """Production Qwen3 train/inference alignment settings."""
    return {
        "mem_fraction_static": mem_fraction_static,
        "skip_server_warmup": True,
        "disable_cuda_graph": False,
        "cuda_graph_max_bs_decode": 16,
        "rl_on_policy_target": "fsdp",
        "attention_backend": "triton",
    }


def run_ar(args: argparse.Namespace) -> dict[str, Any]:
    from unirl.models.qwen3.conditions import Qwen3ARConditions
    from unirl.models.qwen3.config import Qwen3PipelineConfig
    from unirl.models.qwen3.pipeline import Qwen3Pipeline
    from unirl.rollout.engine.sglang.config import SGLangEngineConfig
    from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

    request = _ar_request(args)
    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=args.model,
        model_family="text",
        backend="native",
        tp_size=1,
        concurrency=1,
        samples_pre_expanded=True,
        engine_kwargs=_ar_engine_kwargs(args.mem_fraction_static),
    )
    engine = SGLangRolloutEngine(config=config)
    try:
        first = engine.generate(request).frontier_gen_part(ARSamplingParams)
        second = engine.generate(request).frontier_gen_part(ARSamplingParams)
        _assert_nested_exact("ar.segment", first.segment, second.segment)
        _assert_nested_exact("ar.conditions", first.conditions, second.conditions)

        bitwise_sleep_wake = None
        if args.check_sleep_wake:
            engine.sleep()
            engine.wake_up()
            after_wake = engine.generate(request).frontier_gen_part(ARSamplingParams)
            _assert_nested_exact("ar.sleep_wake.segment", first.segment, after_wake.segment)
            _assert_nested_exact("ar.sleep_wake.conditions", first.conditions, after_wake.conditions)
            bitwise_sleep_wake = True
    finally:
        engine.shutdown()
    _release_cuda()

    pipeline = Qwen3Pipeline.from_config(
        Qwen3PipelineConfig(
            pretrained_model_ckpt_path=args.model,
            model_precision="bf16",
            attn_implementation="sdpa",
            device=torch.device("cuda"),
            autocast_precision="bf16",
            logprob_precision="fp32",
        )
    )
    conditions = Qwen3ARConditions.from_dict(first.conditions)
    replay = pipeline.ar.replay(
        conditions,
        segment=first.segment,
        temperature=args.temperature,
    )
    rollout_log_probs = first.segment.log_probs.to(replay.device, dtype=replay.dtype)
    metrics = _drift(replay, rollout_log_probs)
    if metrics["mean"] >= args.logprob_mean_limit or metrics["max"] >= args.logprob_max_limit:
        raise AssertionError(
            f"AR rollout/replay drift exceeded limits: {metrics}, "
            f"limits mean<{args.logprob_mean_limit}, max<{args.logprob_max_limit}"
        )
    objective, grad_norm = _signed_backward(replay, pipeline.bundle.transformer)
    report = {
        "mode": "ar",
        "model": args.model,
        "tokens_sha256": _tensor_digest(first.segment.tokens),
        "rollout_log_probs_sha256": _tensor_digest(first.segment.log_probs),
        "num_tokens": int(first.segment.tokens.numel()),
        "rollout_replay_absdiff": metrics,
        "signed_objective": objective,
        "gradient_norm": grad_norm,
        "rl_on_policy_target": "fsdp",
        "bitwise_repeat": True,
        "bitwise_sleep_wake": bitwise_sleep_wake,
    }
    del pipeline
    _release_cuda()
    return report


def _sd3_request(args: argparse.Namespace) -> tuple[Sample, DiffusionSamplingParams]:
    from unirl.sde.kernels import FlowSDEStrategy

    root = Part.input(
        ["sd3-alignment"],
        primitives={"text": Texts(texts=[args.prompt])},
    )
    strategy = FlowSDEStrategy()
    sampling = DiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0,
        height=args.height,
        width=args.width,
        num_frames=1,
        seed=args.seed,
        init_same_noise=True,
        init_noise_latent_shape=[16, args.height // 8, args.width // 8],
        eta=args.eta,
        sde_strategy=strategy,
        sde_indices=list(range(args.num_inference_steps)),
        samples_per_prompt=2,
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
    )
    return Sample.request(root).fork(2, sampling_params=sampling), sampling


def run_sd3(args: argparse.Namespace) -> dict[str, Any]:
    from unirl.models.sd3.conditions import SD3Conditions
    from unirl.models.sd3.config import SD3PipelineConfig
    from unirl.models.sd3.pipeline import SD3Pipeline
    from unirl.rollout.engine.sglang_diffusion.config import SGLangDiffusionEngineConfig
    from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
    from unirl.sde.kernels import FlowSDEStrategy

    request, sampling = _sd3_request(args)
    model_config = SD3PipelineConfig(
        pretrained_model_ckpt_path=args.model,
        model_precision="bf16",
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        shift=3.0,
        load_vae=False,
        use_lora=False,
    )
    config = SGLangDiffusionEngineConfig(
        sampling=sampling,
        model_family="sd3",
        populate_conditions=True,
        init_same_noise=True,
        num_gpus=1,
        tp_size=1,
        local_mode=True,
        forward_batch_size=2,
        engine_kwargs={"model_type": "sd3"},
    )
    engine = SGLangDiffusionRolloutEngine(
        config=config,
        strategy=FlowSDEStrategy(),
        model_config=model_config,
    )
    try:
        first = engine.generate(request).frontier_gen_part(DiffusionSamplingParams)
        second = engine.generate(request).frontier_gen_part(DiffusionSamplingParams)
        _assert_nested_exact("sd3.segment", first.segment, second.segment)
        _assert_nested_exact("sd3.conditions", first.conditions, second.conditions)
    finally:
        engine.shutdown()
    _release_cuda()

    if int(first.segment.indices[0].item()) != 0:
        raise AssertionError(f"SD3 trajectory does not contain x_T: indices={first.segment.indices.tolist()}")
    x_t = first.segment.latents[:, 0]
    _assert_tensor_exact("sd3.shared_x_t", x_t[0], x_t[1])
    post = first.segment.latents[:, 1:]
    if torch.equal(post[0], post[1]):
        raise AssertionError("SD3 grouped sibling trajectories are cloned after x_T")

    pipeline = SD3Pipeline.from_config(
        SD3PipelineConfig(
            pretrained_model_ckpt_path=args.model,
            model_precision="bf16",
            device=torch.device("cuda"),
            autocast_precision="bf16",
            trajectory_precision="bf16",
            logprob_precision="fp32",
            shift=3.0,
            load_vae=False,
        ),
        strategy=FlowSDEStrategy(),
    )
    conditions = SD3Conditions.from_dict(first.conditions)
    replay = pipeline.diffusion.replay(
        conditions,
        segment=first.segment,
        params=first.sampling_params,
    )
    rollout_log_probs = first.segment.sde_logp.to(replay.log_probs.device, dtype=replay.log_probs.dtype)
    metrics = _drift(replay.log_probs, rollout_log_probs)
    objective, grad_norm = _signed_backward(replay.log_probs, pipeline.bundle.transformer)
    report = {
        "mode": "sd3",
        "model": args.model,
        "trajectory_sha256": _tensor_digest(first.segment.latents),
        "sigmas_sha256": _tensor_digest(first.segment.sigmas),
        "rollout_log_probs_sha256": _tensor_digest(first.segment.sde_logp),
        "trajectory_shape": list(first.segment.latents.shape),
        "rollout_replay_absdiff": metrics,
        "signed_objective": objective,
        "gradient_norm": grad_norm,
        "bitwise_repeat": True,
        "shared_x_t": True,
        "distinct_grouped_trajectories": True,
    }
    del pipeline
    _release_cuda()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("ar", "sd3"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.3)
    parser.add_argument("--logprob-mean-limit", type=float, default=0.05)
    parser.add_argument("--logprob-max-limit", type=float, default=0.5)
    parser.add_argument("--check-sleep-wake", action="store_true")

    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--eta", type=float, default=0.7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("alignment validation requires CUDA")
    torch.manual_seed(args.seed)
    report = run_ar(args) if args.mode == "ar" else run_sd3(args)
    _write_report(args.output, report)


if __name__ == "__main__":
    main()
