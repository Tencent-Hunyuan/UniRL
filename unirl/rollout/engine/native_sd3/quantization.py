"""Selective Transformer Engine FP8 conversion for Hopper rollout inference."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

from .config import NativeSD3EngineConfig


@dataclass(frozen=True)
class FP8ConversionReport:
    replaced: Tuple[str, ...]
    skipped: Tuple[Tuple[str, str], ...]


class FP8Controller:
    """Per-worker precision route and TE weight-cache generation."""

    def __init__(self, config: NativeSD3EngineConfig) -> None:
        self.config = config
        self.enabled = bool(config.fp8_enabled)
        self._mode = "bf16"
        self._step = 0
        self._total_steps = 0
        self._weights_dirty = True
        self._first_microbatch = False
        self._fp8_active = False
        self.te = None
        self.recipe = None
        if self.enabled:
            try:
                import transformer_engine.pytorch as te
                from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "NativeSD3 FP8 rollout requires a working transformer-engine[pytorch] installation"
                ) from exc
            self.te = te
            if config.fp8_recipe == "current":
                self.recipe = Float8CurrentScaling(fp8_format=Format.E4M3)
            else:
                self.recipe = DelayedScaling(
                    fp8_format=Format.E4M3,
                    amax_history_len=16,
                    amax_compute_algo="max",
                )

    @property
    def first_microbatch(self) -> bool:
        return self._first_microbatch

    @property
    def fp8_active(self) -> bool:
        return self._fp8_active

    def mark_weights_dirty(self) -> None:
        self._weights_dirty = True

    @contextmanager
    def rollout(self, *, mode: str, total_steps: int) -> Iterator[None]:
        if mode not in {"bf16", "fp8"}:
            raise ValueError(f"FP8Controller rollout mode must be bf16|fp8, got {mode!r}.")
        if mode == "fp8" and not self.enabled:
            raise RuntimeError("A request selected rollout_precision=fp8, but this engine has fp8_enabled=false.")
        previous = (self._mode, self._step, self._total_steps)
        self._mode = mode
        self._step = 0
        self._total_steps = int(total_steps)
        try:
            yield
        finally:
            self._mode, self._step, self._total_steps = previous
            self._first_microbatch = False

    def _use_fp8(self) -> bool:
        if not self.enabled or self._mode != "fp8":
            return False
        prefix = int(self.config.bf16_prefix_steps)
        suffix = int(self.config.bf16_suffix_steps)
        return self._step >= prefix and self._step < max(prefix, self._total_steps - suffix)

    @contextmanager
    def transformer_forward(self) -> Iterator[None]:
        use_fp8 = self._use_fp8()
        self._fp8_active = use_fp8
        self._first_microbatch = bool(use_fp8 and self._weights_dirty)
        if not use_fp8:
            context = nullcontext()
        else:
            autocast = getattr(self.te, "autocast", None)
            if autocast is not None:
                context = autocast(enabled=True, recipe=self.recipe)
            else:
                context = self.te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)
        try:
            with context:
                yield
        finally:
            if use_fp8 and self._weights_dirty:
                self._weights_dirty = False
            self._fp8_active = False
            self._first_microbatch = False
            self._step += 1


class RoutedTransformer(nn.Module):
    """Wrap the converted DiT and route each denoising call through FP8/BF16."""

    def __init__(self, module: nn.Module, controller: FP8Controller) -> None:
        super().__init__()
        self.module = module
        self.controller = controller

    def forward(self, *args, **kwargs):
        with self.controller.transformer_forward():
            return self.module(*args, **kwargs)


class TELinear(nn.Module):
    """TE Linear wrapper preserving the ``nn.Linear``-like public attributes."""

    def __init__(self, linear: nn.Module, controller: FP8Controller) -> None:
        super().__init__()
        self.te_linear = linear
        self.controller = controller
        try:
            self._supports_first_microbatch = "is_first_microbatch" in inspect.signature(linear.forward).parameters
        except (TypeError, ValueError):
            self._supports_first_microbatch = False

    @property
    def weight(self):
        return self.te_linear.weight

    @property
    def bias(self):
        return self.te_linear.bias

    @property
    def in_features(self) -> int:
        return int(self.te_linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.te_linear.out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self._supports_first_microbatch:
            kwargs["is_first_microbatch"] = self.controller.first_microbatch
        inputs = inputs.to(torch.bfloat16)
        rows = int(inputs.numel() // inputs.shape[-1])
        # Hopper TE kernels require M divisible by 8. SD3's context stream has
        # sequence length 333, so a normal micro-batch can violate that even
        # though every feature dimension is aligned. Pad only the flattened row
        # axis during FP8 execution and remove the synthetic rows afterward.
        if self.controller.fp8_active and rows % 8:
            flat = inputs.reshape(rows, int(inputs.shape[-1]))
            padding = 8 - rows % 8
            flat = torch.cat((flat, flat.new_zeros((padding, flat.shape[-1]))), dim=0)
            output = self.te_linear(flat, **kwargs)
            return output[:rows].reshape(*inputs.shape[:-1], self.out_features)
        return self.te_linear(inputs, **kwargs)


def convert_transformer_for_fp8(
    model: nn.Module,
    *,
    config: NativeSD3EngineConfig,
    controller: FP8Controller,
) -> tuple[Dict[str, torch.Tensor], FP8ConversionReport]:
    """Replace eligible linears and return original-name weight targets."""

    original_names = tuple(name for name, _ in model.named_parameters())
    original_buffer_names = tuple(name for name, _ in model.named_buffers())
    parameter_targets: Dict[str, torch.Tensor] = {}
    replaced: List[str] = []
    skipped: List[Tuple[str, str]] = []

    def recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if not isinstance(child, nn.Linear):
                recurse(child, fqn)
                continue

            reason: Optional[str] = None
            if any(pattern in fqn for pattern in config.fp8_skip_modules):
                reason = "name"
            elif max(int(child.in_features), int(child.out_features)) <= int(config.fp8_min_dim):
                reason = "small"
            elif int(child.in_features) % int(config.fp8_dim_multiple) or int(child.out_features) % int(
                config.fp8_dim_multiple
            ):
                reason = "alignment"
            if reason is not None:
                skipped.append((fqn, reason))
                continue

            te_linear = controller.te.Linear(
                int(child.in_features),
                int(child.out_features),
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
                device=child.weight.device,
            )
            te_linear = te_linear.to(dtype=torch.bfloat16)
            with torch.no_grad():
                te_linear.weight.copy_(child.weight.to(dtype=torch.bfloat16))
                if child.bias is not None and te_linear.bias is not None:
                    te_linear.bias.copy_(child.bias.to(dtype=torch.bfloat16))
            replacement = TELinear(te_linear, controller)
            setattr(module, name, replacement)
            parameter_targets[f"{fqn}.weight"] = te_linear.weight
            if child.bias is not None and te_linear.bias is not None:
                parameter_targets[f"{fqn}.bias"] = te_linear.bias
            replaced.append(fqn)

    if controller.enabled:
        recurse(model)

    current = dict(model.named_parameters())
    for name in original_names:
        if name in parameter_targets:
            continue
        if name not in current:
            raise RuntimeError(f"FP8 conversion lost parameter {name!r} without registering a replacement target.")
        parameter_targets[name] = current[name]
    current_buffers = dict(model.named_buffers())
    for name in original_buffer_names:
        if name not in current_buffers:
            raise RuntimeError(f"FP8 conversion lost buffer {name!r} without registering a replacement target.")
        parameter_targets[name] = current_buffers[name]
    return parameter_targets, FP8ConversionReport(tuple(replaced), tuple(skipped))


__all__ = [
    "FP8Controller",
    "FP8ConversionReport",
    "RoutedTransformer",
    "convert_transformer_for_fp8",
]
