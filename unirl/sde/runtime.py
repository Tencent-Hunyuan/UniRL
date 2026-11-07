"""Shared SDE runtime entrypoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: Optional[torch.device] = None,
    *,
    mu: Optional[float] = None,
    time_shift_type: str = "exponential",
    shift_terminal: Optional[float] = None,
) -> torch.Tensor:
    """Compute the FlowMatch σ schedule of length ``num_steps + 1``."""
    if mu is None:
        if shift_terminal is not None:
            raise ValueError(
                f"get_sigma_schedule: shift_terminal={shift_terminal!r} is only "
                f"supported on the dynamic branch (mu is not None); no static-"
                f"shift model declares it. Pass mu= or drop shift_terminal."
            )
        t = torch.linspace(1.0, 0.0, num_steps + 1)
        sigmas = (shift * t) / (1 + (shift - 1) * t)
    else:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            time_shift_type=time_shift_type,
            shift_terminal=shift_terminal,
        )
        base_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(num_inference_steps=num_steps, sigmas=base_sigmas, mu=mu)
        sigmas = scheduler.sigmas
    if device is not None:
        sigmas = sigmas.to(device)
    return sigmas


def calculate_dynamic_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear interpolation of dynamic-shift μ from image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file; return ``None`` on any failure (missing / unreadable)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, OSError):
        return None


_DYNAMIC_SCHED_KEYS = (
    "base_shift",
    "max_shift",
    "base_image_seq_len",
    "max_image_seq_len",
    "time_shift_type",
    "shift_terminal",
)


@dataclass
class FlowMatchSchedulePolicy:
    """The model-owned σ schedule policy. Loaded once per actor."""

    shift: float = 3.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    time_shift_type: str = "exponential"
    shift_terminal: Optional[float] = None
    vae_scale_factor: int = 8
    patch_size: int = 2

    def __post_init__(self) -> None:
        self.shift = float(self.shift)
        self.use_dynamic_shifting = bool(self.use_dynamic_shifting)
        self.shift_terminal = self.shift_terminal or None
        if isinstance(self.patch_size, (list, tuple)):
            self.patch_size = self.patch_size[-1] if self.patch_size else 2

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        """Dynamic-shift μ for this policy — the single per-model override point."""
        return calculate_dynamic_mu(
            image_seq_len,
            base_seq_len=self.base_image_seq_len,
            max_seq_len=self.max_image_seq_len,
            base_shift=self.base_shift,
            max_shift=self.max_shift,
        )

    def compute_sigma(
        self,
        *,
        num_inference_steps: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Apply this policy to a request's ``(T, H, W)`` → σ tensor ``[T+1]``."""
        if not self.use_dynamic_shifting:
            return get_sigma_schedule(num_inference_steps, self.shift, device, shift_terminal=self.shift_terminal)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        image_seq_len = (latent_h // self.patch_size) * (latent_w // self.patch_size)
        mu = self.compute_mu(image_seq_len, num_inference_steps)
        return get_sigma_schedule(
            num_inference_steps,
            self.shift,
            device,
            mu=mu,
            time_shift_type=self.time_shift_type,
            shift_terminal=self.shift_terminal,
        )

    @classmethod
    def static_only(cls, shift: float) -> "FlowMatchSchedulePolicy":
        """Build a static-shift-only policy."""
        return cls(shift=shift, use_dynamic_shifting=False)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path, None],
        *,
        shift: float,
        require_dynamic: bool = False,
        dynamic_overrides: Optional[Dict[str, Any]] = None,
    ) -> "FlowMatchSchedulePolicy":
        """Build a policy by reading the diffusers-standard JSON layout."""
        root = Path(path) if path is not None else None
        if root is None or not root.exists():
            if require_dynamic:
                if not dynamic_overrides:
                    raise RuntimeError(
                        f"FlowMatchSchedulePolicy.from_pretrained: caller declared "
                        f"require_dynamic=True for path={path!r} but provided no "
                        f"dynamic_overrides. The checkpoint isn't locally readable "
                        f"so we can't load scheduler_config.json, and without "
                        f"explicit dynamic fields we'd silently produce a static "
                        f"policy (which mis-shifts dynamic-shift models like "
                        f"Qwen-Image). Pre-download the scheduler/scheduler_config.json "
                        f"from HF Hub OR have the model's Pipeline.build_schedule_policy "
                        f"pass dynamic_overrides with use_dynamic_shifting=True + "
                        f"base_shift / max_shift / base_image_seq_len / max_image_seq_len / "
                        f"time_shift_type (+ shift_terminal where the model declares it) fields."
                    )
                allowed = (*_DYNAMIC_SCHED_KEYS, "vae_scale_factor", "patch_size")
                kwargs = {k: dynamic_overrides[k] for k in allowed if dynamic_overrides.get(k) is not None}
                return cls(shift=shift, use_dynamic_shifting=True, **kwargs)
            if root is not None:
                logger.debug(
                    "FlowMatchSchedulePolicy.from_pretrained: %s does not exist "
                    "locally (likely an HF repo ID — bundle.from_pretrained will "
                    "resolve it). Falling back to static_only(shift=%s).",
                    root,
                    shift,
                )
            return cls.static_only(shift)

        sched_path = root / "scheduler" / "scheduler_config.json"
        sched = _read_json(sched_path)
        if sched is None:
            logger.warning(
                "FlowMatchSchedulePolicy.from_pretrained: %s not found; "
                "dynamic-shift fields default to static-only behavior. "
                "If the model wants dynamic shift, σ will drift and the "
                "drift assert will raise at the first rollout.",
                sched_path,
            )
            sched = {}
        trans = _read_json(root / "transformer" / "config.json") or {}
        vae = _read_json(root / "vae" / "config.json") or {}

        kwargs = {k: sched[k] for k in ("use_dynamic_shifting", *_DYNAMIC_SCHED_KEYS) if sched.get(k) is not None}
        if blocks := vae.get("block_out_channels"):
            kwargs["vae_scale_factor"] = 2 ** (len(blocks) - 1)
        if (patch_size := trans.get("patch_size")) is not None:
            kwargs["patch_size"] = patch_size
        return cls(shift=shift, **kwargs)


def ensure_sample_sigmas(sample: Any, policy: FlowMatchSchedulePolicy) -> None:
    """Compute and pin σ onto a Sample's diffusion generation parameters."""
    from unirl.types.sampling import DiffusionSamplingParams

    if not sample.parts or not isinstance(sample.parts[-1].sampling_params, DiffusionSamplingParams):
        return
    gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
    diffusion = gen_part.sampling_params
    if diffusion.sigmas is not None:
        return
    diffusion.sigmas = policy.compute_sigma(
        num_inference_steps=diffusion.num_inference_steps,
        height=diffusion.height,
        width=diffusion.width,
    )


__all__ = [
    "FlowMatchSchedulePolicy",
    "ensure_sample_sigmas",
    "get_sigma_schedule",
    "calculate_dynamic_mu",
]
