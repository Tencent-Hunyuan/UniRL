"""SGLang runtime hooks owned by UniRL."""

from __future__ import annotations

import logging
import os

_DETERMINISTIC_SAMPLER_ENV = "UNIRL_SGLANG_DETERMINISTIC_SAMPLER"
logger = logging.getLogger(__name__)


def configure_deterministic_sampler(name: str) -> None:
    """Select the deterministic sampler inherited by SGLang schedulers."""
    os.environ[_DETERMINISTIC_SAMPLER_ENV] = str(name)


def register_unirl_runtime() -> None:
    """Register UniRL's opt-in deterministic sampler hook."""
    if os.environ.get(_DETERMINISTIC_SAMPLER_ENV, "sglang") != "inverse_cdf":
        return

    logger.info("Registering UniRL inverse-CDF deterministic sampler")
    import torch
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    @torch.compile(dynamic=True)
    def inverse_cdf_sample(
        logprobs: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the counter hash visible to Inductor. Calling SGLang's custom
        # Triton Murmur op inside torch.compile can expose its output before the
        # opaque mutation is ordered, which makes identical calls intermittent.
        mask = 0xFFFFFFFF
        hashes = seeds.to(torch.int64)
        hashes = (hashes & mask) ^ ((hashes >> 32) & mask)
        hashes = hashes ^ ((positions.to(torch.int64) * 0x9E3779B9) & mask)
        hashes = hashes ^ (hashes >> 16)
        hashes = (hashes * 0x85EBCA6B) & mask
        hashes = hashes ^ (hashes >> 13)
        hashes = (hashes * 0xC2B2AE35) & mask
        hashes = hashes ^ (hashes >> 16)
        uniforms = (hashes.to(torch.float64) + 0.5) * (2.0**-32)
        cumulative = torch.cumsum(logprobs.float().exp(), dim=-1)
        totals = cumulative[:, -1]
        thresholds = (uniforms * totals.double()).float()
        thresholds = torch.minimum(
            thresholds,
            torch.nextafter(totals, torch.zeros_like(totals)),
        )
        sampled = torch.searchsorted(cumulative, thresholds[:, None]).view(-1)
        sampled = sampled.clamp_max(logprobs.shape[-1] - 1)
        return torch.where(totals > 0, sampled, torch.zeros_like(sampled)).to(torch.int32)

    def sample_from_logprobs(
        _original_fn,
        sampler,
        logprobs,
        sampling_info,
        positions,
    ):
        del sampler
        seeds = sampling_info.sampling_seed
        assert seeds is not None, "inverse-CDF sampling requires sampling_seed"
        return inverse_cdf_sample(logprobs, seeds, positions)

    HookRegistry.register(
        "sglang.srt.layers.sampler.Sampler._sample_from_logprobs",
        sample_from_logprobs,
        HookType.AROUND,
    )


__all__ = ["configure_deterministic_sampler", "register_unirl_runtime"]
