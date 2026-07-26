"""WAN22Bundle — concrete weights+params holder for WAN 2.2 T2V / I2V.

Implements the empty :class:`Bundle` Protocol. Holds the dual
``WanTransformer3DModel`` pair (high_noise + low_noise) wrapped in a
single :class:`WanDualTransformer` composite, plus the VAE + UMT5 text
encoder shared with WAN 2.1.

**Why a composite transformer?** The new policy stack expects each stage
to expose one trainable module. ``WanDualTransformer`` is that surface:
``named_modules()`` recurses into both branches, so LoRA target matching
and FSDP block discovery see both ``high_noise.*`` and ``low_noise.*``.
FSDPPolicy currently uses block-only wrapping, not root wrapping: it
fully shards each discovered ``WanTransformerBlock`` and leaves the
composite root as a plain ``nn.Module``. The composite ``forward`` still
matters because it centralizes high/low branch routing and keeps callers
from reaching into branch internals.

Reuses :class:`unirl.models.wan21.bundle.WAN21Bundle.from_config`
for the VAE / text encoder / scheduler loading — those components are
unchanged from WAN 2.1, and not duplicating the loading logic keeps
both bundles in sync.

I2V channel conditioning (the 20-channel mask + VAE latent payload) is
shared with WAN 2.1 via the structurally-typed
:class:`WAN21ImageLatentEncodeStage`. WAN 2.2 14B I2V checkpoints
declare ``image_dim == 0`` (no CLIP vision tower), so the CLIP path is
not exercised here — image conditioning travels entirely through the
channel concat inside :class:`WAN22DiffusionStep`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.wan21.bundle import WAN21Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import WAN22PipelineConfig


class WanDualTransformer(nn.Module):
    """Thin ``nn.Module`` wrapper presenting two WAN transformers as one model.

    Exists so the stage exposes a single trainable module while still
    making both sub-transformers visible to recursive tooling.
    ``named_modules()`` recurses into both children, so LoRA target
    discovery and FSDPPolicy's block enumeration see
    ``WanTransformerBlock`` instances in both branches.

    The ``forward()`` method routes to the correct sub-transformer based
    on a ``use_high_noise`` flag. The composite root is not itself
    fully-sharded today; FSDPPolicy shards the discovered transformer
    blocks below it. Keep call sites going through this root anyway so
    branch selection remains a stage-level contract rather than leaking
    high/low transformer internals into sampling code.
    """

    def __init__(self, high_noise: nn.Module, low_noise: nn.Module) -> None:
        super().__init__()
        self.high_noise = high_noise
        self.low_noise = low_noise

    def forward(self, *, use_high_noise: bool, **kwargs: Any) -> Any:
        """Dispatch to the high- or low-noise sub-transformer.

        ``**kwargs`` forwards directly to the chosen sub-transformer
        (the underlying ``WanTransformer3DModel.forward`` signature).
        Returning whatever the sub-transformer returns (typically a
        tuple when ``return_dict=False``).
        """
        target = self.high_noise if use_high_noise else self.low_noise
        return target(**kwargs)


class WAN22Bundle(Bundle):
    """WAN 2.2 T2V bundle: dual transformer + VAE + UMT5 text encoder."""

    def __init__(
        self,
        *,
        transformer: WanDualTransformer,
        high_noise_transformer: nn.Module,
        low_noise_transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        max_sequence_length: int,
        boundary_ratio: float,
        guidance_scale_2: Any,
        num_train_timesteps: int,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        # Sub-transformer handles also exposed for hooks that need to
        # iterate them individually (e.g. checkpoint loading verifiers).
        self.high_noise_transformer = high_noise_transformer
        self.low_noise_transformer = low_noise_transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.max_sequence_length = max_sequence_length
        self.boundary_ratio = float(boundary_ratio)
        self.guidance_scale_2 = guidance_scale_2
        self.num_train_timesteps = int(num_train_timesteps)

    @classmethod
    def from_config(cls, config: WAN22PipelineConfig) -> "WAN22Bundle":
        """Load both WAN 2.2 transformers + reuse WAN 2.1 VAE / text loaders."""
        try:
            from diffusers import WanTransformer3DModel
        except ImportError:
            from diffusers import AutoModel

            WanTransformer3DModel = AutoModel

        # Step 1: reuse ``WAN21Bundle.from_config`` to load the shared
        # WAN 2.x components — VAE, UMT5 text encoder, tokenizer — plus
        # the transformer at the ``transformer/`` subfolder. This works
        # because:
        #   - VAE / text encoder / tokenizer are architecturally
        #     identical between WAN 2.1 and 2.2; reusing one loader keeps
        #     the wiring in sync.
        #   - WAN 2.2 checkpoints store the high-noise transformer at
        #     ``transformer/`` and the low-noise transformer at
        #     ``transformer_2/``. So the proxy's ``.transformer`` field is
        #     intentionally repurposed as our ``high_noise_transformer``
        #     here.
        # We don't expose ``WAN21Bundle`` itself — callers see only the
        # ``WAN22Bundle`` surface — but we pluck the loaded modules off
        # it. ``WAN22PipelineConfig(WAN21PipelineConfig)`` so
        # ``WAN21Bundle.from_config`` accepts our config.
        aux = WAN21Bundle.from_config(config)
        high_noise_transformer = aux.transformer

        # Step 2: load the low-noise transformer separately.
        transformer_2_path = config.transformer_2_pretrained_path or config.pretrained_model_ckpt_path
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        low_noise_transformer = WanTransformer3DModel.from_pretrained(
            transformer_2_path,
            subfolder="transformer_2",
            torch_dtype=dtype,
        )
        # Dtype unification, same reason as in WAN 2.1: diffusers leaves
        # some buffers in fp32; FSDP2 asserts a uniform dtype across the
        # wrapped module.
        low_noise_transformer = low_noise_transformer.to(aux.device, dtype=dtype)

        # Step 3: expose both branches through the composite. The composite
        # is the stage's trainable-module surface; FSDPPolicy then discovers
        # and wraps the WanTransformerBlock children under both branches.
        transformer = WanDualTransformer(
            high_noise=high_noise_transformer,
            low_noise=low_noise_transformer,
        )

        return cls(
            transformer=transformer,
            high_noise_transformer=high_noise_transformer,
            low_noise_transformer=low_noise_transformer,
            vae=aux.vae,
            text_encoder=aux.text_encoder,
            tokenizer=aux.tokenizer,
            dtype=aux.dtype,
            device=aux.device,
            pretrained_path=aux.pretrained_path,
            max_sequence_length=aux.max_sequence_length,
            boundary_ratio=float(config.boundary_ratio),
            guidance_scale_2=config.guidance_scale_2,
            num_train_timesteps=int(config.num_train_timesteps),
        )

    # ------------------------------------------------------------------
    # Weight-sync name mapping (vllm-omni cross-process compatibility)
    # ------------------------------------------------------------------

    def weight_sync_name_map(self) -> Dict[str, str]:
        """Return the prefix-substitution map for cross-process weight sync.

        Train-side parameter names under this bundle look like:

        - ``transformer.high_noise.<block_path>.<weight>``
        - ``transformer.low_noise.<block_path>.<weight>``

        After the standard ``weight_sync_param_name_prefix="transformer."``
        strip on the sync side, the keys become ``high_noise.<...>`` /
        ``low_noise.<...>``. But the vllm-omni-side WAN22 reference
        pipeline (``vllm-omni/.../pipeline_wan2_2.py``) loads the two
        transformers under ``transformer.*`` (high noise) and
        ``transformer_2.*`` (low noise). Without an additional
        substitution, the receiver can't match either bucket.

        This map declares the transform the sync layer should apply
        AFTER the prefix strip:

        ::

            "high_noise."  -> "transformer."
            "low_noise."   -> "transformer_2."

        **Current consumer status**: the weight-sync handlers do
        prefix-prepend only; they do NOT consume ``weight_sync_name_map``.
        So this method is a forward-looking API surfaced on the bundle for
        the eventual WAN22 separate-sampling (vllm-omni rollout) path. For
        trainside rollout (rollout = train Policy stack), no cross-process
        sync runs, so missing the substitution is harmless.

        When wiring the consumer side: the weight-sync handler gains an
        optional ``name_substitutions: Optional[Dict[str, str]]`` ctor
        field and the trainer's weight-sync setup reads
        ``getattr(self.bundle, "weight_sync_name_map", lambda: {})()``
        and passes it through to the sync config.
        """
        return {
            "high_noise.": "transformer.",
            "low_noise.": "transformer_2.",
        }


__all__ = ["WanDualTransformer", "WAN22Bundle"]
