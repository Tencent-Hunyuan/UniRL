"""QwenImagePipeline — ``Sample → Sample`` end-to-end for Qwen-Image.

Implements the four-tier flow::

    Texts ──text_embed──▶ QwenImageConditions ──diffuse──▶ LatentSegment
                                                              │
                                                              ▼
                                                          vae_decode
                                                              │
                                                              ▼
                                                            Images

Hydra constructs a pipeline via
``QwenImagePipeline.from_config(QwenImagePipelineConfig)`` (see
``config.py``); ``from_config`` loads the :class:`QwenImageBundle` then
constructs the four stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen Part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``; this
pipeline reads ``params.sigmas`` and uses it verbatim. The engine builds the
policy with :meth:`FlowMatchSchedulePolicy.from_pretrained(pretrained_path,
shift=pipeline.shift)`; Qwen-Image's
``scheduler/scheduler_config.json`` carries the dynamic-shift block, so
the policy enables dynamic μ derivation automatically. The pipeline
neither owns a σ builder nor reads model-specific scheduler config.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import QwenImageBundle
from .conditions import QwenImageConditions
from .config import QwenImagePipelineConfig
from .diffusion import (
    QwenImageDiffusionStage,
    QwenImageDiffusionStep,
)
from .text_embed import QwenImageTextEmbedStage
from .vae import QwenImageVAEDecodeStage


class QwenImagePipeline(Pipeline):
    """Qwen-Image generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier (last) Part is a pre-forked
    diffusion gen shell carrying ``DiffusionSamplingParams`` (with ``sigmas``
    pinned by the hosting engine). Reads the prompt via ``sample.conditioning()``
    and fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["image"]: Images`` — the decoded images.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay.
    Sample generation uses the model's default CFG negative; direct callers can
    pass explicit negatives through :meth:`build_conditions`.
    """

    def __init__(
        self,
        *,
        bundle: QwenImageBundle,
        text_embed: Optional[QwenImageTextEmbedStage] = None,
        diffusion: Optional[QwenImageDiffusionStage] = None,
        vae_decode: Optional[QwenImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
        batch_replay_steps: bool = False,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        if text_embed is None and bundle.text_encoder is not None:
            text_embed = QwenImageTextEmbedStage(bundle, max_sequence_length=max_sequence_length)
        self.text_embed = text_embed
        if diffusion is None:
            diffusion = QwenImageDiffusionStage(
                model=bundle,
                step=QwenImageDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                batch_replay_steps=batch_replay_steps,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else QwenImageVAEDecodeStage(bundle)
        self.shift = shift

    def build_schedule_policy(self):
        """Build the FlowMatchSchedulePolicy for this pipeline.

        Qwen-Image's transformer was trained with **dynamic shift**
        (per the upstream ``QwenImagePipeline`` reference): μ is
        derived per-request from ``image_seq_len`` via
        :func:`calculate_dynamic_mu`. Static shift would silently
        mis-shift σ and drift the GRPO ratio.

        On real-pod runs where ``pretrained_model_ckpt_path`` is a
        local mount, ``from_pretrained`` reads dynamic-shift fields
        from ``scheduler/scheduler_config.json`` automatically.

        On HF-Hub-repo-id paths (e.g. ``Qwen/Qwen-Image``) the path
        isn't local at policy build time. Without ``require_dynamic``,
        ``from_pretrained`` would fall back to ``static_only(shift)``
        — silently wrong. We pass ``require_dynamic=True`` so that path
        either uses the upstream-canonical ``dynamic_overrides`` below
        OR raises loudly.

        Canonical overrides live in
        :func:`unirl.models.qwen_image.config._qwen_image_dynamic_overrides`
        (mirrors ``Qwen/Qwen-Image``'s ``scheduler_config.json``, incl.
        ``shift_terminal: 0.02``) — one source for the trainside and
        rollout-engine paths.
        """
        from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(self.bundle, "pretrained_path", None),
            shift=float(self.shift),
            require_dynamic=True,
            dynamic_overrides=_qwen_image_dynamic_overrides(),
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. Mirrors
        :meth:`QwenImageDiffusionStage._latent_shape`'s arithmetic so the
        driver-provided noise tensor matches what the stage would
        otherwise generate internally.

        Qwen-Image canonical: ``AutoencoderKLQwenImage`` is 16-channel,
        ``QwenImageTransformer2DModel.in_channels=64`` (packed 16×4); the
        post-VAE latent grid is 8× downsampled with an extra patchify-2×2
        rounding (``latent_h = 2 * (H // 16)``).
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: QwenImagePipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "QwenImagePipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`FlowSDEStrategy`; callers running GRPO with a different
        SDE family (Dance / CPS / DPM2) pass an explicit strategy built
        from ``cfg.sampling.sde_strategy``.
        """
        bundle = QwenImageBundle.from_config(config)
        text_embed = (
            QwenImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
            if bundle.text_encoder is not None
            else None
        )
        step = QwenImageDiffusionStep()
        diffusion = QwenImageDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            batch_replay_steps=config.batch_replay_steps,
        )
        vae_decode = QwenImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> QwenImageConditions:
        """Encode prompts (+ optional CFG negatives) into ``QwenImageConditions``.

        CFG empty negative: Qwen-Image upstream (diffusers v0.37.1
        ``QwenImagePipeline`` docstring at ``pipeline_qwenimage.py:509``)
        recommends a single-space ``" "`` as the canonical empty
        negative. Empty string ``""`` is unsafe here: ``_get_qwen_prompt_embeds``
        wraps the prompt in a chat template and strips the first 34
        tokens of the encoder output (``prompt_template_encode_start_idx``)
        — an empty user-content slot can degenerate into a near-zero
        embedding and divide the norm-corrected CFG blend
        (diffusion.py:248-250) by ~0. Mirrors PR #104 OLD
        ``encode_inputs`` (``[" "] * batch_size``).

        Compare WAN21/WAN22 (which default to ``[""] * B``): WAN's
        text encoder doesn't run a 34-token prefix strip, so ``""``
        is safe there. The empty-negative value is a per-model
        property, not a framework knob — hence hardcoded per pipeline.
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"QwenImagePipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        if self.text_embed is None:
            raise RuntimeError(
                "QwenImagePipeline.build_conditions: no text_embed stage "
                "(load_text_encoder=False); trainside conditioning requires load_text_encoder=True."
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[" "] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return QwenImageConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen-Image t2i end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine (e.g. ``TrainsideRolloutEngine._ensure_sample_sigmas``)
        before the call; see the σ ownership note in ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"QwenImagePipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "QwenImagePipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenImagePipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        qwen_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(
            qwen_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        images = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=qwen_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["QwenImagePipeline"]
