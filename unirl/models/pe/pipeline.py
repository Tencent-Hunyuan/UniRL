"""PEPipeline — ``Sample → Sample`` end-to-end for Prompt Enhancement."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

from .bundle import PEBundle
from .instruction import postprocess_pe_texts

logger = logging.getLogger(__name__)


class PEPipeline(Pipeline):
    """PE generate pipeline: ``Sample → Sample``."""

    def __init__(
        self,
        *,
        diffusion_pipeline: Pipeline,
        llm_pipeline: Pipeline,
        pe_instruction: Optional[str] = None,
        pe_marker: Optional[str] = None,
        pe_max_chars: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.diffusion_pipeline = diffusion_pipeline
        self.llm_pipeline = llm_pipeline
        self.pe_instruction = pe_instruction
        self.pe_marker = pe_marker
        self.pe_max_chars = pe_max_chars
        self.bundle = PEBundle(
            diffusion=diffusion_pipeline.bundle,
            llm=llm_pipeline.bundle,
        )

    @property
    def diffusion(self):
        """The trainable diffusion stage (delegates to the diffusion child)."""
        return self.diffusion_pipeline.diffusion

    @property
    def ar(self):
        """The trainable AR stage (delegates to the LLM child)."""
        return self.llm_pipeline.ar

    def build_schedule_policy(self):
        """σ-schedule policy for the diffusion track (delegates to the diffusion child)."""
        diff = self.diffusion_pipeline
        builder = getattr(diff, "build_schedule_policy", None)
        if callable(builder):
            return builder()
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(diff.bundle, "pretrained_path", None),
            shift=float(diff.shift),
        )

    def generate(self, sample: Sample) -> Sample:
        """Run the PE serial flow over the pre-forked request ``Sample``."""
        input_part, ar_shell, diff_shell = self._unpack_request(sample)

        P = len(input_part.sample_ids)
        if P == 0:
            raise ValueError("PEPipeline.generate: empty Sample")
        prompts = input_part.primitives.get("text")
        if not isinstance(prompts, Texts):
            raise TypeError(
                f"PEPipeline.generate: input Part.primitives['text'] must be Texts; "
                f"got {type(prompts).__name__ if prompts is not None else 'None'}"
            )
        if len(ar_shell.sample_ids) % P != 0:
            raise ValueError(f"PEPipeline.generate: ar shell {len(ar_shell.sample_ids)} not a multiple of P={P}")
        N = len(ar_shell.sample_ids) // P
        if N < 1 or len(diff_shell.sample_ids) % len(ar_shell.sample_ids) != 0:
            raise ValueError(
                f"PEPipeline.generate: diffusion shell {len(diff_shell.sample_ids)} not a multiple of P*N={P * N}"
            )
        M = len(diff_shell.sample_ids) // len(ar_shell.sample_ids)
        if M < 1:
            raise ValueError(f"PEPipeline.generate: diffusion branch M={M} must be >= 1")

        ar_input = Part.input(
            sample_ids=list(input_part.sample_ids),
            primitives={"text": prompts},
            control=self._ar_control(input_part.control or {}),
        )
        ar_out = self.llm_pipeline.generate(Sample(parts=[ar_input, ar_shell]))
        ar_part = ar_out.parts[-1]
        if len(ar_part.sample_ids) != P * N:
            raise RuntimeError(
                f"PEPipeline.generate: LLM child returned {len(ar_part.sample_ids)} samples; expected P*N={P * N}"
            )
        rewritten = ar_part.primitives.get("text")
        if not isinstance(rewritten, Texts) or len(rewritten.texts) != P * N:
            raise RuntimeError(
                "PEPipeline.generate: LLM child must emit Texts in ar Part primitives['text'] "
                f"(expected {P * N}, got {len(rewritten.texts) if isinstance(rewritten, Texts) else 'None'})"
            )

        rewritten = self._extract_pe(rewritten, prompts, N)
        ar_part = ar_part.fill(primitives={"text": rewritten})

        pe_input = Part.input(sample_ids=[f"pe{k}" for k in range(P * N)], primitives={"text": rewritten})
        diff_child_shell = pe_input.fork(
            M,
            sampling_params=diff_shell.sampling_params,
            new_segment=diff_shell.segment,
        )
        diff_out = self.diffusion_pipeline.generate(Sample(parts=[pe_input, diff_child_shell]))
        diff_child = diff_out.parts[-1]
        if len(diff_child.sample_ids) != len(diff_shell.sample_ids):
            raise RuntimeError(
                f"PEPipeline.generate: diffusion child returned {len(diff_child.sample_ids)} "
                f"samples; expected P*N*M={P * N * M}"
            )
        diffusion_part = diff_shell.fill(
            segment=diff_child.segment,
            primitives=dict(diff_child.primitives),
            primitive_metadata=dict(diff_child.primitive_metadata),
            conditions=dict(diff_child.conditions),
            media_preview=diff_child.media_preview,
        )

        return Sample(
            parts=[input_part, ar_part, diffusion_part],
            reward_compute_s=sample.reward_compute_s,
        )

    @staticmethod
    def _unpack_request(sample: Sample) -> tuple:
        """Resolve the pre-forked ``[input, ar_shell, diff_shell]`` request."""
        if not sample.parts or not sample.parts[0].is_root:
            raise ValueError("PEPipeline.generate: requires a root input Part at parts[0]")
        if len(sample.parts) != 3:
            raise ValueError(
                "PEPipeline.generate: requires exactly [input, ar_shell, diffusion_shell]; "
                f"got {len(sample.parts)} Parts."
            )
        input_part, ar_shell, diff_shell = sample.parts
        if not isinstance(ar_shell.sampling_params, ARSamplingParams):
            raise ValueError("PEPipeline.generate: requires an AR gen-shell Part (ARSamplingParams)")
        if not isinstance(diff_shell.sampling_params, DiffusionSamplingParams):
            raise ValueError("PEPipeline.generate: requires a diffusion gen-shell Part (DiffusionSamplingParams)")
        return input_part, ar_shell, diff_shell

    def _ar_control(self, control: Dict[str, Any]) -> Dict[str, Any]:
        """The LLM child input Part's ``control``: the parent's chat + ar subsets with pe_instruction injected."""
        ar_control: Dict[str, Any] = {key: dict(control[key]) for key in ("chat", "ar") if key in control}
        if self.pe_instruction:
            for key in ("ar", "chat"):
                ar_control.setdefault(key, {})["system_instruction"] = self.pe_instruction
        return ar_control

    def _extract_pe(self, pe_texts: Texts, user_prompts: Texts, samples_per_prompt: int) -> Texts:
        """Optional marker-based PE extraction; off-format outputs fall back to the original user prompt."""
        if not self.pe_marker:
            return pe_texts
        cleaned_texts, stats = postprocess_pe_texts(
            pe_texts.texts,
            user_prompts=user_prompts.texts,
            samples_per_prompt=samples_per_prompt,
            marker=self.pe_marker,
            max_chars=self.pe_max_chars,
        )
        if any(stats.values()):
            logger.info(
                "PEPipeline: PE-extract — marker=%r, %d/%d empty, %d truncated, %d fallback_to_original",
                self.pe_marker,
                stats["empty"],
                len(pe_texts.texts),
                stats["truncated"],
                stats["fallback"],
            )
        return Texts(texts=cleaned_texts)


__all__ = ["PEPipeline"]
