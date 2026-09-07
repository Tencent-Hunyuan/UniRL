"""vllm-omni general plugin: flush RL captures after diffusion postprocess."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_unirl_capture_flush"


def register_capture_flush() -> None:
    """Flush pipeline captures into formatter metadata after postprocess sees raw media."""
    from vllm_omni.diffusion import diffusion_engine, output_formatter

    from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
        flush_captures_into_postprocess,
    )

    original = output_formatter.format_diffusion_outputs
    if getattr(original, _PATCH_FLAG, False):
        return

    def patched(*args, **kwargs):
        diffusion_output = kwargs.get("diffusion_output")
        postprocess_output = kwargs.get("postprocess_output")
        if diffusion_output is not None and postprocess_output is not None:
            kwargs["postprocess_output"] = flush_captures_into_postprocess(diffusion_output, postprocess_output)
        return original(*args, **kwargs)

    setattr(patched, _PATCH_FLAG, True)
    output_formatter.format_diffusion_outputs = patched
    # diffusion_engine took a from-import, so it holds its own binding.
    diffusion_engine.format_diffusion_outputs = patched
    logger.info("unirl: diffusion formatter flushes capture metadata after postprocess")
