#!/usr/bin/env python3
"""Trainside rollout+replay smoke for the Sample → Sample Qwen-Image pipeline (LIN-495).

Tier-1 representative for the diffusion-pipeline migration (the 7 SD3-shaped
pipelines: qwen_image / z_image / wan21 / wan22 / hunyuan_video / hunyuan_video15 /
ltx2 all follow the identical 6-step recipe and the same TrainsideRolloutEngine
path this exercises). Clone of ``trainside_sd3_smoke.py`` with the Qwen-Image
pipeline swapped in.

Loads the IN-PROCESS Qwen-Image pipeline, wraps it in a ``TrainsideRolloutEngine``,
builds a request ``Sample`` by hand, runs ``generate`` (rollout), reconstructs
the typed Qwen-Image conditions carried by the filled ``Part``, and runs the
diffusion stage's ``replay``. It asserts that the replayed SDE log-probs reproduce
the rollout's stored ``sde_logp`` (ratio ≈ 1). That shared-bundle invariant —
rollout and replay over the same weights and captured conditions agree — is the
correctness bar for the model bundle.

No external inference server (the trainside engine runs the pipeline's own stages
in-process), no training loop, no reward. Run on a GPU pod (1 free GPU), torch venv:

    PRETRAINED_MODEL=/root/unirl/models/local/Qwen-Image \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_qwen_image_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.algorithms.base import rollout_replay_logp_absdiff
from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.qwen_image.config import QwenImagePipelineConfig
from unirl.models.qwen_image.pipeline import QwenImagePipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

# Loose, pending pod calibration: a gross bug (wrong conditions / trajectory)
# drives |Δlogp| into the hundreds (the SDE logp sums over the latent), so this
# bound catches real breakage while tolerating bf16 forward drift. The actual
# value is logged so the threshold can be tightened once measured.
_ABSDIFF_MEAN_MAX = 0.5


def _log(msg: str) -> None:
    print(f"[trainside-qwen-image] {msg}", flush=True)


def build_request_sample() -> Sample:
    """2-prompt, 1-sample-each Qwen-Image request: ``[input, diffusion gen-shell]``.

    No ``sigmas`` on the params — the trainside engine pins σ via
    ``_ensure_sample_sigmas``. ``sde_indices`` records SDE log-probs so replay has
    something to reproduce. ``guidance_scale=1.0`` keeps CFG off (the migrated
    pipeline synthesizes the ``" "`` empty-negative only when guidance > 1).
    """
    prompts = ["a photo of a red apple on a wooden table", "an astronaut riding a horse on the moon"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)})
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,  # CFG off
        height=512,
        width=512,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        init_same_noise=False,
        sde_indices=[0, 1, 2],  # record SDE log-probs on the first 3 of 4 steps
    )
    return Sample.request(input_part).fork(1, sampling_params=diff_params)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL")
    if not model_path:
        _log("ERROR: set PRETRAINED_MODEL to a local Qwen-Image dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    _log(f"model_path={model_path}")

    config = QwenImagePipelineConfig(
        pretrained_model_ckpt_path=model_path,
        model_precision="bf16",
        shift=3.0,
        device="cuda:0",
    )
    try:
        _log("loading QwenImagePipeline.from_config (bundle on cuda:0) ...")
        pipeline = QwenImagePipeline.from_config(config)
        engine = TrainsideRolloutEngine(pipeline=pipeline)  # stage_attrs=("diffusion",)

        sample = build_request_sample()
        gen_in = sample.parts[-1]
        _log(f"request: {len(sample.parts)} parts; gen ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)

        # ---- rollout: the frontier gen Part is filled ----
        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert list(gen.sample_ids) == list(gen_in.sample_ids), "gen ids changed"
        assert isinstance(gen.segment, LatentSegment), f"segment must be LatentSegment; got {type(gen.segment)}"
        assert gen.segment.latents is not None, "LatentSegment.latents is None (no trajectory captured)"
        assert gen.segment.sde_logp is not None, "LatentSegment.sde_logp is None (no rollout logp to compare)"
        assert isinstance(gen.primitives.get("image"), Images) and len(gen.primitives["image"]) == 2, (
            "decoded Images (2) expected"
        )
        assert gen.conditions, "trainside path must carry replay conditions on the filled Part"
        assert gen.sampling_params.sigmas is not None, "engine must have pinned σ onto the gen params"
        _log(
            f"rollout PASS: latents={tuple(gen.segment.latents.shape)} "
            f"images={len(gen.primitives['image'])} conditions={sorted(gen.conditions)}"
        )

        # ---- replay: use the captured conditions, reproduce sde_logp (ratio ≈ 1) ----
        _log("reconstructing captured conditions and replaying the diffusion stage ...")
        params = gen.sampling_params
        conds = QwenImageConditions.from_dict(gen.conditions)
        model = pipeline.diffusion.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                result = pipeline.diffusion.replay(conds, segment=gen.segment, params=params)
        finally:
            model.train(was_training)

        new_logp = result.log_probs
        old_logp = gen.segment.sde_logp.to(device=new_logp.device, dtype=new_logp.dtype)
        assert new_logp.shape == old_logp.shape, (
            f"replay logp shape {tuple(new_logp.shape)} != rollout {tuple(old_logp.shape)}"
        )
        assert torch.isfinite(new_logp).all(), "replay produced non-finite log-probs"
        m = rollout_replay_logp_absdiff(new_logp, old_logp)
        mean, mx = m["rollout_replay_logp_absdiff_mean"], m["rollout_replay_logp_absdiff_max"]
        _log(f"ratio≈1 check: mean|Δlogp|={mean:.3e} max|Δlogp|={mx:.3e} (threshold mean<{_ABSDIFF_MEAN_MAX})")
        assert mean < _ABSDIFF_MEAN_MAX, f"rollout↔replay logp drift too large: mean|Δlogp|={mean:.3e}"

        _log("TRAINSIDE QWEN-IMAGE SMOKE PASSED ✅  (captured Sample conditions reproduce rollout sde_logp)")
        return 0
    except Exception:
        _log("TRAINSIDE QWEN-IMAGE SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
