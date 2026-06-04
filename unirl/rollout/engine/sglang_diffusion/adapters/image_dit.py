"""``ImageDiTAdapter`` — per-output-shape base adapter holding the conversion logic.

Output shape = a 5-D image-form latent trajectory ``[B, T+1, C, H, W]`` decoded to
``Images``. Holds ``build_inputs`` / ``build_response`` once and exposes the per-model
variation points as overridable methods (``unpack_trajectory``, ``build_segment``,
``build_decoded``, ``build_condition``) and class knobs (``track_name``,
``segment_factory``). Concrete adapters override only what differs.

Ported from the old engine's ``request.py`` / ``response.py`` free functions, with
the model-family branches lifted into overridable methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import ModelAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_diffusion_params
from unirl.types.segments.latent import make_image_segment


class ImageDiTAdapter(ModelAdapter):
    """Conversion for image DiT families (SD3, FLUX, …). Default path end-to-end."""

    #: RolloutResp track key the segment/decoded/conditions are stored under.
    track_name: str = "image"
    #: Segment factory (modality). A video adapter would pass ``make_video_segment``.
    segment_factory = staticmethod(make_image_segment)

    # ------------------------------------------------------------------ #
    # Request side
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq, *, initial_noise: Any) -> Dict[str, Any]:
        text_prim = req.primitives.get("text")
        if not isinstance(text_prim, Texts):
            raise TypeError(
                f"build_inputs: req.primitives['text'] must be Texts; got "
                f"{type(text_prim).__name__ if text_prim is not None else 'None'}"
            )
        prompts = list(text_prim.texts)
        require(bool(prompts), "build_inputs: req.primitives['text'] must be non-empty")
        require(
            len(prompts) == len(req.sample_ids),
            f"build_inputs: text count {len(prompts)} != sample_ids count {len(req.sample_ids)}",
        )

        diffusion = get_diffusion_params(req.sampling_params)
        require(
            diffusion is not None,
            "build_inputs: req.sampling_params must contain diffusion params",
        )

        num_inference_steps = int(diffusion.num_inference_steps)
        guidance_scale = float(diffusion.guidance_scale)
        height = int(diffusion.height)
        width = int(diffusion.width)
        eta = float(diffusion.eta)
        sde_indices_raw = diffusion.sde_indices
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None

        # σ is the SSOT on RolloutReq (pinned by the engine via ensure_req_sigmas
        # before this runs). Never recompute here. req.sigmas is length T+1
        # (terminal 0 included); SGLang's set_timesteps wants the interior T.
        require(
            req.sigmas is not None,
            "build_inputs: req.sigmas must be set by the engine before conversion "
            "(see unirl.sde.runtime.ensure_req_sigmas).",
        )
        require(
            int(req.sigmas.shape[0]) == num_inference_steps + 1,
            f"build_inputs: req.sigmas length {int(req.sigmas.shape[0])} != "
            f"num_inference_steps+1 ({num_inference_steps + 1}).",
        )
        sigmas = req.sigmas.detach().cpu().tolist()[:-1]

        unique_prompts, k = utils.deexpand_prompts_from_groups(prompts, list(req.group_ids))
        prompt_payload: Any = unique_prompts if len(unique_prompts) > 1 else unique_prompts[0]
        num_outputs_per_prompt: Optional[int] = k if k > 1 else None

        sampler_kwargs: Dict[str, Any] = dict(diffusion.sampler_kwargs or {})

        # Negative-prompt CFG invariant: SGLang gates CFG on guidance_scale>1
        # independently of return_negative_prompt_embeds — without pinning the
        # latter, rollout conditions on the negative prompt while replay falls
        # back to zero negative embeds (silent GRPO ratio mismatch). Fail fast.
        neg_prompt = sampler_kwargs.get("negative_prompt")
        return_neg_embeds = bool(sampler_kwargs.get("return_negative_prompt_embeds", False))
        require(
            neg_prompt is None or return_neg_embeds,
            "build_inputs: sampler_kwargs.negative_prompt is set but "
            "return_negative_prompt_embeds is not True — rollout would condition on "
            "the negative prompt while replay uses zero negative embeds (silent GRPO "
            "ratio mismatch). Set return_negative_prompt_embeds=True.",
        )

        # Layer 1: caller escape-hatch (lowest priority).
        kwargs: Dict[str, Any] = dict(sampler_kwargs)
        # Layers 2 + 3: typed/computed + engine pins (override layer 1).
        kwargs.update(
            {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "num_frames": int(diffusion.num_frames),
                "sigmas": sigmas,
                "prompt": prompt_payload,
                "init_same_noise": False,
                "seed": int(diffusion.seed) if diffusion.seed is not None else 0,
                "save_output": False,
                "return_file_paths_only": False,
                "return_trajectory_latents": True,
                "return_trajectory_decoded": False,
                "return_prompt_embeds": True,
            }
        )

        if initial_noise is not None:
            kwargs["initial_noise"] = initial_noise
        if req.group_ids:
            kwargs["noise_group_ids"] = [str(gid) for gid in req.group_ids]

        # Layer 4: SDE-kernel kwargs only when the algorithm requested SDE noise.
        if sde_indices is not None:
            require(
                self._sde_label is not None,
                "build_inputs: SDE mode requires an sde_label (resolved from the strategy)",
            )
            kwargs["rollout"] = True
            kwargs["rollout_sde_type"] = self._sde_label
            kwargs["rollout_noise_level"] = eta
            kwargs["rollout_sde_indices"] = sde_indices

        if num_outputs_per_prompt is not None:
            kwargs["num_outputs_per_prompt"] = num_outputs_per_prompt

        return kwargs

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, raw: List[RawResult]) -> RolloutResp:
        require(bool(raw), "build_response: SGLang returned no results")
        require(req.sigmas is not None, "build_response: req.sigmas must be set")

        diffusion = get_diffusion_params(req.sampling_params)
        num_steps = int(diffusion.num_inference_steps)
        sde_indices_raw = diffusion.sde_indices
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None
        use_native_logprob = self.cfg.logprob_source == "native" and sde_indices is not None

        segment = self.build_segment(
            req,
            raw,
            num_steps=num_steps,
            sde_indices=sde_indices,
            use_native_logprob=use_native_logprob,
        )
        decoded = self.build_decoded(raw)

        conditions: Dict[str, Any] = {}
        if self.cfg.populate_conditions:
            conditions = self.build_condition(raw)

        return RolloutResp(
            tracks={
                self.track_name: RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conditions,
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

    # ------------------------------------------------------------------ #
    # Overridable conversion steps (defaults delegate to utils)
    # ------------------------------------------------------------------ #

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        use_native_logprob: bool,
    ):
        latents = []
        for r in results:
            require(r.trajectory_latents is not None, "SGLang result missing trajectory_latents")
            latents.append(r.trajectory_latents.detach().cpu())
        traj = self.unpack_trajectory(torch.cat(latents, dim=0), req)
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            use_native_logprob=use_native_logprob,
            segment_factory=self.segment_factory,
        )

    def unpack_trajectory(self, traj, req: RolloutReq):
        """Image-form families keep latents 5-D throughout; reject other ranks.

        Only ``flux2_klein`` emits + unpacks the packed 4-D ``[B, T, H*W, C]``
        shape, so it overrides this.
        """
        if traj.ndim == 5:
            return traj
        raise ValueError(
            f"{self.model_family}: expected a 5-D image-form trajectory "
            f"[B, T+1, C, H, W]; got rank {traj.ndim}, shape {tuple(traj.shape)}. "
            f"Only flux2_klein handles the packed 4-D shape."
        )

    def build_decoded(self, results: List[RawResult]):
        return utils.stack_decoded_images(results)

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        text_cond, neg_text_cond = utils.fuse_text_conditions(results)
        out: Dict[str, Any] = {}
        if text_cond is not None:
            out["text"] = text_cond
        if neg_text_cond is not None:
            out["negative_text"] = neg_text_cond
        return out


__all__ = ["ImageDiTAdapter"]
