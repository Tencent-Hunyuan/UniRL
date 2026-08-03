"""Wan2.1 adapter for FastVideo rollout."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.fastvideo.adapters.base import FastVideoModelAdapter, register_adapter
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Video, Videos
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import make_video_segment


@register_adapter("wan2.1", "wan21")
class Wan21FastVideoAdapter(FastVideoModelAdapter):
    """Isolate every Wan schedule/shape/condition assumption from the engine."""

    def validate(self) -> None:
        super().validate()
        require(hasattr(self.model_config, "shift"), "Wan2.1 FastVideo adapter requires model_config.shift")

    def align_runtime_args(self, fastvideo_args: Any) -> None:
        target_shift = float(self.model_config.shift)
        pipeline_config = fastvideo_args.pipeline_config
        pipeline_config.flow_shift = target_shift
        # Consumed by the request-local transformer wrapper in patch_denoising.
        fastvideo_args._unirl_timestep_dtype = "long"
        fastvideo_args._unirl_cfg_combine_dtype = "float32"
        fastvideo_args._unirl_custom_sigmas_dtype = "float32"

    def build_forward_batch(
        self,
        *,
        prompt: str,
        seed: int,
        params: Any,
        sigmas: torch.Tensor,
        fastvideo_args: Any,
    ) -> Any:
        from fastvideo.configs.sample.base import SamplingParam
        from fastvideo.pipelines import ForwardBatch
        from fastvideo.utils import shallow_asdict

        sampling = SamplingParam()
        sampling.prompt = prompt
        sampling.height = int(params.height)
        sampling.width = int(params.width)
        sampling.num_frames = int(params.num_frames)
        sampling.num_inference_steps = int(params.num_inference_steps)
        sampling.guidance_scale = float(params.guidance_scale)
        sampling.seed = int(seed)
        sampling.num_videos_per_prompt = 1
        sampling.save_video = False
        sampling.return_frames = False
        sampling.return_trajectory_latents = True
        sampling.return_trajectory_decoded = False

        # FastVideo's scheduler re-applies flow_shift to custom sigmas. Supply
        # the Wan FlowMatch inverse so the actual worker schedule equals the
        # already-shifted UniRL schedule; verify_engine_used_sigmas checks the
        # round trip on every result.
        flow_shift = float(fastvideo_args.pipeline_config.flow_shift)
        schedule = sigmas.detach().cpu().double()
        denominator = flow_shift - schedule * (flow_shift - 1.0)
        require(bool(torch.all(denominator != 0)), "Wan2.1 FastVideo sigma pre-image has a zero denominator")
        preimage = schedule / denominator
        # Keep the public ForwardBatch contract as a list. The Wan-scoped
        # timestep patch converts it after FastVideo input validation and
        # immediately before the upstream scheduler call.
        sampling.sigmas = preimage.tolist()[:-1]

        raw_indices = getattr(params, "sde_indices", None)
        sde_indices = [int(index) for index in raw_indices] if raw_indices else None
        latent_frames = (sampling.num_frames - 1) // 4 + 1
        n_tokens = latent_frames * (sampling.height // 8) * (sampling.width // 8)
        sampling_dict = shallow_asdict(deepcopy(sampling))
        sampling_dict.pop("eta", None)
        return ForwardBatch(
            **sampling_dict,
            eta=float(params.eta),
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
            rl_data=ForwardBatch.RLData(
                enabled=True,
                collect_log_probs=bool(self.cfg.native_logprob),
                store_trajectory=True,
                keep_trajectory_on_cpu=True,
                sde_step_indices=sde_indices,
                sde_type=str(getattr(self.strategy, "canonical_name", "flow")),
            ),
        )

    def collect_output(self, output: Any) -> Dict[str, Any]:
        rl_data = output.rl_data
        trajectory = getattr(rl_data, "trajectory_latents", None) if rl_data is not None else None
        if trajectory is None:
            trajectory = output.trajectory_latents
        require(torch.is_tensor(trajectory), "FastVideo returned no trajectory tensor")
        if trajectory.dim() == 5:
            trajectory = trajectory.unsqueeze(0)

        decoded = getattr(output, "output", None)
        require(torch.is_tensor(decoded), "FastVideo returned no decoded output")
        if decoded.dim() == 4:
            decoded = decoded.unsqueeze(0)

        prompt_embeds = output.prompt_embeds
        require(
            isinstance(prompt_embeds, (list, tuple)) and len(prompt_embeds) > 0 and torch.is_tensor(prompt_embeds[0]),
            "FastVideo returned no Wan UMT5 prompt embedding",
        )
        text_embed = prompt_embeds[0]
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(0)

        prompt_mask = output.prompt_attention_mask
        text_mask = (
            prompt_mask[0]
            if isinstance(prompt_mask, (list, tuple)) and prompt_mask and torch.is_tensor(prompt_mask[0])
            else None
        )

        negative_embed = None
        negative_mask = None
        raw_negative = output.negative_prompt_embeds
        if isinstance(raw_negative, (list, tuple)) and raw_negative and torch.is_tensor(raw_negative[0]):
            negative_embed = raw_negative[0]
            if negative_embed.dim() == 2:
                negative_embed = negative_embed.unsqueeze(0)
            raw_negative_mask = output.negative_attention_mask
            if (
                isinstance(raw_negative_mask, (list, tuple))
                and raw_negative_mask
                and torch.is_tensor(raw_negative_mask[0])
            ):
                negative_mask = raw_negative_mask[0]

        actual_timesteps = getattr(rl_data, "trajectory_timesteps", None) if rl_data is not None else None
        if actual_timesteps is None:
            actual_timesteps = output.trajectory_timesteps
        log_probs = getattr(rl_data, "log_probs", None) if rl_data is not None else None
        return {
            "trajectory": trajectory.detach().cpu(),
            "decoded": decoded.detach().cpu().float(),
            "actual_timesteps": None if actual_timesteps is None else actual_timesteps.detach().cpu(),
            "log_probs": None if log_probs is None else log_probs.detach().cpu(),
            "text_embed": text_embed.detach().cpu().float(),
            "text_mask": None if text_mask is None else text_mask.detach().cpu(),
            "negative_embed": None if negative_embed is None else negative_embed.detach().cpu().float(),
            "negative_mask": None if negative_mask is None else negative_mask.detach().cpu(),
        }

    @staticmethod
    def _condition(
        samples: List[Dict[str, Any]],
        *,
        embed_key: str,
        mask_key: str,
    ) -> Optional[TextEmbedCondition]:
        if any(sample[embed_key] is None for sample in samples):
            return None
        return TextEmbedCondition.concat(
            [
                TextEmbedCondition(
                    embeds=sample[embed_key],
                    pooled=None,
                    attn_mask=sample[mask_key],
                )
                for sample in samples
            ]
        )

    def build_response(self, req: Any, params: Any, samples: List[Dict[str, Any]]) -> RolloutResp:
        require(len(samples) == int(req.batch_size), "FastVideo returned the wrong number of Wan samples")
        expected_sigmas = req.sigmas.detach().cpu()
        for sample in samples:
            actual = sample["actual_timesteps"]
            if torch.is_tensor(actual) and actual.numel() + 1 == expected_sigmas.numel():
                actual = torch.cat([actual.reshape(-1), actual.new_zeros(1)])
            verify_engine_used_sigmas(actual, expected=expected_sigmas, engine_name="fastvideo/wan2.1")

        trajectory = torch.cat([sample["trajectory"] for sample in samples], dim=0)
        decoded = torch.cat([sample["decoded"] for sample in samples], dim=0)
        transitions = int(trajectory.shape[1]) - 1
        sde_steps = sorted(int(index) for index in (getattr(params, "sde_indices", None) or []))
        if not sde_steps:
            sde_steps = list(range(transitions))
        sde_indices = torch.tensor(sde_steps, dtype=torch.long)

        log_prob_parts = [sample["log_probs"] for sample in samples]
        sde_logp = None
        if all(torch.is_tensor(part) for part in log_prob_parts):
            log_probs = torch.cat(log_prob_parts, dim=0)
            if log_probs.shape[1] == transitions and len(sde_steps) < transitions:
                log_probs = log_probs[:, sde_steps]
            require(
                log_probs.shape[1] == len(sde_steps),
                f"FastVideo native log-prob columns {log_probs.shape[1]} do not match SDE steps {len(sde_steps)}",
            )
            sde_logp = log_probs.contiguous()

        segment = make_video_segment(
            latents=trajectory,
            sigmas=req.sigmas,
            indices=torch.arange(trajectory.shape[1], dtype=torch.long),
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )
        text = self._condition(samples, embed_key="text_embed", mask_key="text_mask")
        require(text is not None, "FastVideo returned no Wan text conditions")
        conditions: Dict[str, Any] = {"text": text}
        negative = self._condition(samples, embed_key="negative_embed", mask_key="negative_mask")
        if negative is not None:
            conditions["negative_text"] = negative

        videos = Videos.from_list(
            [Video(frames=decoded[index].permute(1, 0, 2, 3).contiguous()) for index in range(int(decoded.shape[0]))]
        )
        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conditions,
                    segment=segment,
                    decoded=videos,
                )
            }
        )


__all__ = ["Wan21FastVideoAdapter"]
