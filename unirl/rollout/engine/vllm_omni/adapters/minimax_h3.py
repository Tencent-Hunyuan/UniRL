"""MiniMax-H3 text-to-video+audio adapter for vLLM-Omni rollout."""

from __future__ import annotations

from typing import Any, List

import torch

from unirl.config.require import require
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import pick_stage_output
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Audio, Audios, Video, Videos
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_video_segment

MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}


def _resolve_aspect_ratio(params: DiffusionSamplingParams) -> str:
    configured = dict(getattr(params, "sampler_kwargs", {}) or {}).get("aspect_ratio")
    if configured is not None:
        value = str(configured)
        require(
            value in MINIMAX_H3_ASPECT_RATIOS,
            f"MiniMax-H3 aspect_ratio must be one of {tuple(MINIMAX_H3_ASPECT_RATIOS)}, got {value!r}",
        )
        return value

    height = int(params.height)
    width = int(params.width)
    require(height > 0 and width > 0, f"MiniMax-H3 height and width must be positive, got {height}x{width}")
    ratio = width / height
    for name, expected in MINIMAX_H3_ASPECT_RATIOS.items():
        if abs(ratio - expected) <= 1e-6:
            return name
    raise ValueError(
        "MiniMax-H3 dimensions must match an official aspect ratio "
        f"{tuple(MINIMAX_H3_ASPECT_RATIOS)}; got width={width}, height={height}"
    )


class MiniMaxH3InputAdapter(DitInputAdapter):
    """Add H3's frame count, dual shifts, and sigma-point convention."""

    def __init__(self, modality: str, *, video_shift: float, audio_shift: float) -> None:
        super().__init__(modality)
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        sampling = super().build_sampling(sample)
        params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        kwargs = sampling[0].kwargs
        kwargs["num_frames"] = int(params.num_frames)
        # Upstream H3 interprets this value as sigma points; UniRL counts transitions.
        kwargs["num_inference_steps"] = int(params.num_inference_steps) + 1
        kwargs.pop("sigmas", None)
        extra = dict(kwargs.get("extra_args") or {})
        extra.update(
            task="t2va",
            aspect_ratio=_resolve_aspect_ratio(params),
            flow_shift=self.video_shift,
            audio_flow_shift=self.audio_shift,
        )
        kwargs["extra_args"] = extra
        return sampling


class MiniMaxH3OutputAdapter:
    """Recover sparse joint trajectories and one reward frame per request."""

    final_output_type = "video"
    stage_id = 0

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("MiniMax-H3 rollout returned no request outputs")

        outputs = []
        for request_outputs in per_request:
            output = pick_stage_output(
                request_outputs,
                final_output_type=self.final_output_type,
                stage_id=self.stage_id,
            )
            if output is None:
                raise RuntimeError("MiniMax-H3 rollout request has no video-stage output")
            outputs.append(output)

        payloads = []
        schedules = []
        for output in outputs:
            raw_payload = getattr(output, "trajectory_latents", None)
            raw_schedule = getattr(output, "trajectory_timesteps", None)
            if isinstance(raw_payload, dict):
                payload = raw_payload
            else:
                # Compatibility with vLLM-Omni revisions that carried model
                # extras in DiffusionOutput.custom_output.
                custom = getattr(output, "custom_output", None) or {}
                payload = {
                    "video": raw_payload,
                    "audio": custom.get("minimax_h3_audio_trajectory"),
                    "indices": custom.get("minimax_h3_trajectory_indices"),
                    "text_embeddings": custom.get("minimax_h3_text_embeddings"),
                    "reward_frame": custom.get("minimax_h3_reward_frame"),
                }
            if isinstance(raw_schedule, dict):
                schedule = raw_schedule
            else:
                custom = getattr(output, "custom_output", None) or {}
                schedule = {
                    "video": raw_schedule,
                    "audio": custom.get("minimax_h3_audio_sigmas"),
                    "sde_indices": custom.get("sde_step_indices", []),
                }
            payloads.append(payload)
            schedules.append(schedule)

        required = ("video", "audio", "indices", "text_embeddings", "reward_frame")
        for index, payload in enumerate(payloads):
            missing = [key for key in required if payload.get(key) is None]
            if missing:
                raise RuntimeError(f"MiniMax-H3 rollout output {index} is missing RL captures: {missing}")

        latents = torch.cat([payload["video"] for payload in payloads], dim=0)
        aux_latents = torch.cat([payload["audio"] for payload in payloads], dim=0)
        text_embeddings = torch.cat([payload["text_embeddings"] for payload in payloads], dim=0)

        sigmas = schedules[0].get("video")
        expected = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas
        verify_engine_used_sigmas(sigmas, expected=expected, engine_name="vllm-omni-minimax-h3")
        indices = torch.as_tensor(payloads[0]["indices"], dtype=torch.long)
        sde_indices = torch.as_tensor(schedules[0].get("sde_indices", []), dtype=torch.long)

        segment = make_video_segment(
            latents=latents,
            aux_latents=aux_latents,
            sigmas=sigmas,
            indices=indices,
            sde_indices=sde_indices,
            sde_logp=None,
            initial_latents=latents[:, 0],
        )

        videos = []
        for payload in payloads:
            frame = payload["reward_frame"]
            require(
                torch.is_tensor(frame) and frame.ndim == 5 and int(frame.shape[0]) == 1,
                f"MiniMax-H3 reward frame must be [1,C,1,H,W], got {getattr(frame, 'shape', None)}",
            )
            frames = frame[0].permute(1, 0, 2, 3).to(torch.float32).div_(255.0)
            videos.append(Video(frames=frames))

        # Audio is represented by a one-sample placeholder; replay uses aux_latents.
        audios = Audios.from_list([Audio(waveform=torch.zeros(1, 2)) for _ in payloads])
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"video": Videos.from_list(videos), "audio": audios},
                primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
                conditions={"text": TextEmbedCondition(embeds=text_embeddings)},
            )
        )


@register_adapter("minimax_h3_t2va")
class MiniMaxH3T2VAAdapter(ModelAdapter):
    """MiniMax-H3 t2va, one HSDP4+UP4 engine per external DP replica."""

    needs_driver_tokenizer = False
    # The grouped engine broadcasts one adapter to four subprocesses. Byte-copy
    # avoids reusing a one-shot CUDA IPC file descriptor on ranks 2..4.
    lora_copy_transport = True

    def boot_kwargs(self) -> dict[str, Any]:
        """Use the current vLLM-Omni direct diffusion-stage configuration."""
        require(
            not self.cfg.stage_yaml_override,
            "minimax_h3_t2va uses current direct stage arguments; stage_yaml_override is not supported",
        )
        world_size = int(self.cfg.replica_size)
        return {
            "use_stage_yaml": False,
            "needs_driver_tokenizer": False,
            "clear_cuda_visible": False,
            "omni_kwargs": {
                "model_class_name": "MiniMaxH3Pipeline",
                "task_type": "t2va",
                "num_gpus": world_size,
                "max_num_seqs": 1,
                "diffusion_batch_size": 1,
                "distributed_executor_backend": "mp",
                "enforce_eager": True,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "cfg_parallel_size": 1,
                "allgather_degree": 1,
                "sequence_parallel_size": world_size,
                "ulysses_degree": world_size,
                "ring_degree": 1,
                "text_encoder_tp_size": world_size,
                "vae_patch_parallel_size": world_size,
                "vae_parallel_mode": "tile",
                "vae_use_tiling": True,
                "use_hsdp": True,
                "hsdp_shard_size": world_size,
                "hsdp_replicate_size": 1,
                "custom_pipeline_args": {
                    "pipeline_class": ("unirl.rollout.engine.vllm_omni.pipelines.minimax_h3.MiniMaxH3RLPipeline"),
                },
                "worker_extension_cls": ("unirl.rollout.engine.vllm_omni.worker.dit_extension.DiTWeightSyncExtension"),
            },
        }

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = MiniMaxH3InputAdapter(
            self.modality,
            video_shift=float(model_config.video_shift),
            audio_shift=float(model_config.audio_shift),
        )
        self.output_adapter = MiniMaxH3OutputAdapter(self.modality)

    def validate(self) -> None:
        require(self.model_config is not None, "MiniMaxH3T2VAAdapter requires model_config")
        require(hasattr(self.model_config, "video_shift"), "MiniMaxH3T2VAAdapter requires model_config.video_shift")
        require(hasattr(self.model_config, "audio_shift"), "MiniMaxH3T2VAAdapter requires model_config.audio_shift")
        require(
            int(self.cfg.replica_size) == 4 and int(self.cfg.tp_size) == 4,
            "MiniMaxH3T2VAAdapter currently qualifies only replica_size=tp_size=4; "
            f"got replica_size={self.cfg.replica_size}, tp_size={self.cfg.tp_size}",
        )

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        return FlowMatchSchedulePolicy.static_only(float(self.model_config.video_shift))

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError("minimax_h3_t2va rejects image-bearing requests")

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = [
    "MiniMaxH3InputAdapter",
    "MiniMaxH3OutputAdapter",
    "MiniMaxH3T2VAAdapter",
]
