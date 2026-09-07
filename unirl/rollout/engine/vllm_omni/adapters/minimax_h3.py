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
from unirl.sde.runtime import FlowMatchSchedulePolicy, get_sigma_schedule
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

    def __init__(
        self,
        modality: str,
        *,
        video_shift: float,
        audio_shift: float,
        audio_joint_sde: bool,
    ) -> None:
        super().__init__(modality)
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        self.audio_joint_sde = bool(audio_joint_sde)

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        sampling = super().build_sampling(sample)
        params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        kwargs = sampling[0].kwargs
        kwargs["num_frames"] = int(params.num_frames)
        # Upstream H3 interprets this value as sigma points; UniRL counts transitions.
        kwargs["num_inference_steps"] = int(params.num_inference_steps) + 1
        kwargs.pop("sigmas", None)
        extra = dict(kwargs.get("extra_args") or {})
        sampler_kwargs = dict(getattr(params, "sampler_kwargs", {}) or {})
        extra.update(
            task="t2va",
            aspect_ratio=_resolve_aspect_ratio(params),
            flow_shift=self.video_shift,
            audio_flow_shift=self.audio_shift,
            audio_joint_sde=self.audio_joint_sde,
            capture_transition_means=bool(sampler_kwargs.get("capture_transition_means", False)),
        )
        kwargs["extra_args"] = extra
        return sampling


class MiniMaxH3OutputAdapter:
    """Recover sparse joint trajectories and one reward frame per request."""

    final_output_type = "video"
    stage_id = 0

    def __init__(self, modality: str, *, audio_shift: float) -> None:
        self.modality = modality
        self.audio_shift = float(audio_shift)

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
        rollout_log_probs = []
        for output in outputs:
            raw_payload = getattr(output, "trajectory_latents", None)
            raw_schedule = getattr(output, "trajectory_timesteps", None)
            raw_log_probs = getattr(output, "trajectory_log_probs", None)
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
                    "video_means": custom.get("minimax_h3_video_transition_means"),
                    "reward_video": (
                        custom.get("minimax_h3_reward_video")
                        if custom.get("minimax_h3_reward_video") is not None
                        else custom.get("minimax_h3_reward_frame")
                    ),
                    "reward_audio": custom.get("minimax_h3_reward_audio"),
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
            rollout_log_probs.append(raw_log_probs)

        replay_fields = ("video", "audio", "indices", "text_embeddings")
        replay_ready = [all(payload.get(key) is not None for key in replay_fields) for payload in payloads]
        for index, payload in enumerate(payloads):
            if payload.get("reward_video") is None or payload.get("reward_audio") is None:
                raise RuntimeError(f"MiniMax-H3 rollout output {index} is missing reward video/audio")
        if any(replay_ready) != all(replay_ready):
            raise RuntimeError("MiniMax-H3 rollout mixed replay and evaluation-only outputs")
        if not all(replay_ready):
            videos = []
            for payload in payloads:
                frame = payload["reward_video"]
                require(
                    torch.is_tensor(frame)
                    and frame.ndim == 5
                    and int(frame.shape[0]) == 1
                    and int(frame.shape[2]) >= 1,
                    f"MiniMax-H3 reward video must be [1,C,T,H,W], got {getattr(frame, 'shape', None)}",
                )
                frames = frame[0].permute(1, 0, 2, 3).to(torch.float32).div_(255.0)
                videos.append(Video(frames=frames))
            audios = Audios.from_list([Audio(waveform=payload["reward_audio"]) for payload in payloads])
            frontier = sample.frontier_gen_part(DiffusionSamplingParams)
            return sample.replace_frontier(
                frontier.fill(
                    primitives={"video": Videos.from_list(videos), "audio": audios},
                    primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
                )
            )
        if any(log_prob is None for log_prob in rollout_log_probs):
            raise RuntimeError("MiniMax-H3 training rollout emitted no old-policy log-probs")
        transition_means_present = [payload.get("video_means") is not None for payload in payloads]
        if any(transition_means_present) != all(transition_means_present):
            raise RuntimeError("MiniMax-H3 rollout emitted transition means for only some outputs")

        params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        expected_video_sigmas = params.sigmas
        expected_audio_sigmas = get_sigma_schedule(
            num_steps=int(params.num_inference_steps),
            shift=self.audio_shift,
            device=torch.device("cpu"),
        )
        expected_indices = torch.as_tensor(payloads[0]["indices"], dtype=torch.long)
        expected_sde_indices = torch.as_tensor(schedules[0].get("sde_indices", []), dtype=torch.long)
        for output_index, (payload, schedule, log_prob) in enumerate(
            zip(payloads, schedules, rollout_log_probs, strict=True)
        ):
            verify_engine_used_sigmas(
                schedule.get("video"),
                expected=expected_video_sigmas,
                engine_name=f"vllm-omni-minimax-h3 output {output_index} video",
            )
            verify_engine_used_sigmas(
                schedule.get("audio"),
                expected=expected_audio_sigmas,
                engine_name=f"vllm-omni-minimax-h3 output {output_index} audio",
            )
            indices_i = torch.as_tensor(payload["indices"], dtype=torch.long)
            sde_indices_i = torch.as_tensor(schedule.get("sde_indices", []), dtype=torch.long)
            if not torch.equal(indices_i, expected_indices) or not torch.equal(sde_indices_i, expected_sde_indices):
                raise RuntimeError(
                    f"MiniMax-H3 output {output_index} trajectory index mismatch: "
                    f"indices={indices_i.tolist()} sde_indices={sde_indices_i.tolist()}, "
                    f"expected={expected_indices.tolist()}/{expected_sde_indices.tolist()}"
                )
            for field_name in ("video", "audio"):
                tensor = payload[field_name]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all()
                    or int(tensor.shape[1]) != int(expected_indices.numel())
                ):
                    raise RuntimeError(
                        f"MiniMax-H3 output {output_index} invalid {field_name} trajectory: "
                        f"shape={getattr(tensor, 'shape', None)} dtype={getattr(tensor, 'dtype', None)}"
                    )
            if (
                not torch.is_tensor(log_prob)
                or log_prob.dtype != torch.float32
                or not torch.isfinite(log_prob).all()
                or int(log_prob.shape[1]) != int(expected_sde_indices.numel())
            ):
                raise RuntimeError(
                    f"MiniMax-H3 output {output_index} invalid rollout log-prob: "
                    f"shape={getattr(log_prob, 'shape', None)} dtype={getattr(log_prob, 'dtype', None)}"
                )
            if transition_means_present[output_index]:
                transition_means = payload["video_means"]
                if (
                    not torch.is_tensor(transition_means)
                    or transition_means.dtype != torch.float32
                    or not torch.isfinite(transition_means).all()
                    or int(transition_means.shape[1]) != int(expected_sde_indices.numel())
                ):
                    raise RuntimeError(
                        f"MiniMax-H3 output {output_index} invalid video transition means: "
                        f"shape={getattr(transition_means, 'shape', None)} "
                        f"dtype={getattr(transition_means, 'dtype', None)}"
                    )

        latents = torch.cat([payload["video"] for payload in payloads], dim=0)
        aux_latents = torch.cat([payload["audio"] for payload in payloads], dim=0)
        text_condition = TextEmbedCondition.concat(
            [
                TextEmbedCondition(
                    embeds=payload["text_embeddings"],
                    attn_mask=torch.ones(
                        payload["text_embeddings"].shape[:2],
                        dtype=torch.bool,
                        device=payload["text_embeddings"].device,
                    ),
                )
                for payload in payloads
            ]
        )
        sde_logp = torch.cat(rollout_log_probs, dim=0).to(torch.float32)
        sde_means = (
            torch.cat([payload["video_means"] for payload in payloads], dim=0)
            if all(transition_means_present)
            else None
        )

        sigmas = schedules[0].get("video")
        indices = expected_indices
        sde_indices = expected_sde_indices

        segment = make_video_segment(
            latents=latents,
            aux_latents=aux_latents,
            sigmas=sigmas,
            indices=indices,
            sde_indices=sde_indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            initial_latents=latents[:, 0] if int(indices[0]) == 0 else None,
        )

        videos = []
        for payload in payloads:
            frame = payload["reward_video"]
            require(
                torch.is_tensor(frame) and frame.ndim == 5 and int(frame.shape[0]) == 1 and int(frame.shape[2]) >= 1,
                f"MiniMax-H3 reward video must be [1,C,T,H,W], got {getattr(frame, 'shape', None)}",
            )
            frames = frame[0].permute(1, 0, 2, 3).to(torch.float32).div_(255.0)
            videos.append(Video(frames=frames))

        audios = Audios.from_list([Audio(waveform=payload["reward_audio"]) for payload in payloads])
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"video": Videos.from_list(videos), "audio": audios},
                primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
                conditions={"text": text_condition},
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
            audio_joint_sde=bool(model_config.audio_joint_sde),
        )
        self.output_adapter = MiniMaxH3OutputAdapter(
            self.modality,
            audio_shift=float(model_config.audio_shift),
        )

    def validate(self) -> None:
        require(self.model_config is not None, "MiniMaxH3T2VAAdapter requires model_config")
        require(hasattr(self.model_config, "video_shift"), "MiniMaxH3T2VAAdapter requires model_config.video_shift")
        require(hasattr(self.model_config, "audio_shift"), "MiniMaxH3T2VAAdapter requires model_config.audio_shift")
        require(
            hasattr(self.model_config, "audio_joint_sde"),
            "MiniMaxH3T2VAAdapter requires model_config.audio_joint_sde",
        )
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
