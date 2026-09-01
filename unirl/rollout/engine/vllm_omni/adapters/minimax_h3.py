"""MiniMax-H3 text-to-video+audio adapter for vLLM-Omni rollout."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.config.require import require
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import read_captures
from unirl.rollout.engine.vllm_omni.pipelines.minimax_h3.pipeline import CAPTURE_KEY
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
    """The official aspect-ratio name for these dimensions, or the configured override."""
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
            # The engine derives its canvas from the ratio name and this edge, so
            # without it every recipe would be served at H3's released 768.
            short_edge=min(int(params.height), int(params.width)),
            flow_shift=self.video_shift,
            audio_flow_shift=self.audio_shift,
            audio_joint_sde=self.audio_joint_sde,
            capture_transition_means=bool(sampler_kwargs.get("capture_transition_means", False)),
            # A terminal x0 is ~270 MB per sample at 768px, so a forward-process
            # recipe asks for it rather than every rollout paying to ship one.
            capture_terminal_latents=bool(sampler_kwargs.get("capture_terminal_latents", False)),
        )
        kwargs["extra_args"] = extra
        return sampling


class MiniMaxH3OutputAdapter:
    """Recover sparse joint trajectories and the reward media for each request."""

    final_output_type = "video"
    stage_id = 0

    _MISSING_CAPTURE_MSG = (
        "MiniMax-H3 rollout output carries no unirl capture. The stage config must install "
        "MiniMaxH3RLPipeline, and the capture flush must be installed in the stage process."
    )

    def __init__(self, modality: str, *, audio_shift: float) -> None:
        self.modality = modality
        self.audio_shift = float(audio_shift)

    def _captures(self, per_request: List[List[OmniRawResult]]) -> List[Dict[str, Any]]:
        """One capture bundle per request, in request order."""
        captures = []
        for request_outputs in per_request:
            output = pick_stage_output(
                request_outputs,
                final_output_type=self.final_output_type,
                stage_id=self.stage_id,
            )
            if output is None:
                raise RuntimeError("MiniMax-H3 rollout request has no video-stage output")
            capture = read_captures(output).get(CAPTURE_KEY)
            if not isinstance(capture, dict) or not capture:
                raise RuntimeError(self._MISSING_CAPTURE_MSG)
            captures.append(capture)
        return captures

    @staticmethod
    def _videos(captures: List[Dict[str, Any]]) -> Videos:
        """Decode the uint8 reward frames back to ``[T, C, H, W]`` floats."""
        videos = []
        for capture in captures:
            frame = capture["reward_video"]
            require(
                torch.is_tensor(frame) and frame.ndim == 5 and int(frame.shape[0]) == 1 and int(frame.shape[2]) >= 1,
                f"MiniMax-H3 reward video must be [1,C,T,H,W], got {getattr(frame, 'shape', None)}",
            )
            frames = frame[0].permute(1, 0, 2, 3).to(torch.float32).div_(255.0)
            videos.append(Video(frames=frames))
        return Videos.from_list(videos)

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("MiniMax-H3 rollout returned no request outputs")

        captures = self._captures(per_request)
        for index, capture in enumerate(captures):
            if capture.get("reward_video") is None or capture.get("reward_audio") is None:
                raise RuntimeError(f"MiniMax-H3 rollout output {index} is missing reward video/audio")

        # log_probs is absent by design under a forward-process objective, which trains
        # on the terminal x0 and never asks for a transition density.
        replay_fields = ("video", "audio", "indices", "text_embeddings")
        replay_ready = [all(capture.get(key) is not None for key in replay_fields) for capture in captures]
        if any(replay_ready) != all(replay_ready):
            raise RuntimeError("MiniMax-H3 rollout mixed replay and evaluation-only outputs")

        audios = Audios.from_list([Audio(waveform=capture["reward_audio"]) for capture in captures])
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        if not all(replay_ready):
            return sample.replace_frontier(
                frontier.fill(
                    primitives={"video": self._videos(captures), "audio": audios},
                    primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
                )
            )

        transition_means_present = [capture.get("video_means") is not None for capture in captures]
        if any(transition_means_present) != all(transition_means_present):
            raise RuntimeError("MiniMax-H3 rollout emitted transition means for only some outputs")

        log_probs_present = [capture.get("log_probs") is not None for capture in captures]
        if any(log_probs_present) != all(log_probs_present):
            raise RuntimeError("MiniMax-H3 rollout emitted rollout log-probs for only some outputs")

        params = frontier.sampling_params
        expected_video_sigmas = params.sigmas
        expected_audio_sigmas = get_sigma_schedule(
            num_steps=int(params.num_inference_steps),
            shift=self.audio_shift,
            device=torch.device("cpu"),
        )
        expected_indices = torch.as_tensor(captures[0]["indices"], dtype=torch.long)
        expected_sde_indices = torch.as_tensor(captures[0].get("sde_indices", []), dtype=torch.long)
        # Only a pure-ODE rollout may omit them; a stochastic step that lost its density
        # would otherwise reach the ratio silently.
        if int(expected_sde_indices.numel()) and not all(log_probs_present):
            raise RuntimeError(
                f"MiniMax-H3 rollout has {int(expected_sde_indices.numel())} stochastic step(s) "
                "but returned no rollout log-probs"
            )
        for output_index, capture in enumerate(captures):
            verify_engine_used_sigmas(
                capture.get("video_sigmas"),
                expected=expected_video_sigmas,
                engine_name=f"vllm-omni-minimax-h3 output {output_index} video",
            )
            verify_engine_used_sigmas(
                capture.get("audio_sigmas"),
                expected=expected_audio_sigmas,
                engine_name=f"vllm-omni-minimax-h3 output {output_index} audio",
            )
            indices_i = torch.as_tensor(capture["indices"], dtype=torch.long)
            sde_indices_i = torch.as_tensor(capture.get("sde_indices", []), dtype=torch.long)
            if not torch.equal(indices_i, expected_indices) or not torch.equal(sde_indices_i, expected_sde_indices):
                raise RuntimeError(
                    f"MiniMax-H3 output {output_index} trajectory index mismatch: "
                    f"indices={indices_i.tolist()} sde_indices={sde_indices_i.tolist()}, "
                    f"expected={expected_indices.tolist()}/{expected_sde_indices.tolist()}"
                )
            for field_name in ("video", "audio"):
                tensor = capture[field_name]
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
            if log_probs_present[output_index]:
                log_prob = capture["log_probs"]
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
                transition_means = capture["video_means"]
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

        latents = torch.cat([capture["video"] for capture in captures], dim=0)
        aux_latents = torch.cat([capture["audio"] for capture in captures], dim=0)
        text_condition = TextEmbedCondition.concat(
            [
                TextEmbedCondition(
                    embeds=capture["text_embeddings"],
                    attn_mask=torch.ones(
                        capture["text_embeddings"].shape[:2],
                        dtype=torch.bool,
                        device=capture["text_embeddings"].device,
                    ),
                )
                for capture in captures
            ]
        )
        sde_logp = (
            torch.cat([capture["log_probs"] for capture in captures], dim=0).to(torch.float32)
            if all(log_probs_present)
            else None
        )
        sde_means = (
            torch.cat([capture["video_means"] for capture in captures], dim=0)
            if all(transition_means_present)
            else None
        )

        segment = make_video_segment(
            latents=latents,
            aux_latents=aux_latents,
            sigmas=captures[0].get("video_sigmas"),
            indices=expected_indices,
            sde_indices=expected_sde_indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            initial_latents=latents[:, 0] if int(expected_indices[0]) == 0 else None,
        )

        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"video": self._videos(captures), "audio": audios},
                primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
                conditions={"text": text_condition},
            )
        )


@register_adapter("minimax_h3_t2va")
class MiniMaxH3T2VAAdapter(ModelAdapter):
    """MiniMax-H3 t2va, one HSDP4+UP4 engine per external DP replica."""

    stage_yaml = "minimax_h3_t2va_rl.yaml"
    needs_driver_tokenizer = False
    # The adapter is ~166M parameters, which base64s to a ~440 MB control message; spool it
    # to a file instead. The four sequence-parallel subprocesses each read that path.
    lora_file_transport = True

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
        # The pipeline reads the audio branch off rank 0 of the sequence-parallel group, so a
        # ring-parallel split would leave it reading a partial sequence.
        require(
            int((self.cfg.omni_extra or {}).get("ring_degree", 1)) == 1,
            "MiniMaxH3T2VAAdapter does not support ring parallelism; "
            f"got ring_degree={(self.cfg.omni_extra or {}).get('ring_degree')}",
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
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MiniMaxH3InputAdapter",
    "MiniMaxH3OutputAdapter",
    "MiniMaxH3T2VAAdapter",
]
