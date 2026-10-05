"""MiniMax-H3 text-to-video+audio adapter for vLLM-Omni rollout."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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


_REPLAY_FIELDS = ("video", "audio", "indices", "text_embeddings")
_TRAINER_BLOCKS = re.compile(r"^transformer\.transformer_blocks\.(\d+)\.(.+)\.(lora_[AB])\.weight$")
# Diffusers leaf -> serving leaf. ``ff.net.0.proj`` is the fused SwiGLU input
# projection and is split into gate/up sublayers separately.
_SERVING_LEAVES = {
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.to_out.0": "attn.out_proj",
    "ff.net.2": "mlp.fc2",
}
_SWIGLU_LEAF = "ff.net.0.proj"


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
        extra.update(
            task="t2va",
            aspect_ratio=_resolve_aspect_ratio(params),
            flow_shift=self.video_shift,
            audio_flow_shift=self.audio_shift,
            audio_joint_sde=self.audio_joint_sde,
        )
        kwargs["extra_args"] = extra
        return sampling


def _reward_primitives(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decode the per-request reward clip and waveform into frontier primitives."""
    videos = []
    for payload in payloads:
        frame = payload.get("reward_video")
        require(
            torch.is_tensor(frame) and frame.ndim == 5 and int(frame.shape[0]) == 1 and int(frame.shape[2]) >= 1,
            f"MiniMax-H3 reward video must be [1,C,T,H,W], got {getattr(frame, 'shape', None)}",
        )
        require(payload.get("reward_audio") is not None, "MiniMax-H3 rollout output is missing reward audio")
        videos.append(Video(frames=frame[0].permute(1, 0, 2, 3).to(torch.float32).div_(255.0)))
    audios = Audios.from_list([Audio(waveform=payload["reward_audio"]) for payload in payloads])
    return {"video": Videos.from_list(videos), "audio": audios}


def _require_float32_rows(tensor: Any, *, rows: int, what: str) -> None:
    if (
        not torch.is_tensor(tensor)
        or tensor.dtype != torch.float32
        or not torch.isfinite(tensor).all()
        or int(tensor.shape[1]) != rows
    ):
        raise RuntimeError(
            f"MiniMax-H3 {what}: shape={getattr(tensor, 'shape', None)} dtype={getattr(tensor, 'dtype', None)}, "
            f"expected finite float32 with {rows} entries on dim 1"
        )


class MiniMaxH3OutputAdapter:
    """Recover sparse joint trajectories and the reward clip per request."""

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
                request_outputs, final_output_type=self.final_output_type, stage_id=self.stage_id
            )
            if output is None:
                raise RuntimeError("MiniMax-H3 rollout request has no video-stage output")
            outputs.append(output)

        payloads = [output.trajectory_latents for output in outputs]
        if not all(isinstance(payload, dict) for payload in payloads):
            raise RuntimeError("MiniMax-H3 rollout outputs must carry the RL pipeline's trajectory payload dict")
        primitives = _reward_primitives(payloads)
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        primitive_metadata = {"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}}

        replay_ready = [all(payload.get(key) is not None for key in _REPLAY_FIELDS) for payload in payloads]
        if any(replay_ready) != all(replay_ready):
            raise RuntimeError("MiniMax-H3 rollout mixed replay and evaluation-only outputs")
        if not all(replay_ready):
            return sample.replace_frontier(frontier.fill(primitives=primitives, primitive_metadata=primitive_metadata))

        schedules = [output.trajectory_timesteps for output in outputs]
        rollout_log_probs = [output.trajectory_log_probs for output in outputs]
        if not all(isinstance(schedule, dict) for schedule in schedules):
            raise RuntimeError("MiniMax-H3 training rollout emitted no sigma schedule")
        if any(log_prob is None for log_prob in rollout_log_probs):
            raise RuntimeError("MiniMax-H3 training rollout emitted no old-policy log-probs")

        params = frontier.sampling_params
        expected_audio_sigmas = get_sigma_schedule(
            num_steps=int(params.num_inference_steps),
            shift=self.audio_shift,
            device=torch.device("cpu"),
        )
        expected_indices = torch.as_tensor(payloads[0]["indices"], dtype=torch.long)
        expected_sde_indices = torch.as_tensor(schedules[0]["sde_indices"], dtype=torch.long)
        for index, (payload, schedule, log_prob) in enumerate(zip(payloads, schedules, rollout_log_probs, strict=True)):
            verify_engine_used_sigmas(
                schedule["video"], expected=params.sigmas, engine_name=f"vllm-omni-minimax-h3 output {index} video"
            )
            verify_engine_used_sigmas(
                schedule["audio"],
                expected=expected_audio_sigmas,
                engine_name=f"vllm-omni-minimax-h3 output {index} audio",
            )
            indices = torch.as_tensor(payload["indices"], dtype=torch.long)
            sde_indices = torch.as_tensor(schedule["sde_indices"], dtype=torch.long)
            if not torch.equal(indices, expected_indices) or not torch.equal(sde_indices, expected_sde_indices):
                raise RuntimeError(
                    f"MiniMax-H3 output {index} trajectory index mismatch: indices={indices.tolist()} "
                    f"sde_indices={sde_indices.tolist()}, expected "
                    f"{expected_indices.tolist()}/{expected_sde_indices.tolist()}"
                )
            _require_float32_rows(
                payload["video"], rows=expected_indices.numel(), what=f"output {index} video trajectory"
            )
            _require_float32_rows(
                payload["audio"], rows=expected_indices.numel(), what=f"output {index} audio trajectory"
            )
            _require_float32_rows(log_prob, rows=expected_sde_indices.numel(), what=f"output {index} rollout log-prob")

        latents = torch.cat([payload["video"] for payload in payloads], dim=0)
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
        segment = make_video_segment(
            latents=latents,
            aux_latents=torch.cat([payload["audio"] for payload in payloads], dim=0),
            sigmas=schedules[0]["video"],
            indices=expected_indices,
            sde_indices=expected_sde_indices,
            sde_logp=torch.cat(rollout_log_probs, dim=0),
            initial_latents=latents[:, 0] if int(expected_indices[0]) == 0 else None,
        )
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives=primitives,
                primitive_metadata=primitive_metadata,
                conditions={"text": text_condition},
            )
        )


def remap_minimax_h3_lora(
    lora_tensors: Dict[str, Any], peft_config: Optional[dict]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map a Diffusers-layout H3 block LoRA onto the vLLM-Omni serving DiT's module names."""
    remapped: Dict[str, Any] = {}
    for name, tensor in lora_tensors.items():
        match = _TRAINER_BLOCKS.match(name)
        if match is None:
            raise ValueError(f"MiniMax-H3 rollout LoRA only supports transformer_blocks modules; got {name!r}")
        block, leaf, side = match.groups()
        prefix = f"transformer.blocks.{block}"
        if leaf in _SERVING_LEAVES:
            remapped[f"{prefix}.{_SERVING_LEAVES[leaf]}.{side}.weight"] = tensor
        elif leaf == _SWIGLU_LEAF:
            if side == "lora_A":
                remapped[f"{prefix}.mlp.gate_proj.lora_A.weight"] = tensor
                remapped[f"{prefix}.mlp.up_proj.lora_A.weight"] = tensor.clone()
            else:
                if tensor.shape[0] % 2:
                    raise ValueError(f"MiniMax-H3 SwiGLU LoRA-B rows must split evenly; got {tuple(tensor.shape)}")
                # Diffusers stores the fused SwiGLU rows as [up, gate]; serving fc1 is [gate, up].
                up, gate = tensor.chunk(2, dim=0)
                remapped[f"{prefix}.mlp.gate_proj.lora_B.weight"] = gate.contiguous()
                remapped[f"{prefix}.mlp.up_proj.lora_B.weight"] = up.contiguous()
        else:
            raise ValueError(f"MiniMax-H3 rollout LoRA has no serving target for {leaf!r} (from {name!r})")

    leaves = sorted({name.split(".", 3)[3].rsplit(".", 2)[0] for name in remapped})
    target_modules = r"^transformer\.blocks\.\d+\.(?:" + "|".join(re.escape(leaf) for leaf in leaves) + r")$"
    return remapped, {**dict(peft_config or {}), "target_modules": target_modules}


@register_adapter("minimax_h3_t2va")
class MiniMaxH3T2VAAdapter(ModelAdapter):
    """MiniMax-H3 t2va: one four-GPU engine replica per rollout TP group."""

    deploy_config = "minimax_h3_t2va_rl.yaml"
    needs_driver_tokenizer = False
    # One grouped engine hands the adapter to four worker processes; byte-copy
    # avoids reusing a one-shot CUDA-IPC handle across them.
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = MiniMaxH3InputAdapter(
            self.modality,
            video_shift=float(model_config.video_shift),
            audio_shift=float(model_config.audio_shift),
            audio_joint_sde=bool(model_config.audio_joint_sde),
        )
        self.output_adapter = MiniMaxH3OutputAdapter(self.modality, audio_shift=float(model_config.audio_shift))

    def validate(self) -> None:
        require(self.model_config is not None, "MiniMaxH3T2VAAdapter requires model_config")
        for name in ("video_shift", "audio_shift", "audio_joint_sde"):
            require(hasattr(self.model_config, name), f"MiniMaxH3T2VAAdapter requires model_config.{name}")
        require(
            self.cfg.tp_size == 4,
            f"MiniMaxH3T2VAAdapter qualifies one engine per four GPUs; rollout.config.tp_size must be 4, "
            f"got {self.cfg.tp_size}",
        )

    def serving_lora(
        self, lora_tensors: Dict[str, Any], peft_config: Optional[dict]
    ) -> Tuple[Dict[str, Any], Optional[dict]]:
        return remap_minimax_h3_lora(lora_tensors, peft_config)

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
    "remap_minimax_h3_lora",
]
