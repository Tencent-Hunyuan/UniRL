"""VideoAlign reward scorer — self-contained, no fastvideo dependency."""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from unirl.reward.base import PromptRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest, RewardResponse

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)


_DETAILED_PROMPT_WITH_SPECIAL_TOKEN = "\nYou are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.\n\n**Visual Quality:**  \nEvaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:\n- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.\n- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.\n- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).\n- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.\n- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. \n\nPlease provide the ratings of Visual Quality: <|VQ_reward|>\nEND\n\n**Motion Quality:**  \nAssess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:\n- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).\n- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).\n- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.\n- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.\n- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.\n- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.\n\nPlease provide the ratings of Motion Quality: <|MQ_reward|>\nEND\n\n**Text Alignment:**  \nAssess how well the video matches the textual prompt across the following sub-dimensions:\n- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.\n- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.\n- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.  \n- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.\n- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.\n\nTextual prompt - {text_prompt}\nPlease provide the ratings of Text Alignment: <|TA_reward|>\nEND\n"

_DIMENSION_DESCRIPTIONS = {
    "VQ": ["visual quality", "the quality of the video in terms of clearness, resolution, brightness, and color"],
    "TA": ["text-to-video alignment", "the alignment between the text prompt and the video content and motion"],
    "MQ": ["motion quality", "the quality of the motion in terms of consistency, smoothness, and completeness"],
}

_SIMPLE_PROMPT = (
    "Please evaluate the {dimension_name} of a generated video. "
    "Consider {dimension_description}. "
    'The text prompt used for generation is "{text_prompt}".'
)


def _build_prompt(prompt: str, dimension, template_type: str) -> str:
    """Build an eval prompt, safely inserting the user-supplied text prompt."""
    if template_type == "detailed_special":
        return _DETAILED_PROMPT_WITH_SPECIAL_TOKEN.replace("{text_prompt}", prompt)
    if isinstance(dimension, list) and len(dimension) > 1:
        dim_name = ", ".join([_DIMENSION_DESCRIPTIONS[d][0] for d in dimension])
        dim_name = f"overall performance({dim_name})"
        dim_desc = "the overall performance of the video"
    else:
        if isinstance(dimension, list):
            dimension = dimension[0]
        dim_name = _DIMENSION_DESCRIPTIONS[dimension][0]
        dim_desc = _DIMENSION_DESCRIPTIONS[dimension][1]
    if template_type == "none":
        return prompt
    return _SIMPLE_PROMPT.format(dimension_name=dim_name, dimension_description=dim_desc).replace(
        "{text_prompt}", prompt
    )


def _video_reward_model_class():
    """Build the official Qwen2-VL reward-model class lazily."""
    from transformers import Qwen2VLForConditionalGeneration

    class VideoRewardModel(Qwen2VLForConditionalGeneration):
        def __init__(
            self,
            config,
            *,
            output_dim: int = 1,
            reward_token: str = "special",
            special_token_ids: Optional[List[int]] = None,
        ) -> None:
            super().__init__(config)
            hidden_size = getattr(config, "hidden_size", None) or config.text_config.hidden_size
            self.output_dim = int(output_dim)
            self.reward_token = str(reward_token)
            self.special_token_ids = tuple(
                int(token_id) for token_id in (() if special_token_ids is None else special_token_ids)
            )
            if self.special_token_ids:
                self.reward_token = "special"
            self.rm_head = nn.Linear(hidden_size, self.output_dim, bias=False)

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            video_grid_thw: Optional[torch.Tensor] = None,
            rope_deltas: Optional[torch.Tensor] = None,
            mm_token_type_ids: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Dict[str, torch.Tensor]:
            del labels
            kwargs.pop("return_dict", None)
            kwargs.pop("logits_to_keep", None)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
            token_rewards = self.rm_head(outputs.last_hidden_state)  # [B, L, output_dim]
            batch_size = int(token_rewards.shape[0])

            if self.reward_token == "special":
                if input_ids is None or len(self.special_token_ids) != 3:
                    raise ValueError("VideoAlign special-token pooling requires input_ids and three reward token ids.")
                special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for token_id in self.special_token_ids:
                    special_mask |= input_ids == token_id
                counts = special_mask.sum(dim=1)
                if not bool((counts == 3).all()):
                    raise ValueError(
                        f"VideoAlign expected three reward tokens per sample, got counts={counts.tolist()}."
                    )
                logits = token_rewards[special_mask].view(batch_size, 3, self.output_dim)
                if self.output_dim == 3:
                    logits = logits.diagonal(dim1=1, dim2=2)
                return {"logits": logits.reshape(batch_size, -1)}

            if attention_mask is None:
                attention_mask = token_rewards.new_ones(token_rewards.shape[:2], dtype=torch.long)
            if self.reward_token == "last":
                last = attention_mask.sum(dim=1).sub(1).clamp(min=0)
                return {"logits": token_rewards[torch.arange(batch_size, device=token_rewards.device), last]}
            if self.reward_token == "mean":
                mask = attention_mask.unsqueeze(-1).to(token_rewards.dtype)
                return {"logits": (token_rewards * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)}
            raise ValueError(f"Unsupported VideoAlign reward_token={self.reward_token!r}.")

    return VideoRewardModel


def _lora_target_modules(model: nn.Module, config: dict) -> List[str]:
    """Match the official VideoAlign LoRA target selection."""
    raw_excluded = config.get("lora_namespan_exclude")
    excluded = [] if raw_excluded is None else list(raw_excluded)
    if not config.get("vision_lora", False):
        excluded.append("visual")
    excluded.extend(("lm_head", "rm_head", "embed_tokens"))
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding)) and not any(part in name for part in excluded)
    ]
    limit = int(config.get("num_lora_modules", -1))
    return targets[-limit:] if limit > 0 else targets


def _read_checkpoint(checkpoint_dir: str) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Read the latest official full or split VideoAlign checkpoint."""
    candidates = [checkpoint_dir] if os.path.basename(checkpoint_dir).startswith("checkpoint-") else []
    candidates.extend(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    candidates = sorted(
        {path for path in candidates if os.path.isdir(path)},
        key=lambda p: int(m.group(1)) if (m := re.search(r"checkpoint-(\d+)", p)) else -1,
        reverse=True,
    )
    if not candidates:
        candidates = [checkpoint_dir]

    checkpoint_path = candidates[0]
    full_path = os.path.join(checkpoint_path, "model.pth")
    if os.path.isfile(full_path):
        state = torch.load(full_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"VideoAlign checkpoint {full_path!r} is not a state dict.")
        return state, True

    from safetensors.torch import load_file

    state: Dict[str, torch.Tensor] = {}
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_path = os.path.join(checkpoint_path, "non_lora_state_dict.pth")
    missing_files = [path for path in (adapter_path, non_lora_path) if not os.path.isfile(path)]
    if missing_files:
        raise FileNotFoundError(
            f"VideoAlign found no model.pth and the split checkpoint is incomplete; missing {missing_files}."
        )
    state.update(load_file(adapter_path))
    non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=True)
    if not isinstance(non_lora, dict):
        raise TypeError(f"VideoAlign checkpoint {non_lora_path!r} is not a state dict.")
    state.update(non_lora)
    return state, False


def _destination_key(key: str, model_keys: set[str]) -> Optional[str]:
    """Resolve checkpoint wrapper and adapter-name differences."""
    key = re.sub(r"\.(lora_[AB]|lora_embedding_[AB])\.weight$", r".\1.default.weight", key)
    prefix = "base_model.model." if key.startswith("base_model.model.") else ""
    suffix = key[len(prefix) :]
    for old, new in (
        ("visual.", "model.visual."),
        ("model.embed_tokens.", "model.language_model.embed_tokens."),
        ("model.layers.", "model.language_model.layers."),
        ("model.norm.", "model.language_model.norm."),
    ):
        if suffix.startswith(old):
            suffix = new + suffix[len(old) :]
            break
    converted = prefix + suffix
    candidates = [converted]
    if converted.startswith("base_model.model."):
        candidates.append(converted[len("base_model.model.") :])
    else:
        candidates.append("base_model.model." + converted)
    return next((candidate for candidate in candidates if candidate in model_keys), None)


def _load_checkpoint(inference_obj, checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> None:
    """Load official VideoReward weights into the matching rm_head/PEFT model."""
    full_state, is_full = _read_checkpoint(checkpoint_dir)
    model_state = inference_obj.model.state_dict()
    model_keys = set(model_state)
    filtered_state = {}
    unmatched_keys = []
    for key, value in full_state.items():
        destination = _destination_key(key, model_keys)
        if destination is not None:
            filtered_state[destination] = value
        else:
            unmatched_keys.append(key)

    unmatched_source = len(unmatched_keys)
    critical = [key for key in model_keys if key.endswith(("rm_head.weight", "embed_tokens.weight"))]
    missing_critical = [key for key in critical if key not in filtered_state]
    lora_keys = [key for key in model_keys if ".lora_" in key]
    missing_lora = [key for key in lora_keys if key not in filtered_state]
    if unmatched_source or missing_critical or missing_lora:
        raise RuntimeError(
            f"VideoAlign checkpoint is incompatible: {unmatched_source} source tensors unmatched, "
            f"missing {len(missing_critical)} critical and "
            f"{len(missing_lora)}/{len(lora_keys)} LoRA tensors after key conversion. "
            f"Examples: {(unmatched_keys + missing_critical + missing_lora)[:5]}"
        )

    incompatible = inference_obj.model.load_state_dict(filtered_state, strict=False)
    if is_full:
        parameter_keys = set(dict(inference_obj.model.named_parameters()))
        missing_parameters = [key for key in incompatible.missing_keys if key in parameter_keys]
        if missing_parameters:
            raise RuntimeError(
                f"VideoAlign full checkpoint left {len(missing_parameters)} model parameters unloaded; "
                f"examples: {missing_parameters[:5]}"
            )
    logger.info(
        "Loaded VideoAlign checkpoint: %d/%d tensors matched (%d unmatched source tensors)",
        len(filtered_state),
        len(model_state),
        unmatched_source,
    )
    inference_obj.model.to(device=device, dtype=dtype)
    inference_obj.model.eval()


class _VideoAlignInference:
    """Self-contained inference for the official Qwen2-VL VideoReward model."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.model: Optional[nn.Module] = None
        self.processor = None
        self._inference_cfg: Optional[dict] = None

        self._fps: float = 2.0
        self._num_frames: Optional[int] = None
        self._max_pixels: int = 200704
        self._eval_dim: List[str] = ["VQ", "MQ", "TA"]
        self._prompt_template_type: str = "detailed_special"
        self._sample_type: str = "uniform"

        self._load(checkpoint_dir)

    def _load(self, checkpoint_dir: str) -> None:
        """Load model, config, processor, and checkpoint weights."""
        from transformers import AutoProcessor

        config_root = (
            os.path.dirname(checkpoint_dir.rstrip("/"))
            if os.path.basename(checkpoint_dir.rstrip("/")).startswith("checkpoint-")
            else checkpoint_dir
        )
        config_path = os.path.join(config_root, "model_config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        data_cfg = cfg["data_config"]
        model_cfg = cfg["model_config"]
        peft_cfg = cfg.get("peft_lora_config", {})
        self._inference_cfg = cfg.get("inference_config")

        self._fps = data_cfg.get("fps", 2.0)
        self._num_frames = data_cfg.get("num_frames")
        self._max_pixels = data_cfg.get("max_frame_pixels", 200704)
        self._eval_dim = data_cfg.get("eval_dim", ["VQ", "MQ", "TA"])
        self._prompt_template_type = data_cfg.get("prompt_template_type", "detailed_special")
        self._sample_type = data_cfg.get("sample_type", "uniform")

        qwen_path = os.environ.get("VIDEOALIGN_QWEN_CKPT", "")
        if not qwen_path:
            _sibling = os.path.join(os.path.dirname(config_root.rstrip("/")), "Qwen2-VL-2B-Instruct")
            qwen_path = (
                _sibling
                if os.path.isdir(_sibling)
                else model_cfg.get("model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
            )
        self.processor = AutoProcessor.from_pretrained(qwen_path, padding_side="right")

        special_token_ids = None
        if model_cfg.get("use_special_tokens", False):
            special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
            self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(special_tokens)

        model_cls = _video_reward_model_class()
        self.model = model_cls.from_pretrained(
            qwen_path,
            output_dim=int(model_cfg.get("output_dim", 1)),
            reward_token=str(model_cfg.get("reward_token", "special")),
            special_token_ids=special_token_ids,
            dtype=self.dtype,
        )
        if special_token_ids is not None:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        if peft_cfg.get("lora_enable", False):
            from peft import LoraConfig, get_peft_model

            target_modules = _lora_target_modules(self.model, peft_cfg)
            lora_config = LoraConfig(
                target_modules=target_modules,
                r=int(peft_cfg.get("lora_r", 16)),
                lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
                lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
                task_type=str(peft_cfg.get("lora_task_type", "CAUSAL_LM")),
                use_rslora=bool(peft_cfg.get("use_rslora", False)),
                bias="none",
                modules_to_save=peft_cfg.get("lora_modules_to_save"),
            )
            self.model = get_peft_model(self.model, lora_config)

        _load_checkpoint(self, checkpoint_dir, torch.device(self.device), self.dtype)

    def _norm(self, reward: dict) -> dict:
        if self._inference_cfg is None:
            return reward
        reward["VQ"] = (reward["VQ"] - self._inference_cfg["VQ_mean"]) / (self._inference_cfg["VQ_std"] + 1e-8)
        reward["MQ"] = (reward["MQ"] - self._inference_cfg["MQ_mean"]) / (self._inference_cfg["MQ_std"] + 1e-8)
        reward["TA"] = (reward["TA"] - self._inference_cfg["TA_mean"]) / (self._inference_cfg["TA_std"] + 1e-8)
        return reward

    def reward(self, video_paths: List[str], prompts: List[str]) -> List[dict]:
        from qwen_vl_utils import process_vision_info

        if len(video_paths) != len(prompts):
            raise ValueError(f"VideoAlign got {len(video_paths)} videos for {len(prompts)} prompts.")
        chat_data = []
        for video_path, prompt in zip(video_paths, prompts):
            vid_info: Dict[str, Any] = {
                "type": "video",
                "video": video_path if video_path.startswith("file://") else f"file://{video_path}",
                "max_pixels": self._max_pixels,
                "sample_type": self._sample_type,
            }
            if self._num_frames is not None:
                vid_info["nframes"] = self._num_frames
            else:
                vid_info["fps"] = self._fps
            text = _build_prompt(prompt, self._eval_dim, self._prompt_template_type)
            chat_data.append([{"role": "user", "content": [vid_info, {"type": "text", "text": text}]}])

        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            logits = self.model(return_dict=True, **batch)["logits"]

        rewards = [self._norm({"VQ": r[0].item(), "MQ": r[1].item(), "TA": r[2].item()}) for r in logits]
        for item in rewards:
            item["Overall"] = item["VQ"] + item["MQ"] + item["TA"]
        return rewards


class VideoAlignRewardScorer(LocalRewardBackend):
    """Video-text alignment reward using self-contained VideoAlign inference."""

    canonical_model_name = "videoalign"
    input_kind = "video"

    def __init__(self, *, config: "VideoAlignSpec", base_device: str) -> None:
        self._inferencer: Optional[_VideoAlignInference] = None
        dtype = torch.bfloat16 if str(config.dtype).lower() == "bf16" else torch.float16
        super().__init__(
            device=resolve_device(config.device, base_device),
            dtype=dtype,
            batch_size=config.batch_size,
            prompt_source=config.prompt_source,
            checkpoint_path=config.checkpoint_path,
        )
        self._vq_coef = float(config.vq_coef)
        self._mq_coef = float(config.mq_coef)
        self._ta_coef = float(config.ta_coef)

    def covers_prompt_video(self) -> bool:
        return self._ta_coef > 0.0

    def _load_model(self) -> None:
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
        checkpoint_path = self.model_kwargs["checkpoint_path"] or os.environ.get("VIDEOALIGN_CKPT")
        if not checkpoint_path:
            raise ValueError("VIDEOALIGN_CKPT env var or config.checkpoint_path must be set")
        self._inferencer = _VideoAlignInference(checkpoint_path, device=str(self.device), dtype=self.dtype)
        self.model = self._inferencer.model

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        try:
            rewards, components = self._compute_videoalign_rewards(request)
            return RewardResponse(
                rewards=rewards,
                component_rewards=components,
                successes=[True] * len(rewards),
                errors=[None] * len(rewards),
                compute_time=time.time() - start,
            )
        except Exception as exc:
            logger.warning("VideoAlign reward computation failed: %s", exc)
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=[str(exc)] * request.batch_size,
                compute_time=time.time() - start,
            )
        finally:
            self.offload()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        rewards, _ = self._compute_videoalign_rewards(request)
        return rewards

    def _compute_videoalign_rewards(self, request: RewardRequest) -> Tuple[List[float], Dict[str, List[float]]]:
        from diffusers.utils import export_to_video

        if self.model is not None and next(self.model.parameters()).device.type != "cuda":
            self.model.to(self.device)

        all_rewards: List[float] = []
        components: Dict[str, List[float]] = {"vq": [], "mq": [], "ta": [], "overall": []}
        prompts = self.prompts(request)

        with tempfile.TemporaryDirectory(prefix="unirl_videoalign_") as tmpdir:
            video_paths = []
            for idx, video in enumerate(request.videos):
                path = os.path.join(tmpdir, f"sample_{idx:05d}.mp4")
                v = video.detach().cpu()
                if v.dim() == 5:
                    v = v.squeeze(0)
                if v.dim() != 4:
                    raise ValueError(f"Expected 4D video tensor, got shape={tuple(v.shape)}")
                c0, t0 = v.shape[0], v.shape[1]
                if c0 == 3 and t0 != 3:
                    v = v.permute(1, 0, 2, 3)
                v = v.permute(0, 2, 3, 1)[..., :3].clamp(0.0, 1.0)
                frames = (v * 255).round().to(torch.uint8).numpy()
                export_to_video(list(frames), path, fps=24)
                video_paths.append(path)

            for start in range(0, len(video_paths), self.batch_size):
                batch_paths = video_paths[start : start + self.batch_size]
                batch_prompts = prompts[start : start + self.batch_size]
                results = self._inferencer.reward(batch_paths, batch_prompts)
                for item in results:
                    vq, mq, ta = float(item["VQ"]), float(item["MQ"]), float(item["TA"])
                    components["vq"].append(vq)
                    components["mq"].append(mq)
                    components["ta"].append(ta)
                    components["overall"].append(float(item["Overall"]))
                    all_rewards.append(self._vq_coef * vq + self._mq_coef * mq + self._ta_coef * ta)

        return all_rewards, components


@dataclass
class VideoAlignSpec(PromptRewardComponentSpec):
    """Typed config for the VideoAlign reward component."""

    weight: float = 1.0
    batch_size: int = 1
    device: str = "auto"
    checkpoint_path: str = ""
    dtype: str = "bf16"
    vq_coef: float = 1.0
    mq_coef: float = 1.0
    ta_coef: float = 1.0
