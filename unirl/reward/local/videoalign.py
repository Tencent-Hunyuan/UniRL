"""VideoAlign reward scorer — self-contained, no fastvideo dependency.

Implements the Qwen2-VL-based VideoAlign reward model from DanceGRPO.
All inference logic is inlined to avoid external `fastvideo` imports.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest, RewardResponse

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)


def _as_tensor(x):
    """Coerce a vision-tower output to a tensor.

    Older transformers return the visual embeds tensor directly; newer
    versions wrap them in a model-output object (e.g. ``BaseModelOutputWithPooling``).
    """
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("last_hidden_state", "image_embeds", "video_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise TypeError(f"Cannot extract tensor from visual output of type {type(x).__name__}")


# ============================================================================
# VideoAlign prompt templates (from DanceGRPO prompt_template.py)
# ============================================================================

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
    """Build an eval prompt, safely inserting the user-supplied text prompt.

    Uses ``str.replace`` for the user prompt so that literal curly braces
    (e.g. ``{foo}``) in the prompt text do not crash ``str.format()``.
    """
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


# ============================================================================
# Video reading utilities (from DanceGRPO vision_process.py)
# ============================================================================

IMAGE_FACTOR = 28
FRAME_FACTOR = 2
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 1024 * 28 * 28
VIDEO_TOTAL_PIXELS = 20480 * 28 * 28
FRAME_MIN_PIXELS = 256 * 28 * 28
FRAME_MAX_PIXELS = 1024 * 28 * 28


def _smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = VIDEO_MIN_PIXELS,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> Tuple[int, int]:
    """Resize dimensions to the nearest factor while respecting pixel constraints."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too extreme {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize_video_to_view(
    video: torch.Tensor,
    view_grid: Tuple[int, int],
    target_size: int,
    factor: int = IMAGE_FACTOR,
) -> torch.Tensor:
    """Resize the input video so its frames match target_size on the shortest edge.

    The view_grid is ``(n_frames, grid_t)`` — only the frames in the first
    temporal position of each row are resized, the others are discarded.
    """
    from torchvision.transforms.v2 import functional as F

    _, total_frames, h, w = video.shape  # C, T, H, W
    n_frames, grid_t = view_grid
    if n_frames * grid_t < total_frames:
        video = video[:, : n_frames * grid_t]
    idx = torch.arange(total_frames).view(-1, grid_t)[:, 0]
    keep_frames = video[:, idx]

    # Calculate resize sizes
    current_size = min(h, w)
    scale = target_size / current_size
    new_h = round(h * scale / factor) * factor
    new_w = round(w * scale / factor) * factor

    keep_frames = F.resize(keep_frames, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)
    video[:, idx] = keep_frames
    return video


def _reshape_by_grid(video: torch.Tensor, view_grid: Tuple[int, int]) -> torch.Tensor:
    """Reshape into (grid_t * H, grid_h * W) by slicing frames from the
    first temporal position of each row."""
    total_frames, h, w = video.shape[1:]
    n_frames, grid_t = view_grid

    if n_frames * grid_t < total_frames:
        video = video[:, : n_frames * grid_t]
    elif n_frames * grid_t > total_frames:
        # pad with zeros if we need more frames
        pad = n_frames * grid_t - total_frames
        video = torch.cat([video, torch.zeros(3, pad, h, w, dtype=video.dtype, device=video.device)], dim=1)

    # Pick frames at column 0 of each temporal row
    idx = torch.arange(n_frames * grid_t).view(-1, grid_t)[:, 0]
    video = video[:, idx]

    # Reassemble into grid
    return video.permute(1, 2, 3, 0).reshape(n_frames, h, w * grid_t, 3).permute(0, 3, 1, 2)


# ============================================================================
# Qwen2-VL helper utilities (from DanceGRPO qwen_utils.py)
# ============================================================================


def _get_rope_index_modified(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen2-VL style 3D RoPE indices (from fastvideo qwen_utils.py)."""
    from diffusers.utils import is_npu_available

    if is_npu_available():
        from transformers.models.qwen2_vl.modeling_qwen2_vl import get_rope_index as _get_rope_index
    else:
        from transformers.models.qwen2_vl.processing_qwen2_vl import get_rope_index as _get_rope_index

    return _get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)


def _process_vision_info(
    conversations: List[List[Dict[str, Any]]],
) -> Tuple[List[Any], List[Any]]:
    """Extract image/video data and build pixel/video lists for Qwen-VL.

    Returns ``(images_list, videos_list)`` compatible with the Qwen2-VL processor.
    """
    from qwen_vl_utils import process_vision_info

    return process_vision_info(conversations)


def _build_vl_inputs(
    processor,
    chat_data: List[List[Dict[str, Any]]],
    video_inputs: List[Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build model-ready inputs through the Qwen2-VL processor chain."""
    text = processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)

    # Handle processor return from qwen_vl_utils.process_vision_info
    if video_inputs is not None:
        video_grid_thw, second_per_grid_ts = zip(
            *(processor.video_processor.preprocess(video_inputs, return_tensors="pt"))
        )
        video_grid_thw = torch.cat(video_grid_thw, dim=0)
        second_per_grid_ts = list(second_per_grid_ts)
    else:
        video_grid_thw = None
        second_per_grid_ts = None

    inputs = processor(
        text=text,
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_rescale=True,
        videos_kwargs={"do_rescale": True},
    ).to(device)

    # Compute 3D RoPE position ids
    if video_grid_thw is not None:
        inputs["video_grid_thw"] = video_grid_thw.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        rope_deltas = inputs.get("rope_deltas")
        if rope_deltas is not None:
            rope_deltas = rope_deltas.to(device)
        inputs["position_ids"], inputs["rope_deltas"] = _get_rope_index_modified(
            input_ids=inputs["input_ids"],
            image_grid_thw=None,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            # rope_deltas=rope_deltas,
        )

    return inputs


# ============================================================================
# VideoAlign model definition (from DanceGRPO videoalign_model.py)
# ============================================================================


class _MLP(nn.Module):
    """Lightweight MLP head on top of vision-language hidden states."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        # Store the key name for logging purposes
        self.key = "mlp_head"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _VideoAlignModel(nn.Module):
    """VideoAlign model: Qwen2-VL backbone + three reward MLP heads.

    Each head outputs a scalar per sample — the reward dimension scores.
    """

    def __init__(
        self,
        base_model: nn.Module,
        in_dim: int,
        reg_dim: int = 1024,
        drop_rate: float = 0.2,
        use_special_tokens: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self.use_special_tokens = use_special_tokens
        self.VQ_head = _MLP(in_dim, 1)
        self.MQ_head = _MLP(in_dim, 1)
        self.TA_head = _MLP(in_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        position_ids: torch.Tensor,
        rope_deltas: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(video_grid_thw)
        video_grid_thw_list = [video_grid_thw[i] for i in range(batch_size)]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw_list,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            output_hidden_states=True,
        )

        # Extract last hidden state using model-specific layout
        last_hidden_state = _get_last_hidden_state(outputs, self.model)

        # Pool over visual token positions
        pooled = _pool_visual_tokens(
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask,
            video_grid_thw=video_grid_thw,
            use_special_tokens=self.use_special_tokens,
        )

        vq = self.VQ_head(pooled)
        mq = self.MQ_head(pooled)
        ta = self.TA_head(pooled)
        return {
            "logits": torch.cat([vq, mq, ta], dim=-1),
            "last_hidden_state": last_hidden_state,
        }


def _get_last_hidden_state(outputs, model: nn.Module) -> torch.Tensor:
    """Extract last_hidden_state from model outputs, handling Qwen2-VL's
    composable architecture where the LM is at ``model.model``."""
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    raise ValueError("Cannot extract last_hidden_state from model outputs")


def _pool_visual_tokens(
    *,
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    video_grid_thw: torch.Tensor,
    use_special_tokens: bool = False,
) -> torch.Tensor:
    """Mean-pool the visual token positions from the last hidden state.

    Only tokens after the text prefix (``attention_mask == 1``) are averaged.
    When ``use_special_tokens`` is set, the last three hidden positions are
    used as special reward tokens instead of mean pooling.
    """
    if use_special_tokens:
        # Special-token mode: last 3 positions carry VQ / MQ / TA
        pooled = last_hidden_state[:, -3:, :]  # (B, 3, D)
    else:
        # Mean-pool over visual positions (where attention_mask != 0)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.unsqueeze(1)  # (B, 1, D)
    return pooled


# ============================================================================
# Checkpoint loading (inlined from DanceGRPO training code)
# ============================================================================


def _load_checkpoint(inference_obj, checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> None:
    """Load a VideoAlign checkpoint into an existing _VideoAlignInference object.

    Handles the key remapping needed for Qwen2-VL's composable model
    architecture (transformers >= 4.49 moved vision / language_model
    into sub-modules).
    """
    from safetensors.torch import load_file

    state_dicts = []
    for pattern in ("*.safetensors", "adapter_*.safetensors"):
        for fpath in sorted(glob.glob(os.path.join(checkpoint_dir, pattern))):
            state_dicts.append(load_file(fpath))

    # Merge all safetensor files into a single state dict
    full_state = {}
    for sd in state_dicts:
        full_state.update(sd)

    model_state = inference_obj.model.state_dict()
    filtered_state = full_state.copy()

    # Detect and remap keys for Qwen2-VL's nested architecture
    flat_model_keys = list(model_state.keys())
    flat_filtered_keys = list(filtered_state.keys())

    # Remap ``model.layers`` → ``model.language_model.layers`` (transformers >= 4.49)
    if any("model.language_model" in k for k in flat_model_keys):
        keys_to_remap = [k for k in flat_filtered_keys if k.startswith("model.layers.")]
        for old_key in keys_to_remap:
            new_key = "model.language_model." + old_key[len("model.") :]
            if new_key in flat_model_keys:
                filtered_state[new_key] = filtered_state.pop(old_key)

    # Remap ``visual`` → ``model.visual`` (transformers >= 4.49)
    if any("model.visual" in k for k in flat_model_keys):
        keys_to_remap = [k for k in flat_filtered_keys if k.startswith("visual.")]
        for old_key in keys_to_remap:
            new_key = "model." + old_key
            if new_key in flat_model_keys:
                filtered_state[new_key] = filtered_state.pop(old_key)

    # Log match stats for debugging checkpoint compatibility
    matched = sum(1 for k in filtered_state if k in model_state)
    skipped = len(filtered_state) - matched
    if skipped:
        print(f"[VideoAlign] Checkpoint key remap: {matched} matched, {skipped} skipped", flush=True)

    inference_obj.model.load_state_dict(filtered_state, strict=False)
    inference_obj.model.to(device=device, dtype=dtype)
    inference_obj.model.eval()


# ============================================================================
# Top-level VideoAlign inference class
# ============================================================================


class _VideoAlignInference:
    """Self-contained VideoAlign inference (no fastvideo dependency).

    Builds the Qwen2-VL backbone, MLP heads, loads the checkpoint, and
    provides a ``reward()`` method that returns per-dimension scores.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.model: Optional[_VideoAlignModel] = None
        self.processor = None
        self._inference_cfg: Optional[dict] = None

        # Data config defaults (overridden by model_config.json)
        self._fps: float = 2.0
        self._num_frames: Optional[int] = None
        self._max_pixels: int = 200704
        self._eval_dim: List[str] = ["VQ", "MQ", "TA"]
        self._prompt_template_type: str = "detailed_special"
        self._sample_type: str = "uniform"

        self._load(checkpoint_dir)

    def _load(self, checkpoint_dir: str) -> None:
        """Load model, config, processor, and checkpoint weights."""
        from transformers import AutoConfig, AutoModelForTextToVideo, AutoProcessor

        # ---- Load model config to determine architecture flags ----
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        data_cfg = cfg["data_config"]
        model_cfg = cfg["model_config"]
        self._inference_cfg = cfg.get("inference_config")

        # Data config
        self._fps = data_cfg.get("fps", 2.0)
        self._num_frames = data_cfg.get("num_frames")
        self._max_pixels = data_cfg.get("max_frame_pixels", 200704)
        self._eval_dim = data_cfg.get("eval_dim", ["VQ", "MQ", "TA"])
        self._prompt_template_type = data_cfg.get("prompt_template_type", "detailed_special")
        self._sample_type = data_cfg.get("sample_type", "uniform")

        # Processor. The base Qwen2-VL-2B model location: prefer an explicit
        # env override (so launches from any cwd work), else a sibling of the
        # checkpoint dir, else the original DanceGRPO-relative default.
        qwen_path = os.environ.get("VIDEOALIGN_QWEN_CKPT", "")
        if not qwen_path:
            _sibling = os.path.join(os.path.dirname(checkpoint_dir.rstrip("/")), "Qwen2-VL-2B-Instruct")
            qwen_path = _sibling if os.path.isdir(_sibling) else "./Qwen2-VL-2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(qwen_path, padding_side="right")

        special_token_ids = None
        if model_cfg.get("use_special_tokens", False):
            special_tokens_dict = {"additional_special_tokens": ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]}
            num_added = self.processor.tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                special_token_ids = [
                    self.processor.tokenizer.convert_tokens_to_ids(tok)
                    for tok in ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
                ]
                print(f"[VideoAlign] Added {num_added} special reward tokens; ids={special_token_ids}", flush=True)

        # ---- Build base Qwen2-VL model ----
        qwen_config = AutoConfig.from_pretrained(qwen_path)
        qwen_model = AutoModelForTextToVideo.from_config(qwen_config)

        # Honour the checkpoint's special-token choice
        has_special = model_cfg.get("use_special_tokens", False)
        in_dim = qwen_config.hidden_size
        if special_token_ids is not None:
            qwen_model.resize_token_embeddings(len(self.processor.tokenizer))

        self.model = _VideoAlignModel(
            base_model=qwen_model,
            in_dim=in_dim,
            reg_dim=model_cfg.get("reg_dim", 1024),
            drop_rate=model_cfg.get("drop_rate", 0.2),
            use_special_tokens=has_special,
        )

        _load_checkpoint(self, checkpoint_dir, torch.device(self.device), self.dtype)

    def _norm(self, reward: dict) -> dict:
        if self._inference_cfg is None:
            return reward
        reward["VQ"] = (reward["VQ"] - self._inference_cfg["VQ_mean"]) / (self._inference_cfg["VQ_std"] + 1e-8)
        reward["MQ"] = (reward["MQ"] - self._inference_cfg["MQ_mean"]) / (self._inference_cfg["MQ_std"] + 1e-8)
        reward["TA"] = (reward["TA"] - self._inference_cfg["TA_mean"]) / (self._inference_cfg["TA_std"] + 1e-8)
        return reward

    def reward(
        self, video_paths: List[str], prompts: List[str], fps=None, num_frames=None, max_pixels=None, use_norm=True
    ) -> List[dict]:
        fps = self._fps if fps is None else fps
        num_frames = self._num_frames if num_frames is None else num_frames
        max_pixels = self._max_pixels if max_pixels is None else max_pixels

        # Build chat messages in Qwen2-VL format
        chat_data = []
        for prompt in prompts:
            vid_info: Dict[str, Any] = {
                "type": "video",
                "video": video_paths,
                "max_pixels": max_pixels,
                "sample_type": self._sample_type,
            }
            if num_frames is not None:
                vid_info["nframes"] = num_frames
            else:
                vid_info["fps"] = fps
            text = _build_prompt(prompt, self._eval_dim, self._prompt_template_type)
            chat_data.append([{"role": "user", "content": [vid_info, {"type": "text", "text": text}]}])

        # Process vision
        _, video_inputs = _process_vision_info(chat_data)

        # Tokenize
        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            logits = self.model(return_dict=True, **batch)["logits"]

        rewards = [{"VQ": r[0].item(), "MQ": r[1].item(), "TA": r[2].item()} for r in logits]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]["Overall"] = rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]
        return rewards


# ============================================================================
# DiffusionRL Reward Scorer Interface
# ============================================================================


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
            checkpoint_path=config.checkpoint_path,
        )
        self._vq_coef = float(getattr(config, "vq_coef", 1.0))
        self._mq_coef = float(getattr(config, "mq_coef", 1.0))
        self._ta_coef = float(getattr(config, "ta_coef", 1.0))

    def _load_model(self) -> None:
        # Ensure decord is used (torchvision.io.read_video removed in newer versions)
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
        checkpoint_path = self.model_kwargs.get("checkpoint_path") or os.environ.get("VIDEOALIGN_CKPT")
        if not checkpoint_path:
            raise ValueError("VIDEOALIGN_CKPT env var or config.checkpoint_path must be set")
        self._inferencer = _VideoAlignInference(checkpoint_path, device=str(self.device), dtype=self.dtype)
        self.model = self._inferencer.model

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        if not self._is_loaded:
            return RewardResponse(
                rewards=[0.0] * request.batch_size,
                successes=[False] * request.batch_size,
                errors=["Model not loaded"] * request.batch_size,
                compute_time=0.0,
            )
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
            self._offload_model()

    def _offload_model(self) -> None:
        """Move model to CPU to free GPU memory for training."""
        if self.model is not None:
            self.model.cpu()
        torch.cuda.empty_cache()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        rewards, _ = self._compute_videoalign_rewards(request)
        return rewards

    def _compute_videoalign_rewards(self, request: RewardRequest) -> Tuple[List[float], Dict[str, List[float]]]:
        if self._inferencer is None:
            raise RuntimeError("VideoAlign model not loaded; call onload() first.")

        # Ensure model is on GPU
        if self.model is not None and next(self.model.parameters()).device.type != "cuda":
            self.model.to(self.device)

        all_rewards: List[float] = []
        components: Dict[str, List[float]] = {"vq": [], "mq": [], "ta": [], "overall": []}
        prompts = list(request.prompts)

        with tempfile.TemporaryDirectory(prefix="unirl_videoalign_") as tmpdir:
            video_paths = []
            for idx, video in enumerate(request.videos):
                path = os.path.join(tmpdir, f"sample_{idx:05d}.mp4")
                _export_tensor_video(video, path)
                video_paths.append(path)

            for start in range(0, len(video_paths), self.batch_size):
                batch_paths = video_paths[start : start + self.batch_size]
                batch_prompts = prompts[start : start + self.batch_size]
                results = self._inferencer.reward(batch_paths, batch_prompts, use_norm=True)
                for item in results:
                    vq, mq, ta = float(item["VQ"]), float(item["MQ"]), float(item["TA"])
                    components["vq"].append(vq)
                    components["mq"].append(mq)
                    components["ta"].append(ta)
                    components["overall"].append(float(item["Overall"]))
                    all_rewards.append(self._vq_coef * vq + self._mq_coef * mq + self._ta_coef * ta)

        return all_rewards, components


def _export_tensor_video(video: torch.Tensor, path: str) -> None:
    """Write a decoded video tensor to mp4.

    Accepts the tensor in either [T, C, H, W] (canonical ``Video.frames``) or
    [C, T, H, W] (the layout produced by ``RewardRequest.videos``) and converts
    to [T, H, W, C] for export.
    """
    from diffusers.utils import export_to_video

    video = video.detach().cpu()
    if video.dim() == 5:
        video = video.squeeze(0)
    if video.dim() != 4:
        raise ValueError(f"Expected 4D video tensor, got shape={tuple(video.shape)}")

    # Normalize to [T, C, H, W]. ``RewardRequest.videos`` hands us [C, T, H, W]
    # (channel-first), while a raw ``Video.frames`` is already [T, C, H, W].
    # Disambiguate by the channel axis: the C dimension is the one of size 3.
    c0, t0 = video.shape[0], video.shape[1]
    if c0 == 3 and t0 != 3:
        # [C, T, H, W] -> [T, C, H, W]
        video = video.permute(1, 0, 2, 3)

    # Convert [T, C, H, W] → [T, H, W, C]
    video = video.permute(0, 2, 3, 1)
    video = video[..., :3].clamp(0.0, 1.0)
    frames = (video * 255).round().to(torch.uint8).numpy()
    export_to_video(list(frames), path, fps=24)


# ============================================================================
# Config registration
# ============================================================================


@dataclass
class VideoAlignSpec(BaseRewardComponentSpec):
    """Typed config for the VideoAlign reward component."""

    weight: float = 1.0
    batch_size: int = 1
    device: str = "auto"
    checkpoint_path: str = ""
    dtype: str = "bf16"
    vq_coef: float = 1.0
    mq_coef: float = 1.0
    ta_coef: float = 1.0
