"""Shared helpers for scorer implementations.

Keeps torch dtype mapping, weights path resolution, turn-splitting, and
data-URL image encoding in one place so individual scorers stay focused
on their model wiring.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from PIL import Image

    from reward_service.scorers.base import ScoreItem


DEFAULT_VLLM_MM_LIMIT: dict[str, int] = {"image": 1}


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Translate a dtype string (`float32` / `float16` / `bfloat16`) into a torch dtype."""
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype: {name!r}. expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def resolve_model_path(model_name: str, weights_path: str | None) -> str:
    """Return weights_path if given, else fall back to model_name (HF hub id)."""
    return weights_path if weights_path else model_name


def split_last_turn(items: list["ScoreItem"]) -> tuple[list[str], list["Image.Image"]]:
    """Pull (text, image) from the last history turn of each item, returned as parallel lists."""
    texts: list[str] = []
    images: list = []
    for item in items:
        text, image = item.history[-1]
        texts.append(text)
        images.append(image)
    return texts, images


def image_to_data_url(image: "Image.Image", format: str = "JPEG", quality: int = 95) -> str:
    """Serialise a PIL image to a `data:image/...;base64,...` URL suitable for vLLM chat."""
    image = image.convert("RGB") if image.mode != "RGB" else image
    buf = io.BytesIO()
    save_kwargs: dict = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def build_vllm_llm_kwargs(
    model: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
    dtype: str = "bfloat16",
    enforce_eager: bool = False,
    swap_space: int = 4,
    quantization: str | None = None,
    seed: int | None = None,
    max_num_seqs: int = 256,
    trust_remote_code: bool = True,
    limit_mm_per_prompt: dict[str, int] | None = None,
    extra_llm_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Assemble the kwargs dict for `vllm.LLM(**kwargs)`.

    All named fields below are always included in the result (unless
    their value is `None` and they are optional, like `max_model_len`).
    `extra_llm_kwargs` is merged last — last-writer-wins — so callers
    can override any named field or inject any other vLLM option from
    YAML. Collisions happen silently by design.
    """
    kwargs: dict = {
        "model": model,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "swap_space": swap_space,
        "max_num_seqs": max_num_seqs,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if quantization is not None:
        kwargs["quantization"] = quantization
    if seed is not None:
        kwargs["seed"] = seed
    if limit_mm_per_prompt is not None:
        kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
    if extra_llm_kwargs is not None:
        kwargs.update(extra_llm_kwargs)
    return kwargs
