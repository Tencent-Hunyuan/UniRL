"""Request-side prompt helpers the adapters' ``build_inputs`` steps call.

Pure: no vllm-omni import — tokenization reaches the runtime through the
injected ``tokenize_fn`` (the seam's ``tokenize_prompt`` verb; tests pass a
lambda). The HI3 task mapping mirrors upstream ``_TASK_PRESETS``; the
per-prompt dict shape is the official ``end2end.py`` reference (see the
``ar_dit`` adapter docstring).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import PIL.Image

from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq

# (default_task_key, default_sys_type, modalities) per modality.
_TASK_DEFAULTS: Dict[str, Tuple[str, str, List[str]]] = {
    "t2i": ("t2i_think", "en_unified", ["image"]),
    "it2i": ("it2i_think", "en_unified", ["image"]),
    "i2t": ("i2t", "en_unified", ["text"]),
    "t2t": ("t2t", "en_unified", ["text"]),
    # Two-engine v2 trainer. ``ar_recaption`` builds a think/recaption prompt
    # (task ``t2i_think`` → AR emits <think>…</think><recaption>…) but is served
    # by an AR-only stage. ``dit_recaption`` only reads sys_type from here for
    # use_system_prompt.
    "ar_recaption": ("t2i_think", "en_unified", ["image"]),
    "dit_recaption": ("t2i_think", "en_unified", ["image"]),
}


def resolve_task(modality: str, stage_config: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Resolve ``(task_key, sys_type, modalities)`` with optional overrides.

    ``stage_config["bot_task"]`` swaps the trigger tag used by upstream's
    chat template (``think`` / ``recaption``). ``stage_config["sys_type"]``
    overrides the system-prompt key (``en_unified`` / ``en_vanilla``).
    """
    if modality not in _TASK_DEFAULTS:
        raise ValueError(f"resolve_task: unsupported modality {modality!r}. Choose one of {list(_TASK_DEFAULTS)}.")
    default_task, default_sys, modalities = _TASK_DEFAULTS[modality]

    sys_type = stage_config.get("sys_type") or default_sys

    bot_task = stage_config.get("bot_task")
    if bot_task and modality in ("t2i", "it2i"):
        # think / recaption / vanilla — translate to upstream task key.
        if bot_task == "vanilla" and modality == "t2i":
            return "t2i_vanilla", "en_vanilla", modalities
        if bot_task in ("think", "recaption"):
            return f"{modality}_{bot_task}", sys_type, modalities

    return default_task, sys_type, modalities


def texts_from_req(req: RolloutReq) -> Texts:
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            f"req.primitives['text'] must be Texts, got {type(texts).__name__ if texts is not None else 'None'}"
        )
    if len(texts.texts) != len(req.sample_ids):
        raise ValueError(f"prompt count {len(texts.texts)} != sample_ids count {len(req.sample_ids)}")
    return texts


def pil_images_from_req(req: RolloutReq, n: int) -> List[PIL.Image.Image]:
    """Extract ``req.primitives['image']`` (Images) as a list of PIL images.

    Returns an empty list when there's no image primitive. Asserts batch
    alignment when present; the conversion itself is :meth:`Images.to_pils`.
    """
    images = req.primitives.get("image")
    if images is None:
        return []
    if not isinstance(images, Images):
        raise TypeError(f"req.primitives['image'] must be Images when present, got {type(images).__name__}")
    if len(images) != n:
        raise ValueError(f"image batch {len(images)} != prompt count {n}")
    return images.to_pils()


def build_prompt_entries(
    texts: Texts,
    *,
    task: str,
    sys_type: str,
    modalities_field: List[str],
    tokenize_fn: Optional[Callable[..., List[int]]],
    decorate: Callable[[Dict[str, Any], int], None],
) -> List[Dict[str, Any]]:
    """Build the HI3 per-prompt dicts shared by the AR-bearing modalities.

    Each entry carries the official ``end2end.py`` base fields
    (``prompt_token_ids`` / ``prompt`` / ``use_system_prompt`` /
    ``modalities``); the adapter's ``decorate`` callback then attaches its
    modality-specific extras (``multi_modal_data``, ``height`` / ``width``).
    """
    if tokenize_fn is None:
        raise RuntimeError("build_prompt_entries: tokenize_fn not provided (AR modalities need the driver tokenizer)")
    prompts: List[Dict[str, Any]] = []
    for i, text in enumerate(texts.texts):
        token_ids = tokenize_fn(text, task=task, sys_type=sys_type)
        entry: Dict[str, Any] = {
            "prompt_token_ids": token_ids,
            "prompt": text,
            "use_system_prompt": sys_type,
            "modalities": list(modalities_field),
        }
        decorate(entry, i)
        prompts.append(entry)
    return prompts


__all__ = [
    "build_prompt_entries",
    "pil_images_from_req",
    "resolve_task",
    "texts_from_req",
]
