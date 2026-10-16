"""t2t — text-to-text autoregressive generation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from unirl.models.types.ar import ARSamplingParams
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline

_TOKENIZER_BOT_TASKS = {"think_recaption": "think", "vanilla": "image"}


def _tokenizer_bot_task(bot_task: str) -> str:
    return _TOKENIZER_BOT_TASKS.get(bot_task, bot_task)


def _as_sample_batches(value: Any) -> List[Any]:
    """Split a batched condition tensor along dim 0 into per-sample 1-row tensors."""
    return list(value.split(1, dim=0)) if isinstance(value, torch.Tensor) else list(value)


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """t2t — single AR-stage rollout, no diffusion."""
    frontier = sample.frontier_gen_part(ARSamplingParams)
    ar = frontier.sampling_params

    conditioning = sample.conditioning()
    texts = conditioning[0] if conditioning else None
    if not isinstance(texts, Texts):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (t2t): "
            f"prompt from sample.conditioning()[0] must be Texts, "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )

    ar_cfg = sample.parts[0].control.get("ar") or {}
    bot_task = str(ar_cfg.get("bot_task", "auto"))
    tok_bot_task = _tokenizer_bot_task(bot_task)
    cot_text = ar_cfg.get("cot_text")

    system_prompt = _resolve_system_prompt(
        pipeline.bundle, tok_bot_task, ar_cfg.get("use_system_prompt"), ar_cfg.get("system_prompt")
    )
    n_prompts = len(texts.texts)
    system_prompt_list = [system_prompt] * n_prompts if system_prompt is not None else None

    mm = pipeline.text_embed.embed_for_ar(
        texts,
        bot_task=tok_bot_task,
        system_prompt=system_prompt_list,
        cot_text=[cot_text] * n_prompts if cot_text else None,
    )

    ar_conds = HunyuanImage3ARConditions(
        fused=mm["fused"],
        tokenizer_output=mm["tokenizer_output"],
    )

    stop_ids: List[int] = list(ar_cfg.get("stop_token_ids") or []) or _stop_tokens_for_bot_task(
        pipeline.bundle, bot_task
    )
    sampling_params = ARSamplingParams(
        max_new_tokens=ar.max_new_tokens,
        temperature=ar.temperature,
        top_p=ar.top_p,
        top_k=ar.top_k,
        stop_token_id=stop_ids[0] if stop_ids else None,
    )
    text_seg = pipeline.ar.autoregress(
        ar_conds,
        sampling_params=sampling_params,
        params=HunyuanImage3ARParams(stop_token_ids=stop_ids),
    )

    decoded_texts = pipeline._detokenize_text_segment(text_seg)

    filled = frontier.fill(segment=text_seg, primitives={"text": decoded_texts}, conditions=ar_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])


def _resolve_system_prompt(
    bundle, bot_task: str, use_system_prompt: Optional[str], system_prompt: Optional[str]
) -> Optional[str]:
    """Mirror upstream ``get_system_prompt(sys_type, bot_task, system_prompt)``."""
    transformer = bundle.transformer
    sys_type = (
        use_system_prompt
        if use_system_prompt is not None
        else getattr(transformer.generation_config, "use_system_prompt", None)
    )
    return sys.modules[type(transformer).__module__].get_system_prompt(sys_type, bot_task, system_prompt)


def _stop_tokens_for_bot_task(bundle, bot_task: str) -> List[int]:
    """Mirror upstream's stop-token dict across Base and Instruct tokenizers."""
    transformer = bundle.transformer
    tkw = getattr(transformer, "_tkwrapper", None) or getattr(transformer, "_tokenizer", None)
    if tkw is None:
        return []

    eos = getattr(tkw, "eos_token_id", None)
    boi = getattr(tkw, "boi_token_id", None)
    special_map = getattr(tkw, "special_token_map", None) or getattr(tkw, "added_tokens_encoder", None) or {}
    end_recap = special_map.get("</recaption>")
    end_answer = special_map.get("</answer>")
    end_think = special_map.get("</think>")

    extra_auto_stops = [int(tid) for i in range(33) if (tid := special_map.get(f"<img_ratio_{i}>")) is not None]

    if bot_task == "auto":
        return ([int(eos)] if eos is not None else []) + extra_auto_stops
    if bot_task == "image":
        return [int(eos)] if eos is not None else []
    if bot_task in ("recaption", "think", "think_recaption"):
        out: List[int] = []
        if bot_task == "think" and end_think is not None:
            out.append(int(end_think))
        if end_recap is not None:
            out.append(int(end_recap))
        if end_answer is not None:
            out.append(int(end_answer))
        if eos is not None:
            out.append(int(eos))
        return out
    if bot_task == "img_ratio":
        if extra_auto_stops:
            return extra_auto_stops
        return [int(boi)] if boi is not None else []
    return [int(eos)] if eos is not None else []
