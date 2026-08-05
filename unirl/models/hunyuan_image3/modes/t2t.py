"""t2t — text-to-text autoregressive generation.

Reads ``primitives["text"]: Texts`` and ``stage_params["ar"]: dict``
(optional). Builds the chat-templated input tensors via
``HunyuanImage3TextEmbedStage.embed_for_ar(...)`` (mode="gen_text"),
then runs ``HunyuanImage3ARStage.autoregress`` against the backbone in
``mode="gen_text"`` and detokenizes the resulting ``TextSegment`` back
into a ``Texts`` primitive on the response.

The bot_task knob (``"auto"`` / ``"image"`` / ``"think"`` /
``"recaption"`` / ``"think_recaption"`` / ``"img_ratio"``) drives both
chat-template splicing (in ``embed_for_ar``) and stop-token selection
(via ``_stop_tokens_for_bot_task``). Stop-token sets mirror upstream
``pipeline_hunyuan_image3.py:627-632``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline

_TOKENIZER_BOT_TASKS = {"think_recaption": "think", "vanilla": "image"}


def _tokenizer_bot_task(bot_task: str) -> str:
    return _TOKENIZER_BOT_TASKS.get(bot_task, bot_task)


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

    model_cfg: Dict[str, Any] = dict((sample.parts[0].control or {}).get("ar") or {})
    ar_params = HunyuanImage3ARParams(
        max_tokens=ar.max_new_tokens if ar is not None else model_cfg.get("max_tokens", 2048),
        temperature=ar.temperature if ar is not None else model_cfg.get("temperature", 0.6),
        top_p=ar.top_p if ar is not None else model_cfg.get("top_p", 0.95),
        top_k=ar.top_k if ar is not None else model_cfg.get("top_k", 1024),
        bot_task=model_cfg.get("bot_task", "auto"),
        cot_text=model_cfg.get("cot_text"),
        system_prompt=model_cfg.get("system_prompt"),
        use_system_prompt=model_cfg.get("use_system_prompt"),
        stop_token_ids=model_cfg.get("stop_token_ids", []),
        taylor_cache_interval=model_cfg.get("taylor_cache_interval"),
        taylor_cache_order=model_cfg.get("taylor_cache_order"),
    )
    bot_task = str(ar_params.bot_task)
    tok_bot_task = _tokenizer_bot_task(bot_task)

    system_prompt = _resolve_system_prompt(
        pipeline.bundle, tok_bot_task, ar_params.use_system_prompt, ar_params.system_prompt
    )
    system_prompt_list = [system_prompt] * len(texts.texts) if system_prompt is not None else None

    mm = pipeline.text_embed.embed_for_ar(
        texts,
        bot_task=tok_bot_task,
        system_prompt=system_prompt_list,
        cot_text=([ar_params.cot_text] * len(texts.texts) if ar_params.cot_text else None),
    )

    ar_conds = HunyuanImage3ARConditions(
        fused=mm["fused"],
        tokenizer_output=mm["tokenizer_output"],
    )

    stop_ids: List[int] = list(ar_params.stop_token_ids or [])
    if not stop_ids:
        stop_ids = _stop_tokens_for_bot_task(pipeline.bundle, bot_task)
    sampling_params = ARSamplingParams(
        max_new_tokens=int(ar_params.max_tokens),
        temperature=float(ar_params.temperature),
        top_p=float(ar_params.top_p),
        top_k=int(ar_params.top_k),
        stop_token_id=stop_ids[0] if stop_ids else None,
    )
    ar_params_with_stops = HunyuanImage3ARParams(
        bot_task=ar_params.bot_task,
        max_tokens=ar_params.max_tokens,
        temperature=ar_params.temperature,
        top_p=ar_params.top_p,
        top_k=ar_params.top_k,
        stop_token_ids=stop_ids,
        cot_text=ar_params.cot_text,
        system_prompt=ar_params.system_prompt,
        use_system_prompt=ar_params.use_system_prompt,
        taylor_cache_interval=ar_params.taylor_cache_interval,
        taylor_cache_order=ar_params.taylor_cache_order,
    )

    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=sampling_params, params=ar_params_with_stops)

    decoded_texts = pipeline._detokenize_text_segment(text_seg)

    filled = frontier.fill(segment=text_seg, primitives={"text": decoded_texts}, conditions=ar_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])


def _resolve_system_prompt(
    bundle, bot_task: str, use_system_prompt: Optional[str], system_prompt: Optional[str]
) -> Optional[str]:
    """Mirror upstream ``get_system_prompt(sys_type, bot_task, system_prompt)``.

    Reads ``use_system_prompt`` from the request (or falls back to the
    bundle's gen_config default). ``custom`` -> use explicit
    ``system_prompt`` arg. ``dynamic`` -> per-bot_task preset.
    Named presets (``en_vanilla`` / ``en_recaption`` / ``en_think_recaption``)
    -> static lookup. ``None`` -> no system prompt.
    """
    import importlib
    import sys

    transformer = bundle.transformer
    gen_config = getattr(transformer, "generation_config", None)
    sys_type = use_system_prompt
    if sys_type is None and gen_config is not None:
        sys_type = getattr(gen_config, "use_system_prompt", None)

    try:
        transformer_mod = sys.modules[type(transformer).__module__]
        package = transformer_mod.__package__ or transformer_mod.__name__.rsplit(".", 1)[0]
        sp_mod = importlib.import_module(f"{package}.system_prompt")
        return sp_mod.get_system_prompt(sys_type, bot_task, system_prompt)
    except (AttributeError, ImportError, KeyError):
        logger.debug("Could not resolve upstream HunyuanImage3 system_prompt module.", exc_info=True)
        return system_prompt


def _stop_tokens_for_bot_task(bundle, bot_task: str) -> List[int]:
    """Mirror upstream's stop-token dict at
    ``vllm-omni/.../pipeline_hunyuan_image3.py:627-632``.

    Falls back to an empty list when the bundle has no usable tokenizer
    wrapper (e.g. fake-bundle unit tests). Callers may seed
    ``ar_params.stop_token_ids`` to override.
    """
    transformer = bundle.transformer
    tkw = getattr(transformer, "_tkwrapper", None) or getattr(transformer, "_tokenizer", None)
    if tkw is None:
        return []

    eos = getattr(tkw, "eos_token_id", None)
    boi = getattr(tkw, "boi_token_id", None)
    end_recap = getattr(tkw, "end_recaption_token_id", None)
    end_answer = getattr(tkw, "end_answer_token_id", None)
    special_map = getattr(tkw, "special_token_map", {}) or {}

    extra_auto_stops: List[int] = []
    for i in range(33):
        tid = special_map.get(f"<img_ratio_{i}>")
        if tid is not None:
            extra_auto_stops.append(int(tid))

    if bot_task == "auto":
        return ([int(eos)] if eos is not None else []) + extra_auto_stops
    if bot_task == "image":
        return [int(eos)] if eos is not None else []
    if bot_task in ("recaption", "think", "think_recaption"):
        out: List[int] = []
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
