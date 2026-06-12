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
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_ar_params

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline

# The upstream tokenizer's apply_chat_template asserts
# ``bot_task in {"image", "auto", "think", "recaption", "img_ratio"}`` —
# the composite presets must be mapped before any template call. Mirrors
# vllm-omni ``pipeline_hunyuan_image3.py:1461-1465``. The ORIGINAL value
# still drives stop-token selection and the params record.
_TOKENIZER_BOT_TASKS = {"think_recaption": "think", "vanilla": "image"}


def _tokenizer_bot_task(bot_task: str) -> str:
    return _TOKENIZER_BOT_TASKS.get(bot_task, bot_task)


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """t2t — single AR-stage rollout, no diffusion."""
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (t2t): "
            f"req.primitives['text'] must be Texts, "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )

    # Build HunyuanImage3ARParams from typed sampling params + model-specific stage_config.
    ar = get_ar_params(req.sampling_params)
    model_cfg: Dict[str, Any] = dict(req.stage_config.get("ar") or {})
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

    # Resolve the system prompt. Mirrors upstream's
    # ``HunyuanImage3ForCausalMM.generate_image`` flow: per-bot_task
    # defaults under ``use_system_prompt='dynamic'``, an explicit string
    # under ``use_system_prompt='custom'``, or one of the named presets.
    # Uses the MAPPED bot_task — upstream get_system_prompt's ``dynamic``
    # branch only knows {think, recaption, image}.
    system_prompt = _resolve_system_prompt(
        pipeline.bundle, tok_bot_task, ar_params.use_system_prompt, ar_params.system_prompt
    )
    system_prompt_list = [system_prompt] * len(texts.texts) if system_prompt is not None else None

    # Build the unified-multimodal tensors via the chat-template wrapper.
    # ``mm`` is ``{"fused": HunyuanImage3FusedMultimodalCondition, "tokenizer_output": Any}``.
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

    # Resolve stop tokens. Caller-supplied ``stop_token_ids`` wins; else
    # we derive from ``bot_task`` against the bundle's tokenizer wrapper.
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

    # text_seg.tokens: packed varlen [sum_lengths] long
    # text_seg.cu_seqlens: [B+1] long
    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=sampling_params, params=ar_params_with_stops)

    # Detokenize back to Texts for downstream reward / display consumption.
    decoded_texts = pipeline._detokenize_text_segment(text_seg)

    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=ar_conds.to_dict(),
                segment=text_seg,
                decoded=decoded_texts,
            ),
        }
    )


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

    # Resolve upstream's ``system_prompt`` module via a sibling import on
    # the transformer's own module path. With ``trust_remote_code=True``
    # the transformer lives under e.g. ``transformers_modules.<ckpt>.hunyuan``;
    # the system_prompt.py is at ``transformers_modules.<ckpt>.system_prompt``.
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
        # Bundle hasn't had its tokenizer loaded yet (fake-bundle path
        # or pre-prefill). Return empty -- ``autoregress`` then runs
        # to ``max_tokens`` without an early stop.
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
    # Unknown bot_task -- fall back to eos.
    return [int(eos)] if eos is not None else []
