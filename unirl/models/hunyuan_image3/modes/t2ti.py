"""t2ti — text → CoT text + image (the HunyuanImage3 think_recaption chain).

Two phases in one request:

1. **AR phase** (like t2t): generates ``<think>…</think><recaption>…
   </recaption>`` chain-of-thought text under the ``en_think_recaption``
   system prompt, stopping at the CoT end markers.
2. **Diffusion phase** (like t2i): conditions on prompt + the truncated /
   normalized CoT via ``embed_for_gen_image(cot_text=...)``, then
   diffuses and VAE-decodes the image.

Mirrors vllm-omni's two-stage serving chain (AR stage →
``stage_input_processors/hunyuan_image3.py`` ar2diffusion bridge → DiT
stage). Fidelity caveat: upstream forces ``</think> → <recaption>`` via
stage-transition logits processing; this mode relies on natural sampling
under the system prompt, so the model may occasionally skip the
recaption block (the CoT then degrades to think-only or plain text —
upstream's own no-marker fallback feeds it as a plain text section).

Fills TWO generated Parts in one lineage: the AR Part carries the truncated +
normalized CoT that actually conditioned the image (raw tokens stay in its
``segment`` for replay), followed by the diffusion Part carrying the image.
``samples_per_prompt`` on either sub-params is deliberately NOT honored:
fan-out belongs to the engine adapter, as with the other HI3 modes.

img_ratio auto-prediction (upstream lets the AR pass pick the aspect
ratio) is out of scope — height/width come from the diffusion sampling
params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from unirl.config.require import require
from unirl.models.types.ar import ARSamplingParams
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions, HunyuanImage3DiffusionConditions
from .t2t import _resolve_system_prompt, _stop_tokens_for_bot_task, _tokenizer_bot_task

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def _truncate_at_cot_end(text: str) -> str:
    """Cut the AR output at the first ``</recaption>`` (else ``</think>``).

    Keeps the marker; drops the trailing ``<answer><boi>…`` tail that
    must not leak into the diffusion prompt builder. Port of vllm-omni
    ``stage_input_processors/hunyuan_image3.py:105-117``.
    """
    for marker in ("</recaption>", "</think>"):
        idx = text.find(marker)
        if idx != -1:
            return text[: idx + len(marker)]
    return text


def _normalize_cot_text(cot: str) -> str:
    """Re-add the opening CoT tag the AR trigger consumed.

    AR generation may omit the leading ``<think>`` / ``<recaption>`` (it
    was spliced as the generation trigger); the wrapper's section parsing
    needs matched tag pairs. Port of vllm-omni
    ``pipeline_hunyuan_image3.py:738-755``.
    """
    if not cot:
        return cot
    if "</think>" in cot and not cot.startswith("<think>"):
        return "<think>" + cot
    if "</recaption>" in cot and not cot.startswith("<recaption>"):
        return "<recaption>" + cot
    return cot


def _cot_stop_tokens(bundle, bot_task: str) -> List[int]:
    """Stop tokens for the CoT AR pass with an explicit image size.

    Mirrors vllm-omni ``prompt_utils.resolve_stop_token_ids`` (explicit-
    size branch): think_recaption / recaption stop at ``</recaption>``;
    ``think`` additionally needs ``</think>`` — prepended here since
    ``_stop_tokens_for_bot_task``'s think-family list omits it. The
    inherited ``</answer>`` / eos entries stay as runaway safety nets.
    Empty on fake bundles (no tokenizer wrapper).
    """
    stop_ids = _stop_tokens_for_bot_task(bundle, bot_task)
    if bot_task == "think":
        tkw = getattr(bundle.transformer, "_tkwrapper", None) or getattr(bundle.transformer, "_tokenizer", None)
        end_think = getattr(tkw, "end_think_token_id", None) if tkw is not None else None
        if end_think is None and tkw is not None:
            end_think = (getattr(tkw, "special_token_map", {}) or {}).get("</think>")
        if end_think is not None and int(end_think) not in stop_ids:
            stop_ids = [int(end_think)] + stop_ids
    return stop_ids


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """t2ti — AR CoT phase, then diffusion conditioned on the CoT."""
    if len(sample.parts) < 2:
        raise ValueError("HunyuanImage3Pipeline.generate (t2ti): expected trailing [AR, diffusion] Parts")
    ar_idx = len(sample.parts) - 2
    image_idx = len(sample.parts) - 1
    ar_part = sample.parts[ar_idx]
    image_part = sample.parts[image_idx]
    require(
        isinstance(ar_part.sampling_params, ARSamplingParams)
        and isinstance(image_part.sampling_params, DiffusionSamplingParams),
        "HunyuanImage3Pipeline.generate (t2ti): current trailing Parts must carry "
        f"[ARSamplingParams, DiffusionSamplingParams], got "
        f"[{type(ar_part.sampling_params).__name__}, {type(image_part.sampling_params).__name__}].",
    )
    ar_sp = ar_part.sampling_params
    diff_sp = image_part.sampling_params
    require(
        isinstance(diff_sp, DiffusionSamplingParams),
        "HunyuanImage3Pipeline.generate (t2ti): the diffusion gen Part must carry DiffusionSamplingParams.",
    )
    if diff_sp.sigmas is None:
        raise ValueError(
            "HunyuanImage3 t2ti: diffusion gen part sigmas is None. The hosting engine must "
            "pin σ before pipeline.generate."
        )

    ar_texts = [value for value in sample.conditioning_at(ar_idx) if isinstance(value, Texts)]
    require(
        len(ar_texts) == 1,
        "HunyuanImage3Pipeline.generate (t2ti): expected exactly one Texts input for the "
        f"AR frontier, got {len(ar_texts)}",
    )
    texts = ar_texts[0]
    require(
        len(texts.texts) == len(ar_part.sample_ids),
        f"HunyuanImage3Pipeline.generate (t2ti): AR-aligned prompt count {len(texts.texts)} "
        f"!= AR sample count {len(ar_part.sample_ids)}",
    )

    control = sample.parts[0].control or {}
    ar_cfg: Dict[str, Any] = dict(control.get("ar") or {})
    require(
        "bot_task" not in ar_cfg,
        "HunyuanImage3Pipeline.generate (t2ti): set the single top-level control['bot_task'] "
        "(the chain is one semantic mode); control['ar']['bot_task'] is not read.",
    )
    bot_task = str(control.get("bot_task", "think_recaption"))
    tok_bot_task = _tokenizer_bot_task(bot_task)
    batch = len(texts.texts)

    use_sp = ar_cfg.get("use_system_prompt")
    if use_sp is None and bot_task == "think_recaption":
        use_sp = "en_think_recaption"
    system_prompt = _resolve_system_prompt(pipeline.bundle, tok_bot_task, use_sp, ar_cfg.get("system_prompt"))
    system_prompt_list = [system_prompt] * batch if system_prompt is not None else None

    mm = pipeline.text_embed.embed_for_ar(
        texts,
        bot_task=tok_bot_task,
        system_prompt=system_prompt_list,
    )
    ar_conds = HunyuanImage3ARConditions(
        fused=mm["fused"],
        tokenizer_output=mm["tokenizer_output"],
    )

    stop_ids: List[int] = list(ar_cfg.get("stop_token_ids") or [])
    if not stop_ids:
        stop_ids = _cot_stop_tokens(pipeline.bundle, bot_task)
    ar_sampling = ARSamplingParams(
        max_new_tokens=int(ar_sp.max_new_tokens),
        temperature=float(ar_sp.temperature),
        top_p=float(ar_sp.top_p),
        top_k=int(ar_sp.top_k),
        stop_token_id=stop_ids[0] if stop_ids else None,
    )
    ar_params = HunyuanImage3ARParams(
        bot_task=bot_task,
        max_tokens=int(ar_sp.max_new_tokens),
        temperature=float(ar_sp.temperature),
        top_p=float(ar_sp.top_p),
        top_k=int(ar_sp.top_k),
        stop_token_ids=stop_ids,
        system_prompt=ar_cfg.get("system_prompt"),
        use_system_prompt=use_sp,
        taylor_cache_interval=ar_cfg.get("taylor_cache_interval"),
        taylor_cache_order=ar_cfg.get("taylor_cache_order"),
    )
    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=ar_sampling, params=ar_params)

    raw = pipeline._detokenize_text_segment(text_seg, skip_special_tokens=False)
    cots = [_normalize_cot_text(_truncate_at_cot_end(t)) for t in raw.texts]

    new_parts = list(sample.parts)
    new_parts[ar_idx] = ar_part.fill(
        segment=text_seg, primitives={"text": Texts(texts=cots)}, conditions=ar_conds.to_dict()
    )
    partially_filled = sample.with_parts(new_parts)

    image_texts = [value for value in partially_filled.conditioning() if isinstance(value, Texts)]
    require(
        len(image_texts) >= 2,
        "HunyuanImage3Pipeline.generate (t2ti): image frontier did not surface both the "
        "original prompt and the generated CoT",
    )
    image_prompts = image_texts[0]
    image_cots = list(image_texts[-1].texts)
    n_images = len(image_part.sample_ids)
    require(
        len(image_prompts.texts) == n_images and len(image_cots) == n_images,
        f"HunyuanImage3Pipeline.generate (t2ti): image-aligned conditioning counts "
        f"prompt={len(image_prompts.texts)}, cot={len(image_cots)}, expected={n_images}",
    )
    image_system_prompts = [system_prompt] * n_images if system_prompt is not None else None

    schedule = diff_sp.sigmas.to(pipeline.bundle.device)

    mm2 = pipeline.text_embed.embed_for_gen_image(
        image_prompts,
        cfg=float(diff_sp.guidance_scale) > 1.0,
        height=int(diff_sp.height),
        width=int(diff_sp.width),
        bot_task=tok_bot_task,
        cot_text=image_cots,
        system_prompt=image_system_prompts,
    )
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm2["fused"],
        tokenizer_output=mm2["tokenizer_output"],
    )

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=diff_sp)
    images = pipeline.vae_decode.decode(latent_seg)

    new_parts[image_idx] = image_part.fill(
        segment=latent_seg, primitives={"image": images}, conditions=diff_conds.to_dict()
    )
    return sample.with_parts(new_parts)
