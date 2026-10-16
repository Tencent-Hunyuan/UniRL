"""t2ti — text → CoT text + image (the HunyuanImage3 think_recaption chain)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from unirl.config.require import require
from unirl.models.types.ar import ARSamplingParams
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3ARConditions, HunyuanImage3DiffusionConditions, split_cfg_fused
from .t2t import _resolve_system_prompt, _stop_tokens_for_bot_task, _tokenizer_bot_task

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


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

    control = sample.parts[0].control
    ar_cfg = control.get("ar", {})
    require(
        "bot_task" not in ar_cfg,
        "HunyuanImage3Pipeline.generate (t2ti): set the single top-level control['bot_task'] "
        "(the chain is one semantic mode); control['ar']['bot_task'] is not read.",
    )
    bot_task = control.get("bot_task", "think_recaption")
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

    stop_ids = ar_cfg.get("stop_token_ids") or _stop_tokens_for_bot_task(pipeline.bundle, bot_task)
    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=ar_sp, stop_token_ids=stop_ids)

    raw = pipeline._detokenize_text_segment(text_seg, skip_special_tokens=False)
    cots: List[str] = []
    for text in raw.texts:
        for marker in ("</recaption>", "</think>"):
            idx = text.find(marker)
            if idx != -1:
                text = text[: idx + len(marker)]
                break
        if text:
            if "</think>" in text and not text.startswith("<think>"):
                text = "<think>" + text
            elif "</recaption>" in text and not text.startswith("<recaption>"):
                text = "<recaption>" + text
        cots.append(text)

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
    image_cots = image_texts[-1].texts
    n_images = len(image_part.sample_ids)
    require(
        len(image_prompts.texts) == n_images and len(image_cots) == n_images,
        f"HunyuanImage3Pipeline.generate (t2ti): image-aligned conditioning counts "
        f"prompt={len(image_prompts.texts)}, cot={len(image_cots)}, expected={n_images}",
    )
    image_system_prompts = [system_prompt] * n_images if system_prompt is not None else None

    schedule = diff_sp.sigmas.to(pipeline.bundle.device)

    cfg = diff_sp.guidance_scale > 1.0
    mm2 = pipeline.text_embed.embed_for_gen_image(
        image_prompts,
        cfg=cfg,
        height=diff_sp.height,
        width=diff_sp.width,
        bot_task=tok_bot_task,
        cot_text=image_cots,
        system_prompt=image_system_prompts,
    )
    fused, fused_uncond = split_cfg_fused(mm2["fused"], n_images, cfg=cfg)
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=fused,
        fused_uncond=fused_uncond,
        tokenizer_output=mm2["tokenizer_output"],
    )

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=diff_sp)
    images = pipeline.vae_decode.decode(latent_seg)

    new_parts[image_idx] = image_part.fill(
        segment=latent_seg, primitives={"image": images}, conditions=diff_conds.to_dict()
    )
    return sample.with_parts(new_parts)
