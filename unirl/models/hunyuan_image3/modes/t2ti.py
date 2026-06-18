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

Returns TWO tracks: ``"ar"`` (root; ``decoded`` is the truncated +
normalized CoT that actually conditioned the image — raw tokens stay in
``segment`` for replay) and ``"image"`` (``parent_track="ar"``).
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
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_ar_params, get_diffusion_params

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


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """t2ti — AR CoT phase, then diffusion conditioned on the CoT."""
    texts = req.primitives.get("text")
    require(
        isinstance(texts, Texts),
        f"HunyuanImage3Pipeline.generate (t2ti): input must be Texts, got {type(texts).__name__ if texts is not None else 'None'}",
    )
    require(
        req.primitives.get("negative_text") is None,
        "HunyuanImage3Pipeline.generate (t2ti): negative_text is not supported — "
        "the HI3 tokenizer never consumes negative-prompt text; CFG is derived from "
        "guidance_scale > 1.0 (the unconditional branch is built internally from <cfg> tokens).",
    )

    ar_sp = get_ar_params(req.sampling_params)
    require(
        ar_sp is not None,
        "HunyuanImage3Pipeline.generate (t2ti): AR sampling params missing — t2ti needs "
        "ComposedSamplingParams(ar=ARSamplingParams(...), diffusion=DiffusionSamplingParams(...)).",
    )
    diff_sp = get_diffusion_params(req.sampling_params)
    require(
        isinstance(diff_sp, DiffusionSamplingParams),
        "HunyuanImage3Pipeline.generate (t2ti): diffusion sampling params missing or mistyped — t2ti needs "
        "ComposedSamplingParams(ar=ARSamplingParams(...), diffusion=DiffusionSamplingParams(...)).",
    )

    ar_cfg: Dict[str, Any] = dict(req.stage_config.get("ar") or {})
    require(
        "bot_task" not in ar_cfg,
        "HunyuanImage3Pipeline.generate (t2ti): set the single top-level stage_config['bot_task'] "
        "(the chain is one semantic mode); stage_config['ar']['bot_task'] is not read.",
    )
    bot_task = str(req.stage_config.get("bot_task", "think_recaption"))
    tok_bot_task = _tokenizer_bot_task(bot_task)
    batch = len(texts.texts)

    # ---- AR phase: generate the CoT ----------------------------------
    # use_system_prompt defaults to the en_think_recaption preset for the
    # think_recaption chain (vllm-omni prompt_utils._BOT_TASK_PRESETS
    # parity) — get_system_prompt's ``dynamic`` branch would resolve the
    # same preset via the mapped "think", but only when the checkpoint's
    # gen_config default is ``dynamic``.
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

    # ---- bridge: AR text -> diffusion cot_text ------------------------
    # Markers must survive decoding (skip_special_tokens=False) so the
    # truncate / normalize helpers and the wrapper's section parsing see
    # the literal ``</think>`` / ``</recaption>`` tags.
    raw = pipeline._detokenize_text_segment(text_seg, skip_special_tokens=False)
    cots = [_normalize_cot_text(_truncate_at_cot_end(t)) for t in raw.texts]

    # ---- diffusion phase: condition on prompt + CoT -------------------
    if req.sigmas is None:
        raise ValueError(
            "HunyuanImage3 t2ti: req.sigmas is None. Engine adapter must call "
            "unirl.sde.runtime.ensure_req_sigmas before pipeline.generate."
        )
    schedule = req.sigmas.to(pipeline.bundle.device)

    mm2 = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=float(diff_sp.guidance_scale) > 1.0,
        height=int(diff_sp.height),
        width=int(diff_sp.width),
        bot_task=tok_bot_task,
        cot_text=cots,
        system_prompt=system_prompt_list,
    )
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm2["fused"],
        tokenizer_output=mm2["tokenizer_output"],
    )

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=diff_sp)
    images = pipeline.vae_decode.decode(latent_seg)

    # Two-track lineage: "ar" is the root (sample_ids = the request's,
    # parent_ids = group_ids — same convention as t2t, so engine-side
    # prompt replication keeps GRPO grouping intact); "image" forks 1:1
    # off the CoT with hierarchical ids.
    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=ar_conds.to_dict(),
                segment=text_seg,
                decoded=Texts(texts=cots),
            ),
            "image": RolloutTrack(
                sample_ids=[f"{sid}/i0" for sid in req.sample_ids],
                parent_ids=list(req.sample_ids),
                parent_track="ar",
                conditions=diff_conds.to_dict(),
                segment=latent_seg,
                decoded=images,
            ),
        }
    )
