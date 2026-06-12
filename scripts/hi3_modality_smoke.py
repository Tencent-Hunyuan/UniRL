#!/usr/bin/env python3
"""HunyuanImage3 (hi3) all-modality smoke test.

Loads the hi3 bundle ONCE with ``device_map="auto"`` (the 80B MoE sharded
across the visible GPUs, single process — no Ray, no FSDP), then drives
``HunyuanImage3Pipeline.generate(req)`` for each of the five task topologies:

    t2t  : text         -> text    (AR / gen_text)
    i2t  : image + text -> text    (AR / gen_text, image comprehension)
    t2i  : text         -> image   (diffusion / gen_image)
    it2i : image + text -> image   (diffusion / gen_image, image edit)
    t2ti : text         -> text+image (AR think_recaption CoT, then diffusion)

Each mode is isolated in its own try/except so one failure doesn't mask the
others; a PASS/FAIL summary is printed at the end. Generated images are saved
under OUT_DIR for visual inspection.

This is an INFERENCE smoke (does each modality of *our* bundle/pipeline produce
sane output on real weights), not a training run.

Env knobs:
    PRETRAINED_MODEL  default /dockerdata/HunyuanImage-3-Instruct
    OUT_DIR           default /tmp/hi3_smoke_out
    HI3_STEPS         default 16    (diffusion num_inference_steps)
    HI3_HW            default 1024  (square image side)
    HI3_MAX_TOKENS    default 128   (AR max_new_tokens)
    HI3_COT_TOKENS    default 512   (t2ti CoT max_new_tokens — think+recaption needs room)
    HI3_MODES         default "t2t,i2t,t2i,it2i,t2ti" (comma list to run a subset)
"""

from __future__ import annotations

import os
import sys
import time
import traceback

import torch

# The unirl-v1 pod image documents a global cuDNN-off fix (cuda-compat / driver
# combos crash on Conv2d). Disabling cuDNN sidesteps that whole class and only
# costs a little VAE-decode speed; harmless on cu129 where it isn't required.
torch.backends.cudnn.enabled = False

PRETRAINED = os.environ.get("PRETRAINED_MODEL", "/dockerdata/HunyuanImage-3-Instruct")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/hi3_smoke_out")
STEPS = int(os.environ.get("HI3_STEPS", "16"))
HW = int(os.environ.get("HI3_HW", "1024"))
MAX_TOKENS = int(os.environ.get("HI3_MAX_TOKENS", "128"))
COT_TOKENS = int(os.environ.get("HI3_COT_TOKENS", "512"))
MODES = [m.strip() for m in os.environ.get("HI3_MODES", "t2t,i2t,t2i,it2i,t2ti").split(",") if m.strip()]

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg: str) -> None:
    print(f"[hi3-smoke] {msg}", flush=True)


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"g{i}:{(total - free) / 2**30:.0f}/{total / 2**30:.0f}G")
    return " ".join(parts)


def make_test_image(side: int = 512):
    """A simple, recognizable RGB test image for i2t/it2i inputs."""
    import PIL.Image
    import PIL.ImageDraw

    img = PIL.Image.new("RGB", (side, side), (30, 60, 140))  # blue background
    d = PIL.ImageDraw.Draw(img)
    d.ellipse([int(side * 0.25), int(side * 0.25), int(side * 0.75), int(side * 0.75)], fill=(220, 40, 40))
    d.rectangle([int(side * 0.05), int(side * 0.05), int(side * 0.25), int(side * 0.25)], fill=(240, 220, 40))
    return img


def image_primitive(side: int = 512):
    """Build a fresh Images primitive (one image) from the synthetic test image."""
    import numpy as np

    from unirl.types.primitives import Image, Images

    pil = make_test_image(side)
    arr = np.asarray(pil.convert("RGB"), dtype="float32") / 255.0  # HWC in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]
    return Images.from_list([Image(pixels=t)])


def build_pipeline():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unirl.models.hunyuan_image3.bundle import HunyuanImage3Bundle
    from unirl.models.hunyuan_image3.compat import apply_hi3_transformers5_compat
    from unirl.models.hunyuan_image3.config import HunyuanImage3PipelineConfig
    from unirl.models.hunyuan_image3.pipeline import HunyuanImage3Pipeline
    from unirl.sde.kernels import FlowSDEStrategy

    # transformers-5.x compat shims (replaces the on-disk checkpoint patcher).
    apply_hi3_transformers5_compat()

    dtype = torch.bfloat16
    log(f"loading transformer (device_map=auto, bf16) from {PRETRAINED} ...")
    t0 = time.time()
    transformer = AutoModelForCausalLM.from_pretrained(
        PRETRAINED,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    transformer.eval()
    # The Instruct checkpoint's config has no ``model_version`` (base does), but
    # its modeling.load_tokenizer reads ``self.config.model_version`` (the value
    # is then ignored by Instruct's tokenizer). Default it so the attribute
    # access doesn't AttributeError. Harmless on base (already set).
    if not hasattr(transformer.config, "model_version") or getattr(transformer.config, "model_version", None) is None:
        transformer.config.model_version = "HunyuanImage-3.0"
        log("set config.model_version='HunyuanImage-3.0' (missing on Instruct ckpt)")
    dmap = getattr(transformer, "hf_device_map", {}) or {}
    n_dev = len(set(str(v) for v in dmap.values())) if dmap else 1
    log(f"transformer loaded in {time.time() - t0:.1f}s across {n_dev} device(s); {gpu_mem()}")

    vae = getattr(transformer, "vae", None) or getattr(transformer, "vae_model", None)
    vit = (
        getattr(transformer, "vit", None)
        or getattr(transformer, "vision_tower", None)
        or getattr(transformer, "siglip", None)
        or getattr(transformer, "vision_model", None)
    )
    if vae is None or vit is None:
        raise RuntimeError(f"could not locate vae(found={vae is not None})/vit(found={vit is not None}) on transformer")

    # Anchor everything to the token-embedding shard (cuda:0 under device_map=auto).
    try:
        device = transformer.model.wte.weight.device
    except AttributeError:
        device = torch.device("cuda:0")
    log(f"bundle.device = {device}")

    # Pin the (small) VAE + ViT onto the anchor device and detach any accelerate
    # device-map hooks, mirroring from_config's vae.to(device)/vit.to(device).
    # This keeps cond-image encode + vae decode on one device instead of chasing
    # the sharded transformer's per-layer placement.
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(vae, recurse=True)
        remove_hook_from_module(vit, recurse=True)
    except Exception as e:  # noqa: BLE001
        log(f"(accelerate hook removal skipped: {type(e).__name__}: {e})")
    vae = vae.to(device=device, dtype=dtype).eval()
    vit = vit.to(device=device).eval()

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, trust_remote_code=True)

    scheduler = None
    try:
        from hunyuan_image_3.hunyuan_image_3_pipeline import FlowMatchDiscreteScheduler

        scheduler = FlowMatchDiscreteScheduler.from_pretrained(PRETRAINED)
    except Exception as e:  # noqa: BLE001
        log(f"scheduler import skipped ({type(e).__name__}); sigmas come from sde.runtime")

    # HI3 uses a static flow-match shift; prefer the model's own generation_config.
    shift = 3.0
    try:
        gc = getattr(transformer, "generation_config", None)
        fs = getattr(gc, "flow_shift", None) if gc is not None else None
        if fs is not None:
            shift = float(fs)
    except Exception:  # noqa: BLE001
        pass
    log(f"flow-match shift = {shift}")

    bundle = HunyuanImage3Bundle(
        transformer=transformer,
        vae=vae,
        vit=vit,
        tokenizer=tokenizer,
        scheduler=scheduler,
        dtype=dtype,
        vae_dtype=dtype,
        device=device,
        pretrained_path=PRETRAINED,
        mrope_section=(0, 32, 32),
    )
    config = HunyuanImage3PipelineConfig(
        pretrained_model_ckpt_path=PRETRAINED,
        model_precision="bf16",
        shift=shift,
    )
    pipeline = HunyuanImage3Pipeline.from_bundle(bundle=bundle, config=config, strategy=FlowSDEStrategy())
    return pipeline, shift


def run() -> int:
    from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

    log(f"torch {torch.__version__}; visible GPUs {torch.cuda.device_count()}; modes {MODES}")
    log(f"steps={STEPS} hw={HW} max_tokens={MAX_TOKENS}")

    try:
        pipeline, shift = build_pipeline()
    except Exception as e:  # noqa: BLE001
        log(f"MODEL LOAD FAILED: {e}\n{traceback.format_exc()}")
        return 2

    policy = FlowMatchSchedulePolicy.static_only(shift)
    results: dict[str, tuple[str, str]] = {}

    def ar_params():
        return ARSamplingParams(temperature=0.7, max_new_tokens=MAX_TOKENS, top_p=0.9, top_k=1024)

    def diff_params(gs: float):
        return DiffusionSamplingParams(num_inference_steps=STEPS, guidance_scale=gs, height=HW, width=HW, seed=42)

    def save_first_image(imgs, name: str) -> str:
        outp = os.path.join(OUT_DIR, name)
        imgs.to_list()[0].to_pil().save(outp)
        return outp

    # ---- t2t : text -> text ------------------------------------------------
    if "t2t" in MODES:
        try:
            log("=== t2t (text -> text) ===")
            req = RolloutReq(
                sample_ids=["t2t:0"],
                group_ids=["g_t2t"],
                primitives={"text": Texts(texts=["Write one vivid sentence about a sunrise over the ocean."])},
                sampling_params=ar_params(),
                stage_config={"task": "t2t", "ar": {"bot_task": "auto"}},
            )
            t0 = time.time()
            with torch.no_grad():
                resp = pipeline.generate(req)
            txt = resp.tracks["ar"].decoded.texts[0]
            log(f"t2t output ({time.time() - t0:.1f}s, {len(txt)} chars): {txt!r}")
            results["t2t"] = ("PASS" if txt and txt.strip() else "EMPTY", repr(txt[:240]))
        except Exception as e:  # noqa: BLE001
            log(f"t2t FAILED: {e}\n{traceback.format_exc()}")
            results["t2t"] = ("FAIL", f"{type(e).__name__}: {e}")

    # ---- i2t : image + text -> text ---------------------------------------
    if "i2t" in MODES:
        try:
            log("=== i2t (image + text -> text) ===")
            req = RolloutReq(
                sample_ids=["i2t:0"],
                group_ids=["g_i2t"],
                primitives={"text": Texts(texts=["Describe this image in detail."]), "image": image_primitive(512)},
                sampling_params=ar_params(),
                stage_config={"task": "i2t", "ar": {"bot_task": "auto"}},
            )
            t0 = time.time()
            with torch.no_grad():
                resp = pipeline.generate(req)
            txt = resp.tracks["ar"].decoded.texts[0]
            log(f"i2t output ({time.time() - t0:.1f}s, {len(txt)} chars): {txt!r}")
            results["i2t"] = ("PASS" if txt and txt.strip() else "EMPTY", repr(txt[:240]))
        except Exception as e:  # noqa: BLE001
            log(f"i2t FAILED: {e}\n{traceback.format_exc()}")
            results["i2t"] = ("FAIL", f"{type(e).__name__}: {e}")

    # ---- t2i : text -> image ----------------------------------------------
    if "t2i" in MODES:
        try:
            log("=== t2i (text -> image) ===")
            req = RolloutReq(
                sample_ids=["t2i:0"],
                group_ids=["g_t2i"],
                primitives={
                    "text": Texts(texts=["A photorealistic red panda sitting on a mossy rock in a misty forest."])
                },
                sampling_params=diff_params(gs=7.5),
                stage_config={"task": "t2i", "bot_task": "image", "diffusion": {}},
            )
            ensure_req_sigmas(req, policy)
            t0 = time.time()
            with torch.no_grad():
                resp = pipeline.generate(req)
            px = resp.tracks["image"].decoded.pixels
            outp = save_first_image(resp.tracks["image"].decoded, "t2i.png")
            log(
                f"t2i output ({time.time() - t0:.1f}s): pixels {tuple(px.shape)} {px.dtype} "
                f"min {px.min().item():.3f} max {px.max().item():.3f} -> {outp}"
            )
            ok = px.ndim == 4 and px.shape[0] == 1 and px.shape[1] == 3
            results["t2i"] = ("PASS" if ok else "BADSHAPE", f"{tuple(px.shape)} -> {outp}")
        except Exception as e:  # noqa: BLE001
            log(f"t2i FAILED: {e}\n{traceback.format_exc()}")
            results["t2i"] = ("FAIL", f"{type(e).__name__}: {e}")

    # ---- it2i : image + text -> image -------------------------------------
    if "it2i" in MODES:
        try:
            log("=== it2i (image + text -> image) ===")
            req = RolloutReq(
                sample_ids=["it2i:0"],
                group_ids=["g_it2i"],
                primitives={
                    "text": Texts(texts=["Change the background to a golden sunset sky."]),
                    "image": image_primitive(512),
                },
                sampling_params=diff_params(gs=2.5),
                stage_config={"task": "it2i", "bot_task": "image", "diffusion": {}},
            )
            ensure_req_sigmas(req, policy)
            t0 = time.time()
            with torch.no_grad():
                resp = pipeline.generate(req)
            px = resp.tracks["image"].decoded.pixels
            outp = save_first_image(resp.tracks["image"].decoded, "it2i.png")
            log(
                f"it2i output ({time.time() - t0:.1f}s): pixels {tuple(px.shape)} {px.dtype} "
                f"min {px.min().item():.3f} max {px.max().item():.3f} -> {outp}"
            )
            ok = px.ndim == 4 and px.shape[0] == 1 and px.shape[1] == 3
            results["it2i"] = ("PASS" if ok else "BADSHAPE", f"{tuple(px.shape)} -> {outp}")
        except Exception as e:  # noqa: BLE001
            log(f"it2i FAILED: {e}\n{traceback.format_exc()}")
            results["it2i"] = ("FAIL", f"{type(e).__name__}: {e}")

    # ---- t2ti : text -> CoT text + image (think_recaption chain) -----------
    if "t2ti" in MODES:
        try:
            log("=== t2ti (text -> CoT text + image) ===")
            from unirl.types.sampling import ComposedSamplingParams

            req = RolloutReq(
                sample_ids=["t2ti:0"],
                group_ids=["g_t2ti"],
                primitives={
                    "text": Texts(texts=["A photorealistic red panda sitting on a mossy rock in a misty forest."])
                },
                sampling_params=ComposedSamplingParams(
                    ar=ARSamplingParams(temperature=0.7, max_new_tokens=COT_TOKENS, top_p=0.9, top_k=1024),
                    diffusion=diff_params(gs=7.5),
                ),
                stage_config={"task": "t2ti", "bot_task": "think_recaption", "ar": {}},
            )
            ensure_req_sigmas(req, policy)
            t0 = time.time()
            with torch.no_grad():
                resp = pipeline.generate(req)
            cot = resp.tracks["ar"].decoded.texts[0]
            px = resp.tracks["image"].decoded.pixels
            outp = save_first_image(resp.tracks["image"].decoded, "t2ti.png")
            lineage_ok = resp.tracks["image"].parent_track == "ar" and list(resp.tracks["image"].parent_ids) == [
                "t2ti:0"
            ]
            # Marker presence is informational, not a failure: without upstream's
            # stage-transition forcing the model may skip the recaption block.
            log(
                f"t2ti CoT ({len(cot)} chars; </think>={'</think>' in cot} </recaption>={'</recaption>' in cot}): {cot!r}"
            )
            log(
                f"t2ti output ({time.time() - t0:.1f}s): pixels {tuple(px.shape)} {px.dtype} "
                f"min {px.min().item():.3f} max {px.max().item():.3f} lineage_ok={lineage_ok} -> {outp}"
            )
            ok = px.ndim == 4 and px.shape[0] == 1 and px.shape[1] == 3 and bool(cot.strip()) and lineage_ok
            results["t2ti"] = ("PASS" if ok else "BAD", f"{tuple(px.shape)} cot_chars={len(cot)} -> {outp}")
        except Exception as e:  # noqa: BLE001
            log(f"t2ti FAILED: {e}\n{traceback.format_exc()}")
            results["t2ti"] = ("FAIL", f"{type(e).__name__}: {e}")

    # ---- summary -----------------------------------------------------------
    log("================ SUMMARY ================")
    for m in ["t2t", "i2t", "t2i", "it2i", "t2ti"]:
        if m in results:
            status, detail = results[m]
            log(f"{m:5s}: {status:9s} | {detail}")
    log(f"final GPU mem: {gpu_mem()}")
    log("========================================")
    n_pass = sum(1 for v in results.values() if v[0] == "PASS")
    log(f"{n_pass}/{len(results)} modes PASS")
    return 0 if results and n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(run())
