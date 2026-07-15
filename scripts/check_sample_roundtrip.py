#!/usr/bin/env python3
"""Runtime oracle for the Sample → Sample rollout-engine refactor (LIN-454).

Exercises the PURE Sample-assembly logic the converted engines rely on — the
path-id lineage, the part-extraction routing, ``assemble_sample`` fill, the σ
round-trip verifier, and the OD-2 noise-key derivation — with fabricated
Samples + ``SimpleNamespace`` fakes (no GPU, no backend, no vllm/sglang). It
does NOT boot an engine; it guards the conversion's structural contracts so a
broken fork / fill / route surfaces here instead of mid-rollout.

Only needs the unirl runtime types (torch) — the imported ``vllm_omni.utils``
helpers pull in no engine runtime. Run in the engine venv:

    python scripts/check_sample_roundtrip.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Callable, Tuple

import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.utils import assemble_sample
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_raises(fn: Callable[[], object], exc: type, msg: str) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{msg}: expected {exc.__name__} but none was raised")


def _two_stage_sample() -> Tuple[Sample, Part, Part, Part]:
    """``[input(P=2), ar(P*N, N=1), image(P*N*M, M=1)]`` — the HI3 t2i shape."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})
    ar_shell = inp.fork(1, sampling_params=ARSamplingParams())
    img_shell = ar_shell.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    return Sample(parts=[inp, ar_shell, img_shell]), inp, ar_shell, img_shell


def check_part_extraction() -> None:
    """Stage location by ``sampling_params`` TYPE + the conditioning read."""
    sample, _inp, ar_shell, img_shell = _two_stage_sample()
    _check(sample.gen_part(ARSamplingParams) is ar_shell, "gen_part locates the AR shell")
    _check(sample.gen_part(DiffusionSamplingParams) is img_shell, "gen_part locates the image shell")
    _check(not sample.has_image_input(), "has_image_input False (no Images input Part present)")
    _check(
        list(sample.text_conditioning()[0].content.texts) == ["a cat", "a dog"],
        "text_conditioning surfaces the input prompts",
    )


def check_assemble_sample() -> None:
    """``assemble_sample`` fills the right gen Parts (by params type), preserves ids."""
    sample, inp, _ar_shell, _img_shell = _two_stage_sample()
    ar_seg, img_seg = SimpleNamespace(tag="ar_seg"), SimpleNamespace(tag="img_seg")
    ar_dec, img_dec = Texts(texts=["pe0", "pe1"]), _images(2)
    cond = {"text": SimpleNamespace(tag="cond")}

    out = assemble_sample(
        sample,
        segments_for_track={"ar": ar_seg, "image": img_seg},
        decoded_for_track={"ar": ar_dec, "image": img_dec},
        conditions=cond,
    )
    _check(len(out.parts) == 3, "assemble_sample must preserve the part count")
    out_inp, out_ar, out_img = out.parts
    _check(out_inp is inp, "the input Part passes through unchanged")
    _check(
        out_ar.segment is ar_seg and out_ar.primitives["text"] is ar_dec,
        "ar Part filled by the 'ar' track",
    )
    _check(
        out_img.segment is img_seg and out_img.primitives["image"] is img_dec,
        "image Part filled by the 'image' track",
    )
    _check(out_ar.conditions == cond and out_img.conditions == cond, "conditions replicate onto every filled Part")
    _check(list(out_ar.sample_ids) == ["p0/0", "p1/0"], "ar path ids preserved")
    _check(list(out_img.sample_ids) == ["p0/0/0", "p1/0/0"], "image path ids preserved")
    _check(isinstance(out_ar.sampling_params, ARSamplingParams), "ar sampling_params preserved through fill")
    _check(isinstance(out_img.sampling_params, DiffusionSamplingParams), "image sampling_params preserved through fill")

    # Single-DiT shape ``[input, image]``: only the diffusion Part is targeted.
    inp_s = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    img_s = inp_s.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    single = Sample(parts=[inp_s, img_s])
    out2 = assemble_sample(
        single, segments_for_track={"image": img_seg}, decoded_for_track={"image": img_dec}, conditions={}
    )
    _check(out2.parts[-1].segment is img_seg, "single-DiT: image Part filled")


def check_sigma_roundtrip() -> None:
    """``verify_engine_used_sigmas`` passes on echo == sent, raises on drift / missing."""
    sig = torch.linspace(1.0, 0.0, steps=5)
    verify_engine_used_sigmas(sig.clone(), expected=sig, engine_name="oracle")  # match → no raise

    perturbed = sig.clone()
    perturbed[1] += 0.1
    _expect_raises(
        lambda: verify_engine_used_sigmas(perturbed, expected=sig, engine_name="oracle"),
        RuntimeError,
        "perturbed σ echo must raise",
    )
    _expect_raises(
        lambda: verify_engine_used_sigmas(None, expected=sig, engine_name="oracle"),
        RuntimeError,
        "missing σ echo must raise",
    )
    verify_engine_used_sigmas(None, expected=None, engine_name="oracle")  # expected None → skip, no raise


def check_noise_key_lineage() -> None:
    """OD-2: the x_T noise key is the gen Part's lineage — group id (shared) vs
    sample id (per-sample), both derived from the path-id fork."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["x"])})
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(2, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    _check(
        list(img.sample_ids) == ["p0/0/0", "p0/0/1", "p0/1/0", "p0/1/1"],
        "fork builds group-by-parent contiguous path ids",
    )
    _check(len(set(img.sample_ids)) == 4, "per-sample noise key (sample_ids) is unique → distinct x_T")
    _check(
        img.group_ids == ["p0/0", "p0/0", "p0/1", "p0/1"],
        "group noise key (group_ids) = parent id → siblings share x_T",
    )


def check_split_concat_roundtrip() -> None:
    """``Sample.split`` → ``Sample.concat`` round-trips ids (chunked_engine_generate core)."""
    sample, *_ = _two_stage_sample()
    groups = sample.split()
    _check(len(groups) == 2, "split yields one Sample per root group")
    merged = Sample.concat(groups)
    _check(
        [list(p.sample_ids) for p in merged.parts] == [list(p.sample_ids) for p in sample.parts],
        "split→concat round-trips the per-part sample ids in order",
    )


def _images(n: int) -> Images:
    return Images.from_list([Image(pixels=torch.zeros(3, 8, 8)) for _ in range(n)])


def check_multi_input_image_chain() -> None:
    """IT2I-shaped multi-input ``[text, image_input, ar_gen, image_gen]`` via Part.input_child:
    chains a valid Sample, the helpers route past the image input, conditioning() surfaces both
    primitives, and assemble_sample fills the right gen Parts."""
    text = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["edit the cat", "edit the dog"])})
    image_in = text.input_child({"image": _images(2)})
    sample = (
        Sample.request(text, image_in)
        .fork(1, sampling_params=ARSamplingParams())
        .fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    )
    _check(len(sample.parts) == 4, "multi-input chain has 4 parts [text, image, ar, image]")
    _check(sample.has_image_input(), "has_image_input True (chained image input)")
    _check(sample.gen_part(ARSamplingParams) is sample.parts[2], "gen_part routes past the image input Part")
    _check(sample.gen_part(DiffusionSamplingParams) is sample.parts[3], "gen_part routes past the image input Part")
    cond = sample.conditioning()
    _check(
        len(cond) == 2 and isinstance(cond[0], Texts) and isinstance(cond[1], Images),
        "conditioning() returns [Texts, Images] in turn order (the empty gen shells are skipped)",
    )
    out = assemble_sample(
        sample,
        segments_for_track={"ar": SimpleNamespace(tag="ar"), "image": SimpleNamespace(tag="img")},
        decoded_for_track={"ar": Texts(texts=["r0", "r1"]), "image": _images(2)},
        conditions={"fused": SimpleNamespace(tag="cond")},
    )
    _check(out.parts[0] is sample.parts[0] and out.parts[1] is sample.parts[1], "input Parts pass through unchanged")
    _check(getattr(out.parts[2].segment, "tag", None) == "ar", "ar gen Part filled by the 'ar' track")
    _check(getattr(out.parts[3].segment, "tag", None) == "img", "image gen Part filled by the 'image' track")


def check_cot_text_chain() -> None:
    """dit_recaption-shaped ``[text, cot_text_input, image_gen]``: the chained recaption
    is the second text turn of ``text_conditioning()`` (1:1 with prompts by lineage)."""
    text = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})
    cot_in = text.input_child({"text": Texts(texts=["a fluffy ginger cat", "a happy golden dog"])})
    sample = Sample.request(text, cot_in).fork(
        1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    )
    turns = sample.text_conditioning()
    _check(list(turns[0].content.texts) == ["a cat", "a dog"], "text_conditioning()[0] is the prompt")
    _check(
        list(turns[1].content.texts) == ["a fluffy ginger cat", "a happy golden dog"],
        "text_conditioning()[1] is the chained recaption",
    )


def check_root_group_ids() -> None:
    """``Sample.root_group_ids(i)`` is the ``ancestor_id(sid, 0)`` projection (the
    id's first segment) — coarser than ``Part.group_ids`` (immediate parent). The
    labels stay group-by-parent contiguous, matching what
    ``Part.compute_advantages(group_layer=0)`` groups by."""
    # P=2 prompts, N=2 AR children each, M=1 image each → [input, ar(4), image(4)].
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img])

    _check(list(img.sample_ids) == ["p0/0/0", "p0/1/0", "p1/0/0", "p1/1/0"], "image fan-out ids")
    _check(sample.root_group_ids(2) == ["p0", "p0", "p1", "p1"], "image part grouped by ROOT prompt")
    _check(img.group_ids == ["p0/0", "p0/1", "p1/0", "p1/1"], "Part.group_ids stays immediate-parent (4 groups)")
    _check(sample.root_group_ids(0) == ["p0", "p1"], "root part: each prompt is its own group")
    _check(sample.root_group_ids(1) == ["p0", "p0", "p1", "p1"], "ar part grouped by root prompt")


def check_gen_part_accessors() -> None:
    """``Sample.gen_parts``/``gen_part``/``gen_part_index`` locate stages by
    ``sampling_params`` TYPE (the migration's part-location convention; mirrors
    the engine-side ``ar_gen_part``/``diffusion_gen_part``). ``with_parts`` swaps
    parts while preserving ``reward_compute_s``."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img], reward_compute_s=1.5)

    gps = sample.gen_parts()
    _check(len(gps) == 2 and gps[0] is ar and gps[1] is img, "gen_parts skips the input Part")
    _check(ar.is_gen and img.is_gen and not inp.is_gen, "is_gen classifies gen shells vs the input head")
    _expect_raises(lambda: inp.fork(1), TypeError, "fork without sampling_params must fail loud")
    _check(sample.gen_part(ARSamplingParams) is ar, "gen_part locates the AR stage by type")
    _check(sample.gen_part(DiffusionSamplingParams) is img, "gen_part locates the diffusion stage by type")
    _check(sample.gen_part_index(ARSamplingParams) == 1, "gen_part_index: AR at position 1")
    _check(sample.gen_part_index(DiffusionSamplingParams) == 2, "gen_part_index: diffusion at position 2")

    # A request with no diffusion stage → clear ValueError, not a bare StopIteration.
    try:
        Sample(parts=[inp, ar]).gen_part(DiffusionSamplingParams)
        raised = False
    except ValueError:
        raised = True
    _check(raised, "gen_part raises ValueError when the stage type is absent")

    # with_parts swaps the parts list but carries reward_compute_s forward.
    swapped = sample.with_parts([inp, ar])
    _check(len(swapped.parts) == 2 and swapped.parts[1] is ar, "with_parts replaces the parts list")
    _check(swapped.reward_compute_s == 1.5, "with_parts preserves reward_compute_s")


def check_sample_dp_chunk() -> None:
    """``Sample.chunk``/``slice``/``select`` shard by whole prompt-TREE (the dp>1
    rollout-request fix), not by the ``parts`` list — the inherited ``Batch.slice``
    would slice the length-3 parts list and hand every DP rank the full Sample.
    Reuses the tree-correct ``split``/``concat``."""
    inp = Part.input([f"p{i}" for i in range(4)], primitives={"text": Texts(texts=[f"prompt {i}" for i in range(4)])})
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, ar, img], reward_compute_s=2.0)

    shards = sample.chunk(2)
    _check(len(shards) == 2, "chunk(2) yields 2 shards")
    _check([s.parts[0].batch_size for s in shards] == [2, 2], "each shard holds 2 prompt-trees (not the whole batch)")
    _check([s.parts[1].batch_size for s in shards] == [4, 4], "each shard ar part = 2*N")
    _check([s.parts[2].batch_size for s in shards] == [4, 4], "each shard image part = 2*N*M")
    _check(list(shards[0].parts[0].sample_ids) == ["p0", "p1"], "shard 0 = prompts 0,1")
    _check(
        list(shards[1].parts[2].sample_ids) == ["p2/0/0", "p2/1/0", "p3/0/0", "p3/1/0"],
        "image part is tree-sharded (not list-sliced to the whole part)",
    )
    rt = Sample.concat(shards)
    _check(list(rt.parts[2].sample_ids) == list(sample.parts[2].sample_ids), "chunk -> concat round-trips ids")
    _check(rt.reward_compute_s == 2.0, "chunk -> concat preserves reward_compute_s")
    _check(list(sample.slice(0, 2).parts[0].sample_ids) == ["p0", "p1"], "slice(0,2) = first 2 trees")
    _check(
        list(sample.select(torch.tensor([0, 2])).parts[0].sample_ids) == ["p0", "p2"],
        "select gathers whole trees by index",
    )


def check_unified_dit_noise_ids_unique() -> None:
    """The unified DiT sub-request re-roots from the globally-unique image-shell
    lineage (flatten ``/``), so the engine-derived x_T key stays unique across dp>1
    replicas; the replica-local ``d{k}`` scheme collides (each shard restarts at 0)."""
    inp = Part.input(
        [f"r0:p{i}" for i in range(4)],
        primitives={"text": Texts(texts=[f"prompt {i}" for i in range(4)])},
    )
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    shards = Sample(parts=[inp, ar, img]).chunk(2)  # dp=2 — what run_rollout does

    fixed = [sid.replace("/", "_") for s in shards for sid in s.gen_part(DiffusionSamplingParams).sample_ids]
    _check(len(fixed) == len(set(fixed)), "lineage-derived DiT ids are globally unique across dp shards")
    old = [f"r0:d{k}" for s in shards for k in range(s.gen_part(DiffusionSamplingParams).batch_size)]
    _check(len(old) != len(set(old)), "sanity: the old replica-local d{k} scheme DOES collide")


def check_group_layer_advantages() -> None:
    """``Part.compute_advantages``: ``scope`` picks the normalization mode
    (``"global"`` = batch z-score with unbiased std — the historical convention);
    under ``scope="group"``, ``group_layer`` selects the lineage layer whose
    ancestor id labels the GRPO groups: ``None`` (default) = immediate parent,
    ``0`` = root prompt (PE ``diffusion_group_scope='prompt'``)."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    ar = inp.fork(2, sampling_params=ARSamplingParams())
    img = ar.fork(2, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    image = _part_with_field(img, "rewards", torch.arange(8, dtype=torch.float32))  # 8 = P*N*M

    def _norm(r: torch.Tensor) -> torch.Tensor:  # per-group GRPO z-score, population std
        return (r - r.mean()) / (r.var(unbiased=False) + 1e-8).sqrt()

    rewards = image.rewards
    a_root = image.compute_advantages(normalize=True, group_layer=0)
    expected_root = torch.cat([_norm(rewards[:4]), _norm(rewards[4:])])  # {p0: rows 0-3, p1: rows 4-7}
    _check(torch.allclose(a_root.advantages, expected_root), "group_layer=0 normalizes per ROOT prompt")

    a_def = image.compute_advantages(normalize=True)
    expected_parent = torch.cat([_norm(rewards[i : i + 2]) for i in range(0, 8, 2)])  # one group per rewrite
    _check(torch.allclose(a_def.advantages, expected_parent), "default group_layer normalizes per immediate parent")
    _check(not torch.allclose(a_def.advantages, a_root.advantages), "root layer is coarser than the parent default")

    a_global = image.compute_advantages(normalize=True, scope="global")
    expected_global = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # unbiased std — historical parity
    _check(torch.allclose(a_global.advantages, expected_global), "scope='global' batch z-scores with unbiased std")

    _expect_raises(lambda: image.compute_advantages(group_layer=5), ValueError, "out-of-range group_layer must raise")
    _expect_raises(
        lambda: image.compute_advantages(group_layer=-1),
        ValueError,
        "negative group_layer must raise (global is scope='global')",
    )


def check_rollout_metric_naming() -> None:
    """Multi-stage rollout metrics name the diffusion stage ``diffusion`` (matching
    the train-side track key), not ``image`` — so PE's ``rollout/*`` and ``train/*``
    panels correlate."""
    from unirl.utils.wandb_metrics import compute_rollout_sample_metrics

    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    ar = _part_with_field(
        inp.fork(2, sampling_params=ARSamplingParams()), "rewards", torch.arange(4, dtype=torch.float32)
    )
    img = _part_with_field(
        ar.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)),
        "rewards",
        torch.arange(4, dtype=torch.float32),
    )
    m = compute_rollout_sample_metrics(sample=Sample(parts=[inp, ar, img]))
    _check(any(k.startswith("diffusion_") for k in m), "diffusion stage logs under the 'diffusion_' prefix")
    _check(not any(k.startswith("image_") for k in m), "no stale 'image_' prefix")


def check_disable_driver_xt_flag() -> None:
    """``DiffusionSamplingParams`` carries the ``disable_driver_xt`` opt-out (default
    off), settable per-recipe and via ``dataclasses.replace`` (how the unified trainer
    stamps the env onto the params)."""
    import dataclasses

    d = DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)
    _check(d.disable_driver_xt is False, "disable_driver_xt defaults to False")
    _check(dataclasses.replace(d, disable_driver_xt=True).disable_driver_xt is True, "replace sets the flag")
    _check(
        DiffusionSamplingParams(num_inference_steps=4, height=256, width=256, disable_driver_xt=True).disable_driver_xt
        is True,
        "recipe can construct with the flag set",
    )


def check_noise_recipe_from_sample() -> None:
    """``NoiseRecipe.from_sample`` keys the x_T on the gen Part's lineage
    (sample_ids, or group_ids under ``init_same_noise``) and reads seed + shape off
    its ``DiffusionSamplingParams`` — parity with the engines' ``_resolve_initial_noise``."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    img = inp.fork(
        2,
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=4,
            height=256,
            width=256,
            seed=7,
            init_noise_latent_shape=[16, 32, 32],
            init_same_noise=False,
        ),
    )
    recipe = NoiseRecipe.from_sample(Sample(parts=[inp, img]))
    _check(recipe.noise_group_ids == list(img.sample_ids), "from_sample keys per-sample ids (init_same_noise=False)")
    _check(recipe.base_seed == 7, "from_sample reads seed off the gen params")
    _check(tuple(recipe.latent_shape or ()) == (16, 32, 32), "from_sample reads init_noise_latent_shape")
    _check(recipe.initial_latents is None, "no img2img segment latents → initial_latents None")

    img_shared = inp.fork(
        2,
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=4,
            height=256,
            width=256,
            seed=7,
            init_noise_latent_shape=[16, 32, 32],
            init_same_noise=True,
        ),
    )
    shared = NoiseRecipe.from_sample(Sample(parts=[inp, img_shared]))
    _check(
        shared.noise_group_ids == list(img_shared.group_ids),
        "from_sample keys group ids when init_same_noise=True (siblings share x_T)",
    )


def check_generate_fills_frontier() -> None:
    """The pipeline contract: ``generate`` fills the pre-forked frontier and returns
    ``[input, filled-gen]`` — ids + sampling_params preserved, conditions left empty
    (the trainside re-encode path). Mirrors the Sample reconstruction in
    ``SD3Pipeline.generate`` / ``Qwen3Pipeline.generate`` with fake payloads."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    shell = inp.fork(1, sampling_params=DiffusionSamplingParams(num_inference_steps=4, height=256, width=256))
    sample = Sample(parts=[inp, shell])
    seg, dec = SimpleNamespace(tag="latent_seg"), _images(2)
    filled = sample.parts[-1].fill(segment=seg, primitives={"image": dec})
    out = Sample(parts=[*sample.parts[:-1], filled])
    _check(len(out.parts) == 2, "generate returns [input, gen]")
    _check(out.parts[0] is inp, "input Part passes through unchanged")
    gen = out.parts[-1]
    _check(gen.segment is seg and gen.primitives["image"] is dec, "frontier filled with segment + decoded")
    _check(list(gen.sample_ids) == ["p0/0", "p1/0"], "frontier path ids preserved")
    _check(gen.conditions == {}, "conditions left empty (replay re-encodes)")
    _check(isinstance(gen.sampling_params, DiffusionSamplingParams), "sampling_params preserved through fill")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_part_extraction,
    check_assemble_sample,
    check_sigma_roundtrip,
    check_noise_key_lineage,
    check_split_concat_roundtrip,
    check_multi_input_image_chain,
    check_cot_text_chain,
    check_root_group_ids,
    check_gen_part_accessors,
    check_sample_dp_chunk,
    check_unified_dit_noise_ids_unique,
    check_group_layer_advantages,
    check_rollout_metric_naming,
    check_disable_driver_xt_flag,
    check_noise_recipe_from_sample,
    check_generate_fills_frontier,
)


def main() -> int:
    failures = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("check-sample-roundtrip: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-sample-roundtrip: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
