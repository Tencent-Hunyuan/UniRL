# BAGEL

> **Where it fits:** the BAGEL-7B-MoT model bundle. `pipeline.py` is trainside
> rollout, `diffusion.py` is the replay stage the trainer scores/backprops
> through, `rl_ops.py` is the single home for every helper both of those (and
> the vLLM-Omni worker) must share, `conditions.py` carries what replay needs
> across the Part boundary. Full map: [`../../README.md`](../../README.md).

## Why rl_ops is the only home for prefill helpers

Rollout earns the reward under one conditioning; replay takes the gradient
under another only if the two prefill differently — a mismatch **no importance
ratio corrects**, and nothing crashes: the curve just quietly degrades. So the
source-image transforms and the VAE/ViT/text prefill exist exactly once
(`build_image_transforms`, `resize_input_image`, `update_context_image`,
`update_context_text`), and trainside rollout (`BagelPipeline`), trainer replay
(`BagelDiffusionStage`), and the vLLM-Omni worker (`RLBagelPipeline`) all call
them. Do not fork these paths; extend them in place.

## Gotchas

- **Source images are VAE-encoded by posterior MEAN, not by sampling**
  (`rl_ops._encode_vae_posterior_mean`). Sampling would draw different noise at
  rollout and at replay-rebuild, breaking the ratio unfixably. Consequence:
  since #304 this also holds for trainside it2i rollout — conditioning is
  deterministic where the vendored `AutoEncoder.encode` used to sample. The
  `bagel_editreward.yaml` curve was re-baselined for this (see that PR's run);
  pre-#304 curves are not comparable step-for-step.
- **it2i grad replay rebuilds the source contexts through the CURRENT LoRA
  weights on every replay** (`_resolve_single` bypasses stored contexts when
  grads are enabled and `conditions.input_images` is non-empty). That is the
  point — stored contexts go stale across optimizer updates — but it costs a
  full image+text prefill per replay, and the prefill runs under activation
  checkpointing. `ac_wrap_order: inside` is the composition the it2i smoke
  validated; both EditReward recipes pin it. A recipe that enables
  `activation_checkpointing` on an image-conditioned BAGEL path without that
  pin is running an unvalidated composition.
- **Packed-inference prefill must run in eval dispatch.** Every navit module
  routes `forward_train` vs `forward_inference` on `self.training`, and the
  train stack leaves the model in train mode; `rl_ops.inference_dispatch_scope`
  forces eval for the rebuild. `forward_flow` deliberately restores mode
  asymmetrically — it stays in eval while grads are pending so a later
  `backward()`'s checkpoint recompute also fires in eval. Keep that asymmetry.
- **`force_rebuild` refuses prompt-only conditions** (`_resolve_single`
  requires `input_images`). Stored contexts may bake in text the raw prompt
  cannot reproduce — t2ti stores `init + system + think + prompt` — so a forced
  rebuild there would silently recondition the caller (UniGRPO's `v_ref`).
  Only image-conditioned conditions carry raw material that reproduces the
  stored layout.
- **ViT transform stride must equal the SigLIP patch size (14)**
  (`BAGEL_VIT_TRANSFORM_GEOMETRY`). `patchify` asserts `h % patch_size == 0`;
  flow_grpo ships stride 7 (half a patch), which crashes non-square inputs.
- **The vendored `prepare_vae_images` / `prepare_vit_images` build CPU
  tensors** and the bundle's VAE carries no accelerate hooks, so
  `rl_ops.update_context_image` moves `padded_images` and the packed index
  tensors to the bundle device before the cache update. Dropping those moves
  "works" on trainside colocate and fails only cross-process.
- **Trainside rollout does not strip `<|im_start|>` / `<|im_end|>` from
  prompts; the worker and the grad rebuild do.** Identical behavior for any
  normal prompt; a dataset instruction literally wrapped in those markers would
  condition trainside rollout and its grad replay on different token sequences.
