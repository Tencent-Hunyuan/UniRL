# Qwen-Image-Edit-Plus (Qwen-Image-Edit-2511) recipes

Image-editing RL recipes (DiffusionNFT / FlowGRPO / FlowDPPO) on the
`sglang_diffusion` colocate backend. The model consumes a source image plus an
edit instruction; the source image is VAE-encoded and token-concatenated to the
noise latent inside `predict_noise` (transformer `in_channels=64`).

## Known limitations

### 1. Batches must be single-aspect-ratio (mixed aspect ratios crash)

Upstream sglang's Edit-Plus preprocessing resizes each source image to
~1024x1024 **area while preserving its aspect ratio**, so images with different
aspect ratios produce different latent grid shapes (`H_img x W_img`). The
adapter's `_collect_image_latents`
(`unirl/rollout/engine/sglang_diffusion/adapters/qwen_image_edit_plus.py`)
stacks per-request source-image latents into a uniform `[B, C, H, W]` tensor
and deliberately raises on heterogeneous shapes rather than silently padding:

```
... heterogeneous shapes [...] — expected a uniform grid ...
```

**Consequence:** every rollout batch must contain source images of a single
aspect ratio. Datasets mixing portrait/landscape/square sources will crash
mid-training once a mixed batch is sampled.

**Workarounds until per-sample grids are supported:**
- Preprocess the dataset so all source images share one aspect ratio (e.g.
  center-crop to square), or
- Bucket the dataloader by aspect ratio so each batch is homogeneous.

### 2. `_coalesce_duplicate_single_sample_encodes` shape-heuristic dedup

`unirl/rollout/engine/sglang_diffusion/_patches/patch_conditions.py`
(`_coalesce_duplicate_single_sample_encodes`) deduplicates repeated
single-sample encoder outputs by **shape equality only**. This is correct for
the current models (each encoder in a pipeline produces a distinct shape), but
it is a heuristic: a future model whose pipeline runs two different encoders
that happen to emit same-shape tensors would have one encoder's output
silently dropped.

If you add such a model, harden the dedup with a value check (e.g.
`torch.equal`) before coalescing, or gate the coalescing per model adapter.
