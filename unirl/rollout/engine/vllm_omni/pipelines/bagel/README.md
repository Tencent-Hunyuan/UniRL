# BAGEL worker pipeline (vLLM-Omni)

> **Where it fits:** the worker-side RL pipeline loaded into vLLM-Omni's DiT
> worker via `custom_pipeline_args`. It taps the trajectory (SDE scheduler,
> noise, σ echo) and, for it2i, builds the editing KV contexts itself. The
> trainer-side twin of every prefill it runs lives in
> [`unirl/models/bagel/rl_ops.py`](../../../../../models/bagel/README.md) —
> read that README's one-home rule first.

## Gotchas

All verified against the pinned vllm-omni tag in `pyproject.toml`; re-verify
each on any engine bump.

- **it2i must inject KV and bypass upstream's img2img branch entirely**
  (`_inject_it2i_contexts`; upstream skips ALL of its own prefill once
  `sampling_params.past_key_values` is set). Letting upstream's branch run
  would (1) override the output canvas with the resized SOURCE dims — the
  driver authored x_T for the requested height/width, so the noise tap's shape
  check fails — and (2) preprocess the source with a stride-`latent_downsample`
  resize plus a fixed 980x980 `SiglipImageProcessor` squash (~4x the ViT
  tokens, aspect ratio dropped) instead of the vendored navit transforms the
  trainer replays with. That is a rollout/replay conditioning mismatch no
  importance ratio can correct.
- **`ropes` must ride `kv_metadata`.** An image block advances `kv_lens` by
  `num_img_tokens + 2` but rope by ONE (the whole block shares a single
  position). Upstream's `ropes = [seq_len]` fallback coincides for text-only
  t2i and is off by thousands for image-bearing contexts.
- **Repoint the request at a SHALLOW COPY of its sampling params before
  writing KV into them.** Omni shares one params object across the requests of
  a generate call; writing in place leaks this request's caches into its
  siblings.
- **Packed rollout is t2i-only.** Upstream's grouped `generate_image` is cfg=1
  t2i; `_is_batchable_t2i` here and the adapter's `_is_packable_t2i` both
  reject image-bearing requests, so it2i runs one request per sample. Keep the
  two gates in agreement.
- **`kv_metadata["image_shape"]` carries the REQUESTED canvas**, not the
  source image's dims — the driver's x_T recipe and the σ echo are authored
  for it.
