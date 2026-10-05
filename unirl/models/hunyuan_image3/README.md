# HunyuanImage3 (Tencent HunyuanImage-3.0) — mixed AR + diffusion package

> **Where it fits:** one model package under [`unirl/models/`](../README.md), covering both the
> `gen_text` AR path (`ar.py`) and the `gen_image` diffusion path (`diffusion.py`) of the same
> checkpoint.

## Gotchas

The transformer is upstream `trust_remote_code` (`modeling_hunyuan_image_3.py`), and `ar.py`
drives its `prepare_inputs_for_generation` / `_update_model_kwargs_for_generation` step by step.
Re-check these when the checkpoint revision moves:

- `HunyuanStaticCache(dynamic=True)` trims KV to one scalar end and asserts
  `len(cache_position) == 1`, so only `B == 1` uses it. `B > 1` uses the full static cache, and
  since upstream drops the mask in decode, `step` masks row `i` to keys `[0, real_pos_i + step)`
  and right-pads the prefill mask to `max_cache_len`.
- Upstream sets the decode `position_ids` from `tokenizer_output.real_pos` and reads the next
  input via `input_ids.gather(1, position_ids)`. So `autoregress` requires `tokenizer_output`,
  and the token sampled at step `s` is written to column `real_pos_i + s`, not appended.
- `position_ids` are also the KV-cache write indices (`cache_position`), so
  `HunyuanImage3FusedMultimodalCondition.concat` pads them with continuing indices; a `0` pad
  would overwrite the first real token's KV.
- The Instruct checkpoint tokenizer at revision `2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`
  passes `prompt_list=[[]]` in `apply_general_template(batchify=True)`, so the `zip` in
  `batch_gen_infer` drops all but the first sample. `repair_hi3_tokenizer_batchify` installs
  an instance-local, idempotent wrapper that supplies one empty prompt per message list.
  Non-batched calls and the checkpoint's CFG ordering, `make_batch` padding, image slices,
  and `real_pos` (`[B, 1]`) remain upstream-owned. Both AR and diffusion use this repair
  before building their attention masks over the full padded batch.
  Recheck this shim when the checkpoint changes and remove it once upstream fixes the bug;
  the returned-row-count check remains a tripwire. Shipped recipes still use
  `rollout.forward_batch_size: 1` pending real-checkpoint GPU validation.

## Tokenizer regression check

Run `python -m pytest -q tests/test_hi3_tokenizer_batchify.py` with PyTorch,
transformers, diffusers, huggingface-hub and pytest installed. The first run downloads
only tokenizer code/config/assets from the pinned Instruct revision above (no model
weights). Set `HI3_TOKENIZER_PATH` to a local copy of those three files to run offline.
The tests reproduce truncation before the repair and compare every output field with
the upstream source with only `prompt_list` corrected, covering ragged text, image
conditioning, image generation, B=1/B=3, and CFG factors 1/2/3. They do not validate
model generation, log-probabilities, or GPU attention backends.
