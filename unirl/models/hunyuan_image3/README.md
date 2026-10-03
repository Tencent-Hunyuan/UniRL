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
- The checkpoint's batched chat-template path encodes only the first sample:
  `apply_general_template(batchify=True)` passes a one-element `prompt_list` to `batch_gen_infer`
  and `zip`s it against the per-sample kwargs, so `apply_chat_template(batch_prompt=[...])`
  returns `cfg_factor` rows instead of `len(prompts) * cfg_factor`. `_apply_chat_template` raises
  on that mismatch, which otherwise surfaces as a `Part.fill` count mismatch on the AR path or a
  `fused.input_ids` shape mismatch on the diffusion path. `embed_for_ar` therefore encodes one
  row per call, `concat`s the fused rows, and rebuilds the batched `tokenizer_output` itself:
  upstream decode reads `real_pos` (`[B, 1]`, equal to `fused.prompt_lengths`) and the
  attention-mask builder reads the per-sample image slices. `embed_for_gen_image` still takes the
  batched call, so the gen_image recipes keep `rollout.forward_batch_size: 1` until the ragged
  diffusion concat (fully masked pad query rows, zero-padded `rope_cache`) is checked on the real
  model (issue #520).
