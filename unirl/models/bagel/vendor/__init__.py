"""Vendored BAGEL model code — official ByteDance-Seed/Bagel plus documented local fixes.

Copied verbatim from ByteDance-Seed/Bagel (commit pinned in ``VENDOR_COMMIT.txt``).
The intended deviations from upstream are mechanical, plus the documented
compatibility/grad fixes below:

- import roots rewritten ``modeling.`` / ``data.`` / ``inferencer`` ->
  ``unirl.models.bagel.vendor.{modeling,data,inferencer}`` (9 statements across
  ``modeling/bagel/{bagel,qwen2_navit,siglip_navit}.py`` and ``inferencer.py``);
- the navit files import ``flash_attn_varlen_func`` through
  ``vendor/flash_attn_compat.py``. The shim uses the real fused kernel when available
  and otherwise imports the SDPA reimplementation from
  ``unirl/models/bagel/sdpa_varlen.py`` (no flash-attn, or an incompatible build such
  as flash-attn-4). The GENERATION / inference path goes through this; the training /
  replay path uses SDPA / ``flex_attention`` directly and is unaffected;
- added ``modeling/cache_utils/__init__.py`` (upstream ships ``cache_utils`` as a
  bare dir without an ``__init__``);
- only a subset of upstream ``data/`` is vendored (``data_utils.py`` +
  ``transforms.py`` + ``__init__.py``) — the training datasets are not needed for
  T2I RL rollout/replay;
- ONE grad-safety edit in ``modeling/bagel/qwen2_navit.py`` (``PackedAttention``
  gen-mode output): the upstream inference path writes the per-expert o_proj outputs
  back INPLACE into a view of the flash_attn custom-Function output, which autograd
  forbids during the RL replay BACKWARD ("Output 0 of ViewBackward0 is a view and is
  being modified inplace"). It is fine under ``no_grad`` (rollout / the diffuse↔replay
  ratio test), so only RL training trips it. The edit writes into a fresh
  ``torch.zeros_like`` tensor instead — mathematically identical, just grad-safe
  (mirrors flow_grpo's identical fix). Marked inline in that file.
- transformers 5 compatibility edit in ``modeling/{qwen2,bagel}/qwen2*.py``:
  ``PretrainedConfig`` no longer guarantees generation token-id attributes such as
  ``pad_token_id``. BAGEL's upstream config omitted that field and transformers 4
  exposed it as ``None``; the local edit uses ``getattr(..., None)`` to preserve
  that behavior under transformers 5. Also, transformers 5's
  ``ROPE_INIT_FUNCTIONS`` no longer exposes the ``"default"`` key, so the vendored
  Qwen2 rotary embedding keeps a local default RoPE fallback.
- reference inferencer dtype edit in ``inferencer.py``: UniRL loads the BAGEL VAE
  in bf16 for the pipeline path, while the upstream inferencer may hand fp32
  latents directly to ``vae.decode``. The local edit decodes through a temporary
  fp32 VAE cast, then restores the loaded dtype. This intentionally differs from
  ``BagelVAEDecodeStage``'s sticky-fp32 pipeline path: the standalone inferencer
  encodes through the VAE directly and does not cast encode I/O at the boundary.

Apart from those documented fixes the modeling is byte-pristine. The RL primitives
(SDE step + log-prob, window sampler, replay) live OUTSIDE this tree in
``unirl/models/bagel/rl_ops.py`` and call ``model._forward_flow`` (grad-enabled via
``__wrapped__``), so an upstream bump is a re-vendor + import-rewrite + re-applying
the documented local fixes. This subtree is excluded from repo lint/format.
"""
