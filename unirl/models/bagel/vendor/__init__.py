"""Vendored BAGEL model code — PRISTINE official ByteDance-Seed/Bagel.

Copied verbatim from ByteDance-Seed/Bagel (commit pinned in ``VENDOR_COMMIT.txt``).
The ONLY intended deviations from upstream are mechanical:

- import roots rewritten ``modeling.`` / ``data.`` / ``inferencer`` ->
  ``unirl.models.bagel.vendor.{modeling,data,inferencer}`` (9 statements across
  ``modeling/bagel/{bagel,qwen2_navit,siglip_navit}.py`` and ``inferencer.py``);
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

Apart from that one grad-safety fix the modeling is byte-pristine. The RL primitives
(SDE step + log-prob, window sampler, replay) live OUTSIDE this tree in
``unirl/models/bagel/rl_ops.py`` and call ``model._forward_flow`` (grad-enabled via
``__wrapped__``), so an upstream bump is a re-vendor + import-rewrite + re-applying
the single qwen2_navit grad fix. This subtree is excluded from repo lint/format.
"""
