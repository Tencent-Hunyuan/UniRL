"""Vendored Boogu-Image model code — official boogu-project/Boogu-Image, flattened.

Copied from boogu-project/Boogu-Image (commit pinned in ``VENDOR_COMMIT.txt``).
Unlike bagel's whole-subpackage copy, this vendor dir is FLAT: the 8 files are
cherry-picked from three upstream dirs (``boogu/models/transformers/``,
``boogu/models/``, ``boogu/utils/``). The pipeline, scheduler, LoRA-loader, and
cache/acceleration packages are intentionally NOT vendored — UniRL reimplements
sampling as stages, and the released static-v1 time shift is expressed through
``FlowMatchSchedulePolicy.static_only(exp(1.15))`` (see
``unirl/models/boogu_image/pipeline.py``).

Intended deviations from upstream (all mechanical, marked inline with
"UniRL vendor edit"):

- import flattening: ``...utils.teacache_util`` / ``..attention_processor`` /
  ``..embeddings`` / ``...utils.import_utils`` -> same-dir relative imports;
- triton-RMSNorm conditionals (gated on ``is_triton_available()`` + a ``device``
  env var) -> ``torch.nn.RMSNorm`` (the upstream default-env branch) in
  ``transformer_boogu.py`` and ``block_lumina2.py``;
- flash-swiglu conditional -> pure-torch ``components.swiglu`` (upstream
  default-env branch) in ``block_lumina2.py``;
- attention processors: the four ``os.getenv("device", "cpu")`` selection gates
  in ``BooguImageTransformerBlock.__init__`` and
  ``BooguImageDoubleStreamTransformerBlock.__init__`` are pinned to the SDPA
  processors (``BooguImageAttnProcessor`` /
  ``BooguImageDoubleStreamSelfAttnProcessor``). Training numerics must not
  depend on process environment; the Flash2Varlen processor classes remain in
  ``attention_processor.py`` (its guarded ``flash_attn`` import is pristine)
  for an optional post-load swap via the bundle's ``attention_backend`` config
  — SDPA and Flash2Varlen variants share identical parameter names, so
  checkpoints and LoRA adapters are backend-independent;
- TaylorSeer/TeaCache: ``boogu.cache_functions`` / ``boogu.taylorseer_utils``
  imports -> ``taylorseer_stubs.py`` (raising stubs). Every call site is gated
  on ``enable_taylorseer`` / ``enable_teacache`` flags that default False and
  are asserted False by ``BooguImageBundle.from_config``; ``teacache_util.py``
  is vendored verbatim because ``TeaCacheParams`` is instantiated
  unconditionally in the transformer ``__init__``;
- orphaned ``import os`` / ``import warnings`` removed where the rewrites made
  them unused.

Byte-pristine files: ``rope.py``, ``embeddings.py``, ``components.py``,
``import_utils.py``, ``teacache_util.py``.
"""

from .transformer_boogu import BooguImageTransformer2DModel, PromptEmbedding

__all__ = ["BooguImageTransformer2DModel", "PromptEmbedding"]
