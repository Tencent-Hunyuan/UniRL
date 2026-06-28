"""Vendored Janus model code from DeepSeek-AI/Janus.

Copied from the official DeepSeek-AI/Janus repository at the commit pinned in
``VENDOR_COMMIT.txt``. The intended deviations are mechanical:

- import roots rewritten from ``janus.{models,utils}`` to
  ``unirl.models.janus_pro.vendor.{models,utils}``;
- ``attrdict.AttrDict`` replaced by the local ``vendor.attrdict.AttrDict`` shim;
- ``models/modeling_vlm.py`` uses ``dataclasses.field(default_factory=AttrDict)``
  for config ``params`` fields, because transformers 5.x dataclassifies
  ``PretrainedConfig`` subclasses and rejects mutable class defaults.
- ``models/modeling_vlm.py`` defines an empty ``all_tied_weights_keys`` mapping
  on the multimodal wrapper, matching transformers 5.x loader expectations.
- ``models/modeling_vlm.py`` clones token embeddings before replacing image
  placeholder rows, avoiding an inplace write into a grad-carrying tensor during
  RL replay/backward.

Keep Janus-Pro RL logic outside this subtree; an upstream bump should be a
re-vendor plus the mechanical rewrites above.
"""
