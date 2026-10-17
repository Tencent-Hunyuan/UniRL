"""Vendored Janus model code from DeepSeek-AI/Janus.

Copied from the official DeepSeek-AI/Janus repository at the commit pinned in
``VENDOR_COMMIT.txt``. The declared integration deviations are:

- import roots rewritten from ``janus.{models,utils}`` to
  ``unirl.models.janus_pro.vendor.{models,utils}``;
- ``attrdict.AttrDict`` replaced by a local minimal shim covering the pinned
  Janus-Pro checkpoint configs;
- ``models/modeling_vlm.py`` uses ``dataclasses.field(default_factory=AttrDict)``
  for config ``params`` fields, because transformers 5.x dataclassifies
  ``PretrainedConfig`` subclasses and rejects mutable class defaults.
- ``models/modeling_vlm.py`` defines an empty ``all_tied_weights_keys`` mapping
  on the multimodal wrapper, matching transformers 5.x loader expectations.
- ``models/modeling_vlm.py`` clones grad-carrying token embeddings before
  replacing image-placeholder rows; frozen/no-grad paths retain upstream's
  in-place update.
- ``models/modeling_vlm.py`` exposes aligned image embeddings separately and
  accepts a cached value, allowing UniRL replay to skip the frozen vision tower.
- ``models/vq_model.py`` restores the input dtype after fp32 interpolation
  instead of hardcoding bf16, preserving upstream bf16 behavior while allowing
  fp16 model loading.
- ``models/modeling_vlm.py`` drops the five nested generic-config
  ``AutoConfig.register`` calls (``vision`` / ``aligner`` / ``gen_vision`` /
  ``gen_aligner`` / ``gen_head``); ``MultiModalityConfig.__init__`` constructs
  those sub-configs directly. The three registrations the loader relies on
  stay: ``multi_modality`` on ``AutoConfig`` and ``AutoModelForCausalLM``
  (``modeling_vlm.py``), and ``VLMImageProcessor`` on ``AutoImageProcessor``
  (``image_processing_vlm.py``).

Keep Janus-Pro RL logic outside this subtree; an upstream bump should be a
re-vendor plus the mechanical rewrites above.
"""
