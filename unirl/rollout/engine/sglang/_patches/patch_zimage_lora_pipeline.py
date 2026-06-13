"""Make Z-Image's rollout pipeline LoRA-capable on stock upstream sglang.

Mirror of :mod:`patch_sd3_lora_pipeline`. The sglang LoRA control path
(``GPUWorker.set_lora`` and ``set_lora_from_tensors``) gates on
``isinstance(self.pipeline, LoRAPipeline)`` -- a model is "LoRA-enabled" only
when its pipeline CLASS inherits the ``LoRAPipeline`` mixin. If the installed
``ZImagePipeline`` is declared as ``class ZImagePipeline(ComposedPipelineBase)``
(no LoRA mixin), the separate-adapter weight-sync path
(``LocalLoraWeightSync`` -> ``set_lora_from_tensors``, used by
``z_image_sglang_replay_colocate.yaml``) fails the first sync with
``ValueError: set_lora_from_tensors failed: Lora is not enabled``.

This patch is **defensive and idempotent**: the ``celve/sglang`` fork already
declares ``class ZImagePipeline(LoRAPipeline, ComposedPipelineBase)`` (see
``runtime/pipelines/zimage_pipeline.py``), so when that declaration is present
the ``__bases__`` membership guard short-circuits and only the harmless
``LoRAPipeline.register`` runs. When the installed wheel ships the bare
``(ComposedPipelineBase,)`` form, the patch re-hosts the fork's declaration at
runtime by injecting ``LoRAPipeline`` into ``ZImagePipeline.__bases__``.

``LoRAPipeline`` subclasses ``ComposedPipelineBase``, so the solid instance
layout is unchanged and the ``__bases__`` reassignment is permitted; the
resulting bases ``(LoRAPipeline, ComposedPipelineBase)`` match the fork's.
Z-Image defines no ``__init__`` of its own (only ``create_pipeline_stages``),
so instantiation runs ``LoRAPipeline.__init__``; the AROUND-wrapped
``LoRAPipeline.__init__`` from ``patch_lora_tensors`` then eagerly wraps the
LoRA layers in ``online`` mode so the in-memory adapter has targets before the
first ``set_lora_from_tensors``.

Idempotent; ``__bases__`` injection + an ABCMeta-cache-invalidating
``LoRAPipeline.register`` (see ``patch_sd3_lora_pipeline`` for the ABCMeta cache
gotcha) -- no sglang source edits.
"""

from __future__ import annotations


def patch_zimage_lora_pipeline() -> None:
    from sglang.multimodal_gen.runtime.pipelines.zimage_pipeline import (
        ZImagePipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
        LoRAPipeline,
    )

    # Prepend the mixin to whatever bases the installed wheel declares. The
    # idempotency guard is a direct ``__bases__`` membership test -- deliberately
    # NOT ``issubclass`` (ABCMeta caches a stale negative; see the note below).
    if LoRAPipeline not in ZImagePipeline.__bases__:
        ZImagePipeline.__bases__ = (LoRAPipeline,) + ZImagePipeline.__bases__

    # ABCMeta CACHE GOTCHA: ``ComposedPipelineBase`` is an ABC, so
    # ``isinstance`` / ``issubclass`` route through ABCMeta's per-class cache. A
    # ``__bases__`` reassignment does NOT bump the ABC invalidation counter, so a
    # negative ``issubclass(ZImagePipeline, LoRAPipeline)`` cached before the
    # reassignment STICKS: the bases show ``LoRAPipeline`` yet
    # ``isinstance(pipeline, LoRAPipeline)`` stays False and the worker still
    # rejects ``set_lora_from_tensors``. ``register`` bumps the global counter
    # (invalidating the stale negative) and records Z-Image as a subclass. Safe
    # to call even when ZImagePipeline already inherits LoRAPipeline directly.
    LoRAPipeline.register(ZImagePipeline)


__all__ = ["patch_zimage_lora_pipeline"]
