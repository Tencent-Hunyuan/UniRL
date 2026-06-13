"""Make SD3's rollout pipeline LoRA-capable on stock upstream sglang (LIN-365).

The sglang LoRA control path (``GPUWorker.set_lora`` and the migration's
``set_lora_from_tensors``) gates on ``isinstance(self.pipeline, LoRAPipeline)`` --
a model is "LoRA-enabled" only when its pipeline CLASS inherits the
``LoRAPipeline`` mixin. The fork (``sglang-drl``) declares
``class StableDiffusion3Pipeline(LoRAPipeline, ComposedPipelineBase)``; stock
upstream's is ``class StableDiffusion3Pipeline(ComposedPipelineBase)`` -- LoRA was
never added to SD3 upstream. So on upstream the separate-adapter weight-sync path
(``LocalLoraWeightSync`` -> ``set_lora_from_tensors``, used by
``conf/sd3_sglang_rollout_colocate.yaml``) fails the first sync with
``ValueError: set_lora_from_tensors failed: Lora is not enabled`` -- even though
``server_args.lora_target_modules`` is correctly populated, the pipeline simply
isn't a ``LoRAPipeline`` so the worker rejects the request
(``gpu_worker.py: if not isinstance(self.pipeline, LoRAPipeline): return
OutputBatch(error="Lora is not enabled")``).

Re-host the fork's declaration at runtime: inject ``LoRAPipeline`` into
``StableDiffusion3Pipeline.__bases__``. ``LoRAPipeline`` subclasses
``ComposedPipelineBase``, so the solid instance layout is unchanged and the
``__bases__`` reassignment is permitted; the resulting bases
``(LoRAPipeline, ComposedPipelineBase)`` are exactly the fork's. SD3 defines no
``__init__`` of its own (only ``create_pipeline_stages``), so instantiation now
runs ``LoRAPipeline.__init__`` (``super().__init__`` builds the stages first, then
LoRA setup reads ``server_args``). The AROUND-wrapped ``LoRAPipeline.__init__``
from ``patch_lora_tensors`` then eagerly wraps the LoRA layers in ``online`` mode
(no startup ``lora_path``), so the in-memory adapter has targets before the first
``set_lora_from_tensors``.

The merged-LoRA recipe ``sd3_sglang_full_tensor`` runs the engine with
``use_lora=false``, so ``lora_merge_mode`` stays null and the online prewrap is
skipped -- the pipeline is a ``LoRAPipeline`` holding no LoRA layers, which is the
fork's behaviour too (SD3 was always a ``LoRAPipeline`` regardless of sync mode),
so that path is unaffected.

Idempotent; ``__bases__`` injection + an ABCMeta-cache-invalidating
``LoRAPipeline.register`` (see the note in the body) -- no sglang source edits.
"""

from __future__ import annotations


def patch_sd3_lora_pipeline() -> None:
    from sglang.multimodal_gen.runtime.pipelines.stable_diffusion_3 import (
        StableDiffusion3Pipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
        LoRAPipeline,
    )

    # Mirror the fork's ``(LoRAPipeline, ComposedPipelineBase)`` by prepending the
    # mixin to upstream's existing ``(ComposedPipelineBase,)``. The idempotency
    # guard is a direct ``__bases__`` membership test -- deliberately NOT
    # ``issubclass`` (see the ABCMeta note below).
    if LoRAPipeline not in StableDiffusion3Pipeline.__bases__:
        StableDiffusion3Pipeline.__bases__ = (LoRAPipeline,) + StableDiffusion3Pipeline.__bases__

    # ABCMeta CACHE GOTCHA: ``ComposedPipelineBase`` is an ABC, so
    # ``isinstance`` / ``issubclass`` route through ABCMeta's per-class cache. A
    # ``__bases__`` reassignment does NOT bump the ABC invalidation counter, so a
    # negative ``issubclass(StableDiffusion3Pipeline, LoRAPipeline)`` cached before
    # the reassignment (an ``issubclass`` idempotency guard, or any earlier check
    # during import) STICKS: the bases show ``LoRAPipeline`` yet
    # ``isinstance(pipeline, LoRAPipeline)`` stays False and the worker still
    # rejects ``set_lora_from_tensors``. ``register`` bumps the global counter
    # (invalidating the stale negative) and records SD3 as a subclass.
    LoRAPipeline.register(StableDiffusion3Pipeline)


__all__ = ["patch_sd3_lora_pipeline"]
