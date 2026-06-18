"""Worker-side RL pipeline subclasses (role 8 — worker subprocess).

One package per model, each holding the pipeline subclass the v2 stage YAMLs
install via ``engine_args.custom_pipeline_args.pipeline_class``:

- ``hi3.pipeline.RLHunyuanImage3Pipeline``
- ``hv15.pipeline.RLHunyuanVideo15Pipeline``
- ``sd3.pipeline.RLStableDiffusion3Pipeline``
- ``_shared.flow_match_sde_scheduler`` — the SDE scheduler they share
- ``_shared.interception`` — the shared install/arm/harvest mechanics
  (vllm-omni-free, CPU-tested)

Each ``forward`` follows the interception protocol — install (once) → arm
(every request) → run (upstream's stages, with our taps/injectors firing
inside) → harvest (export trajectory latents / σ echo / SDE log-probs +
the ``custom_output`` condition captures onto the wire) — so a pipeline
here is paired with one adapter in ``adapters/`` that consumes its exports.

The family modules import diffusers / vllm-omni at module level and are only
meant to be imported inside the vllm-omni worker subprocess (qualname
resolution) — do not import them from driver-side code.
"""
