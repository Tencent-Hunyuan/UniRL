"""Worker-side RL pipeline subclasses (role 8 — worker subprocess).

One package per model, each holding the pipeline subclass the v2 stage YAMLs
install via ``engine_args.custom_pipeline_args.pipeline_class``:

- ``hi3.pipeline.RLHunyuanImage3Pipeline`` (+ its SDE scheduler)
- ``hv15.pipeline.RLHunyuanVideo15Pipeline``
- ``sd3.pipeline.RLStableDiffusion3Pipeline``
- ``_shared.flow_match_sde_scheduler`` — the SDE scheduler they share

Each runs the actual denoise loop and captures the dense artifacts the
trainer's replay needs (trajectory latents / σ echo / SDE log-probs, plus the
``custom_output`` condition captures the driver-side adapters consume), so a
pipeline here is paired with one adapter in ``adapters/``.

These modules import diffusers / vllm-omni at module level and are only meant
to be imported inside the vllm-omni worker subprocess (qualname resolution) —
do not import them from driver-side code.
"""
