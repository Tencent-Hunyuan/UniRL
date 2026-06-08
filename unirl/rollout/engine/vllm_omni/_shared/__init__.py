"""Engine-internal utilities shared across model-specific RL pipelines.

Modules here are imported by per-model packages
(``unirl.rollout.engine.vllm_omni.hi3``,
``unirl.rollout.engine.vllm_omni.sd3``) inside the vLLM-Omni worker
subprocess. They depend on diffusers / vllm-omni and will fail outside
that environment.
"""

__all__: list[str] = []
