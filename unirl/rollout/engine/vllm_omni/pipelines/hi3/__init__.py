"""HI3-specific subclasses for the vLLM-Omni rollout engine.

Importing this package's ``RLHunyuanImage3Pipeline`` will fail outside a
vLLM-Omni-equipped environment because the parent class lives in
``vllm_omni``; that's intentional — these modules are only meant to be
imported inside vLLM-Omni's worker subprocess via
``custom_pipeline_args``.
"""

__all__: list[str] = []
