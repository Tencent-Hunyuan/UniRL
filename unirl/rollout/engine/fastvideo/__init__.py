"""FastVideo rollout engine (third dedicated diffusion engine, alongside
``sglang_diffusion`` and ``vllm_omni``).

Wraps FastVideo's ``VideoGenerator`` (RL fork, hao-ai-lab/FastVideo PR #1222)
as an in-process colocate rollout engine for diffusion GRPO. Selected by Hydra
``_target_`` like every other engine; nothing here is imported unless the
FastVideo engine is actually constructed, so the existing engines are
unaffected.
"""
