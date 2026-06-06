"""Shared fakes for the vllm_omni_v2 CPU tests.

Adapters are constructed with cheap stand-ins (the adapter reads a handful of
declared fields); wire results are ``SimpleNamespace`` objects satisfying the
``OmniRawResult`` protocol structurally. No vllm-omni anywhere.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import torch

from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniV2EngineConfig
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams, ComposedSamplingParams, DiffusionSamplingParams


class FakeModelConfig:
    """The declared model-config fields the adapters/engine read."""

    shift = 3.0
    use_lora = True


@pytest.fixture
def model_config() -> FakeModelConfig:
    return FakeModelConfig()


def make_config(modality: str, **overrides: Any) -> VLLMOmniV2EngineConfig:
    return VLLMOmniV2EngineConfig(model_path="/ckpt", modality=modality, **overrides)


def fake_tokenize(text: str, *, task: str, sys_type: str) -> List[int]:
    """Canned tokenizer: length-keyed ids so tests can assert pass-through."""
    return [len(text), len(task), len(sys_type)]


def make_adapter(modality: str, model_config: Any, **cfg_overrides: Any):
    from unirl.rollout.engine.vllm_omni_v2.adapters import get_adapter

    return get_adapter(modality)(make_config(modality, **cfg_overrides), model_config, tokenize_fn=fake_tokenize)


def make_req(
    n: int = 2,
    *,
    modality_params: str = "diffusion",
    sigmas: Optional[torch.Tensor] = None,
    num_inference_steps: int = 2,
    primitives: Optional[Dict[str, Any]] = None,
    **req_overrides: Any,
) -> RolloutReq:
    sample_ids = [f"p{i}/r0" for i in range(n)]
    diffusion = DiffusionSamplingParams(
        num_inference_steps=num_inference_steps,
        height=64,
        width=64,
        guidance_scale=4.0,
        eta=1.0,
        seed=7,
    )
    if modality_params == "diffusion":
        sampling = diffusion
    elif modality_params == "ar":
        sampling = ARSamplingParams(temperature=0.5, max_new_tokens=16, top_p=0.9, top_k=50)
    else:  # composed: AR + diffusion (the HI3 two-stage shape)
        sampling = ComposedSamplingParams(
            diffusion=diffusion,
            ar=ARSamplingParams(temperature=0.5, max_new_tokens=16, top_p=0.9, top_k=50),
        )
    prims = {"text": Texts(texts=[f"prompt {i}" for i in range(n)])}
    prims.update(primitives or {})
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=[f"g{i}" for i in range(n)],
        primitives=prims,
        sampling_params=sampling,
        sigmas=sigmas,
        **req_overrides,
    )


def sigmas_for(num_inference_steps: int) -> torch.Tensor:
    """A valid [T+1] σ schedule ending at the terminal 0."""
    return torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Wire-result fakes (satisfy OmniRawResult structurally)
# --------------------------------------------------------------------------- #


def fake_ar_output(
    idx: int,
    *,
    token_ids: Optional[List[int]] = None,
    text: str = "ar text",
    with_logprobs: bool = True,
    prompt_token_ids: Optional[List[int]] = None,
) -> SimpleNamespace:
    tokens = token_ids if token_ids is not None else [10 + idx, 11 + idx]
    logprobs = [{t: SimpleNamespace(logprob=-0.5 - 0.1 * j)} for j, t in enumerate(tokens)] if with_logprobs else None
    return SimpleNamespace(
        request_id=f"{idx}_uuid",
        stage_id=0,
        final_output_type="text",
        request_output=SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(tokens), logprobs=logprobs, text=text)]),
        prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else [1, 2, 3 + idx],
        images=None,
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        custom_output=None,
    )


def fake_fused_capture(L: int = 4) -> Dict[str, Any]:
    return {
        "input_ids": torch.ones(1, L, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, L, L, dtype=torch.bool),
        "position_ids": torch.arange(L, dtype=torch.long).unsqueeze(0),
        "gen_image_mask": torch.zeros(1, L, dtype=torch.bool),
        "gen_timestep_scatter_index": torch.zeros(1, dtype=torch.long),
        "rope_cache": (torch.zeros(1, L, 8), torch.zeros(1, L, 8)),
    }


def fake_dit_output(
    idx: int,
    *,
    sigmas: torch.Tensor,
    stage_id: int = 1,
    final_output_type: str = "image",
    K: Optional[int] = None,
    sde_step_indices: Optional[List[int]] = None,
    custom_capture: Optional[Dict[str, Any]] = None,
    num_images: int = 1,
) -> SimpleNamespace:
    """One DiT-stage result: dense [1, T+1, C, H, W] trajectory + σ echo."""
    from PIL import Image as PILImage

    T_plus_1 = int(sigmas.shape[0])
    T = T_plus_1 - 1
    k = T if K is None else K
    custom: Dict[str, Any] = {"sde_step_indices": list(range(T))[:k] if sde_step_indices is None else sde_step_indices}
    if custom_capture:
        custom.update(custom_capture)
    return SimpleNamespace(
        request_id=f"{idx}_uuid",
        stage_id=stage_id,
        final_output_type=final_output_type,
        request_output=None,
        prompt_token_ids=None,
        images=[PILImage.new("RGB", (8, 8), color=(idx, 0, 0)) for _ in range(num_images)],
        trajectory_latents=torch.zeros(1, T_plus_1, 4, 2, 2),
        trajectory_timesteps=sigmas.clone(),
        trajectory_log_probs=(torch.zeros(1, k) if k > 0 else torch.zeros(1, 0)),
        custom_output=custom,
    )


def fake_sd3_capture() -> Dict[str, Any]:
    return {
        "text_capture": {
            "prompt_embeds": torch.zeros(1, 6, 8),
            "pooled_prompt_embeds": torch.zeros(1, 8),
        }
    }


def fake_hv15_capture() -> Dict[str, Any]:
    return {
        "text_capture": {
            "prompt_embeds": torch.zeros(1, 6, 8),
            "prompt_embeds_mask": torch.ones(1, 6, dtype=torch.bool),
            "prompt_embeds_2": torch.zeros(1, 4, 8),
            "prompt_embeds_mask_2": torch.ones(1, 4, dtype=torch.bool),
        }
    }
