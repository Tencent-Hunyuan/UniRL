from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.algorithms.bagel_flow_unigrpo import BagelFlowUniGRPO
from unirl.algorithms.base import AlgorithmStepResult
from unirl.models.bagel.ar import BagelARStage
from unirl.models.bagel.conditions import BagelARConditions
from unirl.models.bagel.rl_ops import prefill_prompt_text
from unirl.types.segments import TextSegment, make_image_segment


def test_ar_replay_uses_bundle_compute_device_when_fsdp_shards_are_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_device = torch.device("meta")
    bundle = SimpleNamespace(
        device=execution_device,
        transformer=nn.Linear(2, 2),
        new_token_ids={"bos_token_id": 1},
    )
    stage = object.__new__(BagelARStage)
    stage.model = bundle
    stage.replay_mode = "train"
    stage.logprob_dtype = torch.float32
    captured: dict[str, torch.device] = {}

    def replay_train(_conditions, **kwargs):
        captured["device"] = kwargs["device"]
        return [torch.tensor([0.0])]

    monkeypatch.setattr(stage, "_replay_train", replay_train)
    conditions = BagelARConditions.for_sample(splits=[{"kind": "text", "ids": torch.tensor([2, 3], dtype=torch.long)}])
    segment = TextSegment.pack(tokens=[torch.tensor([4])], log_probs=[torch.tensor([0.0])])

    stage.replay(conditions, segment=segment)

    assert next(bundle.transformer.parameters()).device.type == "cpu"
    assert captured["device"] == execution_device


def test_unigrpo_mse_uses_bundle_compute_device_when_fsdp_shards_are_on_cpu() -> None:
    execution_device = torch.device("meta")

    class StopAfterDeviceCapture(Exception):
        pass

    class FakeStage:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                device=execution_device,
                transformer=nn.Linear(2, 2),
            )
            self.captured_device = None
            self.context_grad_enabled = None

        def build_forward_kwargs(self, _conditions, *, params, device):
            del params
            self.captured_device = device
            self.context_grad_enabled = torch.is_grad_enabled()
            raise StopAfterDeviceCapture

    stage = FakeStage()
    algorithm = BagelFlowUniGRPO(
        params=object(),
        stage=stage,
        mse_weight=1.0,
        ratio_norm=True,
    )
    algorithm._ratio_norm_surrogate = lambda **_kwargs: AlgorithmStepResult(
        loss=0.0,
        metrics={},
        num_steps_or_tokens=1,
        has_backward=True,
    )
    segment = make_image_segment(
        sigmas=torch.tensor([1.0, 0.0]),
        sde_indices=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(StopAfterDeviceCapture):
        algorithm.compute_loss_and_backward(
            conditions={},
            segment=segment,
            advantages=torch.ones(1),
            training_progress=0.0,
            loss_scale=1.0,
        )

    assert next(stage.model.transformer.parameters()).device.type == "cpu"
    assert stage.captured_device == execution_device
    assert stage.context_grad_enabled is False


def test_prompt_prefill_moves_vendor_cpu_tensors_to_compute_device() -> None:
    execution_device = torch.device("meta")
    seen: dict[str, torch.device] = {}

    class FakeBagel:
        def prepare_prompts(self, **_kwargs):
            return (
                {
                    "packed_text_ids": torch.tensor([1, 2]),
                    "packed_text_indexes": torch.tensor([0, 1]),
                    "key_values_lens": torch.tensor([0]),
                },
                [2],
                [2],
            )

        def forward_cache_update_text(self, _past, **generation_input):
            seen.update({name: tensor.device for name, tensor in generation_input.items()})
            return "advanced-cache"

    context = prefill_prompt_text(
        FakeBagel(),
        {"kv_lens": [0], "ropes": [0], "past_key_values": "empty-cache"},
        prompt="hello",
        tokenizer=object(),
        new_token_ids={"bos_token_id": 1, "eos_token_id": 2},
        device=execution_device,
    )

    assert set(seen.values()) == {execution_device}
    assert context == {"kv_lens": [2], "ropes": [2], "past_key_values": "advanced-cache"}
