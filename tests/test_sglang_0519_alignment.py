import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_alignment_module():
    path = Path(__file__).parents[1] / "scripts" / "sglang_0519_alignment.py"
    spec = importlib.util.spec_from_file_location("sglang_0519_alignment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


alignment = _load_alignment_module()


def test_ar_alignment_uses_production_fsdp_target():
    engine_kwargs = alignment._ar_engine_kwargs(0.3)
    replay_kwargs = alignment._ar_replay_kwargs()

    assert engine_kwargs["rl_on_policy_target"] == "fsdp"
    assert engine_kwargs["attention_backend"] == "triton"
    assert engine_kwargs["disable_cuda_graph"] is False
    assert engine_kwargs["cuda_graph_max_bs_decode"] == 16
    assert "enable_memory_saver" not in engine_kwargs
    assert "enable_deterministic_inference" not in engine_kwargs
    assert replay_kwargs["model_precision"] == "fp32"
    assert replay_kwargs["attn_implementation"] == "flex_attention"
    assert replay_kwargs["autocast_precision"] == "bf16"
    assert replay_kwargs["logprob_precision"] == "fp32"


def test_ep_alignment_matches_production_moe_without_fsdp_target():
    engine_kwargs = alignment._ep_engine_kwargs(0.4)

    assert engine_kwargs["mem_fraction_static"] == 0.4
    assert engine_kwargs["attention_backend"] == "triton"
    assert engine_kwargs["enable_deterministic_inference"] is True
    assert "rl_on_policy_target" not in engine_kwargs


def test_tensor_digest_is_bitwise_and_dtype_sensitive():
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)

    assert alignment._tensor_digest(values) == alignment._tensor_digest(values.clone())
    assert alignment._tensor_digest(values) != alignment._tensor_digest(values.to(torch.float16))


def test_nested_exact_rejects_float_drift():
    left = {"value": torch.tensor([1.0], dtype=torch.float32)}
    right = {"value": torch.tensor([1.0 + 1.0e-6], dtype=torch.float32)}

    with pytest.raises(AssertionError, match="not bitwise identical"):
        alignment._assert_nested_exact("root", left, right)


def test_drift_reports_mean_and_max():
    actual = torch.tensor([1.0, 2.0])
    expected = torch.tensor([0.5, 1.0])

    assert alignment._drift(actual, expected) == {"mean": 0.75, "max": 1.0}


def test_output_difference_checks_tokens_and_log_probs():
    baseline = SimpleNamespace(
        segment=SimpleNamespace(
            tokens=torch.tensor([1, 2]),
            log_probs=torch.tensor([-1.0, -2.0]),
        )
    )
    changed_logp = SimpleNamespace(
        segment=SimpleNamespace(
            tokens=torch.tensor([1, 2]),
            log_probs=torch.tensor([-1.0, -2.1]),
        )
    )

    assert not alignment._outputs_differ(baseline, baseline)
    assert alignment._outputs_differ(baseline, changed_logp)


def test_nested_difference_reports_exact_and_changed_values():
    exact = {"nested": [torch.tensor([1.0, 2.0])]}
    changed = {"nested": [torch.tensor([1.0, 3.0])]}

    assert not alignment._nested_differs("exact", exact, exact)
    assert alignment._nested_differs("changed", exact, changed)


def test_signed_backward_requires_and_produces_nonzero_gradient():
    module = torch.nn.Linear(2, 2, bias=False)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    log_probs = module(inputs).reshape(-1)

    objective, grad_norm = alignment._signed_backward(log_probs, module)

    assert torch.isfinite(torch.tensor(objective))
    assert grad_norm > 0.0
