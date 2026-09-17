import importlib.util
from pathlib import Path

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


def test_signed_backward_requires_and_produces_nonzero_gradient():
    module = torch.nn.Linear(2, 2, bias=False)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    log_probs = module(inputs).reshape(-1)

    objective, grad_norm = alignment._signed_backward(log_probs, module)

    assert torch.isfinite(torch.tensor(objective))
    assert grad_norm > 0.0
