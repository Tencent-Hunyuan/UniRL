import pytest
import torch

from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (1.25, torch.float32),
        (1.25, torch.float16),
        (1.25, torch.bfloat16),
        (True, torch.bool),
        (1 + 2j, torch.complex64),
    ],
)
def test_fingerprint_tensor_supports_scalar_dtypes(value, dtype) -> None:
    scalar = torch.tensor(value, dtype=dtype)
    vector = scalar.reshape(1)

    assert fingerprint_tensor(scalar) == fingerprint_tensor(scalar.clone())
    assert fingerprint_tensor(vector) == fingerprint_tensor(vector.clone())
    assert fingerprint_tensor(scalar) != fingerprint_tensor(vector)
