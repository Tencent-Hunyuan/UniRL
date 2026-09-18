from __future__ import annotations

from contextlib import contextmanager

import pytest

from unirl.algorithms.diffusionnft import DiffusionNFT


class _Stage:
    pass


class _EMA:
    @contextmanager
    def use_shadow(self):
        yield


def test_positive_reference_deviation_requires_backend_model() -> None:
    with pytest.raises(ValueError, match="no `backend` was injected"):
        DiffusionNFT(
            params=object(),
            stage=_Stage(),
            nft_lora_policy=_EMA(),
            ref_deviation_coef=1.0e-4,
        )
