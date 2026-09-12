from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unirl.reward.local.hpsv2 import HPSv2RewardScorer, HPSv2Spec


class _Model:
    def __call__(self, image_input: torch.Tensor, text_input: torch.Tensor):
        means = image_input.float().mean(dim=(1, 2, 3))
        image_features = torch.stack((means, torch.ones_like(means)), dim=-1)
        return {
            "image_features": image_features,
            "text_features": text_input.float(),
        }


def test_hpsv2_batches_rows_and_accepts_string_device() -> None:
    scorer = HPSv2RewardScorer.__new__(HPSv2RewardScorer)
    scorer.device = "cpu"
    scorer.batch_size = 2
    scorer.model = _Model()
    scorer._hpsv2_preprocess_val = lambda image: torch.tensor(np.asarray(image, dtype=np.float32) / 255.0).permute(
        2, 0, 1
    )
    text_weights = {"zero": 1.0, "one": 2.0, "three": 3.0}
    scorer._hpsv2_tokenizer = lambda prompts: torch.tensor([[text_weights[prompt], 1.0] for prompt in prompts])
    request = SimpleNamespace(
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ],
        prompts=["zero", "one", "three"],
    )
    assert scorer._compute_model_rewards(request) == [1.0, 3.0, 4.0]

    with pytest.raises(ValueError, match="one prompt per image"):
        scorer._compute_model_rewards(SimpleNamespace(images=request.images, prompts=["zero"]))


def test_hpsv2_uses_matrix_diagonal_score_arithmetic() -> None:
    image_features = torch.tensor([[0.1, 0.2, 0.3], [0.3, -0.1, 0.2]], dtype=torch.float16)
    text_features = torch.tensor([[0.4, -0.2, 0.5], [-0.3, 0.2, 0.1]], dtype=torch.float16)
    scorer = HPSv2RewardScorer.__new__(HPSv2RewardScorer)
    scorer.device = "cpu"
    scorer.batch_size = 2
    scorer.model = lambda image_input, text_input: {
        "image_features": image_features,
        "text_features": text_features,
    }
    scorer._hpsv2_preprocess_val = lambda image: torch.zeros(3, 2, 2)
    scorer._hpsv2_tokenizer = lambda prompts: torch.zeros(len(prompts), 1)
    request = SimpleNamespace(
        images=[np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
        prompts=["first", "second"],
    )
    expected = torch.diagonal(image_features @ text_features.T).float().tolist()
    assert scorer._compute_model_rewards(request) == expected


@pytest.mark.parametrize("batch_size", [0, True, 1.5])
def test_hpsv2_spec_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HPSv2Spec(batch_size=batch_size)
