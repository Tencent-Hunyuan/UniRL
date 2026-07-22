"""Typed and frontier generation-Part selector regressions."""

from __future__ import annotations

import pytest
import torch

from unirl.sde.runtime import ensure_sample_sigmas
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _repeated_ar_sample() -> Sample:
    root = Part.input(["p"], primitives={"text": Texts(texts=["question"])})
    first = ARSamplingParams(max_new_tokens=8)
    second = ARSamplingParams(max_new_tokens=16)
    return (
        Sample.request(root)
        .fork(1, sampling_params=first)
        .observe(Texts(texts=["tool result"]))
        .fork(1, sampling_params=second)
    )


@pytest.mark.parametrize("selector", ["gen_part", "gen_part_index", "gen_part_or_none"])
def test_typed_gen_part_selectors_reject_duplicate_indices(selector: str) -> None:
    sample = _repeated_ar_sample()

    with pytest.raises(ValueError, match=r"multiple Parts.*indices \[1, 3\]"):
        getattr(sample, selector)(ARSamplingParams)


def test_frontier_gen_part_selects_latest_agent_turn() -> None:
    sample = _repeated_ar_sample()

    frontier = sample.frontier_gen_part(ARSamplingParams)

    assert frontier is sample.parts[-1]
    assert frontier.sampling_params.max_new_tokens == 16


def test_frontier_gen_part_rejects_non_generated_or_wrong_type_frontier() -> None:
    root = Part.input(["p"], primitives={"text": Texts(texts=["question"])})
    ar_sample = Sample.request(root).fork(1, sampling_params=ARSamplingParams())

    with pytest.raises(ValueError, match="expected DiffusionSamplingParams"):
        ar_sample.frontier_gen_part(DiffusionSamplingParams)

    observed = ar_sample.observe(Texts(texts=["tool result"]))
    with pytest.raises(ValueError, match="final Part at index 2 is not generated"):
        observed.frontier_gen_part(ARSamplingParams)


def test_base_type_lookup_treats_subclass_matches_as_ambiguous() -> None:
    class SpecializedARSamplingParams(ARSamplingParams):
        pass

    root = Part.input(["p"])
    sample = (
        Sample.request(root)
        .fork(1, sampling_params=ARSamplingParams())
        .fork(1, sampling_params=SpecializedARSamplingParams())
    )

    with pytest.raises(ValueError, match=r"indices \[1, 2\]"):
        sample.gen_part(ARSamplingParams)


def test_sigma_pinning_targets_only_the_repeated_diffusion_frontier() -> None:
    class Schedule:
        def compute_sigma(self, *, num_inference_steps: int, height: int, width: int) -> torch.Tensor:
            assert (num_inference_steps, height, width) == (3, 64, 96)
            return torch.tensor([1.0, 0.5, 0.0])

    root = Part.input(["p"], primitives={"text": Texts(texts=["question"])})
    old_params = DiffusionSamplingParams(num_inference_steps=1, height=32, width=32)
    frontier_params = DiffusionSamplingParams(num_inference_steps=3, height=64, width=96)
    sample = (
        Sample.request(root)
        .fork(1, sampling_params=old_params)
        .observe(Texts(texts=["next turn"]))
        .fork(1, sampling_params=frontier_params)
    )

    ensure_sample_sigmas(sample, Schedule())

    assert old_params.sigmas is None
    assert torch.equal(frontier_params.sigmas, torch.tensor([1.0, 0.5, 0.0]))
