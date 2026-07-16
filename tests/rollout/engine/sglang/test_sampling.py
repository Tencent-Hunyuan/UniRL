from types import SimpleNamespace

from unirl.rollout.engine.sglang.utils.sampling import resolve_sampling
from unirl.types import ARSamplingParams, Part, Sample


def _config(**overrides):
    values = {
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "top_k": 0,
        "system_instruction": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_sampling_uses_frontier_fanout_and_unrestricted_top_k() -> None:
    root = Part.input(
        ["prompt-0", "prompt-1"],
        control={"ar": {"stop": ["</answer>"], "system_instruction": "Be concise"}},
    )
    params = ARSamplingParams(
        samples_per_prompt=3,
        temperature=0.8,
        max_new_tokens=128,
        top_p=0.95,
        top_k=0,
    )
    sample = Sample.request(root).fork(3, sampling_params=params)

    resolved = resolve_sampling(_config(top_k=1024), sample)

    assert resolved.n == 3
    assert resolved.system_instruction == "Be concise"
    assert resolved.block == {
        "temperature": 0.8,
        "max_new_tokens": 128,
        "top_p": 0.95,
        "top_k": -1,
        "n": 3,
        "stop": ["</answer>"],
    }


def test_resolve_sampling_preserves_positive_top_k() -> None:
    params = ARSamplingParams(samples_per_prompt=1, top_k=64)
    sample = Sample.request(Part.input(["prompt-0"])).fork(1, sampling_params=params)

    resolved = resolve_sampling(_config(), sample)

    assert resolved.block["top_k"] == 64
