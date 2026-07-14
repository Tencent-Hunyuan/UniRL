from types import SimpleNamespace

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.utils.sampling import resolve_sampling
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams


def _config(**overrides: object) -> SimpleNamespace:
    values = {
        "samples_pre_expanded": False,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9,
        "system_instruction": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_omitted_ar_top_k_disables_sglang_top_k() -> None:
    req = RolloutReq(sampling_params={"ar": ARSamplingParams()})

    sampling = resolve_sampling(_config(), req)

    assert sampling.block["top_k"] == -1


def test_explicit_positive_ar_top_k_is_preserved() -> None:
    req = RolloutReq(sampling_params={"ar": ARSamplingParams(top_k=1024)})

    sampling = resolve_sampling(_config(), req)

    assert sampling.block["top_k"] == 1024


def test_explicit_zero_ar_top_k_disables_sglang_top_k() -> None:
    req = RolloutReq(sampling_params={"ar": ARSamplingParams(top_k=0)})

    sampling = resolve_sampling(_config(), req)

    assert sampling.block["top_k"] == -1


def test_missing_ar_params_disable_sglang_top_k() -> None:
    sampling = resolve_sampling(_config(), RolloutReq())

    assert sampling.block["top_k"] == -1


def test_sglang_engine_default_top_k_matches_unrestricted_ar_default() -> None:
    config = SGLangEngineConfig(pretrained_model_ckpt_path="model")

    assert config.top_k == 0
