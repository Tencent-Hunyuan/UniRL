from types import SimpleNamespace

from unirl.reward.local.t2av_composite import T2AVCompositeScorer
from unirl.rollout.engine.vllm_omni.pipelines.minimax_h3 import MiniMaxH3RLPipeline
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Part


class _StubScorer:
    def __init__(self, response: RewardResponse) -> None:
        self.response = response

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        del request
        return self.response


def _composite(response: RewardResponse) -> T2AVCompositeScorer:
    scorer = T2AVCompositeScorer.__new__(T2AVCompositeScorer)
    scorer.weights = {"stub": 1.0}
    scorer._scorers = {"stub": _StubScorer(response)}
    return scorer


def _request(batch_size: int = 2) -> RewardRequest:
    return RewardRequest(generated={"video": [object() for _ in range(batch_size)]})


def test_t2av_composite_propagates_inner_failure() -> None:
    result = _composite(
        RewardResponse(
            rewards=[0.4, 0.5],
            successes=[True, False],
            errors=[None, "decode failed"],
        )
    ).compute_rewards(_request())

    assert result.successes == [False, False]
    assert all("decode failed" in str(error) for error in result.errors)


def test_t2av_composite_rejects_non_finite_reward() -> None:
    result = _composite(
        RewardResponse(
            rewards=[0.4, float("nan")],
            successes=[True, True],
            errors=[None, None],
        )
    ).compute_rewards(_request())

    assert result.successes == [False, False]
    assert all("non-finite" in str(error) for error in result.errors)


def test_noise_extra_args_always_include_stable_sample_ids() -> None:
    part = Part(sample_ids=["prompt:7/0", "prompt:7/1"])
    extra: dict[str, object] = {}
    params = SimpleNamespace(disable_driver_xt=True)

    pack_initial_noise_extra_args(extra, part, params, caller="test")

    assert extra["sde_sample_ids"] == ["prompt:7/0", "prompt:7/1"]


def test_h3_sde_key_uses_logical_sample_not_request_uuid() -> None:
    sampling = SimpleNamespace(
        extra_args={"sde_sample_ids": ["prompt:7/0", "prompt:7/1"]},
        num_outputs_per_prompt=1,
    )
    first = SimpleNamespace(sampling_params=sampling, request_id="1_first-uuid")
    retry = SimpleNamespace(sampling_params=sampling, request_id="1_retry-uuid")

    assert MiniMaxH3RLPipeline._sde_sample_key_for_request(first) == "prompt:7/1"
    assert MiniMaxH3RLPipeline._sde_sample_key_for_request(retry) == "prompt:7/1"
