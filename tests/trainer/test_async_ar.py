from unirl.trainer.async_ar import _rollout_dp_size_from_parsed_config


class _SGLangLikeRollout:
    _accepts_rollout_tp_kwargs = True


class _OrdinaryRollout:
    pass


def test_async_rollout_divisibility_uses_dp_size_not_gpu_count() -> None:
    assert (
        _rollout_dp_size_from_parsed_config(
            {
                "role_cls": _SGLangLikeRollout,
                "config": {"tp_size": 8, "pp_size": 1, "ep_size": 8},
            },
            world_size=8,
        )
        == 1
    )
    assert (
        _rollout_dp_size_from_parsed_config(
            {
                "role_cls": _SGLangLikeRollout,
                "config": {"tp_size": 4, "pp_size": 1, "ep_size": 4},
            },
            world_size=8,
        )
        == 2
    )
    assert (
        _rollout_dp_size_from_parsed_config(
            {
                "role_cls": _OrdinaryRollout,
                "config": {"tp_size": 8},
            },
            world_size=8,
        )
        == 8
    )
