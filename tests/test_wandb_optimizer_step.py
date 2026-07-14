from types import SimpleNamespace

from unirl.utils.wandb_logger import UniRLWandBLogger


def _result(*, updates=(), has_backward: bool):
    return SimpleNamespace(
        per_update=tuple(updates),
        has_backward=has_backward,
        loss=1.0,
        grad_norm=2.0,
        lr=3.0,
        metrics={},
    )


def test_disabled_logger_still_tracks_multi_update_optimizer_state() -> None:
    logger = UniRLWandBLogger(enabled=False)
    result = _result(
        updates=tuple({"loss": float(index)} for index in range(4)),
        has_backward=True,
    )
    logger.log_rollout_step(0, result, resp=None)
    assert logger.optimizer_step == 4


def test_disabled_logger_matches_multitrack_axis_count() -> None:
    logger = UniRLWandBLogger(enabled=False, optimizer_step=7)
    results = {
        "diffusion": _result(
            updates=({"loss": 1.0}, {"loss": 2.0}),
            has_backward=True,
        ),
        "ar": _result(has_backward=True),
    }
    logger.log_rollout_step(0, results, resp=None)
    assert logger.optimizer_step == 9


def test_disabled_logger_does_not_advance_without_backward() -> None:
    logger = UniRLWandBLogger(enabled=False, optimizer_step=3)
    result = _result(has_backward=False)
    logger.log_rollout_step(0, result, resp=None)
    assert logger.optimizer_step == 3


def test_disabled_logger_count_matches_live_train_axis_slots() -> None:
    cases = [
        _result(has_backward=True),
        _result(has_backward=False),
        _result(updates=({"loss": 1.0},), has_backward=True),
        _result(updates=({"loss": 1.0},), has_backward=False),
        _result(
            updates=({"loss": 1.0}, {"loss": 2.0}),
            has_backward=False,
        ),
        {
            "diffusion": _result(
                updates=(
                    {"loss": 1.0},
                    {"loss": 2.0},
                    {"loss": 3.0},
                ),
                has_backward=True,
            ),
            "ar": _result(
                updates=({"loss": 1.0}, {"loss": 2.0}),
                has_backward=True,
            ),
        },
        {
            "diffusion": _result(has_backward=False),
            "ar": _result(has_backward=False),
        },
        {},
    ]

    for results in cases:
        disabled = UniRLWandBLogger(enabled=False, optimizer_step=7)
        disabled.log_rollout_step(0, results, resp=None)

        live = UniRLWandBLogger(enabled=False, optimizer_step=7)
        live.enabled = True
        live._initialized = True
        emitted_steps = []

        def record_step(step, _metrics, prefix="train/"):
            del prefix
            emitted_steps.append(step)

        live.log_step = record_step
        live._log_train(results)

        assert live.optimizer_step == disabled.optimizer_step
        assert len(emitted_steps) == live.optimizer_step - 7
