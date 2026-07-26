import argparse

import pytest

from benchmarks.core.score import _geneval2_answer_token_ids
from benchmarks.run import (
    _parse_sim_even_batches,
    _sim_even_metrics,
    _sim_even_order,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, form, *, add_special_tokens):
        self.calls.append((form, add_special_tokens))
        normalized = form.strip().lower()
        return {"two": [2], "2": [2], "yes": [7]}.get(normalized, [])


def test_geneval2_answer_ids_exclude_special_tokens_and_deduplicate() -> None:
    tokenizer = _Tokenizer()

    assert _geneval2_answer_token_ids(tokenizer, "How many cats?", "two") == [2]
    assert tokenizer.calls
    assert all(add_special_tokens is False for _, add_special_tokens in tokenizer.calls)


@pytest.mark.parametrize("value", ["0x32", "32x0", "32", "axb", "-1x2", "1x2x3"])
def test_sim_even_batches_rejects_invalid_dimensions(value) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_sim_even_batches(value)


def test_sim_even_metrics_tolerates_failed_rows_and_fills_a_complete_wave() -> None:
    rows = [{"score": 1.0}, None, {"score": 3.0}]

    assert _parse_sim_even_batches("2X2") == (2, 2)
    assert _sim_even_order(3, world=2, batch_size=2) == [0, 1, 2, 0]
    assert _sim_even_metrics(rows, ["score"], world=2, batch_size=2) == {"score_sim2x2": pytest.approx(5.0 / 3.0)}
