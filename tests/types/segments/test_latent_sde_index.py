"""CPU unit tests for LatentSegment.sde_index_per_sample (FlashGRPO per-sample field).

The field must behave as a ``CONCAT`` per-sample tensor: re-indexed by
``select`` / ``slice`` and stacked along dim 0 by ``concat``, while staying
``None`` (and inert) for the shared-schedule algorithms that never set it.
"""

from __future__ import annotations

import torch

from unirl.types.segments.latent import LatentSegment, make_video_segment


def _segment(n: int, start_step: int) -> LatentSegment:
    """A minimal video segment with ``n`` rows and per-sample steps start_step..."""
    return make_video_segment(
        latents=torch.zeros(n, 2, 3),  # [N, K, feat]
        sde_logp=torch.zeros(n, 1),
        sde_index_per_sample=torch.arange(start_step, start_step + n, dtype=torch.long),
    )


def test_select_reindexes_field() -> None:
    seg = _segment(4, start_step=0)  # steps [0, 1, 2, 3]
    reordered = seg.select(torch.tensor([3, 1, 0, 2]))
    assert reordered.sde_index_per_sample.tolist() == [3, 1, 0, 2]
    assert reordered.sde_index_per_sample.dtype == torch.long


def test_concat_stacks_along_dim0() -> None:
    a = _segment(2, start_step=0)  # [0, 1]
    b = _segment(3, start_step=5)  # [5, 6, 7]
    merged = LatentSegment.concat([a, b])
    assert merged.sde_index_per_sample.tolist() == [0, 1, 5, 6, 7]
    assert merged.sde_index_per_sample.shape == (5,)
    assert merged.sde_index_per_sample.dtype == torch.long
    assert merged.batch_size == 5


def test_slice_subsets_field() -> None:
    seg = _segment(5, start_step=10)  # [10, 11, 12, 13, 14]
    sub = seg.slice(1, 4)
    assert sub.sde_index_per_sample.tolist() == [11, 12, 13]


def test_single_element_concat_is_identity() -> None:
    """concat([x]) short-circuits to x — the field passes through unchanged."""
    seg = _segment(3, start_step=2)
    assert LatentSegment.concat([seg]).sde_index_per_sample.tolist() == [2, 3, 4]


def test_defaults_none_and_stays_inert_through_ops() -> None:
    """Shared-schedule segments never set the field; Batch ops keep it None."""
    seg = make_video_segment(
        latents=torch.zeros(3, 2, 3),
        sde_indices=torch.tensor([4], dtype=torch.long),
    )
    assert seg.sde_index_per_sample is None
    assert seg.select(torch.tensor([2, 0, 1])).sde_index_per_sample is None
    assert seg.slice(0, 2).sde_index_per_sample is None
    assert LatentSegment.concat([seg, seg]).sde_index_per_sample is None
