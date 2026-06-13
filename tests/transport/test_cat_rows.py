"""Unit tests for ``cat_rows`` — the per-ref dim-0 assembly funnel.

``cat_rows`` is the single chokepoint that stitches per-shard tensor parts back
into one tensor along dim 0. Its only non-trivial contract is the ragged
trailing-dim rule: 2D+ parts produced by different shards may have different
widths (e.g. per-worker prompt blocks), so they are right-padded with ZEROS to
the max width before the cat — consumers of such fields must be mask-driven.
Pure-CPU: just plain torch tensors, no transport backend.
"""

import pytest
import torch

from unirl.distributed.tensor.transport import cat_rows

pytestmark = pytest.mark.cpu


def test_empty_list_returns_empty():
    out = cat_rows([])
    assert out.numel() == 0
    assert torch.equal(out, torch.empty(0))


def test_single_part_returned_as_is():
    # len == 1 short-circuits: the exact same object comes back (no copy, no pad).
    a = torch.arange(6).reshape(2, 3).float()
    out = cat_rows([a])
    assert out is a


def test_multiple_1d_parts_plain_cat():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    out = cat_rows([a, b])
    assert out.shape == (5,)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_equal_width_2d_parts_no_padding():
    # all widths equal -> straight torch.cat along dim 0, no zero padding.
    a = torch.arange(6).reshape(2, 3).float()
    b = torch.arange(6, 9).reshape(1, 3).float()
    out = cat_rows([a, b])
    assert out.shape == (3, 3)
    assert torch.equal(out, torch.cat([a, b], dim=0))


def test_ragged_2d_parts_right_padded_with_zeros():
    # parts (2,3) and (1,5) -> (3,5): the narrower part's rows are right-padded
    # with zeros out to the max width before the cat.
    a = torch.full((2, 3), 1.0)
    b = torch.full((1, 5), 2.0)
    out = cat_rows([a, b])
    assert out.shape == (3, 5)
    # real data preserved in the leading columns of every row...
    assert torch.equal(out[0, :3], torch.full((3,), 1.0))
    assert torch.equal(out[1, :3], torch.full((3,), 1.0))
    assert torch.equal(out[2], torch.full((5,), 2.0))
    # ...and the padded region of the narrow rows is exactly zero.
    assert torch.all(out[:2, 3:] == 0)


def test_ragged_2d_pad_applies_to_every_narrower_part():
    # three different widths -> all padded up to max(3, 4, 2) == 4.
    a = torch.full((1, 3), 1.0)
    b = torch.full((2, 4), 2.0)
    c = torch.full((1, 2), 3.0)
    out = cat_rows([a, b, c])
    assert out.shape == (4, 4)
    assert torch.all(out[0, 3:] == 0)  # 'a' row padded col 3
    assert torch.all(out[1:3, 4:] == 0)  # 'b' rows already full width (no cols beyond 4)
    assert torch.all(out[3, 2:] == 0)  # 'c' row padded cols 2..3
    assert torch.equal(out[0, :3], torch.full((3,), 1.0))
    assert torch.equal(out[3, :2], torch.full((2,), 3.0))


def test_ragged_3d_right_pads_on_dim1():
    # 3D+ ragged: differing dim-1 widths, equal trailing dims -> right-pad dim 1.
    # parts (2,1,4) and (1,3,4) -> (3,3,4); narrow part padded along dim 1 only.
    a = torch.full((2, 1, 4), 1.0)
    b = torch.full((1, 3, 4), 2.0)
    out = cat_rows([a, b])
    assert out.shape == (3, 3, 4)
    # 'a' rows: dim-1 index 0 holds real data, indices 1..2 are zero pad.
    assert torch.equal(out[:2, 0, :], torch.full((2, 4), 1.0))
    assert torch.all(out[:2, 1:, :] == 0)
    # 'b' row: all three dim-1 slots are real data (it set the max width).
    assert torch.equal(out[2], torch.full((3, 4), 2.0))


def test_equal_width_3d_parts_no_padding():
    a = torch.arange(24).reshape(2, 3, 4).float()
    b = torch.arange(24, 36).reshape(1, 3, 4).float()
    out = cat_rows([a, b])
    assert out.shape == (3, 3, 4)
    assert torch.equal(out, torch.cat([a, b], dim=0))
