"""Unit tests for the shared video-rollout foundation (PR #135).

Covers the two correctness-critical pure(-ish) helpers the four video-model PRs
(#136-#139) build on, neither of which is exercised by the PR's import smoke:

* ``_write_fused_shard`` — the fused-shard placement math (all-equal AND the new
  HunyuanVideo trailing-unequal ``linear1=[q,k,v,mlp]`` layout, plus the
  out-of-range guard).
* ``stack_decoded_videos`` — canonical ``[C,T,H,W]`` -> frame-major ``[T,C,H,W]``
  permute + ragged ``Videos`` packing (the WAN 2.1 video-reward path), including
  the empty-in -> ``None`` contract and the rank guard.

These need torch; run on a GPU/torch node (the control node has no torch).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from unirl.rollout.engine.sglang_diffusion._patches.patch_weights_updater import (  # noqa: E402
    _write_fused_shard,
)
from unirl.rollout.engine.sglang_diffusion.utils.tracks import (  # noqa: E402
    stack_decoded_videos,
)


class _FakeResult:
    """Minimal RawResult stand-in — the helpers only read ``.samples``."""

    def __init__(self, samples):
        self.samples = samples


# ---------------------------------------------------------------------------
# _write_fused_shard — placement math
# ---------------------------------------------------------------------------


def _fused_param(dim0: int) -> torch.Tensor:
    # A bare param stand-in: no ``weight_loader`` attr, so the manual placement
    # path (the branch under test) runs. ``.data`` is the tensor itself.
    p = torch.zeros(dim0, 8)
    return p


def test_write_fused_shard_all_equal_qkv():
    """q==k==v (Z-Image to_qkv): three equal H-sized shards packed in order."""
    H = 4
    param = _fused_param(3 * H)
    q = torch.full((H, 8), 1.0)
    k = torch.full((H, 8), 2.0)
    v = torch.full((H, 8), 3.0)
    _write_fused_shard(param, q, shard_id=0, num_shards=3)
    _write_fused_shard(param, k, shard_id=1, num_shards=3)
    _write_fused_shard(param, v, shard_id=2, num_shards=3)
    assert torch.equal(param.data[0:H], q)
    assert torch.equal(param.data[H : 2 * H], k)
    assert torch.equal(param.data[2 * H : 3 * H], v)


def test_write_fused_shard_trailing_unequal_linear1():
    """HunyuanVideo linear1=[q,k,v,mlp]: three H shards + one 4H MLP shard at tail.

    The legacy equal-split (dim0//num_shards) would carve four 1.75H chunks and
    crash writing an H tensor into a 1.75H slot; the tail placement is exact.
    """
    H = 4
    total = 3 * H + 4 * H  # q,k,v (H each) + mlp (4H)
    param = _fused_param(total)
    q = torch.full((H, 8), 1.0)
    k = torch.full((H, 8), 2.0)
    v = torch.full((H, 8), 3.0)
    mlp = torch.full((4 * H, 8), 9.0)
    _write_fused_shard(param, q, shard_id=0, num_shards=4)
    _write_fused_shard(param, k, shard_id=1, num_shards=4)
    _write_fused_shard(param, v, shard_id=2, num_shards=4)
    _write_fused_shard(param, mlp, shard_id=3, num_shards=4)  # last -> tail
    assert torch.equal(param.data[0:H], q)
    assert torch.equal(param.data[H : 2 * H], k)
    assert torch.equal(param.data[2 * H : 3 * H], v)
    assert torch.equal(param.data[3 * H : 3 * H + 4 * H], mlp)


def test_write_fused_shard_out_of_range_raises():
    """A trailing shard larger than the fused param (offset<0) is a hard error."""
    param = _fused_param(8)
    too_big = torch.zeros(10, 8)  # last shard, offset = 8-10 = -2 < 0
    with pytest.raises(ValueError, match="does not fit fused param"):
        _write_fused_shard(param, too_big, shard_id=1, num_shards=2)


def test_write_fused_shard_leading_overflow_raises():
    """Leading (non-tail) shard whose offset+size overflows is rejected."""
    param = _fused_param(8)
    tensor = torch.zeros(6, 8)
    # shard_id 0 of 3 -> offset 0, 0+6=6 <= 8 ok; make it overflow via size
    big = torch.zeros(10, 8)
    with pytest.raises(ValueError, match="does not fit fused param"):
        _write_fused_shard(param, big, shard_id=0, num_shards=3)


# ---------------------------------------------------------------------------
# stack_decoded_videos — permute + ragged packing
# ---------------------------------------------------------------------------


def test_stack_decoded_videos_permutes_and_packs_ragged():
    """[C,T,H,W] canonical -> frame-major [T,C,H,W], concatenated along T."""
    C, H, W = 3, 5, 6
    # Two samples with different T (ragged) — same C/H/W.
    v0 = torch.rand(C, 2, H, W)
    v1 = torch.rand(C, 4, H, W)
    out = stack_decoded_videos([_FakeResult(v0), _FakeResult(v1)])
    assert out is not None
    # packed frames are [total_T, C, H, W]
    assert out.frames.shape == (2 + 4, C, H, W)
    # per-sample offsets: [0, 2, 6]
    cu = out.cu_frames
    assert [int(x) for x in cu] == [0, 2, 6]
    # frame-major content check: sample 0 frame 0 == v0 permuted [T,C,H,W][0]
    expected0 = v0.permute(1, 0, 2, 3).contiguous().to(torch.float32)
    assert torch.allclose(out.frames[0:2], expected0)


def test_stack_decoded_videos_none_when_empty():
    """No recognizable samples -> None (mirrors stack_decoded_images)."""
    assert stack_decoded_videos([]) is None
    assert stack_decoded_videos([_FakeResult(None)]) is None


def test_stack_decoded_videos_rejects_non_4d():
    """A 3-D (image) canonical sample in the video path is a hard error."""
    img = torch.rand(3, 5, 6)  # [C,H,W] -> decode_sample returns 3-D
    with pytest.raises(RuntimeError, match="expected 4-D canonical video"):
        stack_decoded_videos([_FakeResult(img)])
