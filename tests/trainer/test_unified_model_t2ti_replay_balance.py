from __future__ import annotations

import torch

from unirl.models.bagel.conditions import BagelT2TIDiffusionConditions, BagelThinkKVReplaySpec
from unirl.trainer.unified_model import _bagel_t2ti_replay_depth_metrics, _bagel_t2ti_replay_permutation
from unirl.types.rollout_resp import RolloutTrack


def _spec(depth: int) -> BagelThinkKVReplaySpec:
    token_ids = tuple(range(depth))
    return BagelThinkKVReplaySpec(
        cache_input_ids=token_ids,
        chunk_offsets=tuple(range(depth + 1)),
        kv_length=depth,
        ropes=(depth,),
        received_kv_length=depth,
        received_ropes=(depth,),
        image_shape=(32, 32),
    )


def _padded_work(depths: list[int], *, num_shards: int) -> int:
    shard_size = len(depths) // num_shards
    return sum(max(depths[rank * shard_size + offset] for rank in range(num_shards)) for offset in range(shard_size))


def test_bagel_t2ti_replay_permutation_balances_depth_and_preserves_updates() -> None:
    depths = [100, 90, 80, 70, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1]
    conditions = BagelT2TIDiffusionConditions(replay_specs=[_spec(depth) for depth in depths])
    image_track = RolloutTrack(
        sample_ids=[f"image-{index}" for index in range(len(depths))],
        parent_ids=[f"ar-{index}" for index in range(len(depths))],
        parent_track="ar",
        conditions=conditions.to_dict(),
        advantages=torch.arange(len(depths), dtype=torch.float32) + 100,
    )

    permutation = _bagel_t2ti_replay_permutation(image_track, num_shards=4, num_updates=2)

    assert permutation is not None
    assert permutation.tolist() == _bagel_t2ti_replay_permutation(image_track, num_shards=4, num_updates=2).tolist()
    assert sorted(permutation.tolist()) == list(range(len(depths)))

    shard_size = 4
    update_size = 2
    for update in range(2):
        original = {
            rank * shard_size + update * update_size + offset for rank in range(4) for offset in range(update_size)
        }
        planned = {
            permutation[rank * shard_size + update * update_size + offset].item()
            for rank in range(4)
            for offset in range(update_size)
        }
        assert planned == original

    balanced_depths = [depths[index] for index in permutation.tolist()]
    assert _padded_work(balanced_depths, num_shards=4) < _padded_work(depths, num_shards=4)


def test_bagel_t2ti_replay_permutation_keeps_ar_image_rows_paired() -> None:
    depths = [12, 2, 10, 4, 9, 3, 8, 5]
    conditions = BagelT2TIDiffusionConditions(replay_specs=[_spec(depth) for depth in depths])
    ar_track = RolloutTrack(
        sample_ids=[f"ar-{index}" for index in range(len(depths))],
        advantages=torch.arange(len(depths), dtype=torch.float32),
    )
    image_track = RolloutTrack(
        sample_ids=[f"image-{index}" for index in range(len(depths))],
        parent_ids=list(ar_track.sample_ids),
        parent_track="ar",
        conditions=conditions.to_dict(),
        advantages=torch.arange(len(depths), dtype=torch.float32) + 100,
    )

    permutation = _bagel_t2ti_replay_permutation(image_track, num_shards=2, num_updates=2)

    assert permutation is not None
    balanced_ar = ar_track.select(permutation)
    balanced_image = image_track.select(permutation)
    assert balanced_image.parent_ids == balanced_ar.sample_ids
    for ar_advantage, image_advantage in zip(balanced_ar.advantages, balanced_image.advantages):
        assert image_advantage.item() == ar_advantage.item() + 100
    balanced_conditions = BagelT2TIDiffusionConditions.from_dict(balanced_image.conditions)
    assert [len(spec.chunk_offsets) - 1 for spec in balanced_conditions.replay_specs] == [
        depths[index] for index in permutation.tolist()
    ]


def test_bagel_t2ti_replay_permutation_skips_non_t2ti_track() -> None:
    track = RolloutTrack(sample_ids=["a", "b"], conditions={})

    assert _bagel_t2ti_replay_permutation(track, num_shards=2, num_updates=1) is None


def test_bagel_t2ti_replay_depth_metrics_report_distribution() -> None:
    assert _bagel_t2ti_replay_depth_metrics([]) == {}
    assert _bagel_t2ti_replay_depth_metrics([1, 2, 3, 4, 100]) == {
        "bagel_t2ti_replay_depth_min": 1.0,
        "bagel_t2ti_replay_depth_mean": 22.0,
        "bagel_t2ti_replay_depth_p50": 3.0,
        "bagel_t2ti_replay_depth_p90": 100.0,
        "bagel_t2ti_replay_depth_p99": 100.0,
        "bagel_t2ti_replay_depth_max": 100.0,
    }
