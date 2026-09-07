import torch

from unirl.train.stack.planner import CountPlanner, UpdatePlanner
from unirl.types.sample import Part


def _part(prefix: str, size: int, *, advantage_offset: int = 0) -> Part:
    return Part(
        sample_ids=[f"{prefix}-{index}" for index in range(size)],
        advantages=torch.arange(advantage_offset, advantage_offset + size),
    )


def _permutation(size: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(size, generator=generator).tolist()


def test_update_planner_preserves_contiguous_order_by_default() -> None:
    planner = UpdatePlanner(CountPlanner())

    arranged, plan = planner.arrange(_part("sample", 8), num_updates=2, micro_batch_size=2)

    assert arranged.sample_ids == [f"sample-{index}" for index in range(8)]
    assert arranged.advantages.tolist() == list(range(8))
    assert plan == [[(0, 2), (2, 4)], [(4, 6), (6, 8)]]


def test_update_planner_shuffles_aligned_parts_with_one_permutation() -> None:
    planner = UpdatePlanner(CountPlanner(), shuffle_updates=True, shuffle_seed=17)

    (ar_part, ar_plan), (image_part, image_plan) = planner.arrange_many(
        (_part("ar", 8), _part("image", 8, advantage_offset=100)),
        num_updates=2,
        micro_batch_size=2,
    )

    permutation = _permutation(8, seed=17)
    assert ar_part.sample_ids == [f"ar-{index}" for index in permutation]
    assert image_part.sample_ids == [f"image-{index}" for index in permutation]
    assert ar_part.advantages.tolist() == permutation
    assert image_part.advantages.tolist() == [100 + index for index in permutation]
    assert ar_plan == image_plan == [[(0, 2), (2, 4)], [(4, 6), (6, 8)]]


def test_single_update_does_not_shuffle_or_advance_rollout_seed() -> None:
    planner = UpdatePlanner(CountPlanner(), shuffle_updates=True, shuffle_seed=23)

    unshuffled, _ = planner.arrange(_part("single", 8), num_updates=1, micro_batch_size=2)
    first_shuffled, _ = planner.arrange(_part("first", 8), num_updates=2, micro_batch_size=2)
    second_shuffled, _ = planner.arrange(_part("second", 8), num_updates=2, micro_batch_size=2)

    assert unshuffled.sample_ids == [f"single-{index}" for index in range(8)]
    assert first_shuffled.sample_ids == [f"first-{index}" for index in _permutation(8, seed=23)]
    assert second_shuffled.sample_ids == [f"second-{index}" for index in _permutation(8, seed=24)]
