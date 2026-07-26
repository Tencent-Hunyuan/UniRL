from unirl.sde.noise import _derive_group_seed, make_prompt_seed_group_id


def test_prompt_seed_is_stable_per_sample_slot_and_distinct_across_siblings() -> None:
    ids = [make_prompt_seed_group_id("same prompt", sample_ordinal=index) for index in range(3)]
    seeds = [_derive_group_seed(42, group_id) for group_id in ids]

    assert len(set(seeds)) == 3
    assert seeds == [_derive_group_seed(42, group_id) for group_id in ids]
    assert seeds[1] == (seeds[0] + 1) % (2**31)
