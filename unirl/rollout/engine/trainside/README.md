# Trainside rollout chunk boundaries

`forward_batch_size` caps the number of request rows in each `pipeline.generate`
call. Results are concatenated in their original row order before group reward
and advantage computation. Siblings do not need to share a model forward for
GRPO grouping to be correct.

Set `validate_group_boundaries: true` on the recipe's `rollout` block to check
that fixed-size chunks contain whole prompt groups or aligned equal sub-groups.
The default is `false`, preserving existing schedules. Validation never changes
chunk sizes or increases the memory budget.

```yaml
rollout:
  _target_: unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine
  forward_batch_size: 4
  validate_group_boundaries: true
```

For group size 16 this keeps four forwards of four rows, rather than requiring a
16-row forward. Group/FBS pairs 16/8, 16/1 and 24/1 are also accepted. A group of
16 with FBS 6 is rejected by the optional check because its sub-chunks are uneven.
Groups must be contiguous; smaller groups must fit wholly within each chunk.
Unset FBS or a frontier fitting in one forward needs no boundary check.

## Gotchas

- This is a configuration check, not a throughput optimization or an algorithmic
  requirement. No reward or optimization regression from separate sibling
  forwards is claimed. Batch geometry can still change RNG assignment and low
  precision numerics; enabling this check leaves that geometry unchanged.
- Per-row heterogeneous geometry and ragged trajectories are not represented by
  the current shared sampling parameters and dense trajectory carrier. Shape
  bucketing and execution-key scheduling are deferred until those carriers exist.
