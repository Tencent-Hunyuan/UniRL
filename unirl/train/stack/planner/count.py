"""Fixed-count micro-batching — the default planner.

Uniform-shape latents (diffusion) make a sample COUNT a good proxy for compute, so
micros are contiguous equal-count slices and the per-rank micro count is identical
across DP ranks — no NCCL micro-count parity collective is needed. This is the
historical ``TrainStack`` behaviour; recipes get it by omitting ``micro_planner``.
"""

from __future__ import annotations

from unirl.algorithms import StageAlgorithm
from unirl.train.stack.planner.types import Plan, _build_micro_batch_slices, _update_ranges
from unirl.types.rollout_resp import RolloutTrack


def _count_plan(*, total: int, num_updates: int, micro_batch_size: int) -> Plan:
    """Fixed-count plan: contiguous equal updates, each split into ``micro_batch_size`` micros.

    The diffusion / FlowGRPO "batched" schedule, and the fallback for the LLM path
    when a segment exposes no per-sample lengths. No collective: every DP rank
    produces the same update/micro counts because the per-rank batch is evenly
    sharded.
    """
    plan: Plan = []
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        plan.append(
            [
                (u_start + ms, u_start + me)
                for ms, me in _build_micro_batch_slices(total_size=u_end - u_start, micro_batch_size=micro_batch_size)
            ]
        )
    return plan


class CountPlanner:
    """Fixed-count micro-batches: every micro holds ``micro_batch_size`` samples.

    The original ``TrainStack`` behaviour. Uniform-shape latents (diffusion) make a
    sample COUNT a good proxy for compute, so micros are contiguous equal-count
    slices and the per-rank micro count is identical across DP ranks — no NCCL
    micro-count parity collective is needed. Never reorders the track; imposes no
    algorithm precondition.
    """

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> tuple[RolloutTrack, Plan]:
        return resp_track, _count_plan(
            total=int(resp_track.batch_size),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None
