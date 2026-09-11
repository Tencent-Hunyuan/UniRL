"""Who holds GPU memory during each phase of a colocated rollout/reward/train loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Step = Tuple[str, Callable[[], None]]
# (onload, offload) for one role.
Transitions = Tuple[Callable[[], None], Callable[[], None]]


class Role(str, Enum):
    """The three things that can hold weights on a colocated slab."""

    TRAIN = "train"
    ROLLOUT = "rollout"
    REWARD = "reward"


@dataclass(frozen=True)
class ResidencyPolicy:
    """Per-role choice between keeping weights on the GPU and parking them on CPU."""

    train_resident: bool = True
    rollout_resident: bool = False
    reward_resident: bool = True

    def resident(self, role: Role) -> bool:
        """Whether ``role`` keeps its weights on the GPU while idle."""
        return {
            Role.TRAIN: self.train_resident,
            Role.ROLLOUT: self.rollout_resident,
            Role.REWARD: self.reward_resident,
        }[role]


class _RoleState:
    """One role's transitions plus whether its weights are on the GPU right now."""

    def __init__(
        self,
        role: Role,
        *,
        pinned: bool,
        onload: Callable[[], None],
        offload: Callable[[], None],
        starts_on_gpu: bool,
    ) -> None:
        self.role = role
        self.pinned = bool(pinned)
        self._onload = onload
        self._offload = offload
        self.on_gpu = bool(starts_on_gpu)

    def request(self, on_gpu: bool) -> Optional[Step]:
        """The labelled transition needed to reach ``on_gpu``, or None if already there."""
        if self.on_gpu == on_gpu:
            return None
        if self.pinned and not on_gpu:
            return None
        action = self._onload if on_gpu else self._offload

        def run() -> None:
            action()
            self.on_gpu = on_gpu

        return f"{self.role.value} {'onload' if on_gpu else 'offload'}", run


class ResidencyPlanner:
    """Drives train/rollout/reward residency for one colocated loop."""

    # Every transition goes through here so that phases ask for the state they
    # need rather than for a transition. A reward phase that follows a rollout
    # phase inherits an already-parked trainer and issues nothing, where
    # independent per-phase flags each did their own offload/onload pair and
    # moved the whole train state across PCIe twice per rollout to no effect.

    def __init__(
        self,
        policy: ResidencyPolicy,
        *,
        train: Optional[Transitions] = None,
        rollout: Optional[Transitions] = None,
        reward: Optional[Transitions] = None,
        run_steps: Callable[[List[Step]], None],
    ) -> None:
        # A role passed as None is not on this slab (a separate reward or rollout
        # placement, or a trainside rollout that is the trainer's own weights).
        # Parking it would move memory that the active role cannot use anyway, so
        # it stays out of the state table entirely rather than being pinned.
        self._policy = policy
        self._run_steps = run_steps
        self._states: Dict[Role, _RoleState] = {}
        for role, transitions in ((Role.TRAIN, train), (Role.ROLLOUT, rollout), (Role.REWARD, reward)):
            if transitions is None:
                continue
            # Train and reward are built on the GPU. A rollout engine's initial
            # state is engine-specific, so assume it needs waking and let the
            # first wake_up() no-op on the engines that were already up, exactly
            # as the unconditional wake this replaces did.
            self._states[role] = _RoleState(
                role,
                pinned=policy.resident(role),
                onload=transitions[0],
                offload=transitions[1],
                starts_on_gpu=role is not Role.ROLLOUT,
            )

    @property
    def policy(self) -> ResidencyPolicy:
        """The policy this planner enforces."""
        return self._policy

    def on_gpu(self, role: Role) -> bool:
        """Whether ``role``'s weights are currently resident."""
        state = self._states.get(role)
        return True if state is None else state.on_gpu

    def parkable(self, role: Role) -> bool:
        """Whether ``role`` is on this slab and can therefore be parked at all."""
        return role in self._states

    def _apply(self, steps: List[Optional[Step]]) -> None:
        pending = [step for step in steps if step is not None]
        if not pending:
            return
        logger.info("lifecycle residency: %s", ", ".join(label for label, _ in pending))
        self._run_steps(pending)

    def set(self, role: Role, on_gpu: bool) -> None:
        """Move one role to ``on_gpu``; a no-op when it is already there or pinned up."""
        state = self._states.get(role)
        if state is not None:
            self._apply([state.request(on_gpu)])

    def enter(self, active: Role) -> None:
        """Park every other role, then make ``active`` resident."""
        # Park unconditionally, including when ``active`` itself is untracked: a
        # trainside rollout or a trainer on a separate slab still runs, and still
        # displaces anything sharing the slab it runs on.
        self._apply([state.request(False) for role, state in self._states.items() if role is not active])
        state = self._states.get(active)
        if state is not None:
            self._apply([state.request(True)])
