"""Reward subsystem entrypoint.

This package holds typed reward component specs and their runtime
constructors. There is no package-root re-export surface — import from the
appropriate submodule directly:

- ``unirl.reward.base`` — ``RewardBackend`` + ``BaseRewardComponentSpec``
- ``unirl.reward.service`` — ``RewardService`` (holds one backend)
- ``unirl.reward.remote`` — ``RemoteRewardBackend`` (remote backend)
- ``unirl.reward.local.<name>`` — per-scorer ``<Name>RewardScorer`` + ``<Name>Spec`` (local backends)
"""
