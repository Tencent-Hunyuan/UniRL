"""Rollout harnesses: worker-side task-internal control flow, as plugins.

A harness owns the sequence dictated by TASK SEMANTICS — how many model/tool
turns, what ends an episode, how multi-stage flows chain — the third kind of
sequence owner next to training policy (trainer-side step loops) and engine
contracts (driver-side pumps). It runs INSIDE the rollout worker, so
multi-turn env state and intermediate tensors never cross to the driver.

``protocol.py`` holds the narrow boundary (``RolloutHarness`` /
``HarnessContext`` / ``HarnessOutcome``); one harness = one module beside it.
Harnesses share the boundary, never a loop — different tasks are different
control flows, and forcing them through one shared loop is false unification.

Deliberately empty otherwise: harness modules must stay ray/torch-free to
import (they are exercised by CPU harnesses with fake engines/envs), so this
init imports nothing and consumers import the submodules directly.
"""
