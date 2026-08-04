"""Rollout engines over the canonical ``Sample`` request type.

``synchronous.py`` records the worker-side contracts (``BaseRolloutEngine`` — the
broad ABC including coordinator engines — and ``SyncRolloutEngine``, the ``Sample``
→ ``Sample`` refinement the per-backend subpackages implement). The driver side
lives in ``../manager/``, which owns admission, acceptance and disposal over time
and holds no model.

Deliberately empty otherwise: this init imports nothing, so consumers import the
module they need directly and the manager package stays ray/torch-free.
"""
