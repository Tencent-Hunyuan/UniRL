"""Rollout engines over the canonical ``Sample`` request type.

Two halves of one design: ``synchronous.py`` records the worker-side sync contracts
(``BaseRolloutEngine`` — the broad ABC including coordinator engines — and
``SyncRolloutEngine``, the ``Sample`` → ``Sample`` refinement the per-backend
subpackages implement); ``asynchronous.py`` records the driver side (the
batch/agentic async engines and their mechanisms).

Deliberately empty otherwise: importing the driver-side ``asynchronous`` module
must stay ray/torch-free, so this init imports nothing and consumers import the
halves directly.
"""
