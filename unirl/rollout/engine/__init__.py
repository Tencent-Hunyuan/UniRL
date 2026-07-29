"""Rollout engines over the canonical ``Sample`` request type.

The broad ABC includes coordinator engines; single-turn engines refine it with
the ``Sample`` → ``Sample`` contract.
"""

from typing import List, Optional

from unirl.rollout.engine.base import BaseRolloutEngine, BaseSingleTurnRolloutEngine
from unirl.types.sample import Sample


def chunked_engine_generate(
    engine: BaseSingleTurnRolloutEngine,
    sample: Sample,
    *,
    chunk_size: Optional[int],
) -> Sample:
    """Call ``engine.generate`` over mini-batch chunks of *sample* and concat outputs.

    Splits the request into per-root-group sub-Samples (:meth:`Sample.split`,
    tree-complete — each shard holds one prompt's whole subtree across all
    parts), regroups them into chunks of ``chunk_size`` roots, calls
    ``engine.generate`` per chunk, and reassembles via :meth:`Sample.concat`
    (segment rows stay 1:1 with samples, so the merge is a plain per-field
    concat).

    Fast path (zero overhead): when ``chunk_size`` is ``None`` or ``>=`` the
    root count, this is a single direct call to ``engine.generate(sample)``.

    Determinism caveat: per-step SDE noise and the x_T recipe inside the engine
    are independent of chunking (keyed by the sample's path id + step index, not
    batch position — and split preserves ids), so chunked vs unchunked runs
    produce bit-identical outputs.
    """
    groups = sample.split()
    n_roots = len(groups)
    if n_roots == 0:
        raise ValueError(f"chunked_engine_generate requires a non-empty Sample; got 0 roots (sample={sample!r}).")
    if chunk_size is None:
        return engine.generate(sample)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive int when set; got {chunk_size!r} (type={type(chunk_size).__name__})."
        )
    if n_roots <= chunk_size:
        return engine.generate(sample)

    outputs: List[Sample] = []
    for start in range(0, n_roots, chunk_size):
        chunk = groups[start : start + chunk_size]
        outputs.append(engine.generate(Sample.concat(chunk)))
    return Sample.concat(outputs)


__all__ = ["BaseRolloutEngine", "BaseSingleTurnRolloutEngine", "chunked_engine_generate"]
