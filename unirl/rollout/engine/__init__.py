"""Rollout engines that speak the ``RolloutReq``/``RolloutResp`` types.

The ABC at :mod:`unirl.rollout.engine.base` is the canonical one.
Trainside, SGLang, and vllm-omni engines all implement this ABC.
"""

from typing import List, Optional

from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


def chunked_engine_generate_req(
    engine: BaseRolloutEngine,
    req: RolloutReq,
    *,
    chunk_size: Optional[int],
) -> RolloutResp:
    """Call ``engine.generate`` over mini-batch chunks of *req* and concat outputs.

    Slices the request via :meth:`Batch.slice` (which understands the
    ``concat_field`` annotations on ``RolloutReq``),
    calls ``engine.generate`` per chunk, and concatenates per-chunk
    responses via :meth:`RolloutResp.concat` (segment rows stay 1:1 with
    samples, so the merge is a plain per-field concat).

    Fast path (zero overhead): when ``chunk_size`` is ``None`` or ``>=
    req.batch_size``, this is a single direct call to ``engine.generate(req)``.

    Determinism caveat: per-step SDE noise inside the engine is independent
    of chunking (it's keyed by request seed + step index, not batch
    position), so chunked vs unchunked runs produce bit-identical outputs.
    """
    n_samples = int(req.batch_size)
    if n_samples == 0:
        raise ValueError(f"chunked_engine_generate_req requires non-empty req; got batch_size=0 (req={req!r}).")
    if chunk_size is None:
        return engine.generate(req)
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive int when set; got {chunk_size!r} (type={type(chunk_size).__name__})."
        )
    if n_samples <= chunk_size:
        return engine.generate(req)

    outputs: List[RolloutResp] = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        outputs.append(engine.generate(req.slice(start, end)))
    return RolloutResp.concat(outputs)


__all__ = ["BaseRolloutEngine", "chunked_engine_generate_req"]
