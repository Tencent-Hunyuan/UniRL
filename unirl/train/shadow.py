from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

from torch import Tensor


@dataclass(frozen=True)
class Shadow:
    """How to access (live, shadow) parameter pairs on the model tree.

    Returned by ``inject_nft`` / ``inject_mirror``.  Consumed by
    :class:`~unirl.train.ema.EMA`.  The closures capture the
    model reference and the specifics of how shadows are stored.  EMA
    calls these without knowing whether shadows are peft adapters or
    mirror parameters.
    """

    iter_pairs: Callable[[], Iterator[Tuple[Tensor, Tensor]]]
    swap_in: Callable[[], None]
    swap_out: Callable[[], None]


__all__ = ["Shadow"]
