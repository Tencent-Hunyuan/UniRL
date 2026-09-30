"""Index schedulers used by GRPO-style algorithms."""

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np

Strategy = Literal["all", "progressive", "random", "decay", "exp_decay"]


@dataclass
class WindowConfig:
    """Configuration for stateless window-based index scheduling."""

    strategy: Strategy = "all"
    window_size: int = 4
    iters_per_window: int = 25
    init_timestep: int = 0
    overlap_size: int = 0
    roll_back: bool = False
    max_iters_per_window: Optional[int] = None
    min_iters_per_window: Optional[int] = None
    exp_decay_threshold: int = 13
    exp_decay_k: float = 0.1

    def __post_init__(self) -> None:
        if self.strategy != "all" and self.window_size < 1:
            raise ValueError(f"WindowConfig requires window_size >= 1, got {self.window_size}")
        if self.strategy in ("progressive", "decay", "exp_decay"):
            if not 0 <= self.overlap_size < self.window_size:
                raise ValueError(
                    f"WindowConfig({self.strategy}) requires 0 <= overlap_size < window_size, "
                    f"got overlap_size={self.overlap_size}, window_size={self.window_size}"
                )
            if not math.isfinite(self.iters_per_window) or self.iters_per_window < 1:
                raise ValueError(f"WindowConfig requires a finite iters_per_window >= 1, got {self.iters_per_window}")
        if self.strategy == "decay":
            if self.max_iters_per_window is None:
                self.max_iters_per_window = self.iters_per_window
            if self.min_iters_per_window is None:
                self.min_iters_per_window = max(1, self.iters_per_window // 4)
            lo, hi = self.min_iters_per_window, self.max_iters_per_window
            if not math.isfinite(lo) or not math.isfinite(hi) or lo < 1 or hi < lo:
                raise ValueError(
                    "WindowConfig(decay) requires finite bounds with "
                    f"1 <= min_iters_per_window <= max_iters_per_window, got ({lo}, {hi})"
                )
        if self.strategy == "exp_decay" and not math.isfinite(self.exp_decay_k):
            raise ValueError(f"WindowConfig(exp_decay) requires a finite exp_decay_k, got {self.exp_decay_k}")


class TimestepScheduler(ABC):
    """Abstract base class for stateless timestep-index schedulers."""

    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps

    @abstractmethod
    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        """Return the selected indices for the given step."""


def normalize_timestep_fraction(
    timestep_fraction: Union[float, Tuple[float, float], List[float]],
) -> Tuple[float, float]:
    """Normalize timestep_fraction to a ``(start, end)`` tuple."""
    if isinstance(timestep_fraction, Sequence):
        if len(timestep_fraction) != 2:
            raise ValueError(f"timestep_fraction tuple must have exactly 2 elements, got {len(timestep_fraction)}")
        start, end = float(timestep_fraction[0]), float(timestep_fraction[1])
    else:
        start, end = 0.0, float(timestep_fraction)
    if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0):
        raise ValueError(f"timestep_fraction values must be in [0.0, 1.0], got ({start}, {end})")
    if start > end:
        raise ValueError(f"timestep_fraction start ({start}) must be <= end ({end})")
    return (start, end)


class AllSDEScheduler(TimestepScheduler):
    """Full-range index scheduler with optional range filtering and sparse sampling."""

    def __init__(
        self,
        num_timesteps: int,
        timestep_fraction: Union[float, Tuple[float, float]] = 1.0,
        num_sde_steps: Optional[int] = None,
    ):
        super().__init__(num_timesteps)
        self.timestep_fraction = timestep_fraction
        self.num_sde_steps = num_sde_steps
        self._fraction_start, self._fraction_end = normalize_timestep_fraction(timestep_fraction)
        self._effective_start = int(num_timesteps * self._fraction_start)
        self._effective_end = int(num_timesteps * self._fraction_end)
        if num_sde_steps is not None:
            pool_size = self._effective_end - self._effective_start
            if num_sde_steps > pool_size:
                raise ValueError(
                    f"num_sde_steps ({num_sde_steps}) exceeds available timesteps "
                    f"in fraction range [{self._effective_start}, {self._effective_end}) "
                    f"(pool_size={pool_size})"
                )
            if num_sde_steps < 0:
                raise ValueError(f"num_sde_steps must be non-negative, got {num_sde_steps}")

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        if self.num_sde_steps == 0:
            return set()
        pool = list(range(self._effective_start, self._effective_end))
        if self.num_sde_steps is None or self.num_sde_steps >= len(pool):
            return set(pool)
        seed = 0 if step is None else int(step)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(pool, size=self.num_sde_steps, replace=False)
        return set(int(i) for i in chosen)


class WindowScheduler(TimestepScheduler):
    """Stateless sliding-window index scheduler."""

    WINDOW_STRATEGY_TO_METHOD_NAME = {
        "all": None,
        "progressive": "_resolve_progressive",
        "random": "_resolve_random",
        "decay": "_resolve_sliding",
        "exp_decay": "_resolve_sliding",
    }

    def __init__(self, num_timesteps: int, config: WindowConfig):
        super().__init__(num_timesteps)
        self.config = config
        if self.config.strategy not in self.WINDOW_STRATEGY_TO_METHOD_NAME:
            raise ValueError(
                f"Bad strategy configuration for WindowScheduler: {self.config.strategy}. "
                f"Available options: {set(self.WINDOW_STRATEGY_TO_METHOD_NAME.keys())}"
            )

    def get_sde_indices(self, step: Optional[int] = None) -> Set[int]:
        if self.config.strategy == "all":
            return set(range(self.num_timesteps))
        resolve_method = getattr(
            self,
            self.WINDOW_STRATEGY_TO_METHOD_NAME[self.config.strategy],
        )
        return resolve_method(0 if step is None else int(step))

    def _resolve_progressive(self, step: int) -> Set[int]:
        starts = self._window_starts()
        window_step = step // self.config.iters_per_window
        if window_step >= len(starts) and not self.config.roll_back:
            window_step = len(starts) - 1
        else:
            window_step = window_step % len(starts)
        cur = starts[window_step]
        return set(range(cur, cur + self.config.window_size))

    def _resolve_random(self, step: int) -> Set[int]:
        rng = np.random.default_rng(step)
        max_start = max(0, self.num_timesteps - self.config.window_size)
        cur_timestep = int(rng.integers(0, max_start + 1))
        return set(range(cur_timestep, cur_timestep + self.config.window_size))

    def _window_starts(self) -> List[int]:
        """Start index of every full window in one sweep, in visiting order."""
        stride = self.config.window_size - self.config.overlap_size
        remaining = self.num_timesteps - self.config.init_timestep - self.config.window_size
        count = max(1, remaining // stride + 1)
        return [self.config.init_timestep + i * stride for i in range(count)]

    def _resolve_sliding(self, step: int) -> Set[int]:
        """Walk full windows whose dwell is the decay or exp_decay value at each start."""
        starts = self._window_starts()
        if self.config.strategy == "decay":
            lo = self.config.min_iters_per_window
            hi = self.config.max_iters_per_window
            dwell = []
            for start in starts:
                progress = start / self.num_timesteps
                dwell.append(max(lo, int(hi * (1.0 - progress) + lo * progress)))
        else:
            base = self.config.iters_per_window
            k = self.config.exp_decay_k
            threshold = self.config.exp_decay_threshold
            dwell = [int(math.ceil(base * math.exp(-k * max(0, start - threshold)))) for start in starts]
        total = sum(dwell)
        if step >= total:
            if not self.config.roll_back:
                cur = starts[-1]
                return set(range(cur, cur + self.config.window_size))
            step = step % total
        cur = starts[bisect.bisect_right(list(accumulate(dwell)), step)]
        return set(range(cur, cur + self.config.window_size))
