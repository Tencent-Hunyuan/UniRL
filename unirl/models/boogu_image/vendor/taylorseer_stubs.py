"""Raising stubs for the upstream TaylorSeer/TeaCache helper imports.

The vendored ``transformer_boogu.py`` imports these names from the upstream
``boogu.cache_functions`` / ``boogu.taylorseer_utils`` packages, but every
call site is gated behind ``enable_taylorseer`` / ``enable_teacache`` flags
that default to ``False`` and are asserted ``False`` by
``BooguImageBundle.from_config``. Inference-time caching approximates
transformer outputs across steps, which would corrupt RL rollout/replay
log-prob parity — so instead of vendoring the cache machinery, each stub
raises if it is ever reached.
"""

from typing import Any, NoReturn

_MSG = (
    "TeaCache/TaylorSeer is disabled in UniRL's vendored Boogu-Image copy: "
    "RL rollout and replay must be cache-free for log-prob correctness. "
    "This stub should be unreachable — check that `enable_teacache` and "
    "`enable_taylorseer` are False on the transformer."
)


def _raise(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise RuntimeError(_MSG)


def cal_type(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


def taylor_cache_init(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


def taylor_formula(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


def taylor_formula_4_double_stream(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


def derivative_approximation(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


def derivative_approximation_4_double_stream(*args: Any, **kwargs: Any) -> NoReturn:
    _raise(*args, **kwargs)


__all__ = [
    "cal_type",
    "taylor_cache_init",
    "taylor_formula",
    "taylor_formula_4_double_stream",
    "derivative_approximation",
    "derivative_approximation_4_double_stream",
]
