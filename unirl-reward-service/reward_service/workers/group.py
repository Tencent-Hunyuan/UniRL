"""WorkerGroup: N replicas of ScorerActor for one reward, round-robin dispatch."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import ray

from reward_service.config import RewardModelCfg
from reward_service.logging_utils import get_logger
from reward_service.scorers import ScoreItem
from reward_service.workers.actor import ScorerActor

logger = get_logger(__name__)

# Ray's actor scheduler accepts "SPREAD" as a string literal to hint at
# distributing actors across nodes. We keep the literal centralised so the
# group layer doesn't leak Ray vocabulary into ad-hoc string constants.
_SPREAD_STRATEGY = "SPREAD"


# Pragma prefix that lets a requirements file pass extra flags to
# `pip install` via Ray's runtime_env. Anything after the prefix is
# whitespace-split and forwarded as ``pip_install_options``. Used by
# scorers whose deps need flags like ``--no-build-isolation`` (e.g.
# flash-attn building against the venv's inherited torch).
_PIP_OPTIONS_PRAGMA = "# pip-options:"


def _build_runtime_env(requirements_path: str) -> dict[str, Any]:
    """Read a pip requirements file and return a Ray runtime_env dict.

    Ray creates a virtualenv **with ``--system-site-packages``** on top of the
    base environment and pip-installs the listed packages.  Packages already
    present in base at a compatible version are reused (not reinstalled);
    packages that conflict with the pin (e.g. ``transformers==4.45.2`` when
    base has 4.57.x) are installed fresh into the venv and shadow the base
    copy at import time.

    We intentionally do NOT pass ``--ignore-installed``: doing so would
    reinstall heavy packages like ``torch`` into the venv, causing ABI
    mismatches with base-compiled extensions (xformers, flash-attn) that
    leak in via ``--system-site-packages``.

    Identical requirement sets (by content hash) share a single cached venv,
    so actors with the same requirements file don't trigger redundant installs.

    A requirements file may include a special pragma line of the form

        # pip-options: --no-build-isolation --foo

    Tokens after the prefix are forwarded as ``pip_install_options`` so
    individual scorer envs can pass build flags (notably
    ``--no-build-isolation`` for flash-attn) without us hardcoding them
    globally.
    """
    path = Path(requirements_path)
    packages: list[str] = []
    pip_install_options: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_PIP_OPTIONS_PRAGMA):
            pip_install_options.extend(line[len(_PIP_OPTIONS_PRAGMA):].split())
            continue
        if line.startswith("#"):
            continue
        packages.append(line)
    pip_cfg: dict[str, Any] = {"packages": packages, "pip_check": False}
    if pip_install_options:
        pip_cfg["pip_install_options"] = pip_install_options
    return {"pip": pip_cfg}


def _actor_options(cfg: RewardModelCfg) -> dict[str, Any]:
    """Build the dict passed to ``ScorerActor.options(...)``."""
    options: dict[str, Any] = {
        "num_gpus": cfg.num_gpus,
        "num_cpus": cfg.num_cpus,
        "max_concurrency": cfg.max_concurrency,
        "runtime_env": _build_runtime_env(cfg.runtime_env),
    }
    if cfg.scheduling == "spread":
        options["scheduling_strategy"] = _SPREAD_STRATEGY
    return options


class WorkerGroup:
    def __init__(self, cfg: RewardModelCfg) -> None:
        self.cfg = cfg
        self.actors = self._spawn_actors()
        self._rr = itertools.cycle(range(len(self.actors)))

    def _spawn_actors(self) -> list[Any]:
        options = _actor_options(self.cfg)
        pip_cfg = options.get("runtime_env", {}).get("pip", {})
        logger.info(
            "WorkerGroup[%s] runtime_env packages=%s pip_options=%s",
            self.cfg.name,
            pip_cfg.get("packages", []),
            pip_cfg.get("pip_install_options", []),
        )
        actor_cls = ScorerActor.options(**options)
        actors = []
        for i in range(self.cfg.num_replicas):
            handle = actor_cls.remote(self.cfg.scorer, self.cfg.params)
            actors.append(handle)
            logger.info(
                "spawned actor[%s#%d] scorer=%s num_gpus=%s num_cpus=%s "
                "max_concurrency=%d scheduling=%s",
                self.cfg.name,
                i,
                self.cfg.scorer,
                self.cfg.num_gpus,
                self.cfg.num_cpus,
                self.cfg.max_concurrency,
                self.cfg.scheduling,
            )
        return actors

    def dispatch(self, items: list[ScoreItem]) -> Any:
        """Return a Ray ObjectRef for the score computation."""
        idx = next(self._rr)
        return self.actors[idx].score.remote(items)

    def ping(self) -> list[str]:
        return ray.get([a.ping.remote() for a in self.actors])

    def shutdown(self) -> None:
        for a in self.actors:
            ray.kill(a)
        self.actors = []
