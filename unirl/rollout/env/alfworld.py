"""AlfworldEnv — stateful ALFWorld environment for the agentic engine (LIN-519).

ALFWorld (text-only ``AlfredTWEnv``) is the field's canonical multi-turn agentic-RL
benchmark: a household task where the agent issues text commands (``go to shelf 1``,
``take mug 1``, ``clean mug 1 with sinkbasin 1`` …) and the simulator returns the next
observation, ending with a binary task-success reward. Unlike the stateless
:class:`~unirl.rollout.env.tool_environment.ToolEnvironment`, each trajectory is its
own **episode with evolving state** and the **reward comes from the simulator**.

Fits the engine's ``reset(sample)->Sample`` / ``step(sample)->(obs, done, info)``
protocol (:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`), which
builds one env per worker and requires re-entrancy across concurrent trajectories.
Re-entrancy here means **per-episode state keyed by a unique id** minted in
:meth:`reset` and carried in the Sample's root *control* bag — because the ``n`` GRPO
siblings of a prompt are fanned as identical tasks (same ``sample_id``), they must
share the *game* (selected by ``metadata['game_index']``, identical across siblings)
but get *separate* episodes. The terminal task-success is emitted as the trajectory
reward through ``info['reward']``; the engine attaches it to the trajectory and
:class:`~unirl.trainer.agentic_env.AgenticEnvTrainer` reads it (no reward backend).

Constructing an ``AlfredTWEnv`` scans all game files (~5s), so we build a small pool of
templates lazily and reuse them (``init_env`` per episode is ~0s); each template is
checked out to one episode at a time (the engine caps concurrency), then released.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from unirl.types.primitives import Texts
from unirl.types.sample import Part, Primitive, Sample

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"[Aa]ction\s*:\s*(.+)")

_SYSTEM = (
    "You are an agent completing a household task in a text-based world. Read the Task, "
    "then EACH TURN reason briefly and issue exactly ONE command on a final line that "
    "starts with 'Action:'. Only use commands from the 'Admissible actions' list. Explore "
    "receptacles to find objects; to examine an object under a lamp, take the object, go "
    "to the lamp, turn it on, then examine the object.\n\n"
    "Here is one worked example:\n"
    "Task: look at the alarm clock under the desklamp.\n"
    "Thought: I need to find the alarm clock, probably on a desk or shelf.\n"
    "Action: go to desk 1\n"
    "Observation: On the desk 1, you see a alarm clock 1, a pencil 1, and a desklamp 1.\n"
    "Thought: The alarm clock and the desklamp are both here. Take the clock.\n"
    "Action: take alarm clock 1 from desk 1\n"
    "Observation: You pick up the alarm clock 1 from the desk 1.\n"
    "Thought: Turn on the desklamp, then look at the clock under it.\n"
    "Action: use desklamp 1\n"
    "Observation: You turn on the desklamp 1. Task complete.\n\n"
    "Now solve the following task the same way."
)


def list_alfworld_games(split: str = "train", data_dir: Optional[str] = None) -> List[str]:
    """Enumerate ALFWorld TextWorld game files for a split (sorted → stable indices).

    Shared by :class:`AlfworldEnv` and ``unirl.utils.prepare_alfworld`` so a data row's
    ``game_index`` maps to the same game on both sides. Globs ``$ALFWORLD_DATA`` (or
    ``data_dir``) for the ``game.tw-pddl`` task games."""
    root = data_dir or os.environ.get("ALFWORLD_DATA", "")
    if not root:
        return []
    patterns = [
        os.path.join(root, "json_2.1.1", split, "**", "game.tw-pddl"),
        os.path.join(root, "json_2.1.1", split, "**", "*.tw-pddl"),
        os.path.join(root, "**", split, "**", "*.tw-pddl"),
    ]
    for pat in patterns:
        games = sorted(glob(pat, recursive=True))
        if games:
            return games
    return []


def _parse_action(text: Optional[str]) -> str:
    """The command after the last ``Action:``; fallback to the last non-empty line."""
    matches = _ACTION_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class _Episode:
    """Per-trajectory ALFWorld episode: the live gym env, its pooled template, and state."""

    __slots__ = ("env", "template", "reward", "steps", "admissible")

    def __init__(self, env: Any, template: Any) -> None:
        self.env = env
        self.template = template
        self.reward = 0.0
        self.steps = 0
        self.admissible: List[str] = []


def _match_admissible(action: str, admissible: List[str]) -> str:
    """Snap the model's action to the closest ADMISSIBLE command (standard ALFWorld
    practice): exact → substring-containment → best token-overlap (Jaccard ≥ 0.5). This
    honors near-miss phrasings ("go to cabinet" → "go to cabinet 1") so a slightly-off
    action isn't a wasted turn, and keeps malformed text out of TextWorld's parser.
    Falls back to the raw action when nothing is close (env replies "Nothing happens.")."""
    if not admissible:
        return action
    a = action.strip().lower()
    if not a:
        return action
    for c in admissible:
        if c.strip().lower() == a:
            return c
    for c in admissible:
        cl = c.strip().lower()
        if a in cl or cl in a:
            return c
    at = set(a.split())
    best, best_score = action, 0.5  # threshold: only snap on a real overlap
    for c in admissible:
        ct = set(c.strip().lower().split())
        if not ct:
            continue
        score = len(at & ct) / len(at | ct)
        if score > best_score:
            best, best_score = c, score
    return best


class AlfworldEnv:
    """Stateful ALFWorld environment (one episode per trajectory, env-sourced reward)."""

    def __init__(
        self,
        *,
        config_file: Optional[str] = None,
        split: str = "train",
        max_steps: int = 30,
        step_penalty: float = 0.0,
        max_obs_chars: int = 2000,
    ) -> None:
        self._config_file = config_file
        self._split = split
        self._max_steps = int(max_steps)
        self._step_penalty = float(step_penalty)
        self._max_obs_chars = int(max_obs_chars)
        self._episodes: Dict[str, _Episode] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._games: List[str] = []
        self._free: List[Any] = []
        self._environment: Any = None
        self._alfworld_cfg: Any = None
        self._ready = False

    def _ensure_backend(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import alfworld.agents.environment as environment
            import alfworld.agents.modules.generic as generic

            cfg_file = self._config_file or os.environ.get("ALFWORLD_CONFIG", "")
            if not cfg_file or not os.path.exists(cfg_file):
                raise FileNotFoundError(
                    "AlfworldEnv needs an ALFWorld base config. Set $ALFWORLD_CONFIG (or the "
                    f"env's config_file) to a readable base_config.yaml; got {cfg_file!r}."
                )
            old_argv = sys.argv
            try:
                sys.argv = ["alfworld", cfg_file]
                cfg = generic.load_config()
            finally:
                sys.argv = old_argv
            cfg["env"]["type"] = "AlfredTWEnv"

            self._alfworld_cfg = cfg
            self._environment = environment
            self._games = list_alfworld_games(self._split)
            if not self._games:
                logger.warning("AlfworldEnv: no game files under $ALFWORLD_DATA (split=%s).", self._split)
            self._ready = True

    def _acquire_template(self) -> Any:
        """A reusable AlfredTWEnv (constructed on demand; the engine caps concurrency)."""
        with self._lock:
            if self._free:
                return self._free.pop()
        return self._environment.get_environment("AlfredTWEnv")(self._alfworld_cfg, train_eval=self._split)

    def _release_template(self, template: Any) -> None:
        if template is None:
            return
        with self._lock:
            self._free.append(template)

    def _open_episode(self, game_file: Optional[str]) -> Tuple[Any, Any]:
        """Check out a template restricted to one game FILE, ``init_env`` it (cheap), and
        return ``(tw_env, template)``. Restricting ``game_files`` to one game makes the
        ``n`` GRPO siblings share the same task deterministically. Overridable in tests."""
        template = self._acquire_template()
        if game_file and hasattr(template, "game_files"):
            template.game_files = [game_file]
        tw = template.init_env(batch_size=1)
        return tw, template

    def reset(self, request: Sample) -> Sample:
        """Start an episode for this trajectory; return a Sample whose input is the
        ReAct instruction + the initial observation + admissible commands."""
        self._ensure_backend()
        root = request.parts[0]
        sid = str(root.sample_ids[0])
        meta = (root.metadata or [None])[0] or {}
        game_file = meta.get("game_file")
        if not game_file and self._games:
            game_file = self._games[int(meta.get("game_index", 0)) % len(self._games)]

        tw, template = self._open_episode(game_file)
        obs, info = tw.reset()

        with self._lock:
            self._counter += 1
            eid = f"{sid}#ep{self._counter}"
            ep = _Episode(tw, template)
            ep.admissible = self._admissible(info)
            self._episodes[eid] = ep

        control = dict(root.control or {})
        control["alfworld"] = {"episode_id": eid}
        new_root = Part.input(
            [sid],
            primitives={"text": Texts(texts=[self._format(obs, info, first=True)])},
            metadata=root.metadata,
            control=control,
        )
        return Sample.request(new_root)

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        """Apply the frontier action to this trajectory's episode; return
        ``(observation, done, info)`` with the (cumulative) env return in
        ``info['reward']`` — the last value the engine sees is the episode return."""
        eid = self._episode_id(sample)
        with self._lock:
            ep = self._episodes.get(eid) if eid is not None else None
        if ep is None:
            return None, True, {"reward": 0.0}

        frontier = sample.parts[-1].primitives.get("text")
        raw = _parse_action(frontier.texts[0] if isinstance(frontier, Texts) and frontier.texts else "")
        action = _match_admissible(raw, ep.admissible)
        try:
            obs, scores, dones, infos = ep.env.step([action])
        except Exception as exc:  # noqa: BLE001 — TextWorld's PDDL parser raises on some states
            logger.warning("AlfworldEnv: env.step failed (%s: %s); ending episode.", type(exc).__name__, exc)
            with self._lock:
                self._episodes.pop(eid, None)
            return None, True, {"reward": float("nan"), "success": 0.0, "steps": ep.steps, "error": True}
        ep.steps += 1
        ep.admissible = self._admissible(infos)

        success = self._success(scores, infos)
        ep.reward = success - self._step_penalty * ep.steps
        done = bool(self._first(dones, False)) or ep.steps >= self._max_steps

        if done:
            with self._lock:
                self._episodes.pop(eid, None)
            self._release_template(ep.template)
            return None, True, {"reward": ep.reward, "success": success, "steps": ep.steps}

        observation = Texts(texts=[self._format(obs, infos, first=False)])
        return observation, False, {"reward": ep.reward}

    def close(self, sample: Sample) -> None:
        """Guaranteed teardown (LIN-533): reclaim this trajectory's episode + pooled template.

        ``step`` already pops + releases on clean ``done`` (and pops-without-reusing on a game
        error), so this only does work for a trajectory that died **in the engine** between turns —
        closing the leak that would otherwise grow ``self._episodes`` and starve the ``_free``
        template pool. Idempotent (a no-op once the episode is gone, so no double-release); runs
        on the trajectory's own drain thread.
        """
        eid = self._episode_id(sample)
        if eid is None:
            return
        self._teardown_episode(eid)

    def _teardown_episode(self, eid: str) -> None:
        with self._lock:
            ep = self._episodes.pop(eid, None)
        if ep is not None:
            self._release_template(ep.template)

    @staticmethod
    def _episode_id(sample: Sample) -> Optional[str]:
        ctrl = sample.parts[0].control or {}
        return (ctrl.get("alfworld") or {}).get("episode_id")

    @staticmethod
    def _first(seq: Any, default: Any) -> Any:
        if isinstance(seq, (list, tuple)):
            return seq[0] if len(seq) else default
        return seq if seq is not None else default

    def _success(self, scores: Any, infos: Any) -> float:
        """ALFWorld terminal success → 1.0/0.0. Prefers ``infos['won']``, falls back
        to a goal-progress score of 1.0."""
        won = self._first(infos.get("won"), None) if isinstance(infos, dict) else None
        if won is not None:
            return 1.0 if bool(won) else 0.0
        try:
            return 1.0 if float(self._first(scores, 0.0)) >= 1.0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _admissible(self, info: Any) -> List[str]:
        """The admissible commands for the current state (batch row 0), as a list."""
        cmds = self._first(info.get("admissible_commands"), None) if isinstance(info, dict) else None
        return [str(c) for c in cmds] if cmds else []

    def _format(self, obs: Any, info: Any, *, first: bool) -> str:
        text = str(self._first(obs, obs))[: self._max_obs_chars]
        cmds = self._first(info.get("admissible_commands"), None) if isinstance(info, dict) else None
        blocks: List[str] = []
        if first:
            blocks.append(_SYSTEM)
        blocks.append(text)
        if cmds:
            blocks.append("Admissible actions: " + ", ".join(str(c) for c in cmds))
        return "\n\n".join(blocks)


__all__ = ["AlfworldEnv", "list_alfworld_games"]
