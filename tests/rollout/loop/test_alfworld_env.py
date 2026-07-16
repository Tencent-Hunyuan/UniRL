"""CPU tests for AlfworldEnv (LIN-519) — no alfworld install needed.

The ALFWorld backend is lazy (``_ensure_backend``) and the per-episode env factory
(``_open_episode``) is isolated, so we inject a mock TextWorld env and exercise the
real protocol: reset builds the ReAct prompt + mints an episode id in the control bag,
step parses the action + steps the sim + emits reward-on-done, and concurrent
trajectories (even siblings with the SAME sample id) get isolated episodes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from unirl.rollout.loop.alfworld_env import AlfworldEnv, _match_admissible, _parse_action  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

_SP = ARSamplingParams(samples_per_prompt=1, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8)


class _FakeTW:
    """Mock ALFWorld TextWorld env: wins when the action equals ``solve_on``."""

    def __init__(self, solve_on: str = "win") -> None:
        self.solve_on = solve_on
        self.closed = False

    def reset(self):
        return (
            ["You are in a room. There is a mug 1."],
            {"admissible_commands": [["look", "take mug 1", "win"]], "won": [False]},
        )

    def step(self, actions):
        won = actions[0] == self.solve_on
        obs = [f"You did: {actions[0]}."]
        return obs, [1.0 if won else 0.0], [won], {"admissible_commands": [["look", "win"]], "won": [won]}

    def close(self):
        self.closed = True


def _env(monkeypatch, solve_on: str = "win", max_steps: int = 5) -> AlfworldEnv:
    env = AlfworldEnv(split="train", max_steps=max_steps)
    env._ready = True  # bypass the lazy alfworld import
    env._games = ["g0", "g1"]
    # _open_episode returns (tw_env, pooled_template); the mock needs no template
    monkeypatch.setattr(env, "_open_episode", lambda gi: (_FakeTW(solve_on), None))
    return env


def _request(sid: str, game_index: int) -> Sample:
    return Sample.request(
        Part.input([sid], primitives={"text": Texts(texts=["ignored"])}, metadata=[{"game_index": game_index}])
    )


def _turn(sample: Sample, action_text: str) -> Sample:
    return sample.fork(1, sampling_params=_SP).with_filled_frontier(primitives={"text": Texts(texts=[action_text])})


def test_parse_action():
    assert _parse_action("Thought: I look around.\nAction: go to cabinet 1") == "go to cabinet 1"
    assert _parse_action("take mug 1") == "take mug 1"  # bare fallback
    assert _parse_action("") == ""


def test_match_admissible():
    adm = ["go to cabinet 1", "take mug 1", "put mug 1 in/on cabinet 1"]
    assert _match_admissible("go to cabinet 1", adm) == "go to cabinet 1"  # exact
    assert _match_admissible("go to the cabinet 1", adm) == "go to cabinet 1"  # token overlap
    assert _match_admissible("take mug", adm) == "take mug 1"  # containment
    assert _match_admissible("fly to the moon", adm) == "fly to the moon"  # no match -> raw
    assert _match_admissible("look", []) == "look"  # no admissible -> raw


def test_reset_builds_react_prompt_and_episode(monkeypatch):
    env = _env(monkeypatch)
    s = env.reset(_request("r0:g0", 0))
    root = s.parts[0]
    assert root.sample_ids[0] == "r0:g0"
    text = root.primitives["text"].texts[0]
    assert "Action:" in text and "mug 1" in text and "Admissible actions:" in text
    assert (root.control or {}).get("alfworld", {}).get("episode_id")
    assert len(env._episodes) == 1


def test_step_reward_on_done(monkeypatch):
    env = _env(monkeypatch, solve_on="win")
    s = env.reset(_request("r0:g0", 0))
    obs, done, info = env.step(_turn(s, "Thought: look.\nAction: look"))
    assert not done and obs is not None and info["reward"] == 0.0
    s = s.observe(obs)
    obs2, done2, info2 = env.step(_turn(s, "Action: win"))
    assert done2 and obs2 is None
    assert info2["reward"] == 1.0 and info2["success"] == 1.0
    assert not env._episodes  # episode popped on done


def test_max_steps_terminates(monkeypatch):
    env = _env(monkeypatch, solve_on="never", max_steps=2)
    s = env.reset(_request("r0:g0", 0))
    _, d1, _ = env.step(_turn(s, "Action: look"))
    s = s.observe(Texts(texts=["o"]))
    _, d2, info2 = env.step(_turn(s, "Action: look"))
    assert not d1 and d2  # hits max_steps=2
    assert info2["reward"] == 0.0 and info2["success"] == 0.0


def test_same_sample_id_siblings_isolated(monkeypatch):
    """The n GRPO siblings are fanned as identical tasks (same sample id); each must
    still get its OWN episode — keyed by the minted episode id, not the sample id."""
    env = _env(monkeypatch)
    s1 = env.reset(_request("same", 0))
    s2 = env.reset(_request("same", 0))
    e1 = s1.parts[0].control["alfworld"]["episode_id"]
    e2 = s2.parts[0].control["alfworld"]["episode_id"]
    assert e1 != e2
    assert len(env._episodes) == 2
    # finishing one leaves the other's episode intact
    env.step(_turn(s1, "Action: win"))
    assert e1 not in env._episodes and e2 in env._episodes
    obs_b, done_b, _ = env.step(_turn(s2, "Action: look"))
    assert not done_b and obs_b is not None


def test_close_reclaims_episode_and_template(monkeypatch):
    """Leak regression (LIN-533): a trajectory that dies IN THE ENGINE never reaches ``step``'s
    done-path, so its episode stays in ``_episodes`` and its pooled template stays checked out.
    The engine's ``finally`` hook (``close``) reclaims both — idempotently."""
    env = _env(monkeypatch)
    template = object()  # a sentinel pooled template to prove it returns to _free
    monkeypatch.setattr(env, "_open_episode", lambda gi: (_FakeTW("win"), template))
    s = env.reset(_request("r0:g0", 0))
    assert len(env._episodes) == 1 and env._free == []  # leased, not yet released

    env.close(s)
    assert env._episodes == {} and env._free == [template]  # episode popped, template pooled

    env.close(s)  # idempotent: no double-release, no error
    assert env._episodes == {} and env._free == [template]
