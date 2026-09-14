# Reward

> **Where it fits:** the *reward* step of the loop —
> rollout → **reward** → advantage → train → sync. In: the rollout engine's
> response `Sample`. Out: per-sample rewards (which the trainer turns into advantages).
> Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/reward-flow-new.png" alt="UniRL reward: RewardService.score_and_attach turns the rollout output (decoded image or text) into a per-sample reward via exactly one backend — local is a single in-process scorer (PickScore, CLIP, HPS, OCR, GenEval2, …) while remote is an HTTP server that runs a panel of reward models (required_rewards) and weight-aggregates them (weighted_sum, mean, min, max) into one reward plus the per-model breakdown; the trainer then z-scores that reward into the advantage" width="100%">
</div>

*One `RewardService` wraps **one backend**: a single in-process scorer (local) or a remote HTTP server that runs and weight-aggregates a **panel** of reward models. The per-sample reward it attaches is what the trainer z-scores into the advantage.*

## What it is

`unirl.reward` scores what a rollout produced — an image, a video, or text — and
writes a per-sample reward back onto the Sample's frontier Part. A `RewardService` wraps
exactly **one** `RewardBackend`: either a local in-process scorer (PickScore, HPS,
CLIP, OCR, GenEval2, math, multiple-choice, video, …) or `RemoteRewardBackend`, a
thin HTTP client for the standalone server in `unirl-reward-service/`.

Turning rewards into advantages is the trainer's job
(`Part.compute_advantages`); generating the media is the rollout engine's.

## Why it exists

RL is only as good as its reward signal, and two things about that signal are
easy to get wrong — so this module owns both:

- **One interface over many scorers.** Local or remote, image/video/text, the
  trainer always calls the same `score_and_attach(sample)` and never touches a
  backend. Swapping PickScore for a remote multi-reward server is a recipe change,
  not a code change.
- **A bad reward must stop the step, not poison it.** A single NaN, null, or
  failed inference call would silently skew a whole GRPO group's advantages. The
  service fails loud: any non-finite or missing reward raises instead of flowing
  into training.

## How it works

Everything goes through one method, `RewardService.score_and_attach(sample)`
(`service.py`). It runs per DP shard (sharding the Sample by prompt-tree), so it
never mutates the input Sample — it returns a fresh one. Per call it:

1. **Refuses precomputed rewards** — raises if the frontier Part already has
   `rewards` (actor-side scoring is the only writer).
2. **Pairs input with output** — the conditioning (`Sample.conditioning`) with the
   media in the frontier Part's `primitive`, already row-aligned (no expansion).
3. **Scores** — hands a typed `RewardRequest` to `backend.compute_rewards`, getting
   back rewards, per-component rewards, and per-sample success flags.
4. **Fails fast** — raises and names the sample if any failed.
5. **Zeroes runaway AR traces** — when the scored Part is itself an AR generation,
   one that hit `max_new_tokens` (never terminated) gets reward 0, so training
   doesn't learn to ramble to the cap.
6. **Attaches** `rewards` + `component_rewards` and returns the Sample.

A backend is just `compute_rewards(request) -> RewardResponse`. Local scorers
(`local/`) subclass `LocalRewardBackend` and implement `_compute_model_rewards`;
the remote backend (`remote.py`) sends one or more bounded `POST /score` calls,
multiplexes every requested reward in each item, and derives success from the
merged response.

### Managed image scorers

`ManagedScorerProcessBackend` is the environment-isolated, rank-affine middle
ground between local and externally deployed rewards. Each reward worker launches
one scorer child with an explicit Python executable; the child inherits that
worker's single visible GPU and serves only its local prompt-tree shard over
loopback HTTP. The initial capability is deliberately limited to image and
image-edit histories.

Its config separates process ownership, scorer construction, and remote-client
semantics:

```yaml
backend:
  _target_: unirl.reward.managed_process.ManagedScorerProcessBackend
  base_device: cpu
  config:
    _target_: unirl.reward.managed_process.ManagedScorerProcessSpec
    process:
      _target_: unirl.reward.managed_process.ManagedProcessConfig
      python_executable: /venvs/reward/bin/python
      service_root: /workspace/UniRL/unirl-reward-service
    scorer:
      _target_: unirl.reward.managed_process.ManagedScorerConfig
      name: editreward
      history_kind: image_edit
      params: {device: cuda, checkpoint_path: /models/EditReward}
    client:
      _target_: unirl.reward.remote.RemoteRewardSpec
      base_url: managed://rank-affine
      required_rewards: [editreward]
      input_kind: image
      request_batch_size: 8
    gpu_residency: resident
```

`request_batch_size` bounds transport/scorer calls independently from the DP
shard size. Identity echo is required for managed children. The parent manages
GPU residency through the child's `onload`, `offload`, and `shutdown` endpoints;
`drain` remains available for explicit synchronization.

**Extending it:** a new local scorer is usually a file in `local/` subclassing
`LocalRewardBackend` (set `canonical_model_name`, implement `_load_model` +
`_compute_model_rewards`, add a `<Name>Spec`), wired in a recipe by `_target_`. A
new remote reward needs no UniRL code — add it to the server and list its name in
`RemoteRewardSpec.required_rewards`.

## Gotchas

- **A non-finite/missing reward fails the whole step, by design** — fix the scorer.
  `raise_on_failure=False` (remote only) does *not* let training continue on it: the
  backend returns zeros with `successes=[False]`, and `score_and_attach`'s fail-fast
  then raises on those flags anyway. So it can't silently zero-poison a group; leave
  it `True`.
- **`input_kind` must match the media** (`image`/`video`/`text`) — it picks which
  decoded key the backend sees. Remote allows only `image`/`video`; local scorers
  may be `text`.
- **`math_verify` grades in a child process, and that is not optional.** Its own
  timeouts are `signal.alarm`-based, so they work only on the main thread — which the
  reward path is not; enabling them there raises and scores every sample 0. Inside the
  child the main thread is the child's own, so they work and are passed through
  (`UNIRL_MATHVERIFY_TIMEOUT_S`, default 10s). One child per reward call grades the
  whole batch and exits — ~1.2s, amortised over the ~16 grades a call carries — and
  shares nothing with the caller, so a child that is OOM-killed can only cost its own
  batch, which is scored 0.0. Three details are load-bearing: `forkserver` rather than
  `fork`, because `fork` runs `logging`'s registered at-fork handler, which acquires
  the logging lock with no timeout and can block the forking thread forever when the
  worker's other threads log; `wait()` on the child's sentinel as well as the pipe, so
  a child that dies immediately costs a round-trip instead of the whole deadline; and
  `proc.start()` inside the `try`, because `forkserver` forks the child *before* the
  parent writes the job payload to it, so a child dying in that window raises
  `BrokenPipeError` out of `start()`. **Do not reintroduce a `multiprocessing.Pool`
  here**: `Pool.terminate()` is unbounded. `_terminate_pool` sends each worker `SIGTERM`
  and then joins it with no timeout (CPython 3.12 `pool.py:732`), and a Python signal
  handler only runs between bytecodes — so a grade inside a long C-level `sympy` call
  never handles the `SIGTERM`, never exits, and the join blocks forever. That is the
  same runaway expression whose slowness tripped the deadline that called `terminate()`,
  so the condition triggering the teardown is the one that makes it hang. Confirmed by a
  captured stack from a 32-GPU reproduction. `proc.kill()` is used instead because
  `SIGKILL` cannot be caught, blocked or ignored and needs no bytecode boundary.
- **A standalone test of the grader needs a real file with an `if __name__ ==
  "__main__":` guard.** `forkserver` re-imports `__main__` in the child, so an unguarded
  script — or a heredoc, where `__main__` is `<stdin>` — makes every grade return
  `False` from a completely healthy grader, which is indistinguishable from the grader
  failing closed. Production is unaffected: in a reward worker `__main__` is Ray's
  `default_worker.py`.
- **`base_device` is ignored by the remote backend** (it's HTTP-only); local
  scorers honor it, falling back to CPU with a warning if CUDA is unavailable.
