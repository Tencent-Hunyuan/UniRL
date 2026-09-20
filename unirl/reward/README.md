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
2. **Builds the request** — pairs the frontier's generated media with its aligned
   prompts and non-text conditioning.
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

### Micro-batching the reward, and optionally overlapping it

`RewardStack` (`stack.py`) is the optional worker-side scheduler for the diffusion
training path. It receives one DP shard, slices the frontier Part into
`micro_batch_size` rows at a time, calls the rollout engine per micro, scores that micro,
and `Part.concat`s the results back into one Sample. Both the engine and the
`RewardService` arrive as sibling roles resolved on the same Worker, so every call is
in-process: no dispatch, hence no `batch_size % dp_size` constraint on a micro.

Wired by a `rewardstack:` block; `DiffusionTrainer` builds it beside the rollout engine
and scores inside the generation residency window instead of `_reward_phase()`.

**It is for a reward that costs real time, and that means a remote one.** Colocated with
a cheap scorer there is nothing to win: on `sd3_trainside` reward is 3.12 s of a 108.8 s
step and the stack returns 0.15 s. With the same scorer moved behind HTTP onto its own
GPU (`sd3_trainside_http`), reward is 15.97 s of a 157.2 s step, and the micro loop alone
takes generate+reward from 62.58 s to 52.40 s and step time from 157.2 s to 144.7 s.

`overlap: true` additionally runs each micro's scoring on a one-thread pool while the
next micro generates. It is **off by default and opt-in**: it improved the phase it
targets by 2.79 s, but step time did not follow, and one run per arm cannot separate that
from noise. Turn it on only with several runs per arm to check it.

Why the serial micro loop wins is **not established**. The measurement changes two things
at once and cannot apportion them: request size (one 128-image POST becomes eight of 16)
and arrival synchronization (the driver's generate-then-score barrier disappears, so the
DP ranks may drift and keep a single shared scorer busy). Setting
`RemoteRewardSpec.request_batch_size` to the same value on a recipe *without* the stack
isolates the first of those, and is the cheap way to find out.

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

- **`rewardstack:` is colocate-only, `sp_size: 1` only, and it takes over two
  driver-side facts** — `RewardStack` resolves the engine and the service as
  siblings on one Worker, so `layout: separate` or `reward_fraction > 0` place them
  in different placement scopes and the trainer rejects the block. The stack also
  inherits the engine's SP layout, so with `sp_size > 1` every rank of an SP group
  would receive the same shard and score all of it; the trainer rejects that too.
  It scores inside the generation window, so `offload_train_during_reward` is
  rejected alongside it, a local GPU scorer now shares its peak with the awake
  engine instead of running after `rollout.sleep()`, and the driver-side
  `perf/generate_time_s` / `perf/reward_time_s` timers stop firing because those
  patch the Handle attributes the stack no longer routes through; the stack logs
  its own `generate_s` / `score_s` split per call instead. Set `micro_batch_size`
  equal to `rollout.forward_batch_size` to keep chunk boundaries — and the per-step
  SDE noise draw order — identical to the serial path.
- **No scorer that draws from the global RNG may run under `overlap: true`** — SD3's
  per-step SDE noise comes from the shared global CUDA generator
  (`sde/kernels.py:300`, `generator=None`, and no model threads one), so a scorer
  drawing concurrently interleaves with the sampler's draws in an unspecified order
  and makes the *rollout* nondeterministic, not just the score. PickScore, CLIP and
  HPS draw nothing; a sampling in-process LLM judge or a diffusion-based scorer does.
  Use `overlap: false` for those — which is the default, so this bites only if you
  turned it on. A colocated GPU scorer also shares the default CUDA stream with
  generation, so only its CPU-side work overlaps; the thread pays off for a remote
  scorer, where the HTTP round trip is what it hides.
- **Two smaller consequences of the stack owning the micro loop** — the engine's own
  chunk loop goes inert, so `sglang_diffusion`'s per-chunk `torch.cuda.empty_cache()`
  (`engine.py:136`) stops firing and peak *reserved* memory can shift on those
  recipes; and a generate failure surfaces up to one scoring call late, because the
  executor drains before the exception leaves the step.
- **A non-finite/missing reward fails the whole step, by design** — fix the scorer.
  `raise_on_failure=False` (remote only) does *not* let training continue on it: the
  backend returns zeros with `successes=[False]`, and `score_and_attach`'s fail-fast
  then raises on those flags anyway. So it can't silently zero-poison a group; leave
  it `True`.
- **`input_kind` must match the media** (`image`/`video`/`text`) — it picks which
  decoded key the backend sees. Remote allows only `image`/`video`; local scorers
  may be `text`.
- **`math_verify` grades each batch in a `forkserver` child.** Its
  `signal.alarm` timeouts require the main thread, which threaded Ray actors do not
  provide. The forkserver preloads `math_verify` and the scorer module, avoiding
  repeated imports without inheriting the worker's threaded process state. The parent
  waits on both the result pipe and child sentinel; keep `proc.start()` inside the
  cleanup boundary so startup failures also release resources. Per-operation timeout
  is `UNIRL_MATHVERIFY_TIMEOUT_S` (default 10s), with a hard batch cap of
  `3 * timeout * jobs + 60s`. Wrong or unparsable answers return 0.0, while child,
  IPC, and batch-timeout failures raise. Teardown uses `SIGKILL` plus a bounded join;
  do not replace it with `multiprocessing.Pool.terminate()`, whose worker join is
  unbounded.
- **`base_device` is ignored by the remote backend** (it's HTTP-only); local
  scorers honor it, falling back to CPU with a warning if CUDA is unavailable.
- **Prompt-based rewards use the generation prompt by default.** Set
  `prompt_source: original` to score against the original user prompt instead.
