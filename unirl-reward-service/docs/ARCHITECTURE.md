# Reward Service — Architecture and Data Flow

This document describes the Reward Service's **system topology**, **request data flow**, **key abstraction layers**, **error/resource isolation semantics**, and **extension points**. It targets three audiences:

- New contributors — understand the whole system in one pass.
- People debugging or extending it — know which file each responsibility lives in.
- Future maintainers — understand the "why" behind the design choices.

Companion documents:
- [`README.md`](../README.md) — how to install and use it.
- [`docs/DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) — development timeline and decision archive.
- [`CHANGELOG.md`](../CHANGELOG.md) — user-facing change log.

---

## 1. Static Topology

One diagram for the entire process structure:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Host (single machine, multi-GPU)                                        │
│                                                                          │
│   ┌──────────────────────────┐                                           │
│   │ uvicorn + FastAPI        │   (1 process, asyncio event loop)         │
│   │   create_app(cfg)        │   source:  reward_service/server.py       │
│   │                          │                                           │
│   │   /score   /health       │                                           │
│   │   /rewards               │                                           │
│   │                          │                                           │
│   │   app.state.pool ────────┼──► WorkerPool  (reward_service/workers/pool.py)
│   └──────────────────────────┘                │                          │
│                                               │ 1 pool : N groups        │
│                                               ▼                          │
│   ┌────────────── Ray runtime (ray.init) ────────────────┐               │
│   │                                                      │               │
│   │  ┌───────────────┐ ┌───────────────┐   ┌──────────┐  │               │
│   │  │ WorkerGroup   │ │ WorkerGroup   │…  │ Worker…  │  │               │
│   │  │  name=clip    │ │ name=geneval2 │   │          │  │               │
│   │  │  (group.py)   │ │  (group.py)   │   │          │  │               │
│   │  │               │ │               │   │          │  │               │
│   │  │  actors[0] ───┼─┼──────────┐    │   │          │  │               │
│   │  │  actors[1] ───┼─┼─────────┐│    │   │          │  │               │
│   │  └───────────────┘ └─────────┼┼────┘   └──────────┘  │               │
│   │                              ││                      │               │
│   │   @ray.remote ScorerActor    ││  (actor.py)          │               │
│   │   ┌──────────────────────┐   ││   ┌──────────────┐   │               │
│   │   │ ScorerActor          │   ││   │ ScorerActor  │   │               │
│   │   │   scorer=ClipScorer  │   ││   │  TP=2 vLLM   │   │               │
│   │   │   num_gpus=1         │   ││   │  num_gpus=2  │   │               │
│   │   └─────────┬────────────┘   ││   └──────┬───────┘   │               │
│   └─────────────┼────────────────┼┼──────────┼───────────┘               │
│                 │                ││          │                           │
│           GPU 0 │                ││          │ GPU 6 + GPU 7             │
│                 ▼                ▼▼          ▼                           │
│            [ CUDA ctx ]   [ CUDA ctx × 2 ]   [ CUDA ctx × 2 ]            │
│                                                                          │
│              GPUs are owned exclusively, never shared across rewards     │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key properties**:

- **1 FastAPI process** · **N WorkerGroups** · **Σ num_replicas ScorerActors** (Ray subprocesses).
- Ray allocates GPUs to an actor as an **integer resource** via `ScorerActor.options(num_gpus=N, num_cpus=C)` — a GPU belongs to exactly one actor at a time.
- For a vLLM-style scorer with TP=N, the actor's `num_gpus=N` must match vLLM's `tensor_parallel_size=N` (both set in YAML).
- Multi-host extension point: just add `cluster.ray_address` to the YAML; the architecture is unchanged. See §5.3.

**Components and their source files**:

| Component | Source | Responsibility |
|---|---|---|
| HTTP gateway | `reward_service/server.py` | The `/score` `/health` `/rewards` endpoints; bucket / dispatch / gather logic |
| Schema | `reward_service/schemas.py` | Pydantic: `ScoreRequest` / `RewardRequest` / `HistoryTurn` / `ScoreResponse` |
| Config | `reward_service/config.py` | YAML → `ServiceCfg` + `RewardModelCfg` dataclasses; validates `num_replicas≥1`, `num_gpus≥0`, unique names |
| WorkerPool | `reward_service/workers/pool.py` | Ray runtime lifecycle + group registry + dispatch by name |
| WorkerGroup | `reward_service/workers/group.py` | N actors + round-robin dispatch (`itertools.cycle`) |
| ScorerActor | `reward_service/workers/actor.py` | `@ray.remote` thin shell: constructs the scorer, forwards `score()` |
| BaseScorer | `reward_service/scorers/base.py` | The abstract contract: `score(items) -> list[dict]` + `sub_metric_names` |
| Registry | `reward_service/scorers/registry.py` | `register(name, cls)` + optional-dep tolerance (`_try_import`) |
| Concrete scorers | `reward_service/scorers/{clip,pickscore,imagereward,hpsv2_scorer,hpsv3_scorer,unified_reward,geneval2,geneval,ocr,wise,videoalign}.py` (+ vendored `_videoalign/`) | One module per reward; each ends with `register("name", Cls)` |

---

## 2. Data Flow (Request → Response)

The full sequence of one `POST /score`, using a batch of 2 requests that each need 2 rewards:

```
Client                 FastAPI              WorkerPool            WorkerGroup          ScorerActor (Ray)
  │                       │                     │                      │                      │
  │ POST /score           │                     │                      │                      │
  │  {requests:[r0,r1]}   │                     │                      │                      │
  ├──────────────────────►│                     │                      │                      │
  │                       │                                                                   │
  │                       │ (1) asyncio.to_thread(_request_to_item × 2)                       │
  │                       │     base64 → PIL.Image  (CPU-heavy, offloaded off the event loop) │
  │                       │                                                                   │
  │                       │ (2) _bucket_by_reward                                             │
  │                       │     r0 needs [clip, hpsv2]                                        │
  │                       │     r1 needs [clip, pickscore]                                    │
  │                       │     → buckets = {                                                 │
  │                       │         "clip":      ([0, 1], [item0, item1]),                    │
  │                       │         "hpsv2":     ([0],    [item0]),                           │
  │                       │         "pickscore": ([1],    [item1]),                           │
  │                       │       }                                                           │
  │                       │                                                                   │
  │                       │ (3) pool.dispatch(name, bucket_items) — once per reward           │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("clip",  ├──► round-robin ─────►│                      │
  │                       │       [item0,item1])│    actors[rr_idx]    ├──► .score.remote()─►│ ObjectRef₀
  │                       │                     │                      │                      │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("hpsv2", ├──► …                 ├──► .score.remote()─►│ ObjectRef₁
  │                       │       [item0])      │                      │                      │
  │                       │                     │                      │                      │
  │                       ├────────────────────►│                      │                      │
  │                       │   dispatch("pick…", │                      │                      │
  │                       │       [item1])      │                      │                      │
  │                       │◄────── 3 ObjectRef ─┤                      │                      │
  │                       │                                                                   │
  │                       │                                            ScorerActor runs:      │
  │                       │                                              scorer.score(items)  │
  │                       │                                              →  list[dict[str,float]]
  │                       │                                                                   │
  │                       │ (4) asyncio.gather(_await_ref per reward)                         │
  │                       │     each ref is awaited with a per-reward timeout —               │
  │                       │     one failure does not affect the others                        │
  │                       │                                                                   │
  │                       │ (5) using (bucket indices, reward name), fill results[i][name]    │
  │                       │     and errors[i]                                                 │
  │                       │                                                                   │
  │  200 OK               │                                                                   │
  │  {results:[…],        │                                                                   │
  │   errors:[…]}         │                                                                   │
  │◄──────────────────────┤                                                                   │
```

**Sequence highlights**:

1. **Decoding and awaiting are fully async**: base64→PIL runs in `asyncio.to_thread` and each Ray ref is awaited via `asyncio.gather` over `_await_ref`, so the event loop is never blocked — `/health` and other `/score` requests can be served concurrently.
2. **One Ray call per reward**: all items under the same reward are packed into a single `score([item0, item1, ...])` call; the bucket records each request index `i`, and the aggregation step uses it to place results back into `results[i]`.
3. **Round-robin only matters when `num_replicas>1`**; with a single replica it always hits `actors[0]`.
4. **Image-copy cost**: when the same `PIL.Image` lands in multiple buckets (e.g. `item0` goes to both clip and hpsv2), Ray pickles it into the object store **twice**. This is a known limitation (see DEVELOPMENT_LOG §7.2); for very large batches you can `ray.put(item)` to deduplicate.

---

## 3. Key Abstraction Layers (Four Layers)

From "most stable, most externally promised" down to "most implementation detail":

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  L0 · BaseScorer                               scorers/base.py              │
│  Contract layer (most stable)                                               │
│                                                                             │
│   class BaseScorer(ABC):                                                    │
│       name: str                                                             │
│       sub_metric_names: tuple[str, ...]                                     │
│       def score(items: list[ScoreItem]) -> list[dict[str, float]]: ...      │
│                                                                             │
│   —— pure Python; knows nothing about Ray, FastAPI, or GPU scheduling        │
│   —— output constraints: len(out) == len(items), sub-metric keys str→float  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ registry.register(name, Cls)
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L1 · Concrete scorers                         scorers/{clip,hpsv2,…}.py    │
│  Model implementation layer                                                 │
│                                                                             │
│   class ClipScorer(BaseScorer):                                             │
│       def __init__(self, model_name, weights_path, dtype, device): ...      │
│       def score(self, items): ...  # pure torch / vLLM inference             │
│                                                                             │
│   end of module: register("clip", ClipScorer)                               │
│                                                                             │
│   —— reuses scorers/_common.py helpers: resolve_dtype / resolve_model_path / │
│      split_last_turn / image_to_data_url / build_vllm_llm_kwargs             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ wrapped by @ray.remote
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L2 · ScorerActor                              workers/actor.py             │
│  Ray thin-shell layer                                                       │
│                                                                             │
│   @ray.remote                                                               │
│   class ScorerActor:                                                        │
│       def __init__(self, scorer_name, params):                              │
│           cls = get_scorer_cls(scorer_name)   # look up registry on the GPU process
│           self.scorer = cls(**params)         # build the model on the GPU process
│       def score(items): return self.scorer.score(items)                     │
│                                                                             │
│   —— key design: pass "name + params", not a scorer instance                 │
│     (avoids serializing GB-scale weights; the model is built from scratch    │
│      on the target GPU instead)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ WorkerGroup spawns N replicas
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│  L3 · WorkerGroup / WorkerPool                 workers/{group,pool}.py      │
│  Scheduling and lifecycle layer                                             │
│                                                                             │
│   WorkerGroup(cfg):                                                         │
│     actors = [ScorerActor.options(num_gpus=n).remote(name, params)          │
│               for _ in range(cfg.num_replicas)]                             │
│     rr = itertools.cycle(range(len(actors)))    # simple round-robin        │
│                                                                             │
│   WorkerPool(ServiceCfg):                                                   │
│     ray.init(…)                                                             │
│     {reward_name: WorkerGroup(reward_cfg) for reward_cfg in cfg.rewards}    │
│                                                                             │
│   —— side effects are concentrated here: Ray init / actor creation / shutdown│
│   —— hands ObjectRefs upward; does no aggregation                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │
                                       │  server.py gets ObjectRefs → gather → assemble
                                       │
                                    ( L4 HTTP gateway already covered in §1 §2 )
```

**Each layer depends only on the one below it.** In principle:
- Replace `BaseScorer`: the whole stack must be rewritten.
- Replace `ScorerActor` with a different RPC (gRPC, Ray Serve, etc.): L0/L1 are untouched; only L3's group/pool implementation changes.
- Replace the HTTP gateway: L0–L3 are fully reused.

---

## 4. Error and Resource Isolation (Isolation Semantics)

The Reward Service's two hard guarantees:

### 4.1 Resource isolation

**Guarantee**: a GPU belongs to exactly one reward group at any moment.

**Mechanism**: `WorkerGroup._spawn_actors()` uses `ScorerActor.options(num_gpus=self.cfg.num_gpus)`, so Ray allocates GPUs as an **integer resource**; two actors are never scheduled onto the same card.

**Reflected in YAML**:
```yaml
- name: clip
  num_gpus: 1        # owns 1 card exclusively
  num_replicas: 1

- name: geneval2
  num_gpus: 2        # owns 2 cards exclusively (for vLLM TP=2)
  num_replicas: 1
  params:
    tensor_parallel_size: 2    # must match num_gpus
```

**Validated in `config.py`**: `num_gpus < 0` and `num_replicas < 1` are rejected at load time (with a reward-named error message). `tensor_parallel_size > num_gpus` is also rejected up front, because vLLM's Ray executor would otherwise crash at actor init.

**Over-budget consequence**: `ray.init()` succeeds but some actor hangs or reports insufficient resources during GPU allocation — a Ray-level error that shows up on `/health` (that group's `ping()` does not return).

### 4.2 Error isolation

**Guarantee**: a single reward's failure (OOM / parse error / actor crash / timeout) does not make the whole batch return 500. The failed reward writes its exception into `ScoreResponse.errors[i][reward_name]`, while the other rewards return scores as usual.

**Mechanism**: in `server.py`, each reward's Ray ref is awaited independently via `_await_ref`, and the per-reward results are collected with `asyncio.gather`:

```python
async def _await_ref(pool, name, ref, timeout_s):
    try:
        result = await asyncio.wait_for(pool.as_awaitable(ref), timeout=timeout_s)
        return name, result
    except TimeoutError as e:
        wrapped = TimeoutError(f"reward {name!r} exceeded {timeout_s}s")
        wrapped.__cause__ = e
        return name, wrapped
    except Exception as e:
        logger.exception("reward %s failed: %s", name, e)
        return name, e
```

Each ref is awaited (and `ray.get` happens) in isolation, with a per-reward `score_timeout_s` deadline — one reward's actor exception is caught and turned into a value, so it **does not poison the other rewards' ObjectRefs**. During assembly, a reward whose gathered value is an `Exception` has `repr(exception)` written into `errors[i][name]` for each of its request indices.

**Example response** (reward `clip` failed, the rest are OK):
```json
{
  "results": [
    {"hpsv2": {"hpsv2": 0.27}},
    {"hpsv2": {"hpsv2": 0.31}, "pickscore": {"pickscore": 0.82}}
  ],
  "errors": [
    {"clip": "RayActorError(...)"},
    {"clip": "RayActorError(...)"}
  ]
}
```

Request `0` asked for `clip + hpsv2`; `clip` failed → only `hpsv2` has a score, and `errors[0]["clip"]` records the exception; `hpsv2` is unaffected.

### 4.3 What the two guarantees mean for clients

- A client can safely pack unrelated rewards into the same batch — **they will not drag each other down**.
- A client must **read both `results[i]` and `errors[i]`** — looking only at `results` would silently miss a failed reward.

---

## 5. Extension Points

### 5.1 Adding a new scorer

1. Create `my_scorer.py` under `reward_service/scorers/`:
   ```python
   from reward_service.scorers.base import BaseScorer, ScoreItem
   from reward_service.scorers.registry import register

   class MyScorer(BaseScorer):
       name = "my_reward"
       sub_metric_names = ("my_metric",)

       def __init__(self, model_name: str, weights_path: str | None = None, ...):
           # load the model on the target GPU
           ...

       def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
           ...

   register("my_reward", MyScorer)
   ```
2. Add a line to the `SCORER_MODULES` dict in `registry.py`: `"my_reward": "reward_service.scorers.my_scorer"`. (ScorerActor imports the module via this mapping inside its own venv and triggers `register()`; **the main process does not import scorer modules**, so do not import heavy dependencies at module top level — put them inside `__init__`/methods, as clip.py / imagereward.py do.)
3. Add the matching section to `configs/service.example.yaml`.
4. Add `test_my_scorer.py` under `tests/scorers/` (CPU pure-function test + GPU smoke test).

**What you do NOT need to change**: `server.py` / `workers/` / `schemas.py` — the new scorer is auto-discovered by the registry, and WorkerPool creates its group automatically from the YAML.

### 5.1.1 Hard limits of the isolation level (how far a new scorer's env can go)

Each scorer gets an isolated venv via `envs/<name>.txt` + Ray `runtime_env`, but the isolation reaches **only the Python-package level, not the Python-interpreter level**, because of two hard constraints of Ray 2.x runtime_env:

1. **The pip backend always uses `--system-site-packages`** (hardcoded in `ray/_private/runtime_env/virtualenv_utils.py`, no knob) → the venv inherits base's torch / compiled extensions and **cannot swap in a different torch build** (doing so would clash, at the ABI level, with the xformers / flash-attn that leak in from base).
2. **The conda/container backend forces `python={the cluster's current version}`** (`ray/_private/runtime_env/conda.py` unconditionally does `deps.append(f"python={py_version}")`) → workers must share the cluster's Python minor version (for cloudpickle compatibility).

**Corollary**: a new scorer's dependencies must be able to run on **base's (Python × torch)** combination (you may only stack/override the pure-Python layer in the venv, e.g. a different `transformers`/`peft` version). A scorer that needs a **different Python or a different torch foundation** (typically: `geneval`'s mmdet 2.x / mmcv-full 1.7.2 needs py3.10 + torch≤2.1) **cannot** be hosted via runtime_env on this cluster (py3.13 / torch 2.x). The only options are: (a) downgrade the whole cluster to a compatible py/torch, or (b) run that scorer as an **out-of-cluster sidecar service** and proxy calls to it through a thin actor. `geneval` is therefore commented out by default in `configs/service.example.yaml`; see `envs/geneval.txt`.

### 5.2 Adding a sub-metric to an existing scorer

- Change the scorer class's `sub_metric_names` tuple + add the new key to the dict its `score()` returns.
- The response schema `ScoreResponse.results[i][reward_name]` is `dict[str, float]`, which naturally supports multiple keys, so **the schema does not change**.
- Add a regression test confirming the new key exists and its value is in a sensible range.

### 5.3 Multi-host scaling

Joining an external Ray cluster requires no code changes, only YAML changes:

1. Bring up a Ray cluster externally (this repo ships a pdsh script):
   ```bash
   export NODE_IP_LIST="ip1:8 ip2:8"   # the first entry is the head
   bash scripts/ray_start.sh
   ```
2. Add a `cluster:` section at the YAML root:
   ```yaml
   cluster:
     ray_address: <head-ip>:6379   # or "auto" if the service runs on the head node
     # namespace: reward-service-prod   # optional, for multiple services sharing one cluster
   ```
3. For any 1-GPU reward that should be distributed across hosts, add to its `RewardModelCfg`:
   ```yaml
   num_replicas: 2
   scheduling: spread   # default is pack = Ray's built-in behaviour; spread passes scheduling_strategy=SPREAD
   ```

**Source mapping**: `reward_service/workers/pool.py::_init_ray` reads `ClusterCfg.ray_address`;
`reward_service/workers/group.py::_actor_options` translates `scheduling` into the Ray actor's `scheduling_strategy` kwarg.

**Caveats**:
- A reward that does cross-node NCCL communication (e.g. vLLM TP=2) must use `scheduling: pack` and rely on the total GPU budget to naturally keep it on a single host — cross-Ethernet TP collapses throughput. See the GenEval2 config in `configs/service.cluster.example.yaml`.
- `scheduling: spread` is only a Ray scheduler hint, not a hard placement-group binding; if replicas still pile onto one machine in practice, consider a placement group.

### 5.4 Speed-up paths (if something becomes a bottleneck)

| Bottleneck | Remedy | Where |
|---|---|---|
| Ray pickles PIL repeatedly | `ray.put(item)` to deduplicate | change `dispatch` in `server.py` to pass an `ObjectRef` |
| Round-robin ignores actor load | load-aware dispatch | replace `itertools.cycle` in `group.py` with a load tracker |
| Slow vLLM first load | dynamic / continuous batching | switch `unified_reward.py` / `geneval2.py` to the vLLM async engine |
| Synchronous `/score` aggregation | stream partial results | change `server.py` to a streaming response |

All of the above are YAGNI deferrals — do them only once a real bottleneck is observed; see the trigger conditions in DEVELOPMENT_LOG §9.

---

## 6. Mapping from Config to Runtime Objects

A reference for tracing "what does this YAML field eventually become" while debugging:

```
YAML                                   config.py            workers/pool.py         scorers/<...>.py
────────────────────────────────────── ──────────────────── ───────────────────── ────────────────────

server:                                ServerCfg
  host: 0.0.0.0                          .host
  port: 8080                             .port                                     ( used when uvicorn starts )

rewards:                               list[RewardModelCfg]
  - name: clip                           .name             ─► WorkerPool._groups[.name]
    scorer: clip                         .scorer           ─► get_scorer_cls(.scorer)   ClipScorer
    num_replicas: 1                      .num_replicas     ─► len(WorkerGroup.actors)
    num_gpus: 1                          .num_gpus         ─► ScorerActor.options(num_gpus=.)
    num_cpus: 2                          .num_cpus         ─► ScorerActor.options(num_cpus=.)
    params:                              .params (dict)    ─► ScorerActor.remote(.scorer, .params)
      model_name: openai/…                                    → ClipScorer(**.params)
      weights_path: /path/to/…                      → resolved by resolve_model_path
      dtype: float32                                         → resolved by resolve_dtype
```

For **vLLM-style scorers** (`unified_reward` / `geneval2` / `wise`), the `dtype / enforce_eager / swap_space / quantization / seed / max_num_seqs / limit_mm_per_prompt / extra_llm_kwargs` keys in `params` flow through `scorers/_common.py::build_vllm_llm_kwargs` and are ultimately passed to `vllm.LLM(**kwargs)`. See DEVELOPMENT_LOG §11.

---

## 7. File Overview (grouped by responsibility)

```
reward_service/
├── server.py            # HTTP gateway — /score /health /rewards
├── __main__.py          # CLI entry — argparse + uvicorn.run
├── config.py            # YAML → ServiceCfg / RewardModelCfg
├── schemas.py           # Pydantic HTTP schemas
├── logging_utils.py     # get_logger(name)
├── client.py            # Python SDK (RewardClient)
│
├── workers/             #  Ray layer
│   ├── pool.py          #  Ray init + group registry
│   ├── group.py         #  N replicas + round-robin
│   └── actor.py         #  @ray.remote ScorerActor
│
└── scorers/             #  Model layer
    ├── base.py          #  BaseScorer + ScoreItem
    ├── registry.py      #  register / get_scorer_cls / _try_import
    ├── _common.py       #  shared helpers: dtype, path, data_url, vLLM kwargs
    ├── clip.py          # ─┐
    ├── pickscore.py     #  │ transformers-style
    ├── imagereward.py   #  │
    ├── hpsv2_scorer.py  #  │
    ├── hpsv3_scorer.py  #  │
    ├── ocr.py           # ─┘
    ├── unified_reward.py# ─┐
    ├── geneval2.py      #  │ vLLM-style
    ├── wise.py          # ─┘
    ├── geneval.py       #  mmdet/mmcv-style (disabled by default)
    └── videoalign.py    #  T2V reward (vendored _videoalign/)
```

---

*The diagrams and prose here defer to the source code. If you find this document inconsistent with actual behaviour, trust the source first and update this document.*
