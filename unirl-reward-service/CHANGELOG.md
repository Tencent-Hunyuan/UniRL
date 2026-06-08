# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed (2026-05-31) — cross-repo reward calling contract (with UniRL)

- **Failure contract made explicit.** `BaseScorer.score` docstring now defines
  the protocol: per-item failures return `float("nan")` (one bad item must not
  fail the batch, since one `score()` call serves the whole reward bucket);
  whole-reward / config failures `raise` (captured in `errors[i][reward]`). The
  consumer (UniRL) treats any non-finite reward as a sample failure
  and fail-fasts, so a NaN never reaches the RL signal as a real score. `ocr`'s
  NaN-on-failure comment reworded to match (no logic change).
- **`geneval` raises on missing metadata** (was a silent `{"geneval": 0.0}`). A
  missing GenEval spec is a caller wiring bug, not a zero reward; raising
  surfaces it through the gateway error channel instead of training on zeros.
- **`videoalign` sub-metrics reordered to `("Overall", "VQ", "MQ", "TA")`** so
  the consumer's default `sub_metric_reduce="first"` trains on Overall, not
  Visual Quality alone.
- **`schemas.py` declared the single source of truth** for the wire protocol.
  UniRL's `RemoteRewardBackend` builds payloads to match it, and a
  contract test there (`tests/reward/test_wire_contract.py`) validates them
  against `ScoreRequest`. This fixes a latent drift: the caller's video payload
  used a flat `{video_b64, prompt}` body this schema rejects with HTTP 422 — it
  now sends history turns. See `docs/DEVELOPMENT_LOG.md §17`.

### Added (2026-05-30) — integration branch `integration/geneval-ocr-clean`

- **`geneval` scorer** (classic GenEval: Mask2Former/mmdet + CLIP color
  classification; distinct from the VQAScore-based `geneval2`) and **`ocr`
  scorer** (GOT-OCR-2.0-hf text-rendering reward, edit-distance based),
  integrated from `feat/add-geneval-ocr-workers`. Both registered in
  `SCORER_MODULES`; `ocr` enabled in `service.example.yaml`, `geneval`
  shipped commented-out (see Known limitation).
- `ocr` rides the shared base env (Tier-1 overlay): pure `transformers`
  stack, no torch pin (`envs/ocr.txt`). Heavy imports deferred into
  `__init__` / function bodies per the dependency-isolation convention.
- **`videoalign` scorer** (VideoReward: VLM-based T2V reward emitting
  VQ/MQ/TA/Overall), integrated from `feat/videoalign`. Adds an optional,
  backward-compatible `videos` field to `ScoreItem`/`HistoryTurn` (existing
  image scorers unaffected); vendored upstream model code under
  `reward_service/scorers/_videoalign/`. Tier-1 overlay env
  (`envs/videoalign.txt`: transformers 4.45.2 + flash-attn). An OCR-style
  silent-failure was avoided: `ocr` now scores NaN (not 0.0) on inference
  error so failures stay distinguishable in the RL signal.

### Known limitation (2026-05-30)

- **`geneval` cannot run on this Python-3.13 / torch-2.x cluster.** Its
  `mmdet 2.28.2` / `mmcv-full 1.7.2` stack needs Python 3.8–3.10 + torch ≤ 2.1.
  Ray's `runtime_env` pins workers to the cluster's Python (`conda.py`) and the
  pip backend always inherits base torch (`virtualenv_utils.py` hardcodes
  `--system-site-packages`), so no isolation backend can host it in-cluster.
  Run on a py3.10 cluster or as an out-of-cluster sidecar. Left commented-out
  in `configs/service.example.yaml`; rationale in `envs/geneval.txt`,
  `docs/ARCHITECTURE.md §5.1.1`, and `docs/DEVELOPMENT_LOG.md §16`.

### Not adopted (2026-05-30)

- The source branch's `ocr_paddle` scorer (PaddlePaddle backend) was **dropped**
  — paddle is a separate non-torch framework; `ocr` (GOT-OCR) covers the OCR
  reward. The source branch's global `pip`→`uv` `runtime_env` switch was **not**
  brought in (kept main's pip + `--system-site-packages` design); that is a
  separate decision for the venv maintainer.

### Breaking (2026-04-29)

- **Per-scorer isolated venv via Ray `runtime_env`**: every reward in YAML
  now **requires** a `runtime_env:` field pointing to a pip requirements
  file (e.g. `envs/clip.txt`). Ray creates a virtualenv per unique
  requirements set and runs the actor inside it. This eliminates the
  `install.sh` `--no-deps` hack and the `_compat.py` monkey-patch layer.
  YAML configs missing `runtime_env:` will fail at startup with a clear
  error. See `docs/DEVELOPMENT_LOG.md §14`.

### Removed (2026-04-29)

- **`reward_service/scorers/_compat.py`**: deleted. Each scorer now runs in
  its own venv with the correct transformers version — shims are no longer
  needed.
- **`install.sh` legacy steps**: removed Step 2 (legacy subdeps), Step 3
  (`--no-deps --force-reinstall` for hpsv2/hpsv3/image-reward), and
  Step 3b (hpsv2 BPE vocab fix). `install.sh` now only installs the base
  environment (`pip install -e ".[server,dev]"`).
- **`pyproject.toml` extras**: removed `[hpsv2]`, `[hpsv3]`,
  `[imagereward]`, `[vllm]`, `[all]` extras and the `transformers` pin
  from `[server]`. Per-scorer deps live in `envs/*.txt`.

### Added (2026-04-29)

- `envs/` directory with 8 requirements files: `base.txt`, `clip.txt`,
  `pickscore.txt`, `imagereward.txt`, `hpsv2.txt`, `hpsv3.txt`,
  `unified_reward.txt`, `geneval2.txt`. Each declares only the delta
  over the base environment (torch/ray/pillow inherited).
- `RewardModelCfg.runtime_env` — required string field in config.
- `_build_runtime_env()` in `reward_service/workers/group.py` — reads
  requirements file, constructs Ray `runtime_env` dict for actor options.
- `install.sh` now supports `uv pip` as primary installer with `pip`
  fallback.
- GenEval2 Soft-TIFA support: `dataset_path` parameter + `_load_vqa_dataset()`
  (cherry-picked from `geneval2-dataset-and-model-paths` branch).
- `datasets/geneval2/geneval2_data.jsonl` — 800 prompts from
  facebookresearch/GenEval2 for Soft-TIFA scoring.

### Fixed (2026-04-29)

- `scripts/ray_stop.sh`: `pkill -f "VLLM::"` self-match bug — bash's own
  cmdline contained the pattern, killing itself before the actual vLLM
  processes. Fixed with regex character class: `pkill -f "VLLM[:]:"`.
  Same fix for `reward_service` → `reward[_]service`.

### Fixed (2026-04-28)

- **`hpsv2` scorer**: complete rewrite to fix `TypeError: score() got an
  unexpected keyword argument 'cp...'`. Model loading (architecture +
  checkpoint + tokenizer) moved to `__init__` — loaded once, not on every
  `score()` call. Inference logic mirrors the upstream `img_score.score()`
  per-item loop exactly (`torch.no_grad` → `unsqueeze(0)` →
  `torch.cuda.amp.autocast` → `features @ features.T` → `diagonal()[0]`).
  PyPI official `hpsv2==1.2.0` reinstalled over a local fork that had an
  incompatible API. 7 new CPU invariant tests added.
  See `docs/DEVELOPMENT_LOG.md §13`.

### Breaking (2026-04-28)

- **`pickscore` params**: removed `processor_name` field. The CLIPProcessor
  is now loaded from the same directory (or HF id) as the model, matching
  clip.py's existing convention. In practice PickScore_v1 already ships
  its processor files (preprocessor_config.json / tokenizer.json /
  vocab.json / merges.txt) alongside the weights, so no separate path is
  needed. YAML configs still listing `processor_name:` under a pickscore
  reward will fail at actor init with a TypeError — remove that line. See
  `docs/DEVELOPMENT_LOG.md §12.13`.

### Added (2026-04-28)

- `scripts/ray_start.sh` now forwards `HTTP_PROXY` / `HTTPS_PROXY` /
  `NO_PROXY` from the launching shell to every raylet it starts (each
  `ray start` invocation is prefixed with these env vars). Raylet
  inherits them, and so do all actor subprocesses — this lets
  ImageReward's hardcoded `BertTokenizer.from_pretrained('bert-base-uncased')`
  reach HuggingFace from worker nodes that need a proxy.
- `reward_service/config.py::load_config` now rejects YAML where
  `params.tensor_parallel_size` exceeds `num_gpus` for a reward. This
  catches the otherwise-confusing "Current node has no GPU available"
  crash from vLLM's Ray executor at startup instead of at actor init.

- `scripts/ray_stop.sh` now tears down the full stack, not just Ray: on
  every node it SIGTERMs `python -m reward_service`, waits up to 10s for
  the FastAPI lifespan `finally` block to run, SIGKILLs stragglers, then
  purges any leftover vLLM child processes that Ctrl+C orphans (they
  hold onto GPU memory otherwise), and finally `ray stop --force`s. The
  vLLM purge pattern matches every setproctitle variant seen in the
  wild (`VLLM::EngineCore` on 0.11, `VllmWorkerProcess` on older
  spawn executor, `vllm::worker-*` / `vllm::engine_core` on multiproc).
  `SKIP_VLLM_PURGE=1` opts out of the vLLM purge when sharing a node
  with other vLLM workloads. See `docs/DEVELOPMENT_LOG.md §12.11` for
  the root-cause analysis.

### Added (2026-04-27 — multi-host session)

- **Multi-host deployment**: join an externally-managed Ray cluster
  without touching code — YAML only.
    * `ClusterCfg` in `reward_service/config.py` (`ray_address`, optional
      `namespace`), parsed from a top-level `cluster:` section.
    * `_init_ray()` in `reward_service/workers/pool.py` branches on the
      address: `None` → local single-host (original behaviour);
      `"auto"` / `"<host>:6379"` → `ray.init(address=..., namespace=...)`.
    * New `scheduling` field on `RewardModelCfg` (`"pack"` default,
      `"spread"` → Ray actor `scheduling_strategy="SPREAD"` hint).
    * `scripts/ray_start.sh` / `ray_stop.sh` / `ray_smoke.sh`
      (+ shared `scripts/_ray_lib.sh`): pdsh-based bootstrap reading
      the `NODE_IP_LIST` env var; IPv6-safe node parsing; per-worker
      `--node-ip-address` interpolation to avoid the `hostname -i`
      loopback footgun in containers; Ray temp-dir defaults to
      `/tmp/ray-$USER` (overridable via `RAY_TMPDIR`) to keep plasma
      store's AF_UNIX socket path under the 107-byte kernel limit
      — see `docs/DEVELOPMENT_LOG.md §12.10`.
    * `configs/service.cluster.example.yaml` — double-host 16 GPU
      reference with `scheduling: spread` on 1-GPU rewards and
      `scheduling: pack` for GenEval2 (TP=2 must stay single-host).
- `install.sh` — added `peft` to the legacy sub-deps (hpsv3 source imports
  it but wheel METADATA does not declare it).
- 13 new unit tests covering `ClusterCfg` parsing, `scheduling` field
  validation, `_init_ray` branches, and `_actor_options` SPREAD
  forwarding.

### Changed (2026-04-27)

- `pyproject.toml` — `transformers>=4.55.2,<4.58` (was `>=4.44`). Lower
  bound matches vLLM 0.11's hard requirement; upper bound avoids the
  `transformers.pytorch_utils` backward-compat alias removals that start
  in 4.58 (which would break the `_compat.py` shims and ImageReward /
  hpsv3). See `docs/DEVELOPMENT_LOG.md §12.2`.
- `reward_service/workers/pool.py::_init_ray` — dropped redundant
  `ignore_reinit_error=True` kwarg now that the `ray.is_initialized()`
  guard already early-returns.
- Repository-internal file references in `docs/` and `.claude/skills/` are
  now all relative to the repo root; YAML external paths (model weight
  mounts etc.) remain absolute.
- `.claude/skills/code-standards/SKILL.md` — added mandatory post-review
  step "同步项目文档": every finished iteration must update
  `docs/DEVELOPMENT_LOG.md` (append iteration chapter) and the "Resume
  入口" section before closing the work out.

### Added

- `reward_service/scorers/_compat.py` — transformers-4.57 API compat shims.
  Re-injects moved symbols into their old locations so legacy reward libs
  (hpsv3, ImageReward) can still import:
    * `transformers.image_utils.VideoInput` ← `transformers.video_utils`
    * `transformers.modeling_utils.apply_chunking_to_forward` ← `pytorch_utils`
    * `transformers.modeling_utils.find_pruneable_heads_and_indices` ← `pytorch_utils`
    * `transformers.modeling_utils.prune_linear_layer` ← `pytorch_utils`
  Shims are independent (each in its own try/except); adding a new moved
  symbol takes one line in the `_shim(...)` call list.
- `hpsv3_scorer.py` and `imagereward.py` now import `_compat` at module top
  so the shims are active before `hpsv3` / `ImageReward` are imported.
- `install.sh` — one-shot installer that works around the `vllm` ↔ `hpsv3`
  transformers pin conflict. Runs `pip install -e ".[vllm,dev]"` then
  explicitly installs 11 non-conflicting sub-deps (ftfy / braceexpand /
  timm / webdataset / clint / diffusers / omegaconf / fire / matplotlib /
  fairscale / openai-clip), then `pip install --no-deps hpsv2 hpsv3
  image-reward` so the three 2023-era reward libs cannot drag transformers
  back to 4.45.2 (vLLM 0.11 needs ≥4.55.2). Legacy installs are attempted
  independently; failures are reported at the end and the matching reward
  should be commented out of `configs/service.yaml`.
- `docs/ARCHITECTURE.md` — full architecture reference: static topology,
  request-flow sequence diagram, four-layer abstraction breakdown,
  isolation semantics, extension points, config-to-runtime mapping.
- YAML-configurable vLLM init parameters for `unified_reward` and `geneval2` scorers.
  New optional `params` fields: `dtype`, `enforce_eager`, `swap_space`,
  `quantization`, `seed`, `max_num_seqs`, `trust_remote_code`,
  `limit_mm_per_prompt`, and a catch-all `extra_llm_kwargs` escape hatch
  (last-writer-wins, lets future vLLM options pass through without code
  changes). See `configs/service.example.yaml`.
- Shared `install_fake_vllm` pytest fixture under `tests/scorers/conftest.py`
  so CPU-only tests can construct vLLM-based scorers with a capturing
  stub and assert what was passed to `vllm.LLM(...)`.
- `DEFAULT_VLLM_MM_LIMIT` constant (`{"image": 1}`) in `scorers/_common.py`
  so vLLM-based scorers share the same fallback.

### Changed

- `build_vllm_llm_kwargs` now treats `extra_llm_kwargs={}` identically to
  `None` (no-op merge) instead of using truthiness (`if extra_llm_kwargs`).
  Same fix applied to `limit_mm_per_prompt` propagation inside the two
  vLLM scorers — `{}` no longer silently collapses to the default.

---

## [0.1.0] — Initial release

### Added

- FastAPI gateway (`reward_service/server.py`) exposing:
  - `POST /score` — batch scoring with per-reward error isolation
  - `GET /health` — async health check (offloaded via `asyncio.to_thread`)
  - `GET /rewards` — list loaded reward names
- Ray worker layer (`reward_service/workers/`) with strict GPU isolation per reward group:
  - `ScorerActor` — thin Ray actor wrapping one scorer
  - `WorkerGroup` — N replicas + round-robin dispatch
  - `WorkerPool` — group lifecycle + Ray init, exposes `has_reward()` / `dispatch()` / `health()` / `shutdown()`
- Scorer abstraction (`reward_service/scorers/`):
  - `BaseScorer` contract with `score(items) -> list[dict]`
  - Registry with lazy optional-dep import and warning on failure
  - 7 concrete scorers: `clip` / `pickscore` / `imagereward` / `hpsv2` / `hpsv3` / `unified_reward` (vLLM) / `geneval2` (vLLM)
  - Shared helpers in `_common.py`: `resolve_dtype` / `resolve_model_path` / `split_last_turn` / `image_to_data_url`
- Python client SDK (`reward_service/client.py`) with default JPEG q=95 encoding
- YAML-driven configuration (`reward_service/config.py`) with validation:
  - Rejects `num_replicas < 1` and `num_gpus < 0` up front
  - Duplicate reward names rejected
- Response schema with error isolation:
  - `ScoreResponse.results[i][reward_name][sub_metric] -> float`
  - `ScoreResponse.errors[i][reward_name] -> str` (for failed rewards)
- CLI entry point: `python -m reward_service --config configs/service.yaml`
- Example config (`configs/service.example.yaml`) listing all 7 rewards with per-reward GPU/replica notes
- 38 CPU-only unit tests + 8 GPU-gated smoke tests
- Development documentation under `docs/DEVELOPMENT_LOG.md`

### Technical notes

- Python ≥ 3.12
- UnifiedReward / GenEval2 images are inlined into vLLM chat messages as `data:image/jpeg;base64,...` URLs (vLLM's `LLM.chat()` does not accept `multi_modal_data`)
- Each Ray actor owns its GPUs exclusively via `num_gpus` option; single-GPU reward groups do not share hardware with each other
- UnifiedReward output is parsed with a tolerant regex; failed sub-metrics degrade to NaN with a warning rather than raising
- GenEval2 supports Soft-TIFA scoring (per-prompt VQA question lists from JSONL dataset) when `dataset_path` is configured; falls back to degenerate single-question template otherwise

### Known limitations

- When one request requires N reward models, its PIL image is pickled into Ray's object store N times. Acceptable for small batches; revisit with `ray.put(item)` deduplication if request throughput becomes a bottleneck.
- Optional-dep breakage: `hpsv2` / `image-reward` are 2023 packages; if installation fails under Python 3.12, fall back to 3.11.
- Default client `timeout=60.0s` may be insufficient for large vLLM batches; override when needed.

### Not included (deferred by YAGNI)

- Rate limiting / auth headers
- Prometheus `/metrics` endpoint
- Load-aware dispatch (current impl is round-robin)
- Continuous / dynamic batching
