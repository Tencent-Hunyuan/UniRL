# Reward Service

A unified T2I reward inference service: a FastAPI gateway in front of Ray worker groups. Each reward model owns its own GPU(s) and is resource-isolated from the others.

## Supported reward models

| Name | Framework | Notes |
|---|---|---|
| `clip` | transformers | openai/clip-vit-large-patch14, cosine similarity |
| `pickscore` | transformers | yuvalkirstain/PickScore_v1 |
| `imagereward` | official `image-reward` pip package | ImageReward-v1.0 |
| `hpsv2` | official `hpsv2` pip package | supports v2.0 / v2.1 |
| `hpsv3` | official `hpsv3` pip package | built on Qwen2-VL |
| `unified_reward` | vLLM | UnifiedReward VLM family (model set in YAML); parses the generated text into Alignment / Coherence / Style |
| `geneval2` | vLLM | VQAScore via Qwen3-VL-8B-Instruct (Soft-TIFA multi-question scoring when `dataset_path` is set; falls back to a single-question template otherwise) |
| `ocr` | transformers | GOT-OCR-2.0-hf; reward = edit-distance similarity between the recognized text and the target span in the prompt |
| `wise` | vLLM | A MoE VLM (e.g. Qwen3.5-35B-A3B) used as a generic WISE-rubric reward judge. Default sub-metrics: a derived `wiscore` headline plus Consistency / Realism / Aesthetic Quality (template overridable in YAML) |
| `geneval` | mmdet / mmcv | Compositional GenEval (Mask2Former detection + CLIP color classification). **Disabled by default** — its stack needs Python 3.10 and cannot run under this service's Python-3.13 Ray cluster (see `envs/geneval.txt`) |
| `videoalign` | transformers (vendored `_videoalign/`) | T2V reward: scores generated **videos** along Overall / VQ / MQ / TA. Requires a video (`video_b64` or `video_path`) on the request. **Disabled by default** in the example config |

## Architecture

```
Client ──HTTP──▶ FastAPI Gateway ──Ray actor call──▶ WorkerGroup(reward=X)
                                                     ├── actor_0 (GPU k)
                                                     └── actor_1 (GPU k+1)
```

- Each reward is one `WorkerGroup` that holds a configurable number of Ray actors (replicas).
- Each actor reserves N GPUs exclusively via Ray's `num_gpus=N`; it never shares them with another reward.
- Routing fans each request out by `required_rewards` to the matching group, picking an actor round-robin.

The full architecture document (topology, sequence, abstraction layers, extension points) lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

### Base environment

```bash
# Prerequisite: a Python 3.12 environment is already active, with torch + nccl pre-installed
./install.sh    # installs the base deps: ray, fastapi, uvicorn, pillow, etc.
```

Per-scorer dependencies (transformers, vllm, hpsv2, ...) are **not** installed here — they are declared in `envs/*.txt` and pip-installed by Ray's `runtime_env` into an isolated virtualenv the first time each actor starts.

### Caller side (HTTP only, no server)

```bash
pip install .    # installs only requests + Pillow
```

### Dependency-isolation architecture

Each scorer has its own `envs/<scorer>.txt` requirements file, referenced from YAML via the `runtime_env` field:

```yaml
rewards:
  - name: imagereward
    scorer: imagereward
    runtime_env: envs/imagereward.txt   # transformers==4.45.2
    ...
  - name: unified_reward
    scorer: unified_reward
    runtime_env: envs/unified_reward.txt  # vllm==0.11.0 (transformers==4.57.0)
    ...
```

This lets imagereward and vllm use different transformers versions without conflicting.

> On first launch Ray pip-installs each venv (which can take a few minutes). Subsequent launches reuse the cache and start in seconds.

## Launch

```bash
cp configs/service.example.yaml configs/service.yaml
# Edit configs/service.yaml: enable the rewards you want, fill in weights_path, tune num_gpus / num_replicas
python -m reward_service --config configs/service.yaml
```

> On first launch Ray pip-installs each scorer's venv (which can take a few minutes). Subsequent launches reuse the cache and start in seconds.

(The console-script entry point `unirl-reward-service --config configs/service.yaml` is equivalent to `python -m reward_service`.)

### Multi-host deployment

```bash
export NODE_IP_LIST="ip1:8 ip2:8"      # the first entry is the head
bash scripts/ray_start.sh              # brings up the Ray cluster over pdsh
python -m reward_service --config configs/service.cluster.example.yaml
# Tear down:
bash scripts/ray_stop.sh
```

Going multi-host only requires adding a `cluster.ray_address` field to the YAML; see the comments in `configs/service.cluster.example.yaml`.

## Calling the service

### Python SDK

```python
from PIL import Image
from reward_service.client import RewardClient, RewardRequest

client = RewardClient("http://localhost:8080")  # use the server's IP when calling across machines
scores = client.score([
    RewardRequest(history=[("a cute dog", Image.open("dog.png"))],
                  required_rewards=["hpsv2", "clip"]),
    RewardRequest(history=[("a cute cat", Image.open("cat.png"))],
                  required_rewards=["hpsv2", "pickscore"]),
])
# scores -> [{"hpsv2": {"hpsv2": 0.27}, "clip": {"clip": 0.31}}, ...]
```

### Batch calls

`client.score([...])` accepts a request list of any length. The server groups the N requests that need the same reward and scores them together in a single actor call.

```python
# One prompt × many candidate images (the most common RLHF pattern)
prompt = "a cute dog running in the park"
candidates = [Image.open(f"cand_{i}.png") for i in range(8)]
scores = client.score([
    RewardRequest(history=[(prompt, img)], required_rewards=["clip", "hpsv2", "pickscore"])
    for img in candidates
])
best = max(range(8), key=lambda i: scores[i]["hpsv2"]["hpsv2"])
```

Concurrency is governed by each reward's `max_concurrency × num_replicas` in `configs/service.example.yaml`. `server.score_timeout_s` (default 120s) is applied per reward independently; on timeout the failure is written to `response.errors[i][reward]` without blocking the other rewards in the same batch.

### Cross-machine calls

Swap the URL for the server's IP; the rest of the code is unchanged:

```bash
python3 scripts/remote_client_example.py --url http://10.1.2.3:8080
# Or, without installing the SDK, using plain requests + Pillow:
python3 scripts/remote_client_zero_deps.py --url http://10.1.2.3:8080 --image cand.jpg
```

### Without the SDK: hand-written HTTP

```python
import base64, io, requests
from PIL import Image

def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")

payload = {"requests": [
    {"history": [{"text": "a cute dog", "image_b64": _b64(Image.open("dog.png"))}],
     "required_rewards": ["clip", "hpsv2"]},
]}
resp = requests.post("http://10.1.2.3:8080/score", json=payload, timeout=120)
resp.raise_for_status()
body = resp.json()
# body["results"][i][reward][sub_metric] -> float
# body["errors"][i][reward] -> str (present only when that reward failed)
```

### API reference

POST `/score` body:

| Field | Type | Description |
|---|---|---|
| `requests[i].history[j].text` | str | prompt text |
| `requests[i].history[j].image_b64` | str \| null | base64 image (any PIL-readable format; the server calls `convert("RGB")`) |
| `requests[i].history[j].video_b64` | str \| null | base64 video bytes (e.g. mp4); decoded to a tempfile server-side. For T2V rewards such as `videoalign` |
| `requests[i].history[j].video_path` | str \| null | absolute path to a video the server can read directly (shared-FS deployments); mutually exclusive with `video_b64` |
| `requests[i].required_rewards` | list[str] | must all appear in the `GET /rewards` list |
| `requests[i].metadata` | dict \| null | optional; passed through unchanged |

Each turn must carry at least one media field (`image_b64`, `video_b64`, or `video_path`); text-only turns are rejected.

Response:

```json
{
  "results": [ { "clip": {"clip": 0.31}, "hpsv2": {"hpsv2": 0.27} } ],
  "errors":  [ { } ]
}
```

`results[i]` and `errors[i]` correspond one-to-one with `requests[i]`. When a reward fails, its score is absent and an error string (the `repr()` of the exception, including the exception class and message) is written to `errors[i][reward]`.

Other endpoints:

- `GET /health` → `{"status": "ok", "rewards": {name: [replica_states...]}}`
- `GET /rewards` → `{"rewards": [...]}`

## Tests

```bash
pytest -m "not gpu and not slow and not integration"   # CPU-only unit tests
pytest tests/integration/ -m integration -v            # venv-install integration tests (need Ray + network)
pytest                                                 # full suite (needs GPU)
```

## Venv check

Verify each scorer's venv install state and isolation:

```bash
python scripts/check_venvs.py --standalone
```

## Load testing

Once the service is up, use `scripts/bench_concurrent.py` to apply controlled concurrency and measure single-request latency distribution and overall throughput.

```bash
# Single point: 1000 requests, 200 concurrent, clip only
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --concurrency 200 --total 1000 --rewards clip

# Sweep: the same 1000 requests, run once each at 100 / 500 / 1000 / 2000 concurrency
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --sweep 100 500 1000 2000 --total 1000 --rewards clip

# Per-reward comparison: run each of N rewards in its own round, then print a side-by-side table.
# When a single /score asks for N rewards at once, latency is pulled to the slowest one (HTTP
# returns once), so use this mode to see each reward's standalone latency.
python3 scripts/bench_concurrent.py \
    --url http://10.1.2.3:8080 \
    --concurrency 200 --total 500 \
    --rewards clip,hpsv2,hpsv3 --per-reward-isolated
```

The output reports each request's min / mean / max latency plus p50/p90/p95/p99, throughput, transport errors, and server-side per-reward failure counts. The sweep and per-reward modes end with a side-by-side comparison table.

## Design conventions

- `history` is a `list[(text, image)]`; T2I scorers look only at the last pair.
- A reward returns `dict[str, float]` (sub-metric name → score), so multiple sub-metrics are supported.
- Resource isolation: a GPU belongs to exactly one reward group at any moment.
- Error isolation: one reward's failure does not affect the other rewards in the same batch.
- Dependency isolation: each scorer's pip deps live in a separate venv, so different transformers versions never conflict.

## Known limitations

- **Slow first launch**: Ray must pip-install each venv (vllm is especially slow, ~10 minutes). Subsequent launches reuse the cache.
- **Ray PIL serialization**: when one request asks for N rewards, the PIL image is pickled N times. For large batches you can switch to `ray.put` + ObjectRef.
- **GenEval2 Soft-TIFA**: requires `dataset_path`; when a prompt has no match it falls back to the single-question template and logs a warning.
- **UnifiedReward parsing**: relies on a regex to extract scores; if the model's output drifts it degrades to NaN and logs a warning.
