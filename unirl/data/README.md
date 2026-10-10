# Runtime data

> **Where it fits:** data selection before RL rollout or SFT target construction.
> In: local prompt files (RL) or supervised manifests (SFT). Out: input-only
> `Sample` requests for RL, or normalized record batches for SFT.
> Full map: [`../README.md`](../README.md).

*Readers normalize raw text, media references, and metadata; RL collation decodes
local condition images and videos. Downstream model and rollout consumers own
tokenization and encoding.*

## What it is

This package owns runtime parsing, record normalization, and batch iteration.
Offline download/conversion belongs to [`datasets/`](../../datasets/README.md).
Reward scoring, optimizer updates, and standalone
[benchmarks](../../benchmarks/README.md) live elsewhere.

## Why it exists

The same raw prompt can feed different rollout engines and model families
without storing model-specific embeddings. Reward labels travel as metadata;
SFT records carry explicit text or media targets.

## How it works

| Path | Reader and source | Output and next consumer |
| --- | --- | --- |
| File-backed RL | [`TextPromptDataset`](datasets.py) → [`MultimodalRLDataSource`](data_source.py) | Input-only [`Sample`](../types/sample.py) → domain [trainer](../trainer/README.md) → rollout |
| Multi-domain RL | `TextPromptDataset` per domain → [`MultiDomainRLDataSource`](data_source.py) | Text-only `Sample`, one domain per batch in round-robin order, stamped with `metadata["domain"]` → trainer |
| SFT | [`SupervisedDataset` → `SupervisedDataSource`](sft.py) | Normalized record list → `SFTTrainer` → [track builder](../train/sft/README.md), which turns records into training targets |
| Hardcoded prompts | [`DefaultDataSource`](data_source.py) | Cycles eight built-in prompts; no checked-in recipe uses it |

### RL prompt files

`TextPromptDataset` reads `.txt`, `.jsonl`, and `.json`; see
[`datasets.py`](datasets.py) for accepted layouts. A labeled JSONL row is:

```json
{"prompt_id":"q0","prompt":"What is 2 + 2?","metadata":{"answer":"4"}}
```

`metadata` carries dataset/reward information; the receiving scorer defines
label keys such as `answer`. Without a `metadata` field, unrecognized top-level
keys become metadata, so `{"prompt": ..., "answer": "4"}` is equivalent. With a
`metadata` field, other top-level keys are dropped rather than merged.
See [DAPO-Math preparation](../../datasets/dapo_math/README.md) and the
[Qwen3 DRPO recipe](../../examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml)
for matching local data and `DATA_PATH` configuration.

### Supervised manifests

`SupervisedDataset` reads JSONL objects or a JSON list of objects. A text SFT row
carries the target in `response`:

```json
{"sample_id":"s0","prompt":"What is 2 + 2?","response":"4"}
```

An image or video SFT row carries exactly one media entry with `role` `target`;
the track builder loads it on the workers:

```json
{"sample_id":"s1","prompt":"A red cube","media":[{"modality":"image","role":"target","uri":"images/s1.png"}]}
```

Agent SFT uses `messages` instead of prompt/response fields; follow the
[agent contract](../train/sft/README.md#agent-chat-stage-contract).
See [manifest converters](../../datasets/sft_manifests/README.md) and the
[Qwen3 SFT recipe](../../examples/sft/qwen3_sft.yaml) for `SFT_DATA`/`SFT_EVAL_DATA`.

### Media references

- **Media uses typed references.** Each `media`/`media_refs` entry has exactly
  `modality`, `role`, and `uri`. Relative paths resolve against the directory
  that contains the prompt file or manifest; readers do not download remote
  files. See [`MediaRef`](../types/media.py).
- **RL collation decodes condition media on the driver.** `(image, condition)` and
  `(video, condition)` refs become `Image`/`Video` primitives when the batch is
  built, so their URIs must be local or shared-storage paths; remote URIs fail
  there. `(image, prompt)`, `(video, prompt)`, and `(audio, prompt)` refs pass
  through as `MediaRefs` for the rollout engine. Every other pair is rejected,
  including the SFT `target` role.
- **Condition media must be uniform within a batch.** Each prompt has at most one
  condition image and at most one condition video. Within a batch, condition
  images are all-or-nothing, and so are condition videos. A batch cannot mix
  `(video, condition)` and `(video, prompt)` refs. `MultiDomainRLDataSource`
  rejects media entirely.

### Batch iteration

- **RL batches hold prompts, not rollouts.** The file-backed source yields
  `prompts_per_rollout` prompts per batch; rollout later repeats each prompt
  `samples_per_prompt` times. It drops the final incomplete batch and cycles.
  The evaluation iterator keeps the tail; when `eval_data_path` is unset,
  evaluation reuses the training prompts, which is not a held-out split.
- **SFT batches can cross epochs.** The source fills the requested batch size.
  Evaluation reads only `eval_manifest_path`; when it is unset, `SFTTrainer` logs a
  warning and disables validation. Iteration and resume contracts are defined in
  [`SupervisedDataSource`](sft.py).

**Extending it:** for another prompt storage layout, subclass
`MultimodalRLDataSource` and return your own `PromptExampleDataset` from
`_build_dataset` (it returns `TextPromptDataset` today). New media or supervised
targets also need a matching model-side consumer or track builder; changing the
reader alone does not add model support.

## Gotchas

- **The RL reader skips malformed rows; the SFT reader fails.** `TextPromptDataset`
  warns and drops lines with invalid JSON and rows with no `prompt`/`caption` or a
  non-dict `metadata`; such a row raises only if it also has `media`/`media_refs`.
  A misnamed prompt field such as `question` therefore loses rows silently.
  `SupervisedDataset` raises on the first invalid row.
- **Inputs carry raw data, not embeddings.** Both readers reject precomputed
  fields such as `prompt_embeds`, `pooled_prompt_embeds`, `text_ids`, and their
  path variants like `prompt_embed_path` (full list in [`datasets.py`](datasets.py)).
- **Media does not go in metadata.** Put model media in `media`/`media_refs`; the
  RL source rejects a `_media_refs` key inside `metadata`.
