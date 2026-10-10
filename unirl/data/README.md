# Runtime data

> **Where it fits:** data selection before RL rollout or SFT target construction.
> In: local prompt files or supervised manifests. Out: input-only `Sample` requests
> for RL, or normalized record batches for SFT.
> Full map: [`../README.md`](../README.md).

*Readers normalize raw text, media references, and metadata; RL collation decodes
local condition images and videos. Downstream model and rollout consumers own
tokenization and encoding.*

## What it is

This package owns runtime parsing, record normalization, and batch iteration.
Offline download/conversion belongs to [`datasets/`](../../datasets/README.md).
RL sources build input `Sample`/`Part` trees; SFT sources return dictionaries for
training-side [track builders](../train/sft/README.md). Reward scoring, optimizer
updates, and standalone [benchmarks](../../benchmarks/README.md) live elsewhere.

## Why it exists

The same raw prompt can feed different rollout engines and model families
without storing model-specific embeddings. Reward labels travel as metadata;
SFT records carry explicit text or media targets for the track builder.

## How it works

| Path | Reader and source | Output and next consumer |
| --- | --- | --- |
| File-backed RL | [`TextPromptDataset`](datasets.py) → [`MultimodalRLDataSource`](data_source.py) | Input-only [`Sample`](../types/sample.py) → domain [trainer](../trainer/README.md) → rollout |
| Multi-domain RL | `TextPromptDataset` per domain → [`MultiDomainRLDataSource`](data_source.py) | Text-only `Sample` batches with domain metadata → trainer |
| Synthetic smoke run | [`DefaultDataSource`](data_source.py) | Cyclic built-in prompt `Sample` batches → trainer |
| SFT | [`SupervisedDataset` → `SupervisedDataSource`](sft.py) | Normalized record list → `SFTTrainer` → supervised track builder |

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

Other objectives use typed media targets. The receiving track builder loads
media and constructs training targets on the workers. Agent SFT uses `messages`
instead of prompt/response fields; follow the
[agent contract](../train/sft/README.md#agent-chat-stage-contract).
See [manifest converters](../../datasets/sft_manifests/README.md) and the
[Qwen3 SFT recipe](../../examples/sft/qwen3_sft.yaml) for `SFT_DATA`/`SFT_EVAL_DATA`.

### Media references

- **Media uses typed references.** Put `modality`, `role`, and `uri` in each
  `media`/`media_refs` entry. Readers check these fields and resolve relative
  paths against the manifest directory; they do not download remote files. See
  [`MediaRef`](../types/media.py).
- **RL collation decodes condition media on the driver.** `(image, condition)` and
  `(video, condition)` refs become `Image`/`Video` primitives when the batch is
  built, so their URIs must be local or shared-storage paths; remote URIs fail at
  collation. `role="prompt"` refs (image, video, audio) pass through as `MediaRefs`
  for the rollout engine. Other `(modality, role)` pairs are rejected; see
  `_SUPPORTED_MEDIA_REF_ROLES` in [`data_source.py`](data_source.py).
- **Condition media must be uniform within a batch.** Each prompt has at most one
  condition image or video, and either every prompt in a batch has one or none
  does. A batch cannot mix `(video, condition)` and `(video, prompt)` refs.
  `MultiDomainRLDataSource` rejects media entirely. The selected
  [recipe](../../examples/README.md) must also support the media it receives.

### Batch iteration

- **RL training batches contain prompts before generation fanout.** The file-backed
  source uses `prompts_per_rollout`, drops the final incomplete batch, and cycles.
  Its evaluation iterator retains the tail; unset `eval_data_path` falls back to
  training prompts, which is not a held-out split. See
  [`MultimodalRLDataSource`](data_source.py) for selection rules.
- **SFT batches can cross epochs.** The source fills the requested batch size.
  Evaluation reads only `eval_manifest_path`; when it is unset, `SFTTrainer` logs a
  warning and disables validation. Iteration and resume contracts are defined in
  [`SupervisedDataSource`](sft.py).

**Extending it:** subclass `PromptExampleDataset` and override
`MultimodalRLDataSource._build_dataset` for another prompt storage layout.
New media or supervised targets also need a matching model-side consumer or
track builder; changing the reader alone does not add model support.

## Gotchas

- **The RL reader skips malformed rows; the SFT reader fails.** `TextPromptDataset`
  logs a warning and drops rows with invalid JSON, no `prompt`/`caption`, or a
  non-dict `metadata`; rows that declare media raise instead. A misnamed prompt field such as
  `question` therefore loses data silently. `SupervisedDataset` raises on the first
  invalid row.
- **Prompt manifests carry raw data.** Readers reject top-level fields listed in
  [`_LEGACY_EMBEDDING_FIELDS`](datasets.py), such as `prompt_embeds` and `text_ids`;
  leave tokenization and encoding to downstream consumers.
- **Reader acceptance does not establish model compatibility.** A normalized SFT
  record still needs targets supported by the receiving track builder.
- **Reward metadata is not a model-input channel.** Put model media in
  `media`/`media_refs`; the RL source rejects `metadata['_media_refs']`.
