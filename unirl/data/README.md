# Runtime data

> **Where it fits:** turns the data files a recipe points at into training
> batches — prompt batches for RL rollout, record batches for SFT.
> Full map: [`../README.md`](../README.md).

This package runs at training time. Downloading and converting public datasets
into these files is a separate offline step in
[`datasets/`](../../datasets/README.md), and standalone evaluation lives in
[`benchmarks/`](../../benchmarks/README.md).

There are two input formats: **prompt files** for RL and **supervised manifests**
for SFT. Both hold raw text and media paths only. Tokenization and encoding
happen later, in the model and rollout code, so one file works across models.

## RL prompt files

An RL recipe points `data_source.args.run.data_path` at a `.txt` (one prompt per
line), `.jsonl`, or `.json` file, and may set `eval_data_path` as well. A JSONL
row with a reward label looks like this:

```json
{"prompt_id":"q0","prompt":"What is 2 + 2?","metadata":{"answer":"4"}}
```

Labels go in `metadata`, and the reward scorer decides which keys it reads (the
math-verify scorer reads `answer`, for example). If a row has no `metadata` field, its extra
top-level keys become metadata, so `{"prompt": "...", "answer": "4"}` also works.
If it does have one, extra top-level keys are dropped.

Each training batch holds `prompts_per_rollout` prompts; rollout then generates
`samples_per_prompt` outputs for each. The last incomplete batch is dropped and
the file starts over. Evaluation reads `eval_data_path` in order and keeps the
last partial batch. If `eval_data_path` is unset, evaluation reuses the training
prompts, so it is not a held-out score.

[DAPO-Math preparation](../../datasets/dapo_math/README.md) builds such a file for
the [Qwen3 DRPO recipe](../../examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml).

## SFT manifests

An SFT recipe points `data_source.train_manifest_path` at a `.jsonl` file or a
`.json` list, and may set `eval_manifest_path`. Text SFT puts the target in
`response`:

```json
{"sample_id":"s0","prompt":"What is 2 + 2?","response":"4"}
```

Image or video SFT puts the target in a single media entry with role `target`:

```json
{"sample_id":"s1","prompt":"A red cube","media":[{"modality":"image","role":"target","uri":"images/s1.png"}]}
```

Agent SFT uses a `messages` list instead; see the
[agent contract](../train/sft/README.md#agent-chat-stage-contract).

Batches always have the requested size and may span two epochs. Without
`eval_manifest_path`, the trainer logs a warning and skips validation. The
[manifest converters](../../datasets/sft_manifests/README.md) produce these
files, and the [Qwen3 SFT recipe](../../examples/sft/qwen3_sft.yaml) reads them
through `SFT_DATA` and `SFT_EVAL_DATA`.

## Media

Both formats list media in a `media` (or `media_refs`) field, as in the SFT
example above. Each entry has exactly `modality`, `role`, and `uri`. A relative
`uri` is resolved against the folder that contains the data file, and nothing is
downloaded. The `role` decides what happens to the file:

| `role` | Modalities | What happens |
| --- | --- | --- |
| `condition` | image, video | RL loads it into pixels when the batch is built, so it must be a local or shared-storage path. In text SFT, one condition image per record becomes the VLM prompt image. |
| `prompt` | image, video, audio | RL passes the path through to the rollout engine. |
| `target` | image, video | The SFT track builder loads it on the training workers. |

RL rejects any other combination. Within one RL batch, either every prompt has a
condition image or none does, and the same holds for condition videos; each
prompt has at most one of each, and condition and prompt videos cannot be mixed.
The multi-domain RL source accepts no media at all.

## Gotchas

- **Bad RL rows disappear quietly.** The RL reader logs a warning and skips rows
  with invalid JSON, a missing `prompt`/`caption`, or a non-dict `metadata`, so a
  typo like `question` instead of `prompt` loses rows without failing. (A bad
  row that has media raises instead.) The SFT reader fails on the first bad row.
- **No precomputed embeddings.** Both readers reject fields such as
  `prompt_embeds`, `pooled_prompt_embeds`, and `text_ids`; the full list is in
  [`datasets.py`](datasets.py).
- **Media does not go in `metadata`.** RL rejects `metadata["_media_refs"]`; use
  the `media` field.

## Code

`TextPromptDataset` ([`datasets.py`](datasets.py)) reads RL prompt files and
`MultimodalRLDataSource` ([`data_source.py`](data_source.py)) batches them;
`MultiDomainRLDataSource` in the same file takes several text-only prompt files
and serves one domain per batch in turn. `SupervisedDataset` and
`SupervisedDataSource` ([`sft.py`](sft.py)) read and batch SFT manifests and hold
the resume state.

To read another RL file layout, subclass `MultimodalRLDataSource` and return your
own `PromptExampleDataset` from `_build_dataset`. A new media type or target also
needs support on the model side; changing the reader alone is not enough.
