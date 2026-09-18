# SearchGen Interleaved Agent SFT Dataset

Data preparation for **interleaved image/text agent-trajectory SFT**: each SearchGen reasoner
trace is linearized into one conversation whose history carries the *retrieved candidate images*
mid-conversation, then expanded into one training record per supervised assistant turn.

- Recipes: [`examples/sft/qwen_vl_agent_sft.yaml`](../../examples/sft/qwen_vl_agent_sft.yaml),
  [`examples/sft/bagel_agent_sft.yaml`](../../examples/sft/bagel_agent_sft.yaml)
- Cooking script: [`prepare_sft.py`](prepare_sft.py)
- Manifest contract: [`unirl/data/sft.py`](../../unirl/data/sft.py) (AR agent rows)

Generated manifests and extracted images are local artifacts and must not be committed. The
repository `.gitignore` excludes `processed/`.

## Source

- **SearchGen-20K** — [`JasperHaozhe/SearchGen-20K`](https://huggingface.co/datasets/JasperHaozhe/SearchGen-20K)
  (`apache-2.0`): the reasoner traces (prompt analysis → image search → reference selection →
  prompt refinement).
- **SearchGen-Corpus-1M** — [`JasperHaozhe/SearchGen-Corpus-1M`](https://huggingface.co/datasets/JasperHaozhe/SearchGen-Corpus-1M):
  the search corpus the traces cite — the candidate images live here, in TAR shards.
- Paper: [arXiv:2607.05382](https://arxiv.org/abs/2607.05382) — *Search Beyond What Can Be
  Taught: Evolving the Knowledge Boundary in Agentic Visual Generation*;
  code: [HaozheH3/SearchGen](https://github.com/HaozheH3/SearchGen).

> **License caveat.** The corpus is **gated** and released under a
> `searchgen-corpus-non-commercial-research-license` — non-commercial research only, no
> redistribution or production deployment. The `apache-2.0` label on SearchGen-20K covers the
> traces, not the corpus imagery. Accept the terms on the Hub before downloading, and keep the
> extracted images local.

## Download

```bash
ROOT=/path/to/Datasets/searchgen

hf download JasperHaozhe/SearchGen-20K        --repo-type dataset --local-dir "$ROOT/searchgen-20k"
hf download JasperHaozhe/SearchGen-Corpus-1M  --repo-type dataset --local-dir "$ROOT/searchgen-corpus-1m"
```

The converter reads exactly these paths under `--searchgen-root`:

```
searchgen-20k/metadata/train_metadata.jsonl     # row_id → user_prompt
searchgen-20k/traces/trace_artifacts.jsonl      # the reasoner traces
searchgen-corpus-1m/metadata/search.sqlite      # query_id → ranked candidate entries
searchgen-corpus-1m/data/cached-images/         # shards.jsonl index + TAR shards
```

The full corpus is large; only the `cached-images` shards actually cited by the traces you cook
are read, so a partial mirror still works — missing shards are reported and their traces
dropped rather than crashing the run.

## Cook

```bash
python datasets/searchgen/prepare_sft.py \
  --searchgen-root "$ROOT" \
  --out-dir datasets/searchgen/processed \
  --candidates-per-query 4
```

Useful flags: `--max-traces N` caps rendered traces (`0` = all, the default), `--max-image-px`
sets the long-side cap for extracted images (default 448), `--extract-workers` sets parallel
shard readers, `--max-target-chars` drops records with overlong target turns, and
`--val-fraction` / `--seed` control the split.

Outputs `train.jsonl`, `val.jsonl` and `images/` under `--out-dir`. Extraction is resumable:
images already present in `images/` are not re-extracted (so changing `--max-image-px` needs a
fresh `--out-dir`).

## Format

One record per supervised assistant turn. History messages may carry OpenAI-style content-part
lists; image parts hold manifest-relative URIs, and only the final assistant turn is supervised:

```json
{
  "sample_id": "<trace_id>:s2",
  "messages": [
    {"role": "system", "content": "You are an image-generation planning agent. ..."},
    {"role": "user", "content": "<user_prompt>"},
    {"role": "assistant", "content": "<analysis>\n\nSearch queries:\n1. ..."},
    {"role": "user", "content": [
      {"type": "text",  "text": "Search results:"},
      {"type": "text",  "text": "Query 1: ..."},
      {"type": "text",  "text": "Image 1: <title>"},
      {"type": "image", "image": "images/<asset>.jpg"}
    ]},
    {"role": "assistant", "content": "Selected Image 3 as Reference Image 1: ..."}
  ],
  "metadata": {"row_id": "...", "trace_id": "...", "reasoner_id": "...", "stage": "s2_selection"}
}
```

Each trace yields three records — `s1_analysis`, `s2_selection`, `s3_refine`.

## Why cook it this way

- **One record per assistant turn.** Matches the `prepare_sft_agent` convention: the history is
  the prompt, the final assistant turn is the CE target. The s2 target is the only one that has
  actually *seen* the candidate images, so splitting the trace this way is what makes the
  interleaved images load-bearing rather than decorative.
- **Every record is validated at prep time.** The script calls
  `unirl.data.sft.normalize_supervised_example` on each record, so a malformed row fails here —
  at the dataset boundary — instead of mid-training. This is the only dependency the converter
  has on UniRL; install the package (`pip install -e .`) before cooking.
- **Image-only candidates.** The candidate SQL filters `search_type = 'image'` and
  `download_status = 'downloaded'`. Web/HTML results have no pixels to interleave, and including
  them silently produced text-only "candidate" turns.
- **Reference numbering comes from `s4_generation_manifest.ordered_references`.** That list is
  the numbering space the release actually cites: on all 50,808 traces passing the converter's
  resolvedness guard, `s3_refined_prompt.selected_reference_indices` is exactly
  `1..len(ordered_references)`, and `used_in_refinement` — where set at all (it is unset on
  ~25% of those traces) — marks exactly those positions. `s2_reference_selection` rows are not
  usable for numbering: they may lack `query_id`/`entry_id` (unresolved placeholders and
  partially-resolved rows), and their `query_index` indexes the **image-search-query
  subsequence**, not the raw mixed web+image query list, so naive mixed-list indexing disagrees
  on ~13% of rows. Filtering s2 rows and renumbering the survivors silently shifted the
  `Reference Image N` citations on ~2.6% of rendered traces.
- **Selections are re-indexed against OUR presentation.** The release does not preserve the
  original candidate ordering the reasoner saw, but it does preserve the reference *identity*.
  Candidates are presented as `Image <global position>` and kept references as
  `Reference Image <ordered_references position>`; the s2 target welds the two numbering spaces
  explicitly (`Selected Image 3 as Reference Image 1`) so the s3 target's `Reference Image N`
  citations stay resolvable. A trace with an unresolved or reordered reference entry, or whose
  reference cannot be located among its shown candidates, is dropped rather than renumbered.
- **448 px long side (~256 vision tokens/image).** A 3–5 query trace with 3–4 candidates each
  then fits `max_prompt_length: 8192`. Overlong histories **raise** at the chat-template stage —
  truncating would desync the vision spans — so re-cook with fewer/smaller candidates instead.
- **Generated images and judge scores are deliberately untouched.** That part of the release is
  GRPO group-shaped preference data, not SFT.

## Train

```bash
ENTRY=train_sft \
SFT_DATA=datasets/searchgen/processed/train.jsonl \
SFT_EVAL_DATA=datasets/searchgen/processed/val.jsonl \
bash examples/run_experiment_single_node.sh sft/qwen_vl_agent_sft
```

Swap `sft/qwen_vl_agent_sft` for `sft/bagel_agent_sft` to train the BAGEL und/AR experts on the
same manifests. Neither backbone has a tool template: manifests carrying `tools`, `role="tool"`
turns, or tool-call targets are rejected by `BagelChatTemplateStage` and
`QwenVLChatTemplateStage` rather than rendered lossily (the stock Qwen2.5-VL chat template
references neither `tools` nor `tool_calls`, so they would otherwise vanish silently).
