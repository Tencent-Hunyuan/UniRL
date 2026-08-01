# Video-R1-260k Video MCQA RL Dataset

Video-only multiple-choice QA prompts for RL training of Qwen3-Omni Thinker (no audio
conditioning). Each sample is one video plus a 4-way A/B/C/D question answered in
`<answer>X</answer>` form.

Used by:

- `examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8.yaml`
- `examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4.yaml`

## Source

- **Video-R1-data**: [Video-R1/Video-R1-data](https://huggingface.co/datasets/Video-R1/Video-R1-data)
  - Code: [tulerfeng/Video-R1](https://github.com/tulerfeng/Video-R1)
  - Paper: [arXiv:2503.21776](https://arxiv.org/abs/2503.21776) — *Video-R1: Reinforcing Video Reasoning in MLLMs* (NeurIPS 2025)
  - Card license: `apache-2.0` — see the caveat below.

The repo ships two annotation files. `Video-R1-260k.json` (263,071 rows) is the RL set and the
only one this converter reads; `Video-R1-COT-165k.json` (165,575 rows) is a CoT-annotated
subset for SFT cold start, not a disjoint split.

`Video-R1-260k.json` mixes images and video: 146,823 image rows and 116,248 video rows, across
five `problem_type`s (multiple choice 168,769; free-form 38,722; numerical 34,354; OCR 15,886;
regression 5,340). Media lives in per-source folders as independent multi-part zips:

| Folder | Modality | Rows | Zip parts | Size |
| --- | --- | ---: | ---: | ---: |
| LLaVA-Video-178K | video | 82,676 | 38 | 196.9 GB |
| STAR | video | 11,455 | 4 | 16.4 GB |
| CLEVRER | video | 8,220 | 1 | 0.5 GB |
| NeXT-QA | video | 7,549 | 4 | 18.0 GB |
| PerceptionTest | video | 6,348 | 6 | 28.3 GB |
| Knowledge / Math / Chart / Spatial / OCR / General | image | 146,823 | 12 | 48.9 GB |

> **License caveat.** The `apache-2.0` label realistically covers the Video-R1 team's own
> curation and annotations. The underlying media is aggregated from CLEVRER, STAR, NeXT-QA,
> PerceptionTest, LLaVA-Video-178K and dozens of upstream sets, several of which are
> research-only. Check the sub-sources individually before any commercial use.

## Download

The full repo is ~310 GB. Fetch only the video sources you intend to train on — the converter
takes a `--sources` allowlist, and `LLaVA-Video-178K` alone is 197 GB.

```bash
ROOT=/path/to/Video-R1-data

# Annotations + the four small/medium video sources (~63 GB).
hf download Video-R1/Video-R1-data --repo-type dataset --local-dir "$ROOT" \
  --include "Video-R1-260k.json" "CLEVRER/*" "STAR/*" "NeXT-QA/*" "PerceptionTest/*"
```

`--repo-type dataset` is mandatory. Use `--dry-run` first to see the footprint. Prefer
`hf download` over `git clone`: it resumes, and it filters.

The `*_part*.zip` files are **independent archives**, not spanned volumes, so each is
extracted on its own and a partial download still yields usable (partial) media:

```bash
for d in CLEVRER STAR NeXT-QA PerceptionTest; do
  for z in "$ROOT/$d"/*_part*.zip; do [ -f "$z" ] && unzip -o -q "$z" -d "$ROOT/$d"; done
done
```

Extract **in place** — `path` fields in the JSON are relative to `$ROOT` and only resolve
against the original directory layout. Budget ~2× the download size, or delete each zip after
extracting it.

Source row schema:

```json
{
  "problem_id": 1,
  "problem": "What happens after the green cube collides with the sphere?",
  "data_type": "video",
  "problem_type": "multiple choice",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "solution": "<answer>D</answer>",
  "path": "./CLEVRER/video_train/video_00042.mp4",
  "data_source": ""
}
```

`solution` is always `<answer>...</answer>` regardless of problem type; `options` is `[]` for
non-multiple-choice rows, and option prefixes are inconsistent across sources (`"A) text"` vs
`"A. text"`).

## Cook

```bash
python datasets/video_r1_260k/convert_video_r1_260k_to_unirl.py \
  --data-root "$ROOT" \
  --out-dir datasets/video_r1_260k \
  --sources CLEVRER,STAR,NeXT-QA,PerceptionTest \
  --max-total 20000 --val-count 200
```

Useful flags: `--max-per-source` caps each folder (keeps a big source from dominating),
`--seed` fixes the shuffle so the train/val split is reproducible, `--keep-missing` emits rows
whose MP4 is not on disk yet. The script prints kept/missing counts per source; if everything
lands under `missing:`, the zips are not extracted yet.

## Format

Each output line:

```json
{
  "prompt": "What happens after the green cube collides with the sphere?\nA) ...\nB) ...\nC) ...\nD) ...\nFirst reason step by step about which option is correct. Then output the final answer letter (A, B, C, or D) on its own in the exact format:\n<answer>X</answer>",
  "prompt_id": "video_r1_260k:CLEVRER:42",
  "media_refs": [{"modality": "video", "role": "prompt", "uri": "/abs/path/video_00042.mp4"}],
  "metadata": {"answer": "D"}
}
```

## Why cook it this way

- **Video multiple-choice rows only.** The converter keeps `data_type == "video"` and
  `problem_type == "multiple choice"`, dropping ~57% of the file. Both filters are hard
  requirements, not preferences:
  - `MultimodalRLDataSource` accepts `(image, condition)`, `(video, condition)` and
    `(video, prompt)` and raises `NotImplementedError` on anything else. Image rows would need
    `(image, condition)`, which is the diffusion I2V path, not an AR prompt image — and a batch
    may not mix condition and prompt media anyway.
  - `MCExactMatchRewardScorer` only compares A–D letters. Free-form, numerical, OCR and
    regression rows would score 0.0 forever under it.
- **`role: "prompt"`, not `"condition"`.** `(video, condition)` decodes the clip into a frame
  tensor for diffusion V2V; `(video, prompt)` hands the URI to the Qwen3-Omni conversation
  builder, which is what an AR prompt video needs.
- **The `<answer>X</answer>` instruction is appended to every prompt.** The 1x4 recipe uses
  strict `require_answer_tag: true`; the 1x8 recipe uses `graded_format_reward: true`, which
  gives 1.0 for a correct tag and 0.5 for a correct answer in another recognized format.
- **`metadata.answer` is a single uppercase letter**, pulled out of `solution` (preferring the
  `<answer>…</answer>` tag, falling back to the first standalone A–D). `MCExactMatchRewardScorer`
  reads `metadata["answer"]` and nothing else, and **returns 0.0 rather than raising** when it
  is missing or malformed — a schema mistake here is indistinguishable from a model that is
  always wrong, so the converter validates at conversion time instead.
- **Absolute URIs.** `path` is repo-relative (`./CLEVRER/...`); the converter joins it with
  `--data-root` and writes an absolute path. Rerun the converter if the media tree moves.
- **Missing files are skipped by default**, so a partially extracted download produces a
  smaller but fully valid dataset rather than crashing mid-rollout.
- **Unique `prompt_id`.** It becomes the root `sample_id` (`prompt:{id}:sample:0`) and is what
  GSPO groups siblings by; duplicates inside a batch raise `ValueError`. The
  `{source}:{problem_id}` form stays unique across folders since `problem_id` is per-file.
- **Deterministic shuffle then split.** Rows are shuffled with `--seed` before the val holdout,
  so `val.jsonl` mixes sources instead of being whichever source happened to land last.

## Usage

```yaml
data_source:
  _target_: unirl.data.data_source.MultimodalRLDataSource
  args:
    run:
      data_path: datasets/video_r1_260k/train.jsonl
      eval_data_path: datasets/video_r1_260k/val.jsonl
      seed: 42
    algorithm:
      prompts_per_rollout: ${batch_size}
```

```bash
QWEN3_OMNI_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=datasets/video_r1_260k/train.jsonl \
EVAL_DATA_PATH=datasets/video_r1_260k/val.jsonl \
ENTRY=train_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8
```

## Notes

- `video_fps: 1.0` × `video_max_frames: 64` caps a clip at 64 sampled frames; frames plus
  question must fit `max_prompt_length` (12288 for 1x8, 16384 for 1x4). Lower
  `video_max_pixels` before lowering `video_max_frames` if you overflow.
