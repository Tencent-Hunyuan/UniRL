# Daily-Omni Audio/Video MCQA RL Dataset

Audio-visual multiple-choice QA prompts for RL training of Qwen3-Omni Thinker with
`use_audio_in_video: true`. Each sample is one video (whose embedded audio track carries
part of the answer) plus a 4-way A/B/C/D question.

Used by:

- `examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x8.yaml`
- `examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x4.yaml`

## Source

- **Daily-Omni**: [liarliar/Daily-Omni](https://huggingface.co/datasets/liarliar/Daily-Omni)
  - Code: [Lliar-liar/Daily-Omni](https://github.com/Lliar-liar/Daily-Omni)
  - Paper: [arXiv:2505.17862](https://arxiv.org/abs/2505.17862) — *Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities*
  - License: **CC BY-NC-SA 4.0** (non-commercial, share-alike)
- 684 YouTube videos (375 × 30 s, 309 × 60 s), 1,197 QA pairs, all 4-way multiple choice.
- QA types: Event Sequence (306), AV Event Alignment (238), Context understanding (193),
  Reasoning (175), Inference (154), Comparative (131).
- Every video ships as an MP4 with an **H.264 video track and an AAC audio track**, plus a
  pre-extracted `.wav` alongside it.

> **Daily-Omni is published as an evaluation benchmark, not a training set.** There is no
> official train/val split. The HuggingFace dataset viewer shows a split named `train` with
> 1,197 rows — that is just HF's default name for a single bare `qa.json`, not a training
> partition. If you train on part of it and evaluate on the rest, your `eval/acc` is **not**
> the published Daily-Omni benchmark number, and any leaderboard comparison is contaminated.
> Treat this recipe as a small-scale audio-in-video RL smoke test, and carve the split
> yourself (see below).

## Download

```bash
hf download liarliar/Daily-Omni --repo-type dataset --local-dir /path/to/Daily-Omni

# Videos.tar is an uncompressed tar — do NOT pass -z.
tar -xf /path/to/Daily-Omni/Videos.tar -C /path/to/Daily-Omni
```

Total ~3.9 GB compressed; budget ~8 GB while the tar and the extracted tree coexist.
After extraction:

```
/path/to/Daily-Omni/
├── qa.json
└── Videos/<video_id>/<video_id>_video.mp4
                     /<video_id>_audio.wav
```

`qa.json` rows use capitalized keys (and note the misspelled `Explaination`):

```json
{
  "Question": "What visual elements were displayed immediately after ...?",
  "Choice": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "Answer": "B",
  "video_id": "Ec_lQgZ9wlg",
  "Type": "Event Sequence",
  "video_duration": "30s"
}
```

`content_parent_category` / `content_fine_category` (960 rows) and `Explaination` (235 rows)
are optional — parse defensively.

## Cook

Two steps. `convert_daily_omni_dataset_format_to_unirl.py` consumes a **verl/EasyR1-style
JSONL**, not the raw `qa.json`, so you first flatten `qa.json` into that intermediate form
and split it.

### Step 1 — `qa.json` → verl-style JSONL

The converter reads four things per row: the user text (`prompt[].content[].text`), the video
path (`videos[0].video`, falling back to a `type: "video"` content part), the gold letter
(`reward_model.ground_truth`), and `extra_info.{video_id,qa_type}`.

```python
import json, os, random

ROOT = "/path/to/Daily-Omni"
INSTRUCTION = (
    "Watch the video and listen to its audio, then answer the multiple-choice question.\n"
    "Reason step by step, then end your reply with the exact phrase: The answer is [X]"
)

rows = []
for i, qa in enumerate(json.load(open(f"{ROOT}/qa.json", encoding="utf-8"))):
    vid = qa["video_id"]
    path = os.path.join(ROOT, "Videos", vid, f"{vid}_video.mp4")
    text = "\n".join([qa["Question"], *qa["Choice"], INSTRUCTION])
    rows.append({
        "prompt": [{"role": "user", "content": [
            {"type": "video", "video": path},
            {"type": "text", "text": text},
        ]}],
        "videos": [{"video": path}],
        "reward_model": {"ground_truth": qa["Answer"]},
        "extra_info": {"video_id": vid, "qa_type": qa.get("Type")},
    })

# Split by video_id so the same clip never lands in both splits.
by_video = {}
for r in rows:
    by_video.setdefault(r["extra_info"]["video_id"], []).append(r)
vids = sorted(by_video)
random.Random(42).shuffle(vids)
val_vids = set(vids[: max(1, len(vids) // 10)])

for name, keep in (("train", lambda v: v not in val_vids), ("val", lambda v: v in val_vids)):
    with open(f"{ROOT}/daily_omni_av_{name}.jsonl", "w", encoding="utf-8") as f:
        for v in vids:
            if keep(v):
                for r in by_video[v]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

Two details that matter:

- **Split by `video_id`, not by row.** Several QA pairs share one clip; a naive row-level
  split leaks the same video into train and val and inflates `eval/acc`.
- **The `The answer is [X]` instruction is required.** Both recipes score with
  `MCExactMatchSpec(require_answer_phrase: true)`, which only accepts a phrase matching
  `(answer|option)\s*(is|:)\s*[\(\[]?([A-D])` or an `<answer>X</answer>` tag. A reply ending
  in a bare `B` scores **0.0**. Daily-Omni's own `Question` field carries no format
  instruction, and the converter copies the user text verbatim, so if you skip this the
  entire run sits at reward 0.

### Step 2 — verl-style JSONL → UniRL JSONL

```bash
python datasets/daily_omni_av/convert_daily_omni_dataset_format_to_unirl.py \
  --train-input /path/to/Daily-Omni/daily_omni_av_train.jsonl \
  --val-input   /path/to/Daily-Omni/daily_omni_av_val.jsonl \
  --out-dir     datasets/daily_omni_av
```

Rows whose MP4 is missing on disk are dropped; pass `--keep-missing` to emit them anyway
(useful for a dry run before the tar finishes extracting). An unparseable ground truth
aborts with the offending `path:line`.

## Format

Each output line:

```json
{
  "prompt": "Question text\nA. ...\nB. ...\nC. ...\nD. ...\nReason step by step, then end your reply with the exact phrase: The answer is [X]",
  "prompt_id": "daily_omni_av:train:000042:Ec_lQgZ9wlg",
  "media_refs": [{"modality": "video", "role": "prompt", "uri": "/abs/path/Ec_lQgZ9wlg_video.mp4"}],
  "metadata": {"answer": "B", "video_id": "Ec_lQgZ9wlg", "qa_type": "Event Sequence"}
}
```

## Why cook it this way

- **One video ref, no audio ref.** `MultimodalRLDataSource` only accepts the pairs
  `(image, condition)`, `(video, condition)` and `(video, prompt)`; anything else — including
  **any `modality: "audio"` entry** — raises `NotImplementedError` at collate time. Audio
  reaches Qwen3-Omni through the MP4's own AAC track: the recipes set
  `use_audio_in_video: true` on the bundle, the pipeline and the vLLM-Omni engine, and the
  processor demuxes it. So the shipped `_audio.wav` files are **not** referenced. Adding them
  as a second media ref would crash the run.
- **`role: "prompt"`, not `"condition"`.** `(video, condition)` decodes the clip into a frame
  tensor for diffusion V2V. `(video, prompt)` passes the URI through to the Qwen3-Omni
  conversation builder, which is what an AR prompt video needs. A batch may not mix the two.
- **At most one video ref per prompt**, or collate raises `ValueError`.
- **Absolute URIs.** The converter calls `os.path.abspath(os.path.expanduser(...))`. Relative
  URIs are resolved against the directory holding the JSONL, so absolute paths keep the file
  relocatable.
- **`metadata.answer` is a single uppercase letter.** `MCExactMatchRewardScorer` reads
  `metadata["answer"]` and nothing else, and the converter unwraps `[B]` → `B` and validates
  membership in `{A,B,C,D}` up front. This matters because the scorer **fails silently**: a
  missing or malformed `answer` returns 0.0 rather than raising, which looks identical to a
  model that never gets anything right.
- **Unique `prompt_id`.** It becomes the root `sample_id` (`prompt:{id}:sample:0`) and is what
  GSPO groups siblings by; duplicates inside one batch raise `ValueError`. The
  `{split}:{index}:{video_id}` form stays unique even when several questions share a clip.

## Usage

```yaml
data_source:
  _target_: unirl.data.data_source.MultimodalRLDataSource
  args:
    run:
      data_path: datasets/daily_omni_av/train.jsonl
      eval_data_path: datasets/daily_omni_av/val.jsonl
      seed: 42
    algorithm:
      prompts_per_rollout: 8   # must equal batch_size
```

```bash
QWEN3_OMNI_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=datasets/daily_omni_av/train.jsonl \
EVAL_DATA_PATH=datasets/daily_omni_av/val.jsonl \
ENTRY=train_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x8
```

## Recommended: pre-filter all-zero-reward groups

GRPO/GSPO advantages are group-normalized (`Part.compute_advantages`, `scope="group"`):

```
adv_i = (r_i - mean(r_group)) / (std(r_group) + 1e-8)
```

When every sample in a group scores the same, `r_i - mean = 0` and the advantage is
**exactly 0** — with or without `normalize_adv_by_std`. That group consumes a full rollout
(here 8 samples × a 30–60 s video through the audio+vision towers, the most expensive thing
in the loop) and contributes no gradient. Two cases produce it:

- **all-zero groups** — the model never gets the question right, or never emits the
  `The answer is [X]` phrase;
- **all-one groups** — the question is already saturated.

UniRL has no DAPO-style dynamic sampling: nothing resamples or skips these at runtime. It
only *reports* them, as `rollout/zero_std_group_ratio` and `rollout/zero_std_group_count` in
W&B. Watch those first; if the ratio is high, filter offline.

The procedure: roll out K samples per prompt with **the exact model you are about to train**
(same checkpoint/adapter, same prompt text, same `temperature`/`top_p`/`max_new_tokens` as
the `sampling:` block), score them with the same `MCExactMatchSpec` settings the recipe uses,
and drop prompts whose K rewards are all identical.

```python
import json

K_LO, K_HI = 1, 7   # keep prompts with 1..7 correct out of K=8
keep = {pid for pid, rs in json.load(open("passrate.json")).items() if K_LO <= sum(rs) <= K_HI}

with open("train.jsonl", encoding="utf-8") as src, \
     open("train.filtered.jsonl", "w", encoding="utf-8") as dst:
    for line in src:
        if json.loads(line)["prompt_id"] in keep:
            dst.write(line)
```

Caveats worth respecting on a set this small:

- Daily-Omni yields only ~1.1k prompts. Aggressive filtering can leave too few rows —
  `MultimodalRLDataSource` refuses to start if the dataset is smaller than
  `prompts_per_rollout`, and a tiny set means the loader cycles the same prompts every few
  rollouts. Prefer dropping only all-zero groups here, and keep the all-one ones.
- The filter is a snapshot of one checkpoint. As the policy improves, previously all-zero
  prompts become learnable and previously mixed ones saturate, so re-estimate every few
  hundred rollouts rather than filtering once.
- Sanity-check the format first. If pass rates are near zero *everywhere*, the cause is
  usually the missing `The answer is [X]` instruction, not difficulty — filtering would
  delete the whole dataset instead of fixing the prompt.

## Notes

- `video_fps: 2.0` × `video_max_frames: 64` caps a clip at 32 s of sampled frames, so 60 s
  videos are subsampled. Frames plus question must fit `max_prompt_length: 16384`; lower
  `video_max_pixels` before lowering `video_max_frames` if you overflow.
- Non-commercial license: do not ship models trained on this without checking CC BY-NC-SA 4.0.
