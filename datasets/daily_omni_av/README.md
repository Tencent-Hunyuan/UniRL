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
- 684 YouTube videos (375 × 30 s, 309 × 60 s) and 1,197 four-way multiple-choice QA pairs.
- Each MP4 contains H.264 video and AAC audio; the release also includes a standalone `.wav`.

> **Daily-Omni is an evaluation benchmark with no official train/val split.** Hugging Face's
> `train` label is only the default name for the single `qa.json`. Training on a subset makes
> the resulting `eval/acc` incomparable with the published benchmark; treat this recipe as an
> audio-in-video RL smoke test.

## Download

```bash
hf download liarliar/Daily-Omni --repo-type dataset --local-dir /path/to/Daily-Omni

# Videos.tar is an uncompressed tar — do NOT pass -z.
tar -xf /path/to/Daily-Omni/Videos.tar -C /path/to/Daily-Omni
```

The download is ~3.9 GB; budget ~8 GB while the tar and extracted tree coexist.
After extraction:

```
/path/to/Daily-Omni/
├── qa.json
└── Videos/<video_id>/<video_id>_video.mp4
                     /<video_id>_audio.wav
```

`qa.json` rows use capitalized keys:

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

## Cook

```bash
python datasets/daily_omni_av/convert_daily_omni_dataset_format_to_unirl.py \
  --qa-json /path/to/Daily-Omni/qa.json \
  --out-dir datasets/daily_omni_av
```

`--videos-root` defaults to `Videos/` next to `qa.json`; point it elsewhere if you unpacked
the tar somewhere else. `--val-ratio` (default `0.1`) and `--seed` (default `42`) control the
holdout. Rows whose MP4 is missing on disk are dropped and counted — pass
`--keep-missing` to emit them anyway, which is useful for a dry run before the tar finishes
extracting. An unparseable ground truth aborts with the offending `qa.json[index]`.

Two details matter:

- **It splits by `video_id`, not by row.** Several QA pairs share one clip; a row-level split
  leaks the same video into train and val and inflates `eval/acc`.
- **It appends an explicit answer-format instruction.** Both recipes use
  `require_answer_phrase: true`, so a bare `B` scores 0; the reply must contain an
  `answer is B` phrase or an `<answer>B</answer>` tag.

## Format

Each output line:

```json
{
  "prompt": "Question text\nA. ...\nB. ...\nC. ...\nD. ...\nWatch the video and listen to its audio, then answer the multiple-choice question.\nReason step by step, then end your reply with the exact phrase: The answer is [X]",
  "prompt_id": "daily_omni_av:train:000042:Ec_lQgZ9wlg",
  "media_refs": [{"modality": "video", "role": "prompt", "uri": "/abs/path/Ec_lQgZ9wlg_video.mp4"}],
  "metadata": {"answer": "B", "video_id": "Ec_lQgZ9wlg", "qa_type": "Event Sequence"}
}
```

## Why cook it this way

- **One video ref, no separate audio ref.** UniRL supports standalone
  `(audio, prompt)` references, but Daily-Omni is intentionally cooked as one
  `(video, prompt)` reference because its task requires aligned image and audio
  from the same MP4. The recipes set `use_audio_in_video: true`, so the processor
  demuxes the MP4's AAC track. The shipped `_audio.wav` files are therefore not
  referenced; adding one would duplicate the audio rather than improve alignment.
- **`role: "prompt"`, not `"condition"`.** `(video, condition)` decodes the clip into a frame
  tensor for diffusion V2V. `(video, prompt)` passes the URI through to the Qwen3-Omni
  conversation builder, which is what an AR prompt video needs. A batch may not mix the two.
- **Absolute URIs.** The converter resolves every clip against `--videos-root` and writes the
  absolute path. This avoids dependence on the training process's working directory, but the
  output is machine-local; rerun the converter if the video tree moves.
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
      prompts_per_rollout: ${batch_size}
```

```bash
QWEN3_OMNI_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=datasets/daily_omni_av/train.jsonl \
EVAL_DATA_PATH=datasets/daily_omni_av/val.jsonl \
ENTRY=train_ar bash examples/run_experiment_single_node.sh \
  ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x8
```

## Notes

- `video_fps: 2.0` × `video_max_frames: 64` caps a clip at 32 s of sampled frames, so 60 s
  videos are subsampled. Frames plus question must fit `max_prompt_length: 16384`; lower
  `video_max_pixels` before lowering `video_max_frames` if you overflow.
- Non-commercial license: do not ship models trained on this without checking CC BY-NC-SA 4.0.
