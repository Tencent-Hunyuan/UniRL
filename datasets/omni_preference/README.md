# Omni-Preference DPO Dataset

Rubric-grounded multimodal preference pairs for Qwen3-Omni Thinker offline DPO.

Used by:

- `examples/ar/qwen3_omni_audio_dpo_lora_1x8.yaml`
- `examples/ar/qwen3_omni_image_dpo_lora_1x8.yaml`
- `examples/ar/qwen3_omni_pooled_dpo_lora_1x8.yaml`

## Source

- Hugging Face: [Omni-RRM/Omni-Preference](https://huggingface.co/datasets/Omni-RRM/Omni-Preference)
- Introduced by *Omni-RRM: Advancing Omni Reward Modeling via Automatic
  Rubric-Grounded Preference Synthesis*.
- Each `dataset_jsonl/<modality>/final_rl_data.jsonl` row carries the media
  path, a rubric judging prompt whose `### Context` block holds the question and
  two candidate answers, and a `solution` object with `score_A`, `score_B` and a
  reconciled `better` verdict.

The dataset card declares no license. Verify the upstream terms before
redistributing converted media or trained artifacts.

## Download

```bash
hf download Omni-RRM/Omni-Preference --repo-type dataset \
  --include 'dataset_jsonl/*' 'audio_files/*' \
  --local-dir /path/to/Omni-Preference
```

Media is large (image ~2.6 GB, audio ~4.0 GB, video ~39 GB). Fetch only the
modalities you intend to train.

## Cook

```bash
python datasets/omni_preference/convert_omni_preference_to_unirl.py \
  --snapshot /path/to/Omni-Preference \
  --modality audio \
  --out-dir datasets/omni_preference_audio_dpo
```

The converter:

1. parses the `### Context` block into media / question / candidate A / candidate B;
2. orients the pair by the teacher verdict (`better="A"` → `chosen=candidate_a`)
   and drops `better="equal"` rows, which carry no preference signal;
3. resolves each medium to an absolute local path;
4. splits train/val by **media basename**, so one medium never appears in both;
5. writes `train.jsonl`, `val.jsonl` and a `manifest.json` of counters.

Emitted rows use the preference shape accepted by `unirl/data/sft.py`, with the
modality placeholder (`<audio>`/`<image>`/`<video>`) prefixed onto the prompt and
the medium referenced at `role="prompt"`:

```json
{"sample_id": "omni_pref_audio_0",
 "prompt": "<audio>does someone crash and fall?",
 "chosen": "No, there is no indication of a crash or fall ...",
 "rejected": "No, there is no crash or fall in the audio.",
 "media_refs": [{"modality": "audio", "role": "prompt", "uri": "/abs/path.wav"}],
 "metadata": {"modality": "audio", "win_score": 8.0, "lose_score": 6.0, "better": "B"}}
```

Observed audio counts: 5707 raw rows → 1335 dropped as `equal` → 4372 pairs
(4155 train / 217 val at the default `--test-ratio 0.05`).

## Pool the modalities

The pooled recipe trains on all three modalities at once. Concatenating the
per-modality manifests as-is would weight each modality by its row count;
`group_by_modality` then emits proportionally many single-modality blocks. Take
the same number of rows from each modality instead — a multiple of the recipe's
`batch_size`, so no modality contributes a partial block — and interleave the
validation splits round-robin so that a `eval_num_samples` prefix stays balanced.

## Gotchas

- **Media basenames in the jsonl are not byte-equal to the filenames on disk.**
  The jsonl spells them with `_` where the file uses a space, and HTML-escapes
  `&` as `&amp;`; some paths also carry a trailing `_` artifact inside the
  context block. Matching basenames literally resolves only 456 of 1167 audio
  files. The converter folds both spellings (`_basename_variants`) and reaches
  all 1167 — a literal matcher silently discards ~61% of the data as "missing
  media" instead of failing.
- **One modality per manifest.** The converter writes one directory per
  `--modality`. Pool them only as described above; mixing modalities inside a
  single batch breaks `Batch.slice` (see `unirl/train/readme.md` Gotchas).
- **A blanket `hf download --include 'video-dataset/*'` is not worth running.**
  The repo holds 11765 video files but `final_rl_data.jsonl` references only
  1572 of them, and the blanket walk resolves roughly one file per two seconds
  against roughly five per second for direct per-file fetches. Fetch the
  referenced basenames instead.
