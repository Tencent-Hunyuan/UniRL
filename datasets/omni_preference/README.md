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
medium referenced at `role="prompt"` and the question carried bare — the chat stage
injects the medium as its own turn, so a `<audio>`/`<image>`/`<video>` marker in the
prompt text would survive as literal tokens (see Gotchas):

```json
{"sample_id": "omni_pref_audio_0",
 "prompt": "does someone crash and fall?",
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

## Learning rate

The pooled recipe uses `1.0e-5`, ten times the `1e-6` the published reference run
uses. At `1e-6` this setup underfits: held-out accuracy is no worse than accuracy on
rows the model trained on (0.784 vs 0.704 on 240 seen pairs), and the eval loss stops
improving around step 150 of 200. Raising the rate closes most of the gap, then
saturates.

| lr | eval loss @200 | balanced accuracy (877-row val) | 95% CI |
|---|---|---|---|
| 1e-6 | 0.5420 | 0.7839 | [0.756, 0.812] |
| 5e-6 | 0.4641 | 0.8148 | [0.788, 0.841] |
| **1e-5** | **0.4378** | **0.8395** | [0.814, 0.865] |
| 2e-5 | 0.4302 | 0.8399 | [0.815, 0.865] |

`2e-5` buys nothing over `1e-5` (+0.0004) and its eval loss is non-monotone early
(0.4714 at step 25, 0.4827 at step 50), so `1e-5` is the recipe default.

The same effect reproduces in the reference implementation itself, which is the
strongest evidence that this is a property of the data and not of this port: running
the reference's own recipe on its own data with `LR` as the only change moves its
step-50 self-reported validation accuracy from 0.8229 to 1.0000 and its margin from
0.399 to 2.039, with matched-step training loss dropping 0.6644 → 0.4649. `1e-6`
leaves real headroom on this dataset.

Raising LoRA rank past 32 does not help either: `rank=128, alpha=256` at `1e-5`
lands at 0.8289 [0.803, 0.855], *below* `rank=32`. Capacity is not the binding
constraint.

Accuracy here weights the three modalities equally, which is what the reference
protocol does (`val_max_samples=96` split evenly across modalities). Averaging over
the 877 rows instead weights by split size (217/306/354) and reads differently — do
not compare the two.

**Score the whole split, not a 96-row prefix.** A 96-row evaluation carries a 95% CI
of roughly ±0.06, which is wider than every effect measured here, and the two
protocols genuinely disagree: `rank=128` scores 0.9167 on the first 96 rows — its
best result, and above the reference — while scoring 0.8289 on all 877, its worst.
Ranking configurations on 96 rows would have picked the wrong one.

**The converter's `--test-ratio` decides which examples are held out, and that
shifts the number by about as much as a learning-rate step.** This converter
defaults to `0.05` and holds out whole media groups one at a time; the reference
implementation uses `0.10` and accumulates groups until a row target is met. The two
produce different held-out *examples*, and one checkpoint scores 0.8395 on the first
and 0.8675 on the second — a 2.8pp swing from the split alone, comparable to the
`5e-6 → 1e-5` learning-rate step (2.5pp). Reproduce the reference's split before
reading anything into a difference against its published number.

**The reference implementation's audio split is much smaller than this one's, but
that turns out not to matter for preference accuracy.** Its converter matches audio
basenames literally and drops 2669 of 4372 pairs as missing media, keeping 1504
training rows where this converter keeps 4155 (see the basename gotcha below).
Rebuilding its parquet with the folded matcher — its own row builder and split, only
the basename lookup swapped — raises its audio training set 1504 → 3935 and changes
balanced accuracy by **−0.0025 (p=0.94)**, with audio itself going 0.6444 → 0.6370.

That is worth stating plainly because the opposite is easy to argue from true facts:
audio is the reference's weakest modality *and* its most data-starved one. Both hold,
and the causal conclusion is still wrong. The recovered pairs are real data, but they
buy nothing on this metric.

**Check for media leakage before comparing two models trained from different
splits.** Two converters with different `--test-ratio` hold out different media, so
one model's validation rows routinely contain media the other model trained on.
Scoring this recipe on the reference's validation split puts **74.5% (1085/1457)** of
those rows on media this recipe trained on, which inflated a measured gap from
+0.062 to +0.118 and its significance from p=0.03 to p=1e-15. Pass
`--exclude-media-in <the other model's train manifest>`, or build an eval set whose
media appear in neither training split. The check costs seconds; without it every
cross-split number is quietly wrong in the favourable direction.

## Gotchas

- **Do not prefix a `<audio>`/`<image>`/`<video>` marker onto the prompt.** Those
  strings are not special tokens — the tokenizer splits each into three ordinary text
  tokens — and `Qwen3OmniChatTemplateStage.embed_sft_prompt` injects the medium as its
  own user turn regardless, so a marker in the prompt text is never consumed and
  simply trains on junk. The sibling `dcase2025_audio_qa` converter writes the
  question bare, which is the convention to follow. This converter carried such a
  prefix through every run recorded above; stripping it moved balanced accuracy by
  −0.010 on a fixed checkpoint, so it is a data-quality defect rather than a large
  metric effect, but the emitted manifests were not what the README claimed.
- **`reward_accuracy` on this dataset is weaker than it looks.** Picking whichever
  answer is longer already scores 72.8% (audio), 85.9% (image) and 82.5% (video), so
  roughly four fifths of the signal is answer length. A model-quality judgement run
  over the full 217-row audio split was an adequately-powered null result
  (78/75/64 win/lose/tie, p=0.872) even though the training objective moved a lot —
  optimising this metric is not the same as improving generations.
- **The cross-stack comparison against verl-omni does not close.** On one
  hand-built split fed to both stacks (4200 train / 600 val, 1400+200 per
  modality, verified byte-identical row sets), same `lr=1e-5`, same 130 data
  batches, same 4 optimizer updates per batch, scored by one harness with a
  deterministic reduction: this implementation reaches balanced accuracy 0.8067
  and verl 0.7483 (+5.8pp, p=0.015), with mean margins 0.9531 vs 0.2236 (4.3×).
  **The residual is unexplained.** What it is *not*: the loss formula (bit-identical
  over 8 variants), the data (one shared file), the reduction nondeterminism (fixing
  it moved neither number in the fourth decimal), or the integrated learning rate
  (verl's cosine over 130 batch-units reused 4× and a cosine over 520 update-units
  both sum to 2.60e-03 — equal, not merely close). Note also that verl's own logged
  `val/reward_accuracy` is inflated by an unweighted `np.mean` over unequal
  micro-batches in `reduce_metrics`: recomputing from its own dumped tensors gives
  0.5000 where it logged 0.6875, and correcting it reverses the apparent ranking.
  Do not read this comparison as cross-stack parity.

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
