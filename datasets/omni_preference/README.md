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
- **Cross-loading an adapter between the two stacks is not a valid comparison, and
  every number below that does so is void.** Each model scores far better under the
  stack it was trained in. Measured on the same val rows, `reward_accuracy`:

  | | this harness | verl's harness |
  |---|---|---|
  | UniRL adapter | 0.8067 | 0.7274 |
  | verl adapter (step 30) | 0.7417 | 0.9271 |

  Both diagonal entries beat both off-diagonal ones: UniRL loses 7.9pp moving to
  verl's harness, verl loses 18.5pp moving to this one. This is not a weight-transfer
  bug — both round-trips are exact. verl's own exported adapter reloaded into verl
  gives `0.9270833333333334` / margin `1.0472586419847276`, bit-identical to its live
  value; and round-tripping a UniRL adapter through this harness's PEFT-dir branch
  reproduces its native reading (0.8150 both). The models are transferred faithfully
  and still score differently, so what differs is the **input pipeline** — the prompt
  rendering, media processing and tokenisation each stack feeds the same row through.
  A model performs best on the rendering it was trained on.

  This is what the long-unexplained ~0.20 was: verl self-reports 0.9670 at step 130
  and this harness reads 0.7483 from its checkpoint, a difference of 0.2187. It is
  the cross-stack evaluation penalty, not a metric bug and not a training difference.
- **The pipeline difference is a missing system turn, and it is the whole of it.**
  Rendering the same row through both stacks' processors, the user turn is
  byte-identical — `<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Are
  multiple birds singing?<|im_end|>` — so media placement and turn structure already
  agree (`build_omni_messages` merges the two same-role turns `embed_sft_prompt`
  emits into one user message, so there is no two-turn problem). What differs is that
  `verl_omni/utils/dataset/qwen3_omni_transform.py` unconditionally prepends
  Qwen3-Omni's canonical system message, while `system_instruction` defaults to
  `None` here and no recipe set one. The chat template does **not** supply a default,
  so the prompt was rendering at **16 tokens where the model expects 56**, with no
  system turn at all — off-distribution for a model whose instruction tuning assumes
  it. The DPO recipe now sets `pipeline.system_instruction` to that exact string
  (verified equal to verl's constant character for character). Note the earlier
  `<audio>` marker finding is consistent with this rather than contradicting it: the
  marker is junk text in this pipeline, which injects media as typed content blocks,
  and is the placeholder in verl's, which passes a raw string plus an `audios` list.
  Both render to the same `<|audio_start|><|audio_pad|><|audio_end|>` span.
  Retraining with the system turn in place moves this implementation from 0.8067 to
  **0.8167** accuracy and 0.9470 to **1.0840** margin (eval loss 0.53899 -> 0.53245),
  by modality +0.5pp audio / +2.5pp image / +0.0pp video. So the missing system turn
  was a real defect worth fixing, but it accounts for only ~1pp of the ~18pp
  cross-stack asymmetry -- it is not the whole of the pipeline difference, and the
  rest is still unattributed. Aligning the remaining piece -- the supervised span,
  where verl marks only the assistant text and this stack also supervises the
  appended EOS (`track_builder.append_eos: false`; see `unirl/algorithms/README.md`)
  -- brings both models onto one objective for the first time: **UniRL 0.7633 vs verl
  0.7217, a gap of +4.2pp (z=1.65, p=0.099)**, margins 0.5976 vs 0.3067. That is the
  first cross-stack reading where both sides are scored under the same input contract
  *and* the same supervision contract, and it is no longer significant at 0.05.
  Note both numbers drop when EOS leaves the objective (UniRL 0.8167 -> 0.7633, verl
  0.7483 -> 0.7217): supervising EOS raises this metric for both, so the earlier
  absolute figures were partly measuring it.
- **The input contract is fully aligned; what remains is forward noise of the same
  order as the gap.** Feeding verl's own transformed tensors and this stack's own
  through one base model with **no adapter** (same weights, same rows) isolates the
  forward path from anything either run learned. The prompt is 303 tokens on both
  sides, the answer tokens match, and the supervised counts match exactly. verl's
  sequence is 2 tokens longer only because it keeps an unsupervised `<|im_end|>\n`
  *after* the supervised span, which under causal attention cannot affect any
  supervised log-prob. Also ruled out by measurement: audio decoding (verl uses
  `librosa.load`, this stack PyAV — waveform correlation ~1.0000, mel relative
  difference 0.03–0.34%) and the adapter name→parameter map (576/576 placed, 0
  unwritten, 0 shape mismatches). What is left is that the same weights on the same
  tokens still give summed response log-probs differing by −0.40 / −0.90 / +1.00 /
  +0.02 across four rows — **mixed signs, mean −0.07**, so zero-mean noise rather
  than a contract difference. Since a raw margin here is 3–6, per-pair noise of that
  size is 20–30% of the quantity being thresholded; it does not bias the mean margin
  but it flips near-tie pairs, which is what accuracy counts. **Correction:** that
  comparison fed verl's tensors through a dense `model(...)` call and this stack's
  through `pipeline.ar.replay(...)`, which are two different code paths *within this
  stack* — packed varlen with `fuse_full_ids` versus a dense forward. Running the
  same row through both paths here, no adapter, gives mixed-sign differences up to
  0.54 (mean +0.19), the same scale as the "cross-stack" spread, so those numbers
  measured my own two paths and say nothing about verl. Position ids are separately
  confirmed **numerically identical** between the stacks (`max |diff| = 0`, zero
  mismatched elements), and verl's negative-sentinel substitution before
  `get_rope_index` is a no-op — raw and sentinel ids return bit-identical positions.
  So rope is not the cause either.
- **The residual gap is reproducible, not run-to-run noise — a control this
  comparison had been missing all along.** Every cross-stack number up to this point
  compared two *single* runs without ever measuring how much one config varies
  against itself. LoRA init is not seeded here, so re-running the identical aligned
  config is a genuine replicate (its step-2 loss differs, 0.69285 vs 0.69600). Two
  replicates land at **0.7633 and 0.7583** accuracy (margins 0.5976 / 0.5865, eval
  loss 0.58609 / 0.58500) — a spread of 0.50pp, against a gap to verl of **+3.92pp**,
  about 8× larger. So the gap survives, and its margin component (≈0.59 vs 0.307,
  still ~1.9×) survives too. Two replicates cannot estimate a variance, so treat
  0.50pp as one draw rather than a confidence bound; but the gap is not explained by
  restart noise, and it is no longer explained by any difference in configuration,
  training process, or input contract, all of which are now aligned or measured
  inert. What remains unattributed is ~4pp of accuracy and a ~1.9× margin ratio.
  Consequently the "+5.8pp for UniRL" reported earlier was verl's model run through a
  foreign pipeline, and does not support any claim about either implementation.
  Comparing own-pipeline numbers (verl 0.9670, UniRL 0.8067) is also not sound: the
  two renderings do not pose equally hard tasks, so that 16pp is a property of the
  pipelines as much as of the training.
- **The length confound below is still real, but it was measured under that invalid
  comparison.** On one hand-built split fed to both
  stacks (4200 train / 600 val, 1400+200 per modality, verified byte-identical row
  sets), same `lr=1e-5`, same 130 data batches, same 4 optimizer updates per batch,
  both adapters scored in one process on the same rows: raw `reward_accuracy`
  0.8133 here vs 0.7483 for verl, paired McNemar significant (77 vs 38 discordant,
  p=0.0004). 497 of the 600 val rows have the longer answer as `chosen`, and the
  difference sits in that skew: **+10.5pp where chosen is longer, −12.6pp where it
  is not**. Balancing the two length buckets gives 0.6064 vs 0.6172, a gap of
  −0.011, 95% CI [−0.067, +0.042]. **Do not read that as equivalence**: the small
  bucket is 103 rows, so the interval is ±5.5pp and cannot exclude any effect below
  ~5pp — it is "not detected", not "not there", and no equivalence margin was set
  in advance. What *is* established, because rejection is not weakened by low power:
  the two models are functionally different. Their length sensitivity differs by
  +0.231, 95% CI [+0.120, +0.335], their per-row margins share only 56% of variance
  (r=0.746), and they disagree on the ranking of 115/600 rows. Ruled out as causes
  of the raw gap, each by measurement: the loss formula (bit-identical over 8
  variants), the data (one shared file), the reduction nondeterminism (fixing it
  moved neither number in the fourth decimal), and the integrated learning rate
  (verl's cosine over 130 batch-units reused 4× and a cosine over 520 update-units
  both sum to 2.60e-03 — equal, not merely close). A weight-space comparison cannot
  answer this: median cosine between the effective `ΔW` is +0.0004 over 288/288
  modules, but the two `lora_A` row spaces overlap exactly as much as two random
  draws (0.1065 vs a null of 0.1072), so PEFT's random `A` init forces that
  orthogonality and it carries no information. Note separately that verl's own
  logged `val/reward_accuracy` is inflated by an unweighted `np.mean` over unequal
  micro-batches in `reduce_metrics`: recomputing from its own dumped tensors gives
  0.5000 where it logged 0.6875.
- **verl-omni's DPO recipe trains on 63% of its own training split.** Its
  `ModalityGroupedBatchSampler` takes `replacement=True` by default and its launch
  line never overrides it, so the training sampler draws each batch's 32 rows with
  `torch.randint` from one modality. Replaying that draw for the published
  `num_batches=130` over a 4200-row split gives **2646 distinct rows (63.0%), 1554
  rows (37.0%) never seen, and 1098 rows repeated up to 6×** — a bootstrap resample,
  not an epoch. Anything comparing against this recipe is comparing against that
  exposure, so a single shuffled pass is a different experiment even when the split
  file is byte-identical. To feed UniRL the same rows, replay the draw into a manifest
  and read it with `shuffle=false`; `group_by_modality` must stay at the batch size
  regardless (a mixed-modality micro-batch raises), and it reorders blocks, so the row
  multiset is reproduced exactly while batch order is not.
- **Aligning every actionable knob to verl-omni takes 3 changes, not 13.** A
  knob-by-knob audit of verl's resolved launch line found 44 settings: 30 already
  matched, 1 is structural (UniRL packs varlen natively vs `use_remove_padding`), and
  7 of the 13 apparent mismatches closed under measurement rather than argument —
  `image_min_pixels`/`sample_rate`/`frame_factor` are already equal; `video_min_pixels`
  3136-vs-100352 is inert (0/12 rows change tokenization, since it is a floor and every
  frame already clears it); `scale_factor` 28 does not even match verl's own processor
  (`patch 16 × merge 2 = 32`); `max_length=4096` truncates 0/12 rows; `exclude_modules`
  selects an identical 192/192 modules and 96/96 parameters, with all three extra
  patterns matching zero modules in a thinker-only load; and verl's saved
  `adapter_config.json` reads `lora_dropout: 0.0`, so its requested 0.05 was silently
  dropped and the two already agree. The three real ones are the sampler above, the LR
  step convention (see `unirl/train/readme.md`), and `attn_implementation` (verl uses
  `sdpa`, measured −0.0197 against `flash_attention_2` here).
- **Applying all three alignments does not close the gap.** A run with verl's exact
  draw (row multiset identical), verl's LR trace (matching its logged 130 values
  bit-for-bit, `max |diff| = 0`), and `attn_implementation: sdpa` scores 0.8050 against
  verl's 0.7483 — a gap of **+5.7pp (p=0.018)**, versus +5.8pp before aligning. The
  three changes together moved the result by **−0.0017**, i.e. nothing beyond noise.
  It reaches log 2 at step 0 (`eval_loss=0.69315`) and its eval loss falls to 0.53899,
  so the run itself is sound. Combined with the 44-knob audit, this rules out
  configuration as the explanation: every setting verl's launch line specifies is now
  either matched, measured inert, or structurally absent.
- **Aligning the training *process*, not just the knobs, also does not close it.**
  Reading verl's engine rather than its config surfaced three process differences the
  knob audit could not see. Two are real and one is not: (i) verl computes the reference
  log-probs in a separate `infer_batch` pass under `module.eval()` while the policy runs
  under `module.train()` — measured a bit-exact no-op here (`max |diff| = 0.000e+00` over
  60 rows, 0 ranking flips), since the thinker has no dropout or batchnorm in the path;
  (ii) its scheduler advances only on the last mini-batch of each data batch
  (`update_lr_scheduler=batch_idx == total_num_iterations - 1`), which is what
  `steps_per_advance` reproduces; and (iii) `forward_backward_batch` accumulates raw
  micro-batch means with no division by `len(micro_batches)`, making its gradient `N×`
  larger — predicted 2× for this geometry and measured **1.93×/1.93×/1.92×/1.95×** on the
  first four steps once `normalize_across_micros=false` was implemented. Adding that
  fourth alignment gives **0.8067**, i.e. back to the unaligned number and still +5.8pp
  over verl. The 2× is absorbed by `clip_grad=1.0`, which fires on 78% of verl's steps
  and 100% of ours. Read together with the length analysis above — the raw gap is carried
  by the 497/600 length skew, and the two models differ in *length sensitivity* (+0.231,
  CI [+0.120, +0.335]) — neither configuration nor training process explains it.

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
