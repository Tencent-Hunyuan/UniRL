# AudioCaps for LTX-2.3 CLAP training

This converter prepares the official human-written captions from
[AudioCaps](https://github.com/cdjkim/audiocaps) for the LTX-2.3 CLAP reward recipe.
AudioCaps was introduced at NAACL 2019 and contains captions for AudioSet clips.

The generated JSONL is a local artifact and must not be committed. This recipe does not
download or consume the source audio: LTX-2.3 generates audio from each caption and CLAP
scores that generated waveform against the same official caption.

The event-grounded variant also joins the AudioCaps clips to their human-confirmed AudioSet
labels. It uses those labels as positive sound-event targets and RMS-normalizes generated
audio before CLAP scoring so increasing waveform gain cannot improve the reward by itself.

## Source and terms

By default, the converter reads the official CSV files at AudioCaps commit
`d004db3ea1b01cf4fd0347dd8d27db90cadc8809`:

- train: 49,838 clips with one caption each
- validation: 495 clips with five captions each (2,475 caption rows)
- test: 975 clips with five captions each (4,875 caption rows)

The upstream repository says its code and dataset are free to use for academic purposes and
asks users to cite the AudioCaps paper. Review the upstream terms before redistributing or
using the data outside that scope.

## Cook

From the repository root:

```bash
python datasets/audiocaps/prepare_audiocaps.py --out-dir data/audiocaps
```

The default manifests contain the full official train split and 64 distinct clips sampled
deterministically from the official validation split. Validation has five captions per clip;
the converter chooses one deterministically so a clip is not counted five times. Pass
`--keep-all-eval-captions --eval-limit 0` to retain every validation caption.

For the event-grounded recipe, download `class_labels_indices.csv`, `eval_segments.csv`,
`balanced_train_segments.csv`, and `unbalanced_train_segments.csv` from the official
[AudioSet download page](https://research.google.com/audioset/download.html), place them in
one directory, and run:

```bash
python datasets/audiocaps/prepare_audiocaps.py \
  --out-dir data/audiocaps-event-grounded \
  --audioset-metadata-dir /path/to/audioset-metadata
```

The converter requires an exact AudioSet segment match for every selected AudioCaps clip and
fails instead of silently emitting an ungrounded row.

Each row uses the original AudioCaps caption as both the generation prompt and CLAP target:

```json
{
  "prompt": "Multiple clanging and clanking sounds",
  "metadata": {
    "negative_audio_captions": ["... seven captions from other rows ..."],
    "audio_event_labels": ["Door", "Sliding door"],
    "audiocap_id": "58146",
    "source_dataset": "AudioCaps",
    "source_split": "train"
  }
}
```

The CLAP scorer reads the positive text directly from `prompt`; no separate
audio-caption field or hand-written rewrite is required for this dataset.

The deterministic negative captions provide the hardest-negative retrieval margin used by
the CLAP recipe and its retrieval diagnostics. The event-grounded recipe additionally scores
the least-aligned positive AudioSet label, encouraging every labeled event to remain audible.

## Train

```bash
DATA_PATH=data/audiocaps/train.jsonl \
EVAL_DATA_PATH=data/audiocaps/eval.jsonl \
bash examples/run_experiment_single_node.sh \
  diffusion/ltx2/ltx2_3_t2av_clap_trainside
```

Train and evaluation use the official AudioCaps train/validation split boundary.

To train with RMS normalization and AudioSet event coverage:

```bash
DATA_PATH=data/audiocaps-event-grounded/train.jsonl \
EVAL_DATA_PATH=data/audiocaps-event-grounded/eval.jsonl \
bash examples/run_experiment_single_node.sh \
  diffusion/ltx2/ltx2_3_t2av_clap_event_grounded_trainside
```
