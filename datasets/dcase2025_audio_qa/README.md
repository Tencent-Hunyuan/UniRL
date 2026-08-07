# DCASE 2025 Standalone-Audio MCQA Dataset

Four-way audio question answering data for reinforcement learning with the
Qwen3-Omni Thinker. Each UniRL record contains one standalone WAV reference,
one multiple-choice prompt, and an A/B/C/D ground-truth answer.

Used by:

- `examples/ar/qwen3_omni_audio_dcase_gspo_lora_vllm_omni_1x4.yaml`

## Source

- Hugging Face: [gijs/dcase2025-audio-qa](https://huggingface.co/datasets/gijs/dcase2025-audio-qa)
- Source split sizes reported by the dataset card:
  - `train`: 8,221 labeled rows
  - `test`: 2,466 labeled rows
  - `eval`: 4,884 competition rows without public answers
- Audio is embedded in parquet and described by the source as mono, 16-bit,
  48 kHz WAV data.

The Hugging Face dataset card does not declare a license. Verify the upstream
DCASE data terms before redistributing converted audio or trained artifacts.

## Download

```bash
hf download gijs/dcase2025-audio-qa \
  --repo-type dataset \
  --local-dir /path/to/dcase2025-audio-qa
```

The converter also accepts a Hugging Face cache snapshot directory whose
`data/` child contains `train-*.parquet`, `test-*.parquet`, and
`eval-*.parquet`.

## Cook

```bash
python datasets/dcase2025_audio_qa/convert_dcase2025_audio_qa_to_unirl.py \
  --snapshot /path/to/dcase2025-audio-qa \
  --out-dir datasets/dcase2025_audio_qa
```

The converter:

1. reads parquet incrementally instead of loading the full dataset into memory;
2. keeps four-choice rows with a labeled A/B/C/D answer;
3. extracts embedded WAV bytes atomically under `audio/<source-split>/`;
4. maps source `train` to `train.jsonl`;
5. maps labeled source `test` to `val.jsonl`;
6. excludes source `eval`, whose answers are not public;
7. writes conversion counts and split provenance to `manifest.json`.

Generated media and JSONL files are local artifacts and should not be committed
to the repository.

## Output layout

```text
datasets/dcase2025_audio_qa/
├── README.md
├── convert_dcase2025_audio_qa_to_unirl.py
├── manifest.json
├── train.jsonl
├── val.jsonl
└── audio/
    ├── train/*.wav
    └── test/*.wav
```

## UniRL format

Each JSONL line has the prompt-first multimodal shape:

```json
{
  "prompt": "Listen to the audio carefully, then answer ...\nA. ...\nB. ...\nC. ...\nD. ...\nReason step by step, then provide the final answer in the exact format: The answer is [X].",
  "prompt_id": "dcase2025:train:train_audio_0000001:1",
  "media_refs": [
    {
      "modality": "audio",
      "role": "prompt",
      "uri": "audio/train/audio_0000001.wav"
    }
  ],
  "metadata": {
    "answer": "B",
    "choices": ["...", "...", "...", "..."],
    "source_split": "train",
    "question_type": "both",
    "subset": "part3"
  }
}
```

Important details:

- `modality: "audio"` selects the standalone-audio path. It is different from
  a video carrying an embedded audio track.
- `role: "prompt"` inserts the audio block into the Qwen3-Omni user message.
- Relative media URIs are resolved against the JSONL directory, keeping a
  cooked dataset relocatable as one directory tree.
- `metadata.answer` is the single uppercase letter consumed by
  `MCExactMatchRewardScorer`.
- The prompt requests `The answer is [X]` because the recipe enables
  `require_answer_phrase: true`.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
QWEN3_OMNI_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=datasets/dcase2025_audio_qa/train.jsonl \
EVAL_DATA_PATH=datasets/dcase2025_audio_qa/val.jsonl \
PYTHONPATH=$(pwd) \
python -m unirl.train_ar \
  --config-name=ar/qwen3_omni_audio_dcase_gspo_lora_vllm_omni_1x4
```

The recipe sets `use_audio_in_video: false` in the bundle, pipeline, and
rollout engine. This flag is required for standalone audio: enabling it asks
the processor to obtain audio from a video input instead.

## Data quality checks

After conversion, inspect `manifest.json` and verify:

- `written_rows` is nonzero for both splits;
- all emitted answers are in `{A, B, C, D}`;
- every referenced WAV exists relative to its JSONL;
- train and validation `prompt_id` values are disjoint.

For GSPO, also monitor `rollout/zero_std_group_ratio`. A prompt whose sampled
rewards are all zero or all one has zero group-normalized advantage and
contributes no policy-gradient signal, even though it still consumes rollout
compute.
