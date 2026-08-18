# Datasets

Offline data preparation: one folder per dataset, each holding its converter CLI, a README
with the source and the exact commands, and any prompt set small enough to commit.

**Nothing here is imported by the framework.** These are standalone scripts you run once,
before training, to turn a public dataset into the local jsonl/manifest layout the runtime
readers consume. The runtime side — data sources, dataset readers, the supervised manifest
contract — lives in [`unirl/data/`](../unirl/data) and is the only half that ships in the
package.

| Kind | Home |
|---|---|
| One-off dataset download/conversion CLI | `datasets/<dataset>/` |
| Runtime data source / reader / manifest contract | `unirl/data/` |
| Downloaded snapshots and generated manifests | local only — never committed |

## Adding a dataset

Create `datasets/<name>/` with the converter and a `README.md` covering: the upstream source
(HF id + link, and any license or gating caveat), the download step, the cook command with its
output layout, and the recipe that trains on it. [`geo3k_mc`](geo3k_mc/README.md) is a short
template; [`searchgen`](searchgen/README.md) is a thorough one.

Converters are run by path from the repo root, not as modules:

```bash
python datasets/<name>/<converter>.py --out-dir data/<name>
```

A converter may import `unirl` (e.g. to validate rows through
`unirl.data.sft.normalize_supervised_example` at the dataset boundary rather than mid-training);
install the package first. The dependency only ever points that way — `unirl/` must not import
anything under `datasets/`.

Converter code and READMEs are tracked by default; no `.gitignore` allowlist entry is needed
for a new folder. Downloaded and generated data is ignored: keep it in `raw/` or `processed/`
under your dataset folder, or write it to the repo-root `data/` that the converters default to.

## Contents

| Folder | What |
|---|---|
| [`arxivqa_mc/`](arxivqa_mc/README.md) | ArxivQA scientific-figure multiple-choice (BAGEL GRPO) |
| [`asearcher/`](asearcher/README.md) | ASearcher deep-research prompts (agentic RL) |
| [`daily_omni_av/`](daily_omni_av/README.md) | Daily-Omni audio-video QA |
| [`dapo_math/`](dapo_math/README.md) | DAPO-Math-17k + AIME 2024/2025 (AR math RL) |
| [`dcase2025_audio_qa/`](dcase2025_audio_qa/README.md) | DCASE 2025 audio QA |
| [`droid100/`](droid100/README.md) | LeRobot DROID-100 → Cosmos3 SFT debug samples |
| [`geo3k_mc/`](geo3k_mc/README.md) | Geometry3K multiple-choice (VLM GRPO) |
| [`searchgen/`](searchgen/README.md) | SearchGen interleaved image/text agent SFT |
| [`sft_manifests/`](sft_manifests/README.md) | Generic text / VLM / T2I / agent SFT manifest builders |
| [`ucf101/`](ucf101/README.md) | UCF101 T2V SFT |
| [`video_r1_260k/`](video_r1_260k/README.md) | Video-R1-260k video reasoning |
| `geneval/`, `image_edit/`, `ocr/`, `pickscore/` | Committed prompt sets (no download step) |
