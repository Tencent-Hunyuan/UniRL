# DROID-100 → Cosmos3 SFT debug samples (droid100)

LeRobot-v3 robot episodes decoded into pre-materialized frame/action tensors for the two
Cosmos3 SFT recipes (video prediction and action behaviour cloning). This is a **debug-scale**
dataset: it exists to make the recipes runnable end to end, not to reproduce a result.

- Recipes: [`examples/sft/cosmos3_droid100_videopred.yaml`](../../examples/sft/cosmos3_droid100_videopred.yaml),
  [`examples/sft/cosmos3_droid100_action_bc.yaml`](../../examples/sft/cosmos3_droid100_action_bc.yaml)
- Converter: [`prepare_droid100.py`](prepare_droid100.py)
- **Output layout, action-dim and normalization caveats:**
  [`unirl/models/cosmos3/README.md`](../../unirl/models/cosmos3/README.md) — read that before
  changing any flag; it owns the contract this script emits against.

Decoded tensors and manifests are local artifacts and must not be committed.

## Source

[`lerobot/droid_100`](https://huggingface.co/datasets/lerobot/droid_100) — 100 DROID/Franka
single-arm episodes in LeRobot v3 format. Downloaded automatically through
`huggingface_hub.snapshot_download`; pass `--hf-local-dir` to reuse an existing snapshot and
`--repo` to point at another LeRobot-v3 dataset.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)) — this one reads LeRobot's parquet metadata with
`pandas` and decodes video with `av`. Both are required lazily, so a missing one exits with
the package to install rather than a traceback.

**Also needs torch**, which `dataset-prep` does not carry (torch enters through an engine
extra so each engine can pin its own CUDA stack). This prep resizes clips and writes `.pt`
tensors, so in a bare `dataset-prep` venv it fails at import with
`ModuleNotFoundError: No module named 'torch'`. Any engine extra supplies it; `pip install
torch` is enough for CPU-only prep. It is the only converter here with this requirement.

```bash
python datasets/droid100/prepare_droid100.py --root datasets/droid100_debug
```

Writes `frames/`, `actions/`, `manifest.jsonl`, `eval_manifest.jsonl` and `stats.json` under
`--root`. Frames are decoded once here so training never touches AV1/mp4. The default
192×320 canvas is the Cosmos3 resolution-tier-256 bin — changing it desyncs training-time
encoding from tier-based action inference.

`--root` defaults to `datasets/droid100_debug`, which stays untracked via the repo-wide
`*debug*/` ignore rule.

## Train

`SFT_DATA` / `SFT_EVAL_DATA` default to this prep's output.

```bash
ENTRY=train_sft PRETRAINED_MODEL=/path/to/Cosmos3-Nano \
bash examples/run_experiment_single_node.sh sft/cosmos3_droid100_videopred
```

Swap `videopred` for `action_bc` to train the action head instead.
