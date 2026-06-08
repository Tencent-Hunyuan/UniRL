# Vendored VideoAlign Inference Subset

This directory contains a **minimal vendored slice** of the upstream
[VideoAlign](https://github.com/KwaiVGI/VideoAlign) repo, just enough to
run the VideoReward Qwen2-VL-2B reward model in **inference mode** inside
this service.

## Provenance

- **Upstream**: https://github.com/KwaiVGI/VideoAlign
- **License**: MIT (LICENSE preserved at the upstream repo root)
- **Vendored at**: 2026-05 (no specific upstream commit pinned; matches
  the `main` branch contents around the 2025-08-14 prompt-set update)
- **Local source**: `github_repo/VideoAlign/` in this workspace

## Why vendor instead of pip install

VideoAlign is **not** packaged on PyPI and the upstream entry points
(`inference.py`, `train_reward.py`, `trainer.py`, …) sit at the repo
root with mutual top-level imports (`from data import …`, `from utils
import …`). Installing the repo as a Python package would require
either patching every import path or shipping a `setup.py` upstream
doesn't have. Vendoring the inference subset under
`reward_service.scorers._videoalign.*` is the simplest way to:

1. Get a stable, importable Python package with absolute imports
2. Strip training-only code (trl Trainer, deepspeed helpers, dataset
   collators, GSB-CSV converters, …) that pulls in heavy deps we don't
   need at inference time
3. Avoid `sys.path` hacks at scorer-actor startup time

## What's vendored vs what's stripped

| File here | Upstream origin | Notes |
|---|---|---|
| `configs.py` | `data.DataConfig` + `utils.{ModelConfig,PEFTLoraConfig}` + `inference.load_configs_from_json` | Stripped: `TrainingConfig` (subclass of `transformers.TrainingArguments`); replaced by a small `MinimalTrainingArgs` dataclass with only the fields the inference path reads. |
| `prompt_template.py` | `prompt_template.py` | **Verbatim** copy. |
| `vision_process.py` | `vision_process.py` | **Verbatim** copy. Reads videos via `decord` (or `torchvision` fallback) and packs them into `(T, C, H, W)` tensors. |
| `model.py` | `trainer.Qwen2VLRewardModelBT` | Kept the model class (`__init__` + `forward`). Stripped: `_convert_A_B_to_chosen_rejected`, `PartialEmbeddingUpdateCallback`, the entire `VideoVLMRewardTrainer` class. |
| `checkpoint.py` | `utils.load_model_from_checkpoint` + `inference.load_configs_from_json` | Stripped: `maybe_zero_3`, `get_peft_state_*` (deepspeed/zero-3 helpers used only during training). |
| `builder.py` | `train_reward.create_model_and_processor` | Stripped: TRL's `get_kbit_device_map` / `get_quantization_config` (inference assumes no quantization unless `load_in_8bit/4bit` is set in `ModelConfig`). LoRA path is preserved. |
| `inferencer.py` | `inference.VideoVLMRewardInference` (class renamed `VideoRewardInferencer`) | Stripped `pdb`, dead imports. Kept the prepare-batch / forward / score-normalisation path. |
| `__init__.py` | new | Re-exports `VideoRewardInferencer` + the dataclasses needed to parse a checkpoint's `model_config.json`. |

## How to update against upstream

1. Pull the latest VideoAlign into a separate clone.
2. Diff the relevant upstream files against this directory.
3. Reapply only the **inference-relevant** changes; ignore changes that
   touch the trainer / data collator / dataset converters.
4. Run `pytest tests/scorers/test_videoalign.py` and the GPU smoke test
   to verify nothing in the inference path regressed.

## Why no `pdb` imports

The upstream code includes leftover `import pdb` calls and
`pdb.set_trace()` checkpoints. Those are removed here — they leak into
the import graph of long-running services and add useless module-load
side effects.
