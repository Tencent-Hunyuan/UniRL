# ReFL VideoAlign T2V prompts (refl_videoalign)

Human-motion / dialogue video captions used as the T2V prompt set for the ReFL VideoAlign
experiment. Committed in full — the whole set is 677 prompts of plain text, so there is no
converter and nothing to cook.

- Recipe: [`experimental/refl/examples/wan21_t2v_videoalign_refl.yaml`](../../experimental/refl/examples/wan21_t2v_videoalign_refl.yaml)
- Reward: `experimental.refl.reward.videoalign.VideoAlignRewardScorer` (Qwen2-VL VQ/MQ/TA, differentiable)
- Loader contract: [`unirl/data/data_source.py`](../../unirl/data/data_source.py) + [`unirl/data/datasets.py`](../../unirl/data/datasets.py)

## Source

This prompt set was internally constructed by the Tencent Hunyuan team.

## Format

One prompt per line, plain UTF-8 text (ASCII in practice), no header and no blank lines —
the same shape as [`ocr/`](../ocr) and [`pickscore/`](../pickscore), read by
`MultimodalRLDataSource` as a bare prompt list with no media and no metadata.

```
A man in a suit looks down thoughtfully before looking up slightly while speaking.
```

Prompts run 67–166 characters, median 99.

## Splits

| File | Prompts |
|---|---|
| `train.txt` | 653 |
| `test.txt` | 24 |

The committed files define the dataset split. `test.txt` is sized at 24 = 3 full rollouts at
the recipe's `batch_size: 8`, deliberately small because an 81-frame 480x832 generation per
prompt makes held-out scoring expensive.

Note that `experimental/refl` does not currently run periodic eval (see the "Deliberately not
ported" section of [`experimental/refl/README.md`](../../experimental/refl/README.md)), so
`test.txt` is a held-out set waiting on an eval loop rather than one the recipe scores today.

## Train

`train.txt` and `test.txt` are the recipe defaults, so no `DATA_PATH` is needed:

```bash
export PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers \
       VIDEOALIGN_MODEL_PATH=/path/to/VideoReward
RAY_ADDRESS=auto python -m experimental.refl.run \
  --config-name=wan21_t2v_videoalign_refl num_devices=8
```

`DATA_PATH` / `EVAL_DATA_PATH` still override them for a different prompt set.
