# Cosmos3 supervised finetuning

The Cosmos3 recipes use UniRL's current shared SFT path:

`SupervisedDataSource → Cosmos3SupervisedTrackBuilder → RolloutTrack → TrainStack → Cosmos3JointFlowMatchSFT`

The model-specific code is limited to the Cosmos3 bundle, packed joint stage,
record-to-track builder, and joint video/action flow-matching loss. Checkpoint,
optimizer, gradient accumulation, evaluation loss, and resume semantics remain
the same as the other SFT recipes.

## Requirements

- `diffusers>=0.39` (install the `cosmos3` extra).
- A diffusers-layout `nvidia/Cosmos3-Nano` checkpoint.
- Eight GPUs for the supplied full-finetune recipes.

## Data and launch

```bash
pip install -e ".[train,cosmos3]"
python -m unirl.utils.prepare_droid100 --root datasets/droid100_debug

PRETRAINED_MODEL=/path/to/Cosmos3-Nano \
  bash examples/run_experiment_single_node.sh sft/cosmos3_droid100_videopred

PRETRAINED_MODEL=/path/to/Cosmos3-Nano \
  bash examples/run_experiment_single_node.sh sft/cosmos3_droid100_action_bc
```

The generated manifests use the standard supervised schema. Each row has a
`prompt`, one `video/target` tensor media reference, one `action/target`
reference, and `fps` metadata. Frames are uint8 `[T,3,H,W]`; actions are
float32 `[T-1,D]`.

## Numeric conventions

- The transformer is stored in fp32 for a stable optimizer master and gathered
  as bf16 by FSDP for compute.
- Sigma is logit-normal and uses the official short-edge shift tiers:
  256 → 3, 480 → 5, 720 → 10.
- Action BC keeps latent frame 0 clean and jointly noises future video and the
  action chunk with one sigma.
- The action recipe weights vision and action MSE equally (10 and 10).

`droid_100` is only a debug dataset: its collapsed 7-D action differs from the
canonical Cosmos3 DROID action layout and it has no proprioceptive stream.
