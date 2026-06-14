# Image Editing RL Dataset

Curated instruction-guided image editing prompts for RL training with EditReward.

## Sources

- **OpenGPT-4o-Image**: [WINDop/OpenGPT-4o-Image](https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image)
  - Subset: `editing/` (instruction-guided image editing pairs)
- **ShareGPT-4o-Image**: [FreedomIntelligence/ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)
  - Subset: image-to-image editing samples

## Format

Each line is a JSON object:

```json
{
  "prompt": "editing instruction text",
  "media": [{"modality": "image", "role": "condition", "uri": "<path_to_source_image>"}],
  "metadata": {"source": "opengpt4o_image_editing_scored", "edit_reward_score": 1.52, "global_id": "opengpt:8670"}
}
```

- `prompt`: The editing instruction (what to change in the image).
- `media[0].uri`: Path to the **source image** (before editing). Set `role: "condition"` so UniRL's `MultimodalRLDataSource` loads it as a conditioning input.
- `metadata.source`: Which dataset the sample came from.
- `metadata.edit_reward_score`: Pre-computed EditReward score of the original dataset's edited output (for reference; not used during RL training).
- `metadata.global_id`: Unique identifier for the sample.

## Usage

In a UniRL training YAML:

```yaml
data_source:
  _target_: unirl.data.data_source.MultimodalRLDataSource
  args:
    run:
      data_path: datasets/image_edit/train.jsonl
      eval_data_path: datasets/image_edit/test.jsonl
      seed: 42
    algorithm:
      prompts_per_rollout: ${batch_size}
```

## Splits

- `train.jsonl`: 19,000 samples
- `test.jsonl`: 1,000 samples

## Processing

The dataset was assembled by:
1. Selecting editing-scored samples from both sources.
2. Remapping image paths to local storage.
3. Each sample provides only the **source image** and **editing instruction** — during RL training the diffusion model generates its own edited images, which are then scored by EditReward.

## Notes

- Image paths in the JSONL must be updated to match your local storage layout.
- The example JSONL files use placeholder paths (`data/OpenGPT-4o-Image/...`, `data/ShareGPT-4o-Image/...`). Replace with your actual download location.
