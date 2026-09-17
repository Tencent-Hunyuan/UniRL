# DyRef / OmniRef training data

Converts the official [DyRef training metadata](https://huggingface.co/datasets/Eason0438/OmniRef-training)
into the raw-media UniRL manifest shared by diffusion SFT and RL.

Download `train_data.json`, `data_2500.zip`, and `data_4500.zip`, then extract both archives
under one local data root. The upstream files are Apache-2.0 licensed and total about 23 GB
compressed. Generated manifests and downloaded media are local artifacts and must not be
committed.

```bash
python datasets/dyref/prepare_dyref.py \
  --input /local/OmniRef-training/train_data.json \
  --data-root /local/OmniRef-training \
  --out-dir data/dyref
```

Use `--val-input` with the official DyRef RL `test.jsonl` and `--val-data-root` with an
extracted [OmniRef-Bench](https://huggingface.co/datasets/Eason0438/OmniRef-Bench) to keep
the benchmark as a separate validation source. Without it, the converter makes a
deterministic 5% validation split.

FLUX.2 requires uniform dense reference slots. Cook one count into a separate directory:

```bash
python datasets/dyref/prepare_dyref.py \
  --input /local/OmniRef-training/train_data.json \
  --data-root /local/OmniRef-training \
  --reference-count 3 \
  --out-dir data/dyref_n3
```

Each output row keeps every `edit_image` entry as an ordered
`{"modality":"image","role":"condition"}` media ref and appends the clean `image` as one
separate `role="target"` ref. `MultimodalRLDataSource` materializes only the condition refs
as `ImageSets`; it exposes the target to rewards through reserved metadata, so the clean
target cannot leak into model conditioning. `DiffusionSupervisedTrackBuilder` consumes the
same row by encoding the condition set and target through separate model paths.

The converter records reference count, ordered subject/background/style/lighting/pose
types, and the official `style/reference` position under `metadata.dyref`. DyRef's
SigLIPv2/CSD reward uses those fields for DRS, DAR, and style-reference selection.
