# UCF-101 T2V SFT Dataset

This directory contains the data preparation code for the 18-class UCF-101
sports/action subset used by the WAN2.1 text-to-video full-transformer SFT
recipe:

- Recipe: [`examples/sft/wan21_t2v_ucf101_full.yaml`](../../examples/sft/wan21_t2v_ucf101_full.yaml)
- Cooking script: [`prepare_t2v_sft.py`](prepare_t2v_sft.py)
- Download mirror: [`quchenyuan/UCF101-ZIP`](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP)

Raw videos and generated manifests are local artifacts and must not be
committed. The repository `.gitignore` excludes `raw/` and `processed/`.

Install UniRL using one of the supported environments in
[`INSTALL.md`](../../INSTALL.md). For example, a supported vLLM/CUDA 13.0
training environment can be installed with:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -e ".[vllm,train,infer]" --prerelease=allow
```

Both supported engine extras (`vllm` and `sglang`) install PyAV for raw video
decoding, so no separate `av` installation or runtime selection is needed.

## 1. Download and place the data

Run the following commands from the UniRL repository root. The `hf` command is
provided by the `huggingface_hub` package.

```bash
mkdir -p datasets/ucf101/raw

hf download quchenyuan/UCF101-ZIP UCF-101.zip \
  --repo-type dataset \
  --local-dir datasets/ucf101/raw

unzip -q -o datasets/ucf101/raw/UCF-101.zip \
  -d datasets/ucf101/raw
```

The extracted videos should have this layout:

```text
datasets/ucf101/
├── README.md
├── prepare_t2v_sft.py
└── raw/
    ├── UCF-101.zip
    └── UCF-101/
        ├── Archery/*.avi
        ├── Basketball/*.avi
        ├── BasketballDunk/*.avi
        └── ...
```

UCF-101 is an external dataset. Review its upstream license and terms before
using or redistributing the videos.

## 2. Cook the UniRL manifests

```bash
python datasets/ucf101/prepare_t2v_sft.py \
  --data-root datasets/ucf101/raw/UCF-101 \
  --out-dir datasets/ucf101/processed
```

By default, the script:

1. selects the 18 sports/action classes used by the recipe;
2. generates a simple caption from each class name;
3. creates a deterministic, class-stratified split with seed `42`;
4. caps validation at 128 videos;
5. writes absolute target-video paths into UniRL SFT JSONL manifests.

The resulting layout is:

```text
datasets/ucf101/processed/
├── train.jsonl
└── val.jsonl
```

Each row has the supervised target-video format:

```json
{
  "sample_id": "v_Archery_g01_c01",
  "prompt": "a person is archery",
  "media": [
    {
      "modality": "video",
      "role": "target",
      "uri": "/absolute/path/to/UCF-101/Archery/v_Archery_g01_c01.avi"
    }
  ],
  "metadata": {
    "source": "UCF-101",
    "ucf101_class": "Archery"
  }
}
```

Because media paths are absolute, rerun the cooking script if the extracted
video directory moves. Use `python datasets/ucf101/prepare_t2v_sft.py --help`
to see options for classes, validation fraction, validation cap, and seed.

## 3. Check the cooked data

The default source release should produce 2,367 training rows and 128
validation rows:

```bash
wc -l \
  datasets/ucf101/processed/train.jsonl \
  datasets/ucf101/processed/val.jsonl
```

## 4. Run WAN2.1 SFT

The training command and all model/training settings live in
[`examples/sft/wan21_t2v_ucf101_full.yaml`](../../examples/sft/wan21_t2v_ucf101_full.yaml).
Launch it on one 8-GPU node with:

```bash
ENTRY=train_sft \
SFT_DATA=datasets/ucf101/processed/train.jsonl \
SFT_EVAL_DATA=datasets/ucf101/processed/val.jsonl \
OUTPUT_DIR=outputs/wan21_t2v_ucf101_full \
bash examples/run_experiment_single_node.sh sft/wan21_t2v_ucf101_full
```

The recipe defaults to
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`. To use a local/shared checkpoint, add:

```bash
export PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers
```

Before launching a training job, the fully resolved config can be checked
without allocating GPUs:

```bash
SFT_DATA=datasets/ucf101/processed/train.jsonl \
SFT_EVAL_DATA=datasets/ucf101/processed/val.jsonl \
python -m unirl.train_sft \
  --config-name=sft/wan21_t2v_ucf101_full \
  --cfg job --resolve
```
