"""Convert this repo's ``datasets/pickscore/{train,test}.txt`` into verl-omni parquet.

Both sides of the speed pair must consume the same prompts; verl-omni's loader
wants verl's parquet schema (mirrors its ``examples/flowgrpo_trainer/data_process/``).
For the PickScore reward, ``reward_model.ground_truth`` carries the prompt text —
``compute_score_pickscore`` scores (ground_truth, image) pairs.

    python benchmarks/speed_benchmarks/verl_omni/make_pickscore_parquet.py [--out ~/data/pickscore_sd3]
"""

import argparse
from pathlib import Path

import datasets

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "datasets" / "pickscore"


def build(split: str, filename: str) -> datasets.Dataset:
    prompts = [line.strip() for line in (SRC / filename).read_text().splitlines() if line.strip()]
    return datasets.Dataset.from_list(
        [
            {
                "data_source": "unirl/pickscore",
                "prompt": [{"role": "user", "content": p}],
                "ability": "t2i",
                "reward_model": {"style": "model", "ground_truth": p},
                "extra_info": {"split": split, "index": i, "raw_prompt": p},
            }
            for i, p in enumerate(prompts)
        ]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/pickscore_sd3")
    out = Path(ap.parse_args().out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train, test = build("train", "train.txt"), build("test", "test.txt")
    train.to_parquet(str(out / "train.parquet"))
    test.to_parquet(str(out / "test.parquet"))
    print(f"wrote {len(train)} train / {len(test)} test to {out}")
