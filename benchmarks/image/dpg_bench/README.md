# image/dpg_bench

[DPG-Bench](https://github.com/TencentQQGYLab/ELLA) (ELLA, Apache-2.0): 1065 dense
paragraph-length prompts. The 7.4 MB prompts+questions CSV is fetched, not vendored:

```bash
bash benchmarks/image/dpg_bench/fetch.sh
python -m benchmarks.run -b image/dpg_bench --ckpt <base> [--lora <ckpt>]   # generate only
```

Scoring is external (official protocol): tile each prompt's 4 images into a 2×2 grid
named `<item_id>.png`, then run the ELLA repo's `dpg_bench/dist_eval.sh` (mPLUG-large
VQA judge; DPG score = mean accuracy × 100). Our prompt index `p` maps to `item_id` via
the CSV's unique-`text` order — the same first-seen order `run.py` generates in:

```python
import csv, pathlib
from PIL import Image

src = pathlib.Path("benchmarks_results/<ckpt-tag>/image_dpg_bench/images")
out = pathlib.Path("dpg_grids"); out.mkdir(exist_ok=True)
ids: dict = {}  # text -> item_id, first-seen order == prompt index p (matches load_prompts)
for r in csv.DictReader(open("benchmarks/image/dpg_bench/data/dpg_bench.csv")):
    ids.setdefault(r["text"], r["item_id"])
for p, item_id in enumerate(ids.values()):
    tiles = [Image.open(src / f"p{p:05d}_s{s}.png") for s in range(4)]
    w, h = tiles[0].size
    grid = Image.new("RGB", (2 * w, 2 * h))
    for s, tile in enumerate(tiles):
        grid.paste(tile, (w * (s % 2), h * (s // 2)))
    grid.save(out / f"{item_id}.png")
```
