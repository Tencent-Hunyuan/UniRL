# text/gpqa

GPQA-Diamond (198 graduate-level MC questions, CC BY 4.0). The dataset is **gated and
carries a canary string — never commit it**. Accept the terms at
[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa), then:

```bash
HF_TOKEN=<token> python benchmarks/text/gpqa/fetch.py   # writes data/gpqa_diamond.jsonl (git-ignored)
python -m benchmarks.run -b text/gpqa --endpoint http://127.0.0.1:30000
```

Protocol: 4 options shuffled with a fixed per-question seed; avg@4, temperature 0.6;
graded by extracting the final answer letter (`\boxed{X}` or "the answer is (X)", the
official baseline's format).
