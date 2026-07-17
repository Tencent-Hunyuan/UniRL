# text/aime24 · text/aime25

AIME competition math, 30 problems per year. `data/aime2024.jsonl` is vendored
(upstream HF card: MIT); AIME 2025's card declares no license, so fetch it instead:
`python benchmarks/text/aime/fetch_aime2025.py`.

Protocol: avg@16, temperature 0.6, top-p 0.95, **32768 generation tokens** — the
server's context window must cover prompt + 32768 or every request 400s (Qwen3's
native 40960 fits; don't cap `--max-model-len` lower). Answers graded with
[math-verify](https://github.com/huggingface/Math-Verify) (accepts unboxed final
answers). Serve the checkpoint first, e.g.
`python -m sglang.launch_server --model-path <ckpt> --port 30000`, then:

```bash
python -m benchmarks.run -b text/aime24,text/aime25 --endpoint http://127.0.0.1:30000
```
