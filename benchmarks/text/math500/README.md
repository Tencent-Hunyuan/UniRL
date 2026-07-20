# text/math500

[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (MIT, vendored in
`data/`): the 500-problem PRM800K test split of Hendrycks MATH.

Protocol: avg@4, temperature 0.6, top-p 0.95; graded with math-verify.

```bash
python -m benchmarks.run -b text/math500 --endpoint http://127.0.0.1:30000
```
