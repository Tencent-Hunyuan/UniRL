# ASearcher deep-research (asearcher)

Open-ended research questions with short verifiable answers, converted into the local
`{"prompt", "metadata": {"answer"}}` jsonl the agentic trainer reads.

- Recipe: [`examples/deep_research/deep_research_search_judge.yaml`](../../examples/deep_research/deep_research_search_judge.yaml)
- Converter: [`prepare_asearcher.py`](prepare_asearcher.py)
- Loader contract: [`unirl/data/data_source.py`](../../unirl/data/data_source.py)

Generated jsonl is a local artifact and must not be committed.

## Source

[`inclusionAI/ASearcher-train-data`](https://huggingface.co/datasets/inclusionAI/ASearcher-train-data),
split `ASearcherBase35k`. Streamed from the Hub through `datasets.load_dataset`; set
`HF_ENDPOINT` for a mirror. Pass `--source <file.jsonl>` to convert a local dump instead of
downloading.

Question and answer columns are matched by name, so an equivalent dump works unchanged:
question from `question` / `prompt` / `query`, answer from `answer` / `gt` /
`golden_answers` / `ground_truth` / `label`. Rows missing either are dropped.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)) unless you pass `--source` and skip the download.

```bash
python datasets/asearcher/prepare_asearcher.py --out-dir data/asearcher
```

Writes `train.jsonl` under `--out-dir`. `--limit N` caps kept rows for smoke runs (0 = all).

## Train

The recipe also needs live search/judge credentials — see its header for the full list.

```bash
QWEN3_INSTRUCT_PATH=... DATA_PATH=data/asearcher/train.jsonl \
SERPER_KEY_ID=... JINA_API_KEYS=... JUDGE_URL=... JUDGE_MODEL=... \
python -m unirl.train_agentic --config-name=deep_research/deep_research_search_judge num_devices=2
```
