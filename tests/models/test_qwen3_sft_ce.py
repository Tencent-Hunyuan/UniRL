"""CPU tests for the Qwen3 SFT task's loss assembly (no checkpoints, no GPU).

We don't load Qwen3-4B; instead we duck-type the bundle so ``compute_loss``'s
CE assembly is exercised with tiny fake tensors. The reused
``_replay_aware_forward`` (which returns per-token log-probs) is stubbed by a
fake transformer that returns a known logp tensor, so we assert the task turns
per-token logp into ``-mean(logp)`` and that the record→token plumbing is right.
"""

import types

import pytest
import torch

from unirl.models.qwen3.sft_task import Qwen3SFTTask


class _FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, return_dict=False):
        # Deterministic: one token per character of the user content, + a marker.
        # `return_dict` mirrors the transformers-5.x kwarg the task passes
        # (Qwen3SFTTask.load_record uses return_dict=False -> flat id list).
        content = messages[0]["content"]
        return [1] + [ord(c) % 50 for c in content]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i % 26) + 97) for i in ids)


class _FakeTransformer:
    """Records the forward kwargs and returns a fixed per-token logp [1, T]."""

    def __init__(self, logp_row):
        self._logp_row = logp_row
        self.last_kwargs = None

    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        t = int(kwargs["response_tokens"].shape[1])
        return self._logp_row[:, :t]


def _make_task(logp_row):
    task = Qwen3SFTTask.__new__(Qwen3SFTTask)  # bypass __init__ (no real bundle)
    transformer = _FakeTransformer(logp_row)
    bundle = types.SimpleNamespace(transformer=transformer, tokenizer=_FakeTokenizer(), device=torch.device("cpu"))
    task.bundle = bundle
    task.config = types.SimpleNamespace(autocast_precision="bf16")
    task.tokenizer = bundle.tokenizer
    task.autocast_dtype = torch.bfloat16
    return task


def test_load_record_builds_prompt_and_response_ids():
    task = _make_task(torch.zeros(1, 8))
    loaded = task.load_record({"prompt": "hi", "response": "yo"})
    assert loaded["prompt_ids"][0] == 1  # chat-template marker
    # response ids end with the EOS the task appends.
    assert loaded["response_ids"][-1] == _FakeTokenizer.eos_token_id


def test_load_record_rejects_non_string():
    task = _make_task(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        task.load_record({"prompt": 123, "response": "yo"})


def test_compute_loss_is_negative_mean_logp():
    # Known per-token log-probs -> loss must equal -mean(logp).
    logp_row = torch.tensor([[-0.5, -1.5, -2.0]])
    task = _make_task(logp_row)
    loaded = task.load_record({"prompt": "ab", "response": "cd"})
    loss, metrics = task.compute_loss(loaded)
    # response = 2 chars + EOS -> 3 response tokens; fake returns first 3 cols.
    expected = -logp_row.mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert metrics["loss/total"] == pytest.approx(float(expected))
    assert metrics["train/ppl"] == pytest.approx(float(torch.exp(expected)))


def test_compute_loss_passes_prompt_len_and_response_tokens():
    task = _make_task(torch.zeros(1, 16))
    loaded = task.load_record({"prompt": "hello", "response": "world"})
    task.compute_loss(loaded)
    kw = task.bundle.transformer.last_kwargs
    # prompt_len must equal the tokenized prompt length; full_ids = prompt+resp.
    assert kw["prompt_len"] == len(loaded["prompt_ids"])
    assert kw["input_ids"].shape[1] == len(loaded["prompt_ids"]) + len(loaded["response_ids"])
    assert kw["response_tokens"].shape[1] == len(loaded["response_ids"])
    assert kw["temperature"] == 1.0


def test_compute_loss_rejects_empty():
    task = _make_task(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        task.compute_loss({"prompt_ids": [], "response_ids": [5]})
