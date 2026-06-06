"""Shared CPU fakes for the ``sglang_v2`` suite — no sglang, no GPU, no network.

``RawResult`` is a structural protocol, so canned results are
``SimpleNamespace``s; the tokenizer/processor are injected at adapter
construction, so plain stubs stand in; the ``Backend`` protocol is faked by a
recorder. Everything here exercises the package exactly the way the engine
wires it, minus the runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from unirl.rollout.engine.sglang_v2.config import SGLangV2EngineConfig


def make_raw(
    text: str = "hello",
    token_ids: Optional[List[int]] = None,
    logprobs: Optional[List[float]] = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """One canned wire result satisfying the ``RawResult`` protocol."""
    if token_ids is None:
        token_ids = [1, 2, 3]
    if logprobs is None:
        logprobs = [-0.1] * len(token_ids)
    return SimpleNamespace(text=text, token_ids=token_ids, logprobs=logprobs, finish_reason=finish_reason)


class StubTokenizer:
    """Chat-template tokenizer stub: ids are deterministic from the messages."""

    pad_token_id = 7
    eos_token_id = 9
    chat_template = "stub-template"  # templated mode; raw-mode tests set None

    def __init__(self, fail_template: bool = False):
        self.fail_template = fail_template
        self.template_calls: List[Dict[str, Any]] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        self.template_calls.append({"messages": messages, "kwargs": kwargs})
        if self.fail_template:
            raise ValueError("no chat template")
        # One token per message + a length marker, so system-vs-no-system and
        # different prompts produce different ids.
        flat = "|".join(str(m["content"]) for m in messages)
        return [len(messages), len(flat) % 97, 42]

    def encode(self, text: str) -> List[int]:
        return [len(text) % 97, 5]

    def decode(self, ids, **kwargs) -> str:
        return f"<{len(list(ids))} tokens>"


class StubProcessor:
    """VLM processor stub mirroring the HF AutoProcessor call shapes."""

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize) -> str:
        assert tokenize is False
        return "templated:" + "|".join(
            c["text"] for m in messages if isinstance(m["content"], list) for c in m["content"] if c["type"] == "text"
        )

    def __call__(self, *, text, images, return_tensors):
        import torch

        assert return_tensors == "pt"
        n = len(text[0])
        return {
            # EXPANDED ids — longer than any tokenizer encoding, like the real
            # processor's vision-token expansion.
            "input_ids": torch.arange(n % 13 + 8, dtype=torch.long).unsqueeze(0),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class RecordingBackend:
    """A ``Backend`` fake recording every verb call in order."""

    def __init__(self, generate_results: Optional[List[Any]] = None):
        self.calls: List[tuple] = []
        self._generate_results = generate_results if generate_results is not None else []

    def _record(self, name: str, **kwargs):
        self.calls.append((name, kwargs))

    def names(self) -> List[str]:
        return [name for name, _ in self.calls]

    # generation
    def generate(self, requests):
        self._record("generate", requests=requests)
        return list(self._generate_results)

    # memory / lifecycle / health
    def flush_cache(self):
        self._record("flush_cache")

    def release_memory(self, *, tags=None):
        self._record("release_memory", tags=tags)

    def resume_memory(self, *, tags=None):
        self._record("resume_memory", tags=tags)

    def shutdown(self):
        self._record("shutdown")

    def ping(self):
        self._record("ping")
        return True

    # weight sync
    def update_from_tensor(self, **kwargs):
        self._record("update_from_tensor", **kwargs)

    def init_weights_group(self, **kwargs):
        self._record("init_weights_group", **kwargs)

    def update_from_distributed(self, **kwargs):
        self._record("update_from_distributed", **kwargs)

    def destroy_weights_group(self, **kwargs):
        self._record("destroy_weights_group", **kwargs)

    def set_lora(self, **kwargs):
        self._record("set_lora", **kwargs)


@pytest.fixture
def text_config() -> SGLangV2EngineConfig:
    return SGLangV2EngineConfig(pretrained_model_ckpt_path="stub/model")


@pytest.fixture
def stub_tokenizer() -> StubTokenizer:
    return StubTokenizer()
