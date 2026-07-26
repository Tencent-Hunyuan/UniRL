import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from unirl.train.sft.track_builder import ARSupervisedTrackBuilder
from unirl.utils import prepare_sft_agent


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": [1, 2, 3]}

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        add_generation_prompt,
        enable_thinking,
        tokenize,
        return_dict,
        truncation,
    ):
        del tools, enable_thinking, tokenize, return_dict, truncation
        prompt_ids = [10, 11]
        if add_generation_prompt:
            return prompt_ids
        target_length = int(messages[-1].get("_token_count", 2))
        return prompt_ids + [20] * (target_length - 1) + [self.eos_token_id]


class _ChatStage:
    enable_thinking = False

    def embed(self, texts):
        return texts


def _builder(*, append_eos: bool, max_response_length: int) -> ARSupervisedTrackBuilder:
    bundle = SimpleNamespace(tokenizer=_Tokenizer(), device=torch.device("cpu"))
    pipeline = SimpleNamespace(bundle=bundle, chat_template=_ChatStage())
    return ARSupervisedTrackBuilder(
        pipeline=pipeline,
        append_eos=append_eos,
        max_response_length=max_response_length,
    )


def _agent_example(*, sample_id: str, target_length: int, tool_call: bool = False):
    assistant = {
        "role": "assistant",
        "content": None if tool_call else "done",
        "_token_count": target_length,
    }
    if tool_call:
        assistant["tool_calls"] = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "user", "content": "help"},
            assistant,
        ],
    }


def test_overlong_tool_call_is_filtered_and_never_truncated() -> None:
    long_tool_call = _agent_example(sample_id="long", target_length=5, tool_call=True)
    short_answer = _agent_example(sample_id="short", target_length=2)

    kept, dropped = prepare_sft_agent._filter_overlong_targets(
        [long_tool_call, short_answer],
        tokenizer=_Tokenizer(),
        max_response_length=4,
        enable_thinking=False,
    )

    assert kept == [short_answer]
    assert dropped == {"tool_call": 1, "final_answer": 0}
    with pytest.raises(ValueError, match="filter overlong agent targets"):
        _builder(append_eos=True, max_response_length=4)._tokenize_responses([long_tool_call])


def test_append_eos_false_preserves_last_retained_token() -> None:
    tokens, _ = _builder(append_eos=False, max_response_length=2)._tokenize_responses(
        [{"sample_id": "legacy", "response": "three tokens"}]
    )

    assert tokens[0].tolist() == [1, 2]


def test_recipe_defaults_match_preparation_outputs(monkeypatch) -> None:
    monkeypatch.delenv("SFT_DATA", raising=False)
    monkeypatch.delenv("SFT_EVAL_DATA", raising=False)
    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "examples/sft/validation/qwen3_agent_sft_lora.yaml")

    assert cfg.data_source.manifest_path == "datasets/sft_agent_toolcall_12k/train.jsonl"
    assert cfg.data_source.eval_manifest_path == "datasets/sft_agent_toolcall_12k/val.jsonl"


def test_prepare_keeps_train_and_validation_nonempty_for_high_val_fraction(
    monkeypatch,
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": "answer", "_token_count": 2},
            ]
        }
        for index in range(2)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: str(source)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _: _Tokenizer())
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    out_dir = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_sft_agent",
            "--out-dir",
            str(out_dir),
            "--max-trajectories",
            "2",
            "--val-fraction",
            "0.9",
        ],
    )
    prepare_sft_agent.main()

    assert len((out_dir / "train.jsonl").read_text().splitlines()) == 1
    assert len((out_dir / "val.jsonl").read_text().splitlines()) == 1
