from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unirl.models.qwen3.chat_template import Qwen3ChatTemplateStage
from unirl.models.qwen_vl.chat_template import QwenVLChatTemplateStage
from unirl.train.sft.track_builder import ARSupervisedTrackBuilder
from unirl.train.stack.base import TrainStack
from unirl.trainer.sft import SFTTrainer
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Turn
from unirl.types.segments.text import TextSegment


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.messages = []

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": [10 + i for i, _ in enumerate(text)]}

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return torch.tensor([[len(messages), len(str(messages))]], dtype=torch.long)


class _Processor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        has_image = any(
            block.get("type") == "image"
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )
        result = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
        if has_image:
            result["pixel_values"] = torch.ones((1, 3, 2, 2))
            result["image_grid_thw"] = torch.ones((1, 3), dtype=torch.long)
        return result


class _Conditions:
    def to_dict(self):
        return {}


class _ChatStage:
    def embed(self, texts: Texts):
        assert isinstance(texts, Texts)
        return _Conditions()


def test_ar_supervised_builder_returns_part_with_metadata_and_pad_mask():
    tokenizer = _Tokenizer()
    pipeline = SimpleNamespace(
        bundle=SimpleNamespace(tokenizer=tokenizer, device=torch.device("cpu")),
        chat_template=_ChatStage(),
    )
    builder = ARSupervisedTrackBuilder(pipeline=pipeline)
    records = [
        {"sample_id": "a", "prompt": "p0", "response": "xy", "metadata": {"source": "one"}},
        {"sample_id": "b:eval-pad:1", "prompt": "p1", "response": "z", "_eval_pad": True},
    ]

    part = builder.build(records)

    assert isinstance(part, Part)
    assert part.sample_ids == ["a", "b:eval-pad:1"]
    assert part.metadata == [{"source": "one"}, {}]
    assert part.segment is not None
    cu = part.segment.cu_seqlens
    assert cu is not None
    assert float(part.segment.loss_mask[int(cu[0]) : int(cu[1])].sum()) == 3.0
    assert float(part.segment.loss_mask[int(cu[1]) : int(cu[2])].sum()) == 0.0


def test_qwen3_texts_and_single_user_turn_share_rendering():
    tokenizer = _Tokenizer()
    stage = Qwen3ChatTemplateStage(
        SimpleNamespace(tokenizer=tokenizer, device=torch.device("cpu")),
        system_instruction="system",
    )
    texts = Texts(texts=["hello", "world"])

    from_texts = stage.embed(texts)
    messages_from_texts = list(tokenizer.messages)
    tokenizer.messages.clear()
    from_turn = stage.embed([Turn(role="user", content=texts)])

    assert tokenizer.messages == messages_from_texts
    assert torch.equal(from_texts.prompt.input_ids, from_turn.prompt.input_ids)
    assert torch.equal(from_texts.prompt.attention_mask, from_turn.prompt.attention_mask)


def test_qwen_vl_texts_support_mixed_optional_images_and_turn_parity():
    processor = _Processor()
    stage = QwenVLChatTemplateStage(
        SimpleNamespace(processor=processor, device=torch.device("cpu"), dtype=torch.float32),
        system_instruction="system",
    )
    texts = Texts(texts=["with", "without"])
    marker = object()

    conditions = stage.embed(texts, [marker, None])

    assert conditions.pixel_values[0] is not None
    assert conditions.pixel_values[1] is None
    assert processor.messages[0][1]["content"][0] == {"type": "image", "image": marker}
    assert processor.messages[1][1]["content"] == [{"type": "text", "text": "without"}]

    processor.messages.clear()
    text_only = stage.embed(Texts(texts=["same"]))
    texts_messages = list(processor.messages)
    processor.messages.clear()
    turn_only = stage.embed([Turn(role="user", content=Texts(texts=["same"]))])
    assert processor.messages == texts_messages
    assert torch.equal(text_only.prompt.input_ids, turn_only.prompt.input_ids)

    with pytest.raises(ValueError, match="images length"):
        stage.embed(texts, [None])
    with pytest.raises(ValueError, match="only valid with Texts"):
        stage.embed([Turn(role="user", content=Texts(texts=["same"]))], [None])


class _LossBackend:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)

    def gradient_average_world_size(self) -> int:
        return 1

    def all_reduce_loss_sums(self, values):
        return list(values)

    def trainable_module(self):
        return self.model


def test_train_stack_sample_and_token_weighting_are_part_native():
    segment = TextSegment.pack(
        tokens=[torch.ones(2), torch.ones(1), torch.ones(3)],
        loss_mask=[torch.ones(2), torch.zeros(1), torch.tensor([1.0, 0.0, 1.0])],
    )
    part = Part(sample_ids=["a", "b", "c"], segment=segment)
    stack = TrainStack.__new__(TrainStack)
    stack.fsdp_backend = _LossBackend()
    stack.rank_info = SimpleNamespace(sp_size=1)

    stack.algorithm = SimpleNamespace(loss_weighting="sample")
    scales, global_weight = stack._resolve_loss_scales(part, micros=[(0, 1), (1, 3)])
    assert scales == pytest.approx([1 / 3, 2 / 3])
    assert global_weight is None

    stack.algorithm = SimpleNamespace(loss_weighting="token")
    scales, global_weight = stack._resolve_loss_scales(part, micros=[(0, 1), (1, 3)])
    assert scales == pytest.approx([0.5, 0.5])
    assert global_weight == pytest.approx(4.0)


def test_train_stack_eval_slices_parts_and_restores_model_mode():
    class _EvalAlgorithm:
        def __init__(self) -> None:
            self.seen_ids = []

        def evaluate_loss(self, *, conditions, segment, sample_ids):
            self.seen_ids.append(sample_ids)
            return 2.0 * len(sample_ids), len(sample_ids)

    part = Part(
        sample_ids=["a", "b", "c"],
        segment=TextSegment.pack(tokens=[torch.ones(1), torch.ones(2), torch.ones(3)]),
    )
    stack = TrainStack.__new__(TrainStack)
    stack.fsdp_backend = _LossBackend()
    stack.algorithm = _EvalAlgorithm()
    stack.micro_batch_size = 2
    stack.fsdp_backend.model.train()

    metrics = stack.eval_track(part)

    assert metrics == {"loss": 2.0, "weight": 3.0}
    assert stack.algorithm.seen_ids == [["a", "b"], ["c"]]
    assert stack.fsdp_backend.model.training


def test_sft_padding_and_cursor_sidecar_are_lineage_safe_and_atomic(tmp_path: Path):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.dp_size = 4
    trainer.data_source = SimpleNamespace(state_dict=lambda: {"epoch": 2, "position": 3, "seed": 7})

    padded = trainer._pad_to_dp([{"sample_id": "root", "prompt": "p"}])
    assert len(padded) == 4
    assert [row["sample_id"] for row in padded[1:]] == [
        "root:eval-pad:1",
        "root:eval-pad:2",
        "root:eval-pad:3",
    ]
    assert all("/" not in row["sample_id"] for row in padded)

    trainer._save_data_state(0, 1, save_interval=1, save_dir=str(tmp_path))
    path = tmp_path / "checkpoint-1" / "sft_data_state.json"
    assert json.loads(path.read_text()) == {"epoch": 2, "position": 3, "seed": 7}
    assert not list(path.parent.glob("*.tmp.*"))


def test_migrated_sft_sources_do_not_reference_retired_rollout_types():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "unirl/train/sft/track_builder.py",
        root / "unirl/trainer/sft.py",
        root / "unirl/train/stack/base.py",
    ]
    forbidden = {
        "Rollout" + "Req",
        "Rollout" + "Resp",
        "Rollout" + "Track",
        "resp_" + "track",
        "parent_" + "track",
        "parent_" + "ids",
    }
    for path in paths:
        source = path.read_text()
        assert "<" * 7 not in source
        tree = ast.parse(source)
        executable_names = {
            value
            for node in ast.walk(tree)
            for value in (
                getattr(node, "id", None),
                getattr(node, "attr", None),
                getattr(node, "arg", None),
                getattr(node, "name", None),
            )
            if isinstance(value, str)
        }
        assert forbidden.isdisjoint(executable_names), path
