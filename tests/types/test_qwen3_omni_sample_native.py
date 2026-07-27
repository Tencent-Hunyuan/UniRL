"""CPU tests for Qwen3-Omni's Sample-native conversation and adapter boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
from unirl.models.qwen3_omni.pipeline import Qwen3OmniPipeline
from unirl.models.types.conversations import build_video_messages
from unirl.rollout.engine.vllm_omni.adapters.qwen3_omni import (
    Qwen3OmniThinkerInputAdapter,
    Qwen3OmniThinkerOutputAdapter,
)
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import SegmentStatus, TextSegment


def _request(
    *,
    prompts: List[str] | None = None,
    fanout: int = 2,
    with_video: bool = False,
    control: Dict[str, Any] | None = None,
) -> Sample:
    prompts = prompts or ["question 0", "question 1"]
    ids = [f"p{i}" for i in range(len(prompts))]
    root = Part.input(
        ids,
        primitives={"text": Texts(texts=prompts)},
        control=control,
    )
    parts = [root]
    if with_video:
        video = root.input_child({"video": Videos.from_uris([f"/videos/{i}.mp4" for i in range(len(prompts))])})
        parts.append(video)
    return Sample.request(*parts).fork(
        fanout,
        sampling_params=ARSamplingParams(
            samples_per_prompt=fanout,
            temperature=0.8,
            top_p=0.95,
            top_k=0,
            max_new_tokens=32,
        ),
    )


def test_video_messages_fuse_initial_user_parts_and_preserve_agent_roles() -> None:
    request = _request(with_video=True)
    first = request.parts[-1].fill(
        segment=TextSegment.pack(
            tokens=[torch.tensor([7])] * 4,
            log_probs=[torch.tensor([-0.1])] * 4,
        ),
        primitives={"text": Texts(texts=["first"] * 4)},
    )
    trajectory = request.replace_frontier(first)
    trajectory = trajectory.observe(Texts(texts=["tool result"] * 4))
    trajectory = trajectory.fork(1, sampling_params=ARSamplingParams())

    conversations = build_video_messages(trajectory.turns(), "configured system")

    assert len(conversations) == 4
    messages = conversations[0]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[0]["content"] == "configured system"
    assert [block["type"] for block in messages[1]["content"]] == ["video", "text"]
    assert messages[1]["content"][0]["video"] == "/videos/0.mp4"
    assert messages[1]["content"][1]["text"] == "question 0"
    assert messages[2]["content"] == [{"type": "text", "text": "first"}]
    assert messages[3]["content"] == [{"type": "text", "text": "tool result"}]


def test_explicit_system_turn_wins_over_configured_prefix() -> None:
    system = Part.input(
        ["p0"],
        primitives={"text": Texts(texts=["trajectory system"])},
        role="system",
    )
    user = system.input_child({"text": Texts(texts=["question"])}, role="user")
    sample = Sample.request(system, user).fork(1, sampling_params=ARSamplingParams())

    messages = build_video_messages(sample.turns(), "configured system")[0]

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": "trajectory system"}]


def test_multiple_video_turns_fail_loudly() -> None:
    root = Part.input(["p0"], primitives={"text": Texts(texts=["question"])})
    video0 = root.input_child({"video": Videos.from_uris(["/videos/0.mp4"])})
    video1 = video0.input_child({"video": Videos.from_uris(["/videos/1.mp4"])})
    sample = Sample.request(root, video0, video1).fork(1, sampling_params=ARSamplingParams())

    with pytest.raises(ValueError, match="at most one"):
        build_video_messages(sample.turns())


def test_packed_video_rows_follow_frontier_fanout() -> None:
    frames = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    root = Part.input(["p0"], primitives={"text": Texts(texts=["question"])})
    video = root.input_child({"video": Videos.from_list([Video(frames=frames)])})
    sample = Sample.request(root, video).fork(2, sampling_params=ARSamplingParams(samples_per_prompt=2))

    conversations = build_video_messages(sample.turns())

    assert len(conversations) == 2
    for messages in conversations:
        assert torch.equal(messages[0]["content"][0]["video"], frames)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: List[tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
        self.video_processor = SimpleNamespace(size={"shortest_edge": 64})

    def apply_chat_template(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, torch.Tensor]:
        self.calls.append((messages, kwargs))
        row = len(self.calls)
        return {
            "input_ids": torch.tensor([[row, row + 10]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


def _input_adapter() -> Qwen3OmniThinkerInputAdapter:
    adapter = object.__new__(Qwen3OmniThinkerInputAdapter)
    adapter.modality = "qwen3_omni_thinker"
    adapter.model_path = "unused"
    adapter.video_fps = 1.0
    adapter.video_max_frames = None
    adapter.video_max_pixels = None
    adapter.max_prompt_length = 128
    adapter.system_instruction = "default system"
    adapter.chat_template_kwargs = {"tools": [{"type": "function", "function": {"name": "calc"}}]}
    adapter._processor = _FakeProcessor()
    adapter._tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    adapter._last_encodings = []
    return adapter


def test_vllm_input_adapter_uses_frontier_rows_without_second_fanout() -> None:
    request = _request(
        fanout=2,
        control={"chat": {"template_kwargs": {"enable_thinking": False}}},
    )
    adapter = _input_adapter()

    calls = adapter.build(request)

    assert len(calls) == 1
    assert len(calls[0].prompts) == 4
    assert len(request.parts[-1].sample_ids) == 4
    sampling = calls[0].sampling[0].kwargs
    assert sampling["top_k"] == -1
    assert sampling["max_tokens"] == 32
    assert "n" not in sampling
    assert len(adapter._last_encodings) == 4
    assert len(adapter._processor.calls) == 4
    template_kwargs = adapter._processor.calls[0][1]
    assert template_kwargs["tools"][0]["function"]["name"] == "calc"
    assert template_kwargs["enable_thinking"] is False
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["return_dict"] is True


def _raw_result(index: int, *, finish_reason: str = "stop") -> Any:
    token = 20 + index
    completion = SimpleNamespace(
        token_ids=[token],
        logprobs=[{token: SimpleNamespace(logprob=-0.25)}],
        text=f"answer {index}",
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        stage_id=0,
        final_output_type="text",
        request_output=SimpleNamespace(outputs=[completion]),
        prompt_token_ids=[index + 1, index + 2],
    )


def test_vllm_output_adapter_fills_existing_frontier_and_conditions() -> None:
    request = _request(fanout=2)
    input_adapter = _input_adapter()
    input_adapter.build(request)
    output_adapter = Qwen3OmniThinkerOutputAdapter("qwen3_omni_thinker", input_adapter)
    per_request = [
        [_raw_result(0, finish_reason="stop")],
        [_raw_result(1, finish_reason="length")],
        [_raw_result(2, finish_reason="abort")],
        [_raw_result(3, finish_reason="unknown")],
    ]

    generated = output_adapter.build(request, per_request)
    frontier = generated.parts[-1]

    assert len(generated.parts) == len(request.parts)
    assert frontier.sample_ids == request.parts[-1].sample_ids
    assert frontier.sampling_params == request.parts[-1].sampling_params
    assert frontier.primitives["text"].texts == [f"answer {i}" for i in range(4)]
    assert frontier.segment.tokens.tolist() == [20, 21, 22, 23]
    assert torch.allclose(frontier.segment.log_probs, torch.full((4,), -0.25))
    assert frontier.conditions["prompt"].input_ids.shape == (4, 2)
    assert frontier.status.tolist() == [
        int(SegmentStatus.COMPLETED),
        int(SegmentStatus.TRUNCATED),
        int(SegmentStatus.ABORTED),
        int(SegmentStatus.PENDING),
    ]


def test_vllm_output_adapter_rejects_missing_stage_result() -> None:
    request = _request(prompts=["question"], fanout=1)
    input_adapter = _input_adapter()
    input_adapter.build(request)
    output_adapter = Qwen3OmniThinkerOutputAdapter("qwen3_omni_thinker", input_adapter)

    with pytest.raises(RuntimeError, match="no stage-0"):
        output_adapter.build(request, [[]])


class _FakeChatStage:
    system_instruction = None
    chat_template_kwargs: Dict[str, Any] = {}
    max_prompt_length = 128
    pad_to_max_length = False
    video_fps = 1.0
    video_max_frames = None
    video_max_pixels = None

    def __init__(self) -> None:
        self.turns = None

    def embed(self, turns: Any) -> Qwen3OmniARConditions:
        self.turns = turns
        batch = len(turns[0].content)
        return Qwen3OmniARConditions(
            prompt=TextTokenCondition(
                input_ids=torch.ones((batch, 3), dtype=torch.long),
                attention_mask=torch.ones((batch, 3), dtype=torch.long),
            )
        )


class _FakeARStage:
    def autoregress(self, conditions: Any, *, sampling_params: Any, params: Any) -> TextSegment:
        batch = int(conditions.prompt.input_ids.shape[0])
        return TextSegment.pack(
            tokens=[torch.tensor([30 + i], dtype=torch.long) for i in range(batch)],
            log_probs=[torch.tensor([-0.5], dtype=torch.float32) for _ in range(batch)],
        )


def test_trainside_pipeline_is_sample_endomorphism() -> None:
    request = _request(fanout=2)
    chat = _FakeChatStage()
    bundle = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens: f"decoded:{ids[0]}"))
    pipeline = Qwen3OmniPipeline(bundle=bundle, chat_template=chat, ar=_FakeARStage())

    generated = pipeline.generate(request)

    assert generated is not request
    assert generated.parts[:-1] == request.parts[:-1]
    assert generated.parts[-1].sample_ids == request.parts[-1].sample_ids
    assert generated.parts[-1].primitives["text"].texts == [
        "decoded:30",
        "decoded:31",
        "decoded:32",
        "decoded:33",
    ]
    assert generated.parts[-1].conditions["prompt"].input_ids.shape == (4, 3)
    assert chat.turns is not None
