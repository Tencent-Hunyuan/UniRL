from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from unirl.data.data_source import DefaultDataSource, MultimodalRLDataSource, _input_sample
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import prepare_input_sample
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.trainer.pe import PETrainer
from unirl.trainer.refl import _text_inputs
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.types.primitives import Audio, Audios, Image, Images, Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _inputs(*, count: int = 2, image: bool = False, video: bool = False) -> Sample:
    primitives = {"text": Texts(texts=[f"prompt-{i}" for i in range(count)])}
    if image:
        primitives["image"] = Images.from_list([Image(pixels=torch.full((3, 2, 2), float(i))) for i in range(count)])
    if video:
        primitives["video"] = Videos.from_list(
            [Video(frames=torch.full((i + 1, 3, 2, 2), float(i))) for i in range(count)]
        )
    return _input_sample(
        primitives,
        sample_ids=[f"sample-{i}" for i in range(count)],
        metadata=[{"row": i} for i in range(count)],
    )


def test_data_source_builds_tree_complete_multimodal_samples() -> None:
    inputs = _inputs(image=True, video=True)

    assert inputs.batch_size == 2
    assert [list(part.primitives) for part in inputs.parts] == [["text"], ["image"], ["video"]]
    assert [part.sample_ids for part in inputs.parts] == [
        ["sample-0", "sample-1"],
        ["sample-0/0", "sample-1/0"],
        ["sample-0/0/0", "sample-1/0/0"],
    ]
    assert inputs.parts[0].metadata == [{"row": 0}, {"row": 1}]
    assert all(not part.is_gen for part in inputs.parts)

    second = inputs.slice(1, 2)
    assert second.parts[0].metadata == [{"row": 1}]
    assert second.parts[1].primitives["image"].pixels.shape[0] == 1
    restored = Sample.concat([inputs.slice(0, 1), second])
    assert [part.sample_ids for part in restored.parts] == [part.sample_ids for part in inputs.parts]
    assert restored.parts[0].metadata == inputs.parts[0].metadata
    assert restored.parts[2].primitives["video"].cu_frames.tolist() == [0, 1, 3]

    shards = inputs.chunk(2)
    assert [shard.batch_size for shard in shards] == [1, 1]
    assert all(len(shard.parts) == 3 for shard in shards)
    assert Sample.concat(shards).parts[2].primitives["video"].cu_frames.tolist() == [0, 1, 3]


def test_default_data_source_emits_an_input_sample() -> None:
    source = DefaultDataSource(SimpleNamespace())
    inputs = source.get_samples(2)

    assert isinstance(inputs, Sample)
    assert inputs.batch_size == 2
    assert len(inputs.parts) == 1
    assert isinstance(inputs.parts[0].primitives["text"], Texts)
    assert not inputs.parts[0].is_gen


def test_multimodal_data_source_train_and_eval_boundaries_emit_samples(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [{"prompt": f"prompt-{i}", "prompt_id": f"id-{i}", "metadata": {"row": i}} for i in range(4)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    args = SimpleNamespace(
        run=SimpleNamespace(data_path=str(path), eval_data_path=str(path), seed=0),
        algorithm=SimpleNamespace(prompts_per_rollout=2),
    )
    source = MultimodalRLDataSource(args)

    train = source.get_samples(2)
    assert isinstance(train, Sample)
    assert train.batch_size == 2
    assert all(not part.is_gen for part in train.parts)

    eval_batches = list(source.iter_eval_batches(3, eval_num_prompts=4))
    assert [batch.batch_size for batch in eval_batches] == [3, 1]
    assert eval_batches[0].parts[0].sample_ids == [
        "prompt:id-0:sample:0",
        "prompt:id-1:sample:0",
        "prompt:id-2:sample:0",
    ]
    assert eval_batches[0].parts[0].metadata == [{"row": 0}, {"row": 1}, {"row": 2}]


def test_prepare_input_sample_namespaces_every_part_and_merges_root_control() -> None:
    inputs = _inputs(count=1, image=True, video=True)
    inputs.parts[0].control = {"dataset": "kept", "ar": {"temperature": 0.5}}

    prepared = prepare_input_sample(
        inputs,
        7,
        allowed_primitives={"text", "image", "video"},
        caller="test",
        root_control={"ar": {"stop": ["done"]}},
    )

    assert [part.sample_ids for part in prepared.parts] == [
        ["r7:sample-0"],
        ["r7:sample-0/0"],
        ["r7:sample-0/0/0"],
    ]
    assert prepared.parts[0].control == {"dataset": "kept", "ar": {"stop": ["done"]}}
    assert prepared.parts[0].metadata == [{"row": 0}]
    assert inputs.parts[0].sample_ids == ["sample-0"]

    forked = prepared.fork(3, sampling_params=ARSamplingParams(samples_per_prompt=3))
    assert forked.parts[-1].sample_ids == ["r7:sample-0/0/0/0", "r7:sample-0/0/0/1", "r7:sample-0/0/0/2"]
    assert forked.root_group_ids(-1) == ["r7:sample-0"] * 3


def test_prepare_input_sample_rejects_generated_and_unsupported_parts() -> None:
    generated = _inputs(count=1).fork(1, sampling_params=ARSamplingParams())
    with pytest.raises(ValueError, match="input-only"):
        prepare_input_sample(generated, 0, allowed_primitives={"text"}, caller="test")

    root = Part.input(["sample"], primitives={"text": Texts(texts=["prompt"])})
    audio = root.input_child({"audio": Audios.from_list([Audio(waveform=torch.zeros(4))])})
    with pytest.raises(ValueError, match="unsupported input primitive keys.*audio"):
        prepare_input_sample(Sample.request(root, audio), 0, allowed_primitives={"text"}, caller="test")

    image_root = Part.input(
        ["sample"],
        primitives={"image": Images.from_list([Image(pixels=torch.zeros(3, 2, 2))])},
    )
    with pytest.raises(TypeError, match=r"root requires primitives\['text'\]: Texts"):
        prepare_input_sample(
            Sample.request(image_root),
            0,
            allowed_primitives={"text", "image"},
            caller="test",
        )

    with pytest.raises(ValueError, match="unique root sample_ids"):
        _input_sample(
            {"text": Texts(texts=["a", "b"])},
            sample_ids=["duplicate", "duplicate"],
        )


def test_request_builders_preserve_input_tree_and_apply_fanout_and_control() -> None:
    ar = ARTrainer.__new__(ARTrainer)
    ar.sampling_params = {"ar": ARSamplingParams(samples_per_prompt=2)}
    ar_sample = ar._build_request_sample(_inputs(count=1, image=True), rollout_id=2)
    assert [list(part.primitives) for part in ar_sample.parts] == [["text"], ["image"], []]
    assert ar_sample.parts[-1].sample_ids == ["r2:sample-0/0/0", "r2:sample-0/0/1"]

    diffusion = DiffusionTrainer.__new__(DiffusionTrainer)
    diffusion.sampling_params = {"diffusion": DiffusionSamplingParams(samples_per_prompt=2)}
    diffusion._stage_config = {"task": "it2i"}
    diffusion._noise_latent_shape = None
    diffusion_sample = diffusion._build_request_sample(_inputs(count=1, image=True), rollout_id=3)
    assert diffusion_sample.parts[0].control == {"task": "it2i"}
    assert diffusion_sample.parts[-1].sample_ids == ["r3:sample-0/0/0", "r3:sample-0/0/1"]

    pe = PETrainer.__new__(PETrainer)
    pe.sampling_params = {
        "ar": ARSamplingParams(samples_per_prompt=2),
        "diffusion": DiffusionSamplingParams(samples_per_prompt=3),
    }
    pe_sample = pe._build_request_sample(_inputs(count=1), rollout_id=4)
    assert pe_sample.parts[0].control == {"ar": {}, "chat": {}}
    assert [part.batch_size for part in pe_sample.parts] == [1, 2, 6]

    unified = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    unified.sampling_params = pe.sampling_params
    unified_sample = unified._build_request_sample(_inputs(count=1), rollout_id=5)
    assert [part.batch_size for part in unified_sample.parts] == [1, 2, 6]
    assert unified_sample.parts[-1].sample_ids[-1] == "r5:sample-0/1/2"

    agentic = AgenticTrainer.__new__(AgenticTrainer)
    agentic._stop = ["</tool_call>"]
    agentic_sample = agentic._build_request_sample(_inputs(count=1), rollout_id=6)
    assert len(agentic_sample.parts) == 1
    assert agentic_sample.parts[0].sample_ids == ["r6:sample-0"]
    assert agentic_sample.parts[0].control == {"ar": {"stop": ["</tool_call>"]}}


def test_single_input_consumers_reject_chains_their_runners_cannot_preserve() -> None:
    root = Part.input(["sample"], primitives={"text": Texts(texts=["prompt"])})
    followup = root.input_child({"text": Texts(texts=["follow-up"])})
    inputs = Sample.request(root, followup)
    sampling = {
        "ar": ARSamplingParams(samples_per_prompt=1),
        "diffusion": DiffusionSamplingParams(samples_per_prompt=1),
    }

    pe = PETrainer.__new__(PETrainer)
    pe.sampling_params = sampling
    with pytest.raises(ValueError, match="requires exactly one input Part"):
        pe._build_request_sample(inputs, rollout_id=0)

    unified = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    unified.sampling_params = sampling
    with pytest.raises(ValueError, match="requires exactly one input Part"):
        unified._build_request_sample(inputs, rollout_id=0)

    # Agentic engines consume the entire role-aware input tree, so this path is
    # intentionally preserved rather than forced into the single-input contract.
    agentic = AgenticTrainer.__new__(AgenticTrainer)
    agentic._stop = ["</tool_call>"]
    prepared = agentic._build_request_sample(inputs, rollout_id=1)
    assert [part.sample_ids for part in prepared.parts] == [["r1:sample"], ["r1:sample/0"]]


def test_ar_eval_padding_rewrites_whole_tree_with_unique_lineage() -> None:
    trainer = ARTrainer.__new__(ARTrainer)
    trainer.rollout = SimpleNamespace(dp_size=4)
    trainer.reward = SimpleNamespace(dp_size=2)
    inputs = _inputs(count=3, image=True, video=True)

    padded = trainer._pad_eval_inputs(inputs)

    assert padded.batch_size == 4
    roots = padded.parts[0].sample_ids
    assert len(roots) == len(set(roots)) == 4
    assert ":eval-pad:0" in roots[-1]
    for previous, current in zip(padded.parts, padded.parts[1:]):
        assert all(parent_id(sample_id) in set(previous.sample_ids) for sample_id in current.sample_ids)
    assert len(set(padded.parts[-1].sample_ids)) == 4
    assert padded.parts[0].metadata[-1] == {"row": 2}


def test_refl_extracts_text_from_input_sample() -> None:
    inputs = _inputs(count=2, image=True)
    assert _text_inputs(inputs).texts == ["prompt-0", "prompt-1"]

    with pytest.raises(ValueError, match="no input Parts"):
        _text_inputs(Sample())
