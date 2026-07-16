import dataclasses

import pytest
import torch

from unirl.algorithms import AlgorithmStepResult
from unirl.reward.service import _build_reward_request
from unirl.train.unified_model_stack import UnifiedModelTrainStack
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import BaseTrainer
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.types.media_preview import build_media_preview_for_part
from unirl.types.primitives import Audio, Audios, Image, Images, Texts, Video, Videos
from unirl.types.prompts import RolloutInputs
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _text_inputs(*, with_image: bool = False, with_video: bool = False) -> RolloutInputs:
    primitives = {"text": Texts(texts=["prompt"])}
    if with_image:
        primitives["image"] = Images.from_list([Image(pixels=torch.zeros(3, 2, 2))])
    if with_video:
        primitives["video"] = Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2))])
    return RolloutInputs(
        primitives=primitives,
        sample_ids=["sample-0"],
        group_ids=["prompt-0"],
        metadata=[{"tag": "test"}],
    )


def _diffusion_trainer(*, samples_per_prompt: int = 2) -> DiffusionTrainer:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.sampling_params = {"diffusion": DiffusionSamplingParams(samples_per_prompt=samples_per_prompt)}
    trainer._stage_config = {}
    trainer._noise_latent_shape = None
    return trainer


def test_diffusion_request_sample_chains_all_supported_media() -> None:
    trainer = _diffusion_trainer()

    sample = trainer._build_request_sample(
        _text_inputs(with_image=True, with_video=True),
        rollout_id=3,
    )

    assert [list(part.primitives) for part in sample.parts] == [
        ["text"],
        ["image"],
        ["video"],
        [],
    ]
    assert [part.batch_size for part in sample.parts] == [1, 1, 1, 2]

    bad = _text_inputs()
    bad.primitives["audio"] = Audios.from_list([Audio(waveform=torch.zeros(8))])
    with pytest.raises(ValueError, match="unsupported input primitive keys"):
        trainer._build_request_sample(bad, rollout_id=3)


def test_reward_request_carries_joint_video_audio_and_rate() -> None:
    root = Part.input(
        ["prompt-0"],
        primitives={"text": Texts(texts=["prompt"])},
        metadata=[{"tag": "test"}],
    )
    sample = Sample.request(root).fork(1, sampling_params=DiffusionSamplingParams())
    sample = sample.with_filled_frontier(
        primitives={
            "video": Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2))]),
            "audio": Audios.from_list([Audio(waveform=torch.zeros(8))]),
        },
        primitive_metadata={"audio": {"sample_rate": 48_000}},
    )

    request = _build_reward_request(sample, "video")

    assert set(request.generated) == {"video", "audio"}
    assert request.audio_sample_rate == 48_000
    assert request.prompts == ["prompt"]
    assert request.metadata == [{"tag": "test"}]

    preview = build_media_preview_for_part(
        part=sample.parts[-1],
        max_items=1,
        prompts=["prompt"],
    )
    assert preview is not None
    assert len(preview.videos) == 1
    assert len(preview.audios) == 1
    assert preview.audio_sample_rate == 48_000

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.wandb_logger = None
    trainer._drop_decoded(sample, rollout_id=0)
    assert sample.parts[-1].primitives == {}
    assert sample.parts[-1].primitive_metadata == {}


class _ImageRollout:
    def generate(self, sample: Sample) -> Sample:
        n = sample.parts[-1].batch_size
        return sample.with_filled_frontier(primitives={"image": Images(pixels=torch.zeros(n, 3, 2, 2))})


class _FixedReward:
    def __init__(self, rewards: list[float]) -> None:
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        self.inputs: list[Sample] = []

    def score_and_attach(self, sample: Sample) -> Sample:
        assert sample.parts[-1].rewards is None
        self.inputs.append(sample)
        frontier = dataclasses.replace(sample.parts[-1], rewards=self.rewards.clone())
        return sample.replace_frontier(frontier)


def test_diffusion_eval_scores_each_suite_from_same_unscored_sample() -> None:
    trainer = _diffusion_trainer(samples_per_prompt=2)
    trainer.eval_chunk_prompts = 1
    trainer.rollout = _ImageRollout()
    first = _FixedReward([1.0, 3.0])
    second = _FixedReward([2.0, 4.0])

    class _DataSource:
        def get_eval_samples(self, num_prompts: int) -> RolloutInputs:
            assert num_prompts == 1
            return _text_inputs()

    metrics = trainer._eval_pass(
        _DataSource(),
        1,
        [("first", first), ("second", second)],
        trainer.sampling_params,
        step=7,
    )

    assert metrics == {"first": 2.0, "second": 3.0}
    assert first.inputs[0] is second.inputs[0]
    assert first.inputs[0].parts[-1].rewards is None


def test_ar_evaluate_runs_all_batches_under_one_wake_sync_boundary() -> None:
    trainer = ARTrainer.__new__(ARTrainer)
    trainer.sampling_params = {"ar": ARSamplingParams(samples_per_prompt=1)}
    trainer.eval_samples_per_prompt = 2
    trainer.eval_temperature = 0.5
    trainer.eval_batch_size = 1
    trainer.eval_num_prompts = 2

    batches = [
        RolloutInputs(
            primitives={"text": Texts(texts=[f"prompt-{i}"])},
            sample_ids=[f"sample-{i}"],
            group_ids=[f"prompt-{i}"],
        )
        for i in range(2)
    ]

    class _DataSource:
        def iter_eval_batches(self, batch_size: int, *, eval_num_prompts: int):
            assert batch_size == 1
            assert eval_num_prompts == 2
            yield from batches

    class _Rollout:
        wakes = 0
        sleeps = 0
        generates = 0

        def wake_up(self) -> None:
            self.wakes += 1

        def sleep(self) -> None:
            self.sleeps += 1

        def generate(self, sample: Sample) -> Sample:
            self.generates += 1
            n = sample.parts[-1].batch_size
            return sample.with_filled_frontier(primitives={"text": Texts(texts=["answer"] * n)})

    class _Sync:
        calls = 0

        def sync(self) -> None:
            self.calls += 1

    class _Logger:
        logged = None

        def log_eval(self, step: int, metrics) -> None:
            self.logged = (step, metrics)

    trainer.data_source = _DataSource()
    trainer.rollout = _Rollout()
    trainer.weight_sync = _Sync()
    trainer.reward = _FixedReward([1.0, 3.0])
    trainer.wandb_logger = _Logger()

    accuracy = trainer.evaluate(rollout_id=4)

    assert accuracy == 2.0
    assert trainer.rollout.wakes == 1
    assert trainer.rollout.sleeps == 1
    assert trainer.rollout.generates == 2
    assert trainer.weight_sync.calls == 1
    assert trainer.wandb_logger.logged == (5, {"acc": 2.0, "reward": 2.0})


def test_ar_evaluate_pads_ragged_tail_without_counting_replicas(caplog) -> None:
    caplog.set_level("INFO", logger="unirl.trainer.ar")
    trainer = ARTrainer.__new__(ARTrainer)
    trainer.sampling_params = {"ar": ARSamplingParams(samples_per_prompt=1)}
    trainer.eval_samples_per_prompt = 2
    trainer.eval_temperature = 0.5
    trainer.eval_batch_size = 8
    trainer.eval_num_prompts = 60

    final_batch = RolloutInputs(
        primitives={"text": Texts(texts=[f"prompt-{i}" for i in range(4)])},
        sample_ids=[f"sample-{i}" for i in range(4)],
        group_ids=[f"prompt-{i}" for i in range(4)],
    )

    class _DataSource:
        def iter_eval_batches(self, batch_size: int, *, eval_num_prompts: int):
            assert batch_size == 8
            assert eval_num_prompts == 60
            yield final_batch

    class _Rollout:
        dp_size = 8
        dispatched: Sample | None = None

        def wake_up(self) -> None:
            pass

        def sleep(self) -> None:
            pass

        def generate(self, sample: Sample) -> Sample:
            self.dispatched = sample
            n = sample.parts[-1].batch_size
            return sample.with_filled_frontier(primitives={"text": Texts(texts=["answer"] * n)})

    class _Logger:
        logged = None

        def log_eval(self, step: int, metrics) -> None:
            self.logged = (step, metrics)

    trainer.data_source = _DataSource()
    trainer.rollout = _Rollout()
    trainer.weight_sync = None
    # Four real roots * two outputs score 1. Padding contributes eight 100s;
    # including it would produce 50.5 instead of 1.0.
    trainer.reward = _FixedReward([1.0] * 8 + [100.0] * 8)
    trainer.reward.dp_size = 8
    trainer.wandb_logger = _Logger()

    accuracy = trainer.evaluate(rollout_id=2)

    assert accuracy == 1.0
    assert trainer.wandb_logger.logged == (3, {"acc": 1.0, "reward": 1.0})
    dispatched = trainer.rollout.dispatched
    assert dispatched is not None
    assert dispatched.batch_size == 8
    root_ids = dispatched.parts[0].sample_ids
    assert len(set(root_ids)) == 8
    assert root_ids[:4] == [f"r2:sample-{i}" for i in range(4)]
    assert all("eval-pad" in sample_id for sample_id in root_ids[4:])
    assert trainer.reward.inputs[0].batch_size == 8
    assert "over 4 prompts" in caplog.text


def test_unified_m1_copies_ar_advantages_to_image_part() -> None:
    ar_params = ARSamplingParams(samples_per_prompt=2)
    diffusion_params = DiffusionSamplingParams(samples_per_prompt=1)
    root = Part.input(["prompt-0"], primitives={"text": Texts(texts=["prompt"])})
    sample = Sample.request(root).fork(2, sampling_params=ar_params)
    sample = sample.with_filled_frontier(primitives={"text": Texts(texts=["a", "b"])})
    sample = sample.fork(1, sampling_params=diffusion_params)
    sample = sample.with_filled_frontier(primitives={"image": Images(pixels=torch.zeros(2, 3, 2, 2))})

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._shared_advantage = True
    trainer._enable_fsdp_offload = False
    trainer.dump_dir = None
    trainer.run_rollout = lambda request: request
    trainer.reward = _FixedReward([1.0, 3.0])
    trainer._drop_decoded = lambda scored, rollout_id: None

    class _Stack:
        trained: Sample | None = None

        def train_track(self, scored: Sample, *, training_progress: float):
            self.trained = scored
            return {"ok": training_progress}

    class _Logger:
        def log_rollout_step(self, *args, **kwargs) -> None:
            pass

    trainer.stack = _Stack()
    trainer.wandb_logger = _Logger()

    _, mean_reward = trainer.train_step(sample, training_progress=0.5)

    trained = trainer.stack.trained
    assert trained is not None
    ar_part = trained.gen_part(ARSamplingParams)
    image_part = trained.gen_part(DiffusionSamplingParams)
    assert mean_reward == 2.0
    assert torch.equal(image_part.advantages, ar_part.advantages)
    assert not torch.allclose(image_part.advantages, torch.zeros_like(image_part.advantages))


def test_unified_stack_pairs_proportional_ar_image_updates_in_one_step() -> None:
    events: list[tuple[str, list[float]] | tuple[str]] = []

    class _Algorithm:
        supports_multi_update = True

        def compute_loss_and_backward(
            self,
            *,
            conditions,
            segment,
            advantages,
            training_progress,
            loss_scale,
        ) -> AlgorithmStepResult:
            del conditions, segment, training_progress
            name = "ar" if float(advantages.min()) < 10 else "image"
            events.append((name, advantages.tolist()))
            return AlgorithmStepResult(
                loss=float(loss_scale),
                metrics={},
                num_steps_or_tokens=1,
                has_backward=True,
            )

    class _Optimizer:
        param_groups = [{"lr": 1e-4}]

    class _Backend:
        _device = torch.device("cpu")
        optimizer = _Optimizer()
        scheduler = None

        def zero_grad(self) -> None:
            events.append(("zero",))

        def optimizer_step(self, *, max_grad_norm: float) -> float:
            assert max_grad_norm == 1.0
            events.append(("step",))
            return 0.5

        def on_rollout_end(self) -> None:
            events.append(("end",))

    stack = UnifiedModelTrainStack.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = _Backend()
    stack.algorithms = {"ar": _Algorithm(), "image": _Algorithm()}
    stack.micro_batch_size = 1
    stack.max_grad_norm = 1.0
    stack.num_updates_per_batch = 2

    root = Part.input(
        ["p0", "p1"],
        primitives={"text": Texts(texts=["zero", "one"])},
    )
    sample = Sample.request(root).fork(2, sampling_params=ARSamplingParams(samples_per_prompt=2))
    sample = sample.replace_frontier(
        dataclasses.replace(sample.parts[-1], advantages=torch.arange(4, dtype=torch.float32))
    )
    sample = sample.fork(2, sampling_params=DiffusionSamplingParams(samples_per_prompt=2))
    sample = sample.replace_frontier(
        dataclasses.replace(sample.parts[-1], advantages=torch.arange(10, 18, dtype=torch.float32))
    )

    result = UnifiedModelTrainStack.train_track.__wrapped__(stack, sample, training_progress=0.25)

    assert set(result) == {"ar", "image"}
    assert events == [
        ("zero",),
        ("ar", [0.0]),
        ("ar", [1.0]),
        ("image", [10.0]),
        ("image", [11.0]),
        ("image", [12.0]),
        ("image", [13.0]),
        ("step",),
        ("zero",),
        ("ar", [2.0]),
        ("ar", [3.0]),
        ("image", [14.0]),
        ("image", [15.0]),
        ("image", [16.0]),
        ("image", [17.0]),
        ("step",),
        ("end",),
    ]
