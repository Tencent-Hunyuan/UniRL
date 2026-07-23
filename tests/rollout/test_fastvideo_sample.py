"""Sample/Part and lifecycle regressions for the FastVideo rollout engine."""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

import unirl.rollout.engine.fastvideo.engine as engine_module
from unirl.rollout.engine.fastvideo.engine import FastVideoRolloutEngine, _resolve_sde_window
from unirl.sde.noise import _derive_group_seed
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


class _Schedule:
    def compute_sigma(self, *, num_inference_steps: int, height: int, width: int) -> torch.Tensor:
        assert (num_inference_steps, height, width) == (2, 8, 8)
        return torch.tensor([1.0, 0.5, 0.0])


def _sample(*, init_same_noise: bool = False, sde_indices=None, seed: int | None = 17) -> Sample:
    root = Part.input(
        ["p0", "p1"],
        primitives={"text": Texts(texts=["alpha", "beta"])},
    )
    params = DiffusionSamplingParams(
        samples_per_prompt=2,
        num_inference_steps=2,
        height=8,
        width=8,
        num_frames=1,
        seed=seed,
        init_same_noise=init_same_noise,
        sde_indices=sde_indices,
    )
    return Sample.request(root).fork(2, sampling_params=params)


def _bare_engine(*, forward_batch_size: int | None = None, weight_version: int = 0):
    engine = object.__new__(FastVideoRolloutEngine)
    engine.cfg = SimpleNamespace(forward_batch_size=forward_batch_size, native_logprob=False)
    engine._is_offloaded = False
    engine._generator = object()
    engine.schedule_policy = _Schedule()
    engine._weight_version = weight_version
    engine._generate_lock = threading.Lock()
    engine._shutdown_lock = threading.Lock()
    engine._shutdown_requested = False
    engine._shutdown_complete = False
    return engine


def test_generate_chunks_only_the_frontier_and_fills_canonical_video_output() -> None:
    sample = _sample(sde_indices=[])
    original_frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    engine = _bare_engine(forward_batch_size=2, weight_version=7)
    calls: list[tuple[list[str], list[int]]] = []
    next_row = 0

    def drive(prompts, params, sigmas, seeds):
        nonlocal next_row
        assert params is original_frontier.sampling_params
        torch.testing.assert_close(sigmas, torch.tensor([1.0, 0.5, 0.0]))
        calls.append((list(prompts), list(seeds)))
        batch = len(prompts)
        rows = torch.arange(next_row, next_row + batch, dtype=torch.float32)
        next_row += batch
        return {
            "trajectory": rows.view(batch, 1, 1, 1, 1, 1).expand(batch, 3, 1, 1, 1, 1).clone(),
            "decoded": rows.view(batch, 1, 1, 1, 1).expand(batch, 1, 1, 2, 2).clone(),
            "log_probs": None,
            "text_embeds": [row.view(1, 1, 1).expand(1, 2, 3).clone() for row in rows],
            "text_masks": [torch.ones(1, 2) for _ in rows],
            "neg_embeds": [],
            "neg_masks": [],
        }

    engine._drive_fastvideo = drive
    generated = engine.generate(sample)
    frontier = generated.frontier_gen_part(DiffusionSamplingParams)

    assert generated.parts[0] is sample.parts[0]
    assert frontier.sample_ids == original_frontier.sample_ids
    assert frontier.sampling_params is original_frontier.sampling_params
    assert frontier.weight_version == 7
    assert list(frontier.primitives) == ["video"]
    assert [float(video.frames[0, 0, 0, 0]) for video in frontier.primitives["video"].to_list()] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert frontier.segment is not None
    assert frontier.segment.latents[:, 0, 0, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert frontier.segment.sde_indices is None
    assert isinstance(frontier.conditions["text"], TextEmbedCondition)
    assert frontier.conditions["text"].embeds[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert [prompts for prompts, _seeds in calls] == [["alpha", "alpha"], ["beta", "beta"]]
    assert [seed for _prompts, seeds in calls for seed in seeds] == [
        _derive_group_seed(17, sample_id) for sample_id in original_frontier.sample_ids
    ]


def test_seed_keys_follow_sample_or_group_lineage() -> None:
    engine = _bare_engine()

    distinct = _sample(init_same_noise=False)
    distinct_params = distinct.frontier_gen_part(DiffusionSamplingParams).sampling_params
    distinct_seeds = engine._per_sample_seeds(distinct, distinct_params)
    assert len(set(distinct_seeds)) == 4

    shared = _sample(init_same_noise=True)
    shared_params = shared.frontier_gen_part(DiffusionSamplingParams).sampling_params
    shared_seeds = engine._per_sample_seeds(shared, shared_params)
    assert shared_seeds[0] == shared_seeds[1]
    assert shared_seeds[2] == shared_seeds[3]
    assert shared_seeds[0] != shared_seeds[2]

    no_seed = _sample(seed=None)
    no_seed_params = no_seed.frontier_gen_part(DiffusionSamplingParams).sampling_params
    assert engine._per_sample_seeds(no_seed, no_seed_params) == [
        _derive_group_seed(0, sample_id) for sample_id in no_seed.frontier_gen_part(DiffusionSamplingParams).sample_ids
    ]


def test_sde_window_preserves_none_versus_explicit_empty() -> None:
    assert _resolve_sde_window(None, 3) == (None, [0, 1, 2])
    assert _resolve_sde_window([], 3) == ([], [])
    assert _resolve_sde_window([2, 0, 2], 3) == ([0, 2], [0, 2])
    with pytest.raises(ValueError, match="out of range"):
        _resolve_sde_window([3], 3)


@pytest.mark.parametrize("sde_indices", [None, []])
def test_fastvideo_wire_unshifts_sigmas_and_preserves_sde_spelling(
    monkeypatch: pytest.MonkeyPatch,
    sde_indices,
) -> None:
    class SamplingParam:
        pass

    class RLData:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class ForwardBatch:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    ForwardBatch.RLData = RLData

    modules = {
        "fastvideo": ModuleType("fastvideo"),
        "fastvideo.configs": ModuleType("fastvideo.configs"),
        "fastvideo.configs.sample": ModuleType("fastvideo.configs.sample"),
        "fastvideo.configs.sample.base": ModuleType("fastvideo.configs.sample.base"),
        "fastvideo.pipelines": ModuleType("fastvideo.pipelines"),
        "fastvideo.utils": ModuleType("fastvideo.utils"),
    }
    modules["fastvideo.configs.sample.base"].SamplingParam = SamplingParam
    modules["fastvideo.pipelines"].ForwardBatch = ForwardBatch
    modules["fastvideo.utils"].shallow_asdict = lambda value: dict(vars(value))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    captured = []

    class Executor:
        def execute_forward(self, batch, _args):
            captured.append(batch)
            rl_data = SimpleNamespace(
                trajectory_latents=torch.zeros(3, 1, 1, 1, 1),
                log_probs=None,
            )
            return SimpleNamespace(
                rl_data=rl_data,
                trajectory_latents=None,
                output=torch.zeros(1, 1, 1, 2, 2),
                prompt_embeds=[torch.zeros(1, 2, 3)],
                prompt_attention_mask=[torch.ones(1, 2)],
                negative_prompt_embeds=[],
                negative_attention_mask=[],
            )

    engine = object.__new__(FastVideoRolloutEngine)
    engine.cfg = SimpleNamespace(native_logprob=False)
    engine.model_config = SimpleNamespace(shift=5.0)
    engine.strategy = SimpleNamespace(canonical_name="dance")
    engine._fastvideo_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(flow_shift=5.0),
        VSA_sparsity=0.0,
    )
    engine._generator = SimpleNamespace(executor=Executor())
    params = DiffusionSamplingParams(
        num_inference_steps=2,
        height=8,
        width=8,
        num_frames=1,
        seed=None,
        eta=0.7,
        sde_indices=sde_indices,
    )
    shifted_sigmas = torch.tensor([1.0, 5.0 / 6.0, 0.0])

    raw = engine._drive_fastvideo(["alpha"], params, shifted_sigmas, [23])

    assert raw["trajectory"].shape[0] == 1
    assert len(captured) == 1
    assert captured[0].seed == 23
    assert captured[0].sigmas == pytest.approx([1.0, 0.5])
    if sde_indices is None:
        assert captured[0].rl_data.sde_step_indices is None
    else:
        assert captured[0].rl_data.sde_step_indices == []


def test_checkpoint_version_changes_only_after_a_successful_load() -> None:
    engine = _bare_engine(weight_version=4)
    engine._last_weights_path = None

    class Generator:
        fail = True

        def update_transformer_weights_from_path(self, path):
            if self.fail:
                raise RuntimeError(path)

    generator = Generator()
    engine._generator = generator

    with pytest.raises(RuntimeError, match="bad"):
        engine.update_weights_from_path("bad")
    assert engine._weight_version == 4
    assert engine._last_weights_path is None

    generator.fail = False
    engine.update_weights_from_path("good")
    assert engine._weight_version == 5
    assert engine._last_weights_path == "good"


def test_weight_update_waits_for_generation_and_version_stamp_is_atomic() -> None:
    engine = _bare_engine(weight_version=4)
    sample = _sample(sde_indices=[])
    generation_entered = threading.Event()
    release_generation = threading.Event()
    update_attempted = threading.Event()
    update_loaded = threading.Event()
    results = []
    errors = []

    def generate_core(value):
        generation_entered.set()
        assert release_generation.wait(timeout=2.0)
        return value

    class Generator:
        def update_transformer_weights_from_path(self, _path):
            update_loaded.set()

    engine._generate_core = generate_core
    engine._generator = Generator()

    def run_generate():
        try:
            results.append(engine._generate_locked(sample))
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    def run_update():
        update_attempted.set()
        try:
            engine.update_weights_from_path("next")
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    generate_thread = threading.Thread(target=run_generate)
    update_thread = threading.Thread(target=run_update)
    generate_thread.start()
    assert generation_entered.wait(timeout=2.0)
    update_thread.start()
    assert update_attempted.wait(timeout=2.0)
    assert not update_loaded.wait(timeout=0.05)

    release_generation.set()
    generate_thread.join(timeout=2.0)
    update_thread.join(timeout=2.0)

    assert not generate_thread.is_alive()
    assert not update_thread.is_alive()
    assert errors == []
    assert results[0].frontier_gen_part(DiffusionSamplingParams).weight_version == 4
    assert update_loaded.is_set()
    assert engine._weight_version == 5


def test_wake_restores_cached_weights_without_bumping_version_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[str] = []

    class Generator:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.shutdown_calls = 0

        def update_transformer_weights_from_path(self, path):
            restored.append(path)
            if self.fail:
                raise RuntimeError("restore failed")

        def shutdown(self):
            self.shutdown_calls += 1

    next_generator = Generator()

    class VideoGenerator:
        @staticmethod
        def from_fastvideo_args(_args):
            return next_generator

    monkeypatch.setitem(sys.modules, "fastvideo", SimpleNamespace(VideoGenerator=VideoGenerator))
    monkeypatch.setattr(
        engine_module.FastVideoPorts,
        "reserve",
        classmethod(lambda _cls: SimpleNamespace(master_port=12345)),
    )

    engine = _bare_engine(weight_version=9)
    engine._is_offloaded = True
    engine._fastvideo_args = SimpleNamespace(master_port=0)
    engine._last_weights_path = "cached"
    engine.wake_up()
    assert restored == ["cached"]
    assert engine._generator is next_generator
    assert not engine._is_offloaded
    assert engine._weight_version == 9

    failing_generator = Generator(fail=True)
    next_generator = failing_generator
    engine._generator = None
    engine._is_offloaded = True
    with pytest.raises(RuntimeError, match="restore failed"):
        engine.wake_up()
    assert engine._generator is None
    assert engine._is_offloaded
    assert failing_generator.shutdown_calls == 1
    assert engine._weight_version == 9


def test_shutdown_is_idempotent_and_closes_generation_admission() -> None:
    engine = _bare_engine()

    class Generator:
        shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    generator = Generator()
    engine._generator = generator
    engine.shutdown()
    engine.shutdown()

    assert generator.shutdown_calls == 1
    assert engine._shutdown_requested
    assert engine._shutdown_complete
    assert engine._is_offloaded
    with pytest.raises(RuntimeError, match="called after shutdown"):
        engine._generate_locked(_sample())
    with pytest.raises(RuntimeError, match="called after shutdown"):
        engine.wake_up()
    with pytest.raises(RuntimeError, match="called after shutdown"):
        engine.update_weights_from_path("late")
