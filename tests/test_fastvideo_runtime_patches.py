from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch


def _load_patch_module(name: str):
    path = Path(__file__).parents[1] / "unirl" / "rollout" / "engine" / "fastvideo" / "_patches" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_fastvideo_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_conditions = _load_patch_module("conditions")
_denoising = _load_patch_module("denoising")
_ForwardResultPipe = _conditions._ForwardResultPipe
_CONTEXT = _denoising._CONTEXT
_TransitionContext = _denoising._TransitionContext
_sde_step_with_logprob = _denoising._sde_step_with_logprob


class _Scheduler:
    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.8, 0.4, 0.0])

    def index_for_timestep(self, timestep: float) -> int:
        values = [1.0, 0.8, 0.4]
        return min(range(len(values)), key=lambda index: abs(values[index] - float(timestep)))


def test_dance_transition_uses_explicit_eta() -> None:
    scheduler = _Scheduler()
    sample = torch.ones(1, 1, 1, 1, 1)
    model_output = torch.zeros_like(sample)
    prev_sample = torch.ones_like(sample)

    _, log_prob, _, transition_std = _sde_step_with_logprob(
        scheduler,
        model_output,
        torch.tensor(0.8),
        sample,
        prev_sample=prev_sample,
        eta=0.25,
        sde_type="dance",
    )

    assert torch.allclose(transition_std.flatten(), torch.tensor([0.25 * (0.4**0.5)]))
    assert torch.isfinite(log_prob).all()


def test_flow_first_step_uses_second_sigma_as_guard() -> None:
    scheduler = _Scheduler()
    sample = torch.ones(1, 1, 1, 1, 1)
    model_output = torch.zeros_like(sample)

    _, _, _, transition_std = _sde_step_with_logprob(
        scheduler,
        model_output,
        torch.tensor(1.0),
        sample,
        prev_sample=torch.ones_like(sample),
        eta=0.3,
        sde_type="flow",
    )

    expected = (1.0 / (1.0 - 0.8)) ** 0.5 * 0.3 * (0.2**0.5)
    assert torch.allclose(transition_std.flatten(), torch.tensor([expected]), rtol=1e-5, atol=1e-6)


def test_transition_context_makes_tail_deterministic() -> None:
    scheduler = _Scheduler()
    sample = torch.ones(1, 1, 1, 1, 1)
    model_output = torch.full_like(sample, 2.0)
    token = _CONTEXT.set(
        _TransitionContext(
            eta=0.3,
            sde_type="dance",
            sde_step_indices=frozenset({0}),
            collect_kl=False,
            timestep_dtype="long",
            generator=None,
        )
    )
    try:
        result, log_prob, mean, transition_std = _sde_step_with_logprob(
            scheduler,
            model_output,
            torch.tensor(0.8),
            sample,
        )
    finally:
        _CONTEXT.reset(token)

    expected = sample + (0.4 - 0.8) * model_output
    assert torch.equal(result, expected)
    assert torch.equal(mean, expected)
    assert torch.count_nonzero(log_prob) == 0
    assert torch.count_nonzero(transition_std) == 0


def test_contract_patch_is_idempotent() -> None:
    @dataclass
    class RLData:
        enabled: bool = False
        collect_log_probs: bool = True
        store_trajectory: bool = True
        keep_trajectory_on_cpu: bool = False

    class ForwardBatch:
        pass

    ForwardBatch.RLData = RLData
    module = types.ModuleType("fastvideo.pipelines.pipeline_batch_info")
    RLData.__module__ = module.__name__
    ForwardBatch.__module__ = module.__name__
    module.ForwardBatch = ForwardBatch
    names = ("fastvideo", "fastvideo.pipelines", "fastvideo.pipelines.pipeline_batch_info")
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules["fastvideo"] = types.ModuleType("fastvideo")
        sys.modules["fastvideo.pipelines"] = types.ModuleType("fastvideo.pipelines")
        sys.modules["fastvideo.pipelines.pipeline_batch_info"] = module

        contracts = _load_patch_module("contracts")
        contracts.patch_contracts()
        patched = ForwardBatch.RLData
        contracts.patch_contracts()

        assert ForwardBatch.RLData is patched
        assert {"sde_step_indices", "sde_type"} <= {field.name for field in fields(patched)}
        assert patched(sde_step_indices=[0, 1], sde_type="flow").sde_type == "flow"
        restored = pickle.loads(pickle.dumps(patched(sde_step_indices=[0], sde_type="dance")))
        assert restored.sde_step_indices == [0]
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_forward_pipe_carries_exact_conditions_to_cpu() -> None:
    sent = []

    class Connection:
        def send(self, payload):
            sent.append(payload)

    output = types.SimpleNamespace(
        rl_data=types.SimpleNamespace(log_probs=torch.ones(1)),
        trajectory_latents=torch.ones(1),
        trajectory_timesteps=torch.ones(1),
        prompt_embeds=[torch.ones(1)],
        negative_prompt_embeds=None,
        prompt_attention_mask=[torch.ones(1)],
        negative_attention_mask=None,
    )
    wrapper = types.SimpleNamespace(_unirl_last_forward_batch=output)
    owner = types.SimpleNamespace(worker=wrapper)
    pipe = _ForwardResultPipe(Connection(), owner)

    pipe.send({"output_batch": torch.zeros(1)})

    assert torch.equal(sent[0]["prompt_embeds"][0], torch.ones(1))
    assert torch.equal(sent[0]["rl_data"].log_probs, torch.ones(1))
    assert owner.worker._unirl_last_forward_batch is None


def test_timestep_patch_scopes_numpy_sigmas_to_forward_call() -> None:
    observed = []

    class TimestepPreparationStage:
        def __init__(self):
            self.scheduler = types.SimpleNamespace(
                sigmas=torch.tensor([1.0, 0.5005, 0.0]),
                timesteps=torch.tensor([1000, 500]),
                config=types.SimpleNamespace(num_train_timesteps=1000),
            )

        def forward(self, batch, fastvideo_args):
            observed.append(batch.sigmas)
            batch.timesteps = self.scheduler.timesteps
            return batch

    module_name = "fastvideo.pipelines.stages.timestep_preparation"
    module = types.ModuleType(module_name)
    module.TimestepPreparationStage = TimestepPreparationStage
    previous = sys.modules.get(module_name)
    try:
        sys.modules[module_name] = module
        timesteps = _load_patch_module("timesteps")
        timesteps.patch_timesteps()
        batch = types.SimpleNamespace(sigmas=[1.0, 0.5])
        args = types.SimpleNamespace(_unirl_custom_sigmas_dtype="float32")
        stage = TimestepPreparationStage()
        stage.forward(batch, args)
        assert isinstance(observed[0], np.ndarray)
        assert observed[0].dtype == np.float32
        assert batch.sigmas == [1.0, 0.5]
        assert torch.allclose(batch.timesteps, torch.tensor([1000.0, 500.5]))
        assert torch.equal(stage.scheduler.timesteps, batch.timesteps)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def test_multiproc_executor_retries_address_collision() -> None:
    multiproc = _load_patch_module("multiproc")
    attempts = []

    def original_init(executor):
        attempts.append(executor.fastvideo_args.master_port)
        if len(attempts) == 1:
            raise RuntimeError("TCPStore failed to bind: EADDRINUSE")

    previous = multiproc._ORIGINAL_INIT_EXECUTOR
    try:
        multiproc._ORIGINAL_INIT_EXECUTOR = original_init
        executor = types.SimpleNamespace(fastvideo_args=types.SimpleNamespace(master_port=29500))
        multiproc._patched_init_executor(executor)
    finally:
        multiproc._ORIGINAL_INIT_EXECUTOR = previous

    assert attempts == [29500, None]
