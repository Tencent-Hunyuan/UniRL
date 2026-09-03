from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from unirl.sde.unipc import UniPCStrategy


def _load_module(name: str, relative_path: str):
    path = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_unipc = _load_module(
    "_test_fastvideo_unipc",
    "unirl/rollout/engine/fastvideo/_patches/unipc.py",
)
_sigma_verify = _load_module(
    "_test_fastvideo_sigma_verify",
    "unirl/rollout/engine/sigma_verify.py",
)
FastVideoUniPCPlan = _unipc.FastVideoUniPCPlan
verify_engine_used_sigmas = _sigma_verify.verify_engine_used_sigmas


def _stub_wan_timestep_scale(monkeypatch) -> None:
    model_module = ModuleType("unirl.models.wan21.diffusion")
    model_module.WAN21DiffusionStep = type("WAN21DiffusionStep", (), {"TIMESTEP_SCALE": 1000.0})
    monkeypatch.setitem(sys.modules, "unirl.models.wan21.diffusion", model_module)


@pytest.mark.parametrize("value", ["1", "true", "yes", " TRUE "])
def test_fastvideo_rejects_legacy_integer_timestep_env(monkeypatch, value: str) -> None:
    monkeypatch.setenv("DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG", value)

    with pytest.raises(RuntimeError, match="floating WAN timesteps"):
        _unipc._require_float_wan_timesteps()


def test_fastvideo_requires_wan_timestep_scale(monkeypatch) -> None:
    _stub_wan_timestep_scale(monkeypatch)

    valid = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    assert _unipc._wan_timestep_scale(valid) == 1000.0

    invalid = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=999))
    with pytest.raises(RuntimeError, match="does not match"):
        _unipc._wan_timestep_scale(invalid)


@pytest.mark.parametrize(
    ("sde_type", "indices", "message"),
    [
        ("cps", (), "SDE type"),
        ("dance", (2, 1), "sorted unique"),
        ("flow", (1, 1), "sorted unique"),
        ("flow", (-1,), "sorted unique"),
    ],
)
def test_fastvideo_plan_rejects_unsupported_or_ambiguous_dispatch(
    sde_type: str,
    indices: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FastVideoUniPCPlan(sde_type=sde_type, sde_indices=indices)


def test_fastvideo_sigma_verifier_accepts_float_ticks_and_rejects_truncation() -> None:
    expected = torch.tensor([1.0, 5.0 / 6.0, 0.0])
    verify_engine_used_sigmas(
        torch.tensor([1000.0, 1000.0 * 5.0 / 6.0, 0.0]),
        expected=expected,
        engine_name="fastvideo test",
    )

    with pytest.raises(RuntimeError, match="value mismatch"):
        verify_engine_used_sigmas(
            torch.tensor([1000, 833, 0]),
            expected=expected,
            engine_name="fastvideo test",
        )


def test_fastvideo_external_sigmas_are_consumed_without_shift(monkeypatch) -> None:
    _stub_wan_timestep_scale(monkeypatch)
    scheduler_module = SimpleNamespace()

    class FlowUniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                num_train_timesteps=1000,
                final_sigmas_type="zero",
                solver_order=2,
                solver_type="bh2",
                lower_order_final=True,
                prediction_type="flow_prediction",
                thresholding=False,
            )
            self.predict_x0 = True
            self.disable_corrector = []
            self.solver_p = None

        def set_timesteps(
            self,
            num_inference_steps=None,
            device=None,
            sigmas=None,
            mu=None,
            shift=None,
            use_karras_sigmas=None,
            use_kerras_sigma=None,
        ) -> None:
            raise AssertionError("external sigmas must not reach FastVideo's shifting implementation")

    scheduler_module.FlowUniPCMultistepScheduler = FlowUniPCMultistepScheduler
    monkeypatch.setitem(
        sys.modules,
        "fastvideo.models.schedulers.scheduling_flow_unipc_multistep",
        scheduler_module,
    )

    _unipc._patch_scheduler_set_timesteps()
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(sigmas=[1.0, 5.0 / 6.0], device="cpu")

    torch.testing.assert_close(
        scheduler.sigmas,
        torch.tensor([1.0, 5.0 / 6.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scheduler.timesteps,
        torch.tensor([1000.0, 1000.0 * 5.0 / 6.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert scheduler.timesteps.dtype == torch.float32
    with pytest.raises(ValueError, match="strictly decreasing"):
        scheduler.set_timesteps(sigmas=[1.0, 1.0], device="cpu")


def test_fastvideo_helper_dispatches_sde_or_canonical_unipc(monkeypatch) -> None:
    calls = []
    denoising = ModuleType("fastvideo.pipelines.stages.denoising")

    def original(
        scheduler,
        model_output,
        timestep,
        sample,
        prev_sample=None,
        generator=None,
        deterministic=False,
        return_pixel_log_prob=False,
        return_dt_and_std_dev_t=False,
        eta=None,
        sde_type="dance",
    ):
        del scheduler, model_output, timestep, prev_sample, generator
        del deterministic, return_pixel_log_prob, eta
        calls.append(sde_type)
        output = (sample + 10, torch.ones(sample.shape[0]), sample, torch.ones(1))
        return (*output, torch.ones(1)) if return_dt_and_std_dev_t else output

    denoising.sde_step_with_logprob = original
    stages = ModuleType("fastvideo.pipelines.stages")
    stages.denoising = denoising
    pipelines = ModuleType("fastvideo.pipelines")
    pipelines.__path__ = []
    monkeypatch.setitem(sys.modules, "fastvideo.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "fastvideo.pipelines.stages", stages)
    monkeypatch.setitem(sys.modules, "fastvideo.pipelines.stages.denoising", denoising)
    _unipc._patch_denoising_step()

    class Scheduler:
        def __init__(self) -> None:
            self.sigmas = torch.tensor([1.0, 0.5, 0.0])
            self.timesteps = torch.tensor([1000.0, 500.0])
            self._unirl_unipc_strategy = UniPCStrategy()
            self._unirl_unipc_strategy.init_schedule(self.sigmas)

        def index_for_timestep(self, timestep) -> int:
            return int((self.timesteps == timestep).nonzero()[0].item())

    scheduler = Scheduler()
    sample = torch.tensor([[[[0.25, -0.5]]]])
    pred = torch.tensor([[[[0.1, -0.2]]]])

    plan = FastVideoUniPCPlan(sde_type="dance", sde_indices=(1,))
    token = _unipc._PLAN_CONTEXT.set(plan)
    try:
        deterministic_result = denoising.sde_step_with_logprob(
            scheduler,
            pred,
            scheduler.timesteps[0],
            sample,
            return_dt_and_std_dev_t=True,
        )
        torch.testing.assert_close(
            deterministic_result[0],
            sample + (scheduler.sigmas[1] - scheduler.sigmas[0]) * pred,
        )
        assert calls == []

        sde_result = denoising.sde_step_with_logprob(
            scheduler,
            pred,
            scheduler.timesteps[1],
            deterministic_result[0],
            return_dt_and_std_dev_t=True,
        )
    finally:
        _unipc._PLAN_CONTEXT.reset(token)
    assert calls == ["dance"]
    assert torch.equal(sde_result[0], deterministic_result[0] + 10)
