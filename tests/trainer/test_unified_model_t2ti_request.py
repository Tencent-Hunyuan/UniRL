from __future__ import annotations

from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.sampling import ARSamplingParams
from unirl.utils.scheduler_utils import AllSDEScheduler


def test_unified_request_authors_prompt_level_bagel_noise_recipe() -> None:
    trainer = object.__new__(UnifiedModelTrainer)
    trainer.sampling_params = {
        "ar": ARSamplingParams(samples_per_prompt=3, seed=11),
        "diffusion": BagelDiffusionParams(
            samples_per_prompt=1,
            num_inference_steps=2,
            sde_indices=[0],
            seed=17,
            height=32,
            width=48,
        ),
    }
    trainer._noise_latent_shape = [6, 64]
    inputs = RolloutInputs(
        sample_ids=["prompt-0", "prompt-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["zero", "one"])},
        metadata=[{"row": 0}, {"row": 1}],
    )

    req = UnifiedModelTrainer._build_req(trainer, inputs, rollout_id=9)

    assert req.sample_ids == inputs.sample_ids
    assert req.stage_config == {"rollout_id": 9}
    assert req.init_noise_group_ids == ["r9:prompt-0", "r9:prompt-1"]
    assert req.init_noise_latent_shape == [6, 64]
    assert req.sampling_params["diffusion"].sde_indices == [0]
    assert req.sampling_params["diffusion"].scheduler is None


def test_unified_request_can_disable_driver_authored_noise() -> None:
    trainer = object.__new__(UnifiedModelTrainer)
    trainer.sampling_params = {
        "ar": ARSamplingParams(samples_per_prompt=2),
        "diffusion": BagelDiffusionParams(num_inference_steps=2, sde_indices=[]),
    }
    trainer._noise_latent_shape = None
    inputs = RolloutInputs(
        sample_ids=["prompt-0"],
        group_ids=["group-0"],
        primitives={"text": Texts(texts=["zero"])},
    )

    req = UnifiedModelTrainer._build_req(trainer, inputs, rollout_id=4)

    assert req.stage_config == {"rollout_id": 4}
    assert req.init_noise_group_ids == []
    assert req.init_noise_latent_shape is None


def test_deterministic_eval_clears_training_sde_schedule() -> None:
    trainer = object.__new__(UnifiedModelTrainer)
    trainer.eval_eta = 0.0
    trainer.eval_cfg_text_scale = 1.0
    trainer.sampling_params = {
        "ar": ARSamplingParams(samples_per_prompt=2),
        "diffusion": BagelDiffusionParams(
            num_inference_steps=4,
            eta=0.8,
            scheduler=AllSDEScheduler(4, timestep_fraction=[0.0, 0.5], num_sde_steps=1),
        ),
    }

    eval_params = trainer._eval_sampling_params()

    assert eval_params["diffusion"].eta == 0.0
    assert eval_params["diffusion"].scheduler is None
    assert eval_params["diffusion"].sde_indices == []
