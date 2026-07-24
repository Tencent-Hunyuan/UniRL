from pathlib import Path

from omegaconf import OmegaConf


def test_trainside_recipe_threads_hidden_state_skip_to_pipeline():
    repo_root = Path(__file__).parents[2]
    recipe = OmegaConf.load(repo_root / "examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml")

    assert recipe.pipeline.hidden_state_skip_layer == 2
    assert (recipe.sampling.height, recipe.sampling.width, recipe.sampling.num_frames) == (720, 1280, 5)
    recipe.bundle.config.hidden_state_skip_layer = 0
    assert recipe.pipeline.hidden_state_skip_layer == 0


def test_sglang_recipe_uses_expected_hunyuan_settings():
    repo_root = Path(__file__).parents[2]
    recipe = OmegaConf.load(repo_root / "examples/diffusion/hunyuan_video/hunyuan_video_t2v_sglang.yaml")

    assert recipe.rollout.config.model_family == "hunyuan_video"
    assert recipe.rollout.config.populate_conditions is True
    assert recipe.model_config.use_lora is False
    assert recipe.model_config.hidden_state_skip_layer == 2
    assert recipe.pipeline.hidden_state_skip_layer == 2
    assert recipe.backend.lora_cfg.rank == 64
    assert recipe.backend.lora_cfg.alpha == 256
    assert recipe.batch_size == 8
    assert recipe.sampling.samples_per_prompt == 8
    assert recipe.sampling.num_inference_steps == 10
    assert (recipe.sampling.height, recipe.sampling.width, recipe.sampling.num_frames) == (720, 1280, 5)
    assert recipe.sampling.scheduler.num_sde_steps == 4
    assert recipe.algorithm.old_logp_source == "rollout"
    assert recipe.sync._target_.endswith("TensorWeightSync")
    assert recipe.sync.lora_merged is True

    recipe.model_config.hidden_state_skip_layer = 0
    assert recipe.pipeline.hidden_state_skip_layer == 0
