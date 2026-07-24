from pathlib import Path

from omegaconf import OmegaConf


def test_trainside_recipe_threads_hidden_state_skip_to_pipeline():
    repo_root = Path(__file__).parents[2]
    recipe = OmegaConf.load(repo_root / "examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml")

    assert recipe.pipeline.hidden_state_skip_layer == 2
    recipe.bundle.config.hidden_state_skip_layer = 0
    assert recipe.pipeline.hidden_state_skip_layer == 0
