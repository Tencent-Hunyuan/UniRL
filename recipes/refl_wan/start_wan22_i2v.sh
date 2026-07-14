export PRETRAINED_MODEL="/apdcephfs_gy8/share_301869871/ysunlin/mmrl/.model_cache/Wan2.2-I2V-A14B-Diffusers"
export DATA_PATH="/apdcephfs_gy8/share_301869871/yohunawu/gy6/yohunawu/refl_wan_data/wan22_face_refl_prompts.jsonl"
export EVAL_DATA_PATH="${DATA_PATH}"
export FACE_MODEL_PATH="/apdcephfs_gy8/share_301869871/yohunawu/gy6/yohunawu/refl_wan_data/antelodev2"
export OUTPUT_DIR="outputs/wan22_face_refl_recipe_opt"

export REPORT_TO_WANDB=true
export WANDB_PROJECT="unirl-refl"
export WANDB_RUN_NAME="wan22_face_refl_recipe_opt"

RAY_ADDRESS=auto python -m recipes.refl_wan.run \
  num_devices=8 \
  2>&1 | tee ./wan22_i2v_refl.log