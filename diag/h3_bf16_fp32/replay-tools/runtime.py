"""本任务私有Ray与原配置的诊断入口。"""
import os
from pathlib import Path

from omegaconf import OmegaConf

T = Path(os.environ['CHECKUP_ROOT'])
R = Path(os.environ['CHECKUP_OLD_ROOT'])
# Ray首次init会写HOME/.ray/auth_token；后续阶段将HOME也隔离到任务目录。
os.environ['HOME'] = str(T / 'runtime-home')
Path(os.environ['HOME']).mkdir(exist_ok=True)


def configuration(variant='capture', micro=8):
    cfg = OmegaConf.load(R / 'runs/long192-klema/hydra/.hydra/config.yaml')
    assert cfg.batch_size == 24 and cfg.sampling.samples_per_prompt == 8 and cfg.algorithm.beta == 0.004
    cfg.backend._target_ = 'unirl.utils.h3_checkup_capture.CheckupBackend'
    cfg.stack._target_ = ('unirl.utils.h3_checkup_capture.CaptureStack' if variant == 'capture'
                         else 'unirl.utils.h3_checkup_replay.ReplayStack')
    del cfg.backend.ema_cfg
    cfg.logging.report_to_wandb = False
    cfg.eval_interval = cfg.save_interval = 0
    cfg.load_dir = None
    cfg.save_dir = str(T / 'unused-checkpoints')
    cfg.stack.micro_batch_size = micro
    if variant in ['V2', 'V3']:
        cfg.bundle.config.frozen_adaln_cache = False
        cfg.bundle.config.rollout_keep_unsharded_layers = 0
        for key in ['forward_prefetch', 'copy_engine_all_gather', 'defer_grad_sync', 'use_torch_compile']:
            OmegaConf.update(cfg, 'backend.fsdp_cfg.' + key, False, force_add=True)
    if variant == 'V3':
        cfg.bundle.config.model_precision = 'fp32'
        cfg.backend.fsdp_cfg.param_dtype = 'fp32'
        cfg.backend.fsdp_cfg.master_dtype = 'fp32'
        cfg.backend.fsdp_cfg.mixed_precision = False
        cfg.sampling.autocast_precision = 'fp32'
        # 仅重放时升精度，不改变已保存的rollout数值。
        cfg.bundle.config.trajectory_precision = cfg.sampling.trajectory_precision = 'fp32'
    return cfg


def start_ray():
    import ray
    stage = os.environ['CHECKUP_STAGE']
    shm = Path(os.environ['CHECKUP_SHM']) / stage
    shm.mkdir(parents=True, exist_ok=True)
    os.environ.pop('RAY_ADDRESS', None)
    ray.init(num_gpus=8, include_dashboard=False, _temp_dir=str(shm / 'ray'),
             object_store_memory=28_000_000_000, _plasma_directory=str(shm),
             log_to_driver=True)


def build_trainer(cfg):
    from unirl.trainer.diffusion import DiffusionTrainer
    return DiffusionTrainer(cfg=cfg, batch_size=cfg.batch_size, bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline, backend_cfg=cfg.backend, rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward, algorithm_cfg=cfg.algorithm, stack_cfg=cfg.stack,
        reward_stack_cfg=cfg.reward_stack, data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling, logging_cfg=cfg.logging, adv_use_global_std=cfg.adv_use_global_std,
        eval_interval=0)


def close(trainer):
    import ray
    if trainer is not None:
        trainer.pool.shutdown()
    ray.shutdown()
