"""单变体full/A/B重放；独立进程加载同一live LoRA和固定Part。"""
import argparse
import os
import time

from omegaconf import OmegaConf

from runtime import R, T, build_trainer, close, configuration, start_ray
from unirl.trainer.residency import Role
from unirl.utils.h3_checkup_data import atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['V1','V3','V2'],required=True)
parser.add_argument('--micro',type=int,required=True)
args=parser.parse_args()
assert (T/'batch/DONE').exists()
assert args.micro in ([8,1] if args.variant=='V1' else [1,8] if args.variant=='V2' else [4,2])
cfg=configuration(args.variant,args.micro)
config_dir=T/'effective-configs'
config_dir.mkdir(exist_ok=True)
OmegaConf.save(cfg,config_dir/f'{args.variant}-m{args.micro}.yaml')
start_ray()
trainer=None
started=time.time()
try:
    trainer=build_trainer(cfg)
    trainer.backend.checkup_load_live(str(R/'runs/long192-klema/ckpt/checkpoint-20'),
                                     str(T/'verification'/args.variant/f'm{args.micro}'))
    # 重放不采样；显式让原rollout引擎释放VAE等辅助组件，避免占用训练显存。
    trainer.rollout.sleep()
    trainer._residency.enter(Role.TRAIN)
    result=trainer.stack.replay_saved(str(T/'batch'),str(T/'gradients'),args.variant,args.micro)
    unchanged=trainer.backend.checkup_verify()
    name=os.environ.get('CHECKUP_OUT_NAME',args.variant)
    assert all((T/'gradients'/name/split/'DONE').exists() for split in os.environ.get('CHECKUP_SPLITS','full,A,B').split(','))
    atomic_json(T/'results'/f'{name}-complete.json', {'variant':args.variant,'micro':args.micro,
        'wall_seconds':time.time()-started,'rank_results':result,'verification':unchanged,'optimizer_updates':0})
    print(f'CHECKUP_VARIANT_DONE variant={args.variant}',flush=True)
finally:
    close(trainer)
