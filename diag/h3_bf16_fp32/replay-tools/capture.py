"""checkpoint-20单次192条采样，保留原训练入口的数据/采样/打分方法。"""
import json
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf

from runtime import R, T, build_trainer, close, configuration, start_ray
from unirl.trainer.residency import Role
from unirl.distributed.tensor.ref import hydrate
from unirl.utils.h3_checkup_data import atomic_json

cfg = configuration()
(T / 'batch').mkdir(exist_ok=True)
OmegaConf.save(cfg, T / 'batch/config.yaml')
start_ray()
trainer = None
started = time.time()
try:
    trainer = build_trainer(cfg)
    print('CHECKUP_MODEL_BUILT', flush=True)
    loaded = trainer.backend.checkup_load_live(str(R / 'runs/long192-klema/ckpt/checkpoint-20'), str(T / 'batch'))
    print('CHECKUP_LIVE_LOADED', loaded, flush=True)
    for _ in range(20):
        trainer.data_source.get_samples(24)
    inputs = trainer.data_source.get_samples(24)
    prompts = [{'index': i, 'sample_id': sid, 'prompt': text, 'half': 'A' if i < 12 else 'B'}
               for i, (sid, text) in enumerate(zip(inputs.parts[0].sample_ids,
                                                  inputs.parts[0].primitives['text'].texts, strict=True))]
    expected = json.loads((T / 'results/capture-cpu-prompts.json').read_text())
    assert prompts == expected
    atomic_json(T / 'batch/prompts.json', prompts)
    request = trainer._build_per_sample_engine_request(inputs, 20)
    print('CHECKUP_SAMPLING_BEGIN prompts=24 rows=192 rollout_id=20', flush=True)
    sample, reward = trainer._rollout_and_score([request], sync_weights=False, rollout_id=20)
    print(f'CHECKUP_SAMPLING_END mean_reward={reward}', flush=True)
    part = sample.parts[-1]
    assert len(part.sample_ids) == 192 and len(set(part.sample_ids)) == 192
    rows = [{'sample_id': sid, 'root_id': root, 'reward': float(r), 'advantage': float(a),
             'sde_index': int(k), 'prompt_index': i // 8, 'half': 'A' if i < 96 else 'B'}
            for i, (sid, root, r, a, k) in enumerate(zip(part.sample_ids, sample.root_group_ids(-1),
                 part.rewards.tolist(), part.advantages.tolist(), hydrate(part.segment.sde_index_per_sample).tolist(), strict=True))]
    assert all(len({x['root_id'] for x in rows[i:i+8]}) == 1 for i in range(0, 192, 8))
    atomic_json(T / 'batch/rows.json', rows)
    trainer._residency.enter(Role.TRAIN)
    trainer.stack.capture_part(part, str(T / 'batch'))
    unchanged = trainer.backend.checkup_verify()
    assert (T / 'batch/RANKS_DONE').exists()
    atomic_json(T / 'batch/manifest.json', {'schema': 1, 'rollout_id': 20, 'checkpoint_step': 20,
        'source_commit': 'eabc9eb026368cbe53c2db5c3d5b2db27914ab00', 'samples': 192, 'prompts': 24,
        'beta': 0.004, 'mean_reward': reward, 'wall_seconds': time.time()-started,
        'verification': unchanged, 'rank_files': [f'state_rank{i}.pt' for i in range(8)]})
    (T / 'batch/DONE').write_text('capture completed, all ranks verified, zero updates\n')
    print('CHECKUP_CAPTURE_DONE', flush=True)
finally:
    close(trainer)
