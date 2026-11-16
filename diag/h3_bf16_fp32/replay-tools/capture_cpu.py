"""capture最小CPU关卡：真实LoRA结构、真实游标、真实Part序列化。"""
import hashlib
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

from runtime import R, T, configuration
from unirl.models.minimax_h3.vendor.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from unirl.train.lora import inject_lora
from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment
from unirl.utils.h3_checkup_data import assert_same, atomic_json, atomic_save, canonical, live_state

started = time.time()
cfg = configuration()
state = live_state(R / 'runs/long192-klema/ckpt/checkpoint-20')
model_cfg = Path(cfg.bundle.config.pretrained_model_ckpt_path) / 'transformer/config.json'
with torch.device('meta'):
    model = MiniMaxH3Transformer3DModel.from_config(json.loads(model_cfg.read_text()))
    lc = cfg.backend.lora_cfg
    inject_lora(model, rank=lc.rank, alpha=lc.alpha, target_modules=list(lc.target_modules),
                module_prefix=lc.module_prefix, dropout=lc.dropout, bias=lc.bias, task_type=lc.task_type)
expected = {canonical(n): list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
assert expected.keys() == state.keys(), (expected.keys()-state.keys(),state.keys()-expected.keys())
assert all(expected[n] == list(t.shape) for n, t in state.items())
print('PASS live LoRA: 600 names and full shapes match real model on meta', flush=True)
orders = []
for _ in range(2):
    source = instantiate(cfg.data_source)
    batches = [source.get_samples(24) for _ in range(21)]
    chosen = batches[-1].parts[0]
    assert len(chosen.sample_ids) == 24
    orders.append([{'index': i, 'sample_id': sid, 'prompt': text, 'half': 'A' if i < 12 else 'B'}
                   for i, (sid, text) in enumerate(zip(chosen.sample_ids,
                                                       chosen.primitives['text'].texts, strict=True))])
assert orders[0] == orders[1]
atomic_json(T / 'results/capture-cpu-prompts.json', orders[0])
print('PASS cursor: independent DataLoaders agree on batch21, 24 prompts', flush=True)
part = Part(sample_ids=[f'prompt:{i//8}/sample:{i%8}' for i in range(192)],
            segment=LatentSegment(latents=torch.randn(192,3,2,4,dtype=torch.float16),
              aux_latents=torch.randn(192,3,2,4,dtype=torch.float16), sigmas=torch.linspace(1,0,11),
              sde_logp=torch.randn(192,1), sde_indices=torch.tensor([0]),
              sde_index_per_sample=torch.arange(24).remainder(5).repeat_interleave(8),
              policy_loss_weights=torch.ones(192,1)), rewards=torch.randn(192), advantages=torch.randn(192))
payload = {'part': part, 'parameters': {'one_real_parameter': state[next(iter(state))].clone()}}
p = T / 'results/capture-cpu-roundtrip.pt'
atomic_save(p, payload)
assert_same(payload, torch.load(p, weights_only=False, map_location='cpu'))
p.unlink()
result = {'status':'PASS', 'gpu_used':False, 'live_parameters':600, 'live_shape_match':True,
          'cursor_batch':21, 'cursor_prompts':24, 'two_dataloaders_identical':True,
          'part_roundtrip_exact':True, 'seconds':time.time()-started,
          'source_data_sha256':hashlib.sha256(Path(cfg.data_source.args.run.data_path).read_bytes()).hexdigest()}
atomic_json(T / 'results/cpu-checks-capture.json', result)
print(json.dumps(result), flush=True)
