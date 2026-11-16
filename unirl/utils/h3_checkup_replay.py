"""同一批Part上运行真实FlashGRPO loss；仅导出梯度，不调用优化器更新。"""
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.train.stack.single import TrainStack
from unirl.utils.h3_checkup_data import (
    assert_same, atomic_json, atomic_save, canonical, layout, local_tensor, map_tensors, split_factor,
)
from unirl.utils.h3_checkup_ratios import write_ratio_json


def tensors(obj):
    found = []
    map_tensors(obj, lambda x: found.append(x) or x)
    return found


def audit_fp32(model):
    from unirl.models.minimax_h3.vendor.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed, MiniMaxH3Transformer3DModel,
    )

    # Counts are cumulative module-boundary tensor encounters, not unique tensors.
    counts = {'forward_tensors': 0, 'backward_tensors': 0,
              'exempt_position_ids_float64': 0, 'exempt_rope_inputs_float64': 0}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point():
            assert value.dtype == torch.float32, (name, value.dtype)

    def check(obj, key):
        assert not torch.is_autocast_enabled('cuda'), 'autocast must be disabled'
        for value in tensors(obj):
            if value.is_floating_point():
                assert value.dtype == torch.float32, (key, value.dtype, value.shape)
                counts[key] += 1

    def backward(grad):
        check(grad, 'backward_tensors')
        return grad

    def coordinate(value, key):
        # packing.py deliberately builds the shared A/V rotary clock in float64;
        # MiniMaxH3RotaryPosEmbed.forward converts it to float32 before arithmetic.
        # Never exempt lower precision, differentiable activations, or other shapes.
        if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            assert value.ndim == 2 and value.shape[-1] == 3 and not value.requires_grad, (
                key, value.dtype, value.shape, value.requires_grad)
            counts[key] += 1
        else:
            check(value, 'forward_tensors')

    def before(module, args, kwargs):
        if module is model and isinstance(module, MiniMaxH3Transformer3DModel):
            if 'position_ids' in kwargs:
                coordinate(kwargs['position_ids'], 'exempt_position_ids_float64')
                kwargs = {key: value for key, value in kwargs.items() if key != 'position_ids'}
        elif isinstance(module, MiniMaxH3RotaryPosEmbed):
            if args:
                coordinate(args[0], 'exempt_rope_inputs_float64')
                args = args[1:]
            if 'position_ids' in kwargs:
                coordinate(kwargs['position_ids'], 'exempt_rope_inputs_float64')
                kwargs = {key: value for key, value in kwargs.items() if key != 'position_ids'}
        check((args, kwargs), 'forward_tensors')

    def after(module, args, kwargs, output):
        check(output, 'forward_tensors')
        for value in tensors(output):
            if value.requires_grad and value.is_floating_point():
                value.register_hook(backward)

    for module in model.modules():
        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after, with_kwargs=True)
    return counts


def write_full_meta(directory, variant, split, micro):
    metas = [json.loads((directory / f'meta_rank{r}.json').read_text()) for r in range(8)]
    parameters = {}
    for name, info in metas[0]['parameters'].items():
        shared = {key: info[key] for key in ['global_shape', 'placements', 'mesh_shape', 'mesh_rank_order']}
        shards = {}
        for rank, meta in enumerate(metas):
            other = meta['parameters'][name]
            assert all(other[key] == value for key, value in shared.items())
            shards[str(rank)] = {key: other[key] for key in ['offset', 'local_shape']}
        parameters[name] = {**shared, 'shards': shards}
    for meta in metas:
        assert meta['parameters'].keys() == parameters.keys()
        assert meta['lora_unchanged'] and meta['optimizer_updates'] == 0
    result = {'schema': 1, 'variant': variant, 'split': split, 'micro': micro, 'world_size': 8,
        'mesh_rank_order': list(range(8)), 'gradient_dtype': 'float32',
        'compute_dtype': 'fp32' if variant == 'V3' else 'bf16',
        'loss': {'old_logp_source': 'this_forward_new_logp.detach()', 'design_revision': 'r3b',
                 'beta': 0.004, 'population_rows': 192, 'half_rows': 96, 'local_rows': 24,
                 'micro_mean_weight': micro / 24, 'rank_multipliers': [split_factor(split, r) for r in range(8)]},
        'parameters': parameters, 'seconds': max(m['seconds'] for m in metas),
        'optimizer_updates': 0, 'lora_unchanged': True, 'precision_audits': [m['precision_audit'] for m in metas]}
    write_ratio_json(directory, variant, split, micro)
    atomic_json(directory / 'meta.json', result)
    (directory / 'DONE').write_text('all ranks complete; live LoRA unchanged; zero optimizer updates\n')


class ReplayStack(TrainStack):
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def replay_saved(self, batch_directory, output_directory, variant, micro):
        rank = dist.get_rank()
        assert dist.get_world_size() == 8
        assert variant in ['V1','V3','V2'] and self.micro_batch_size == micro
        backend = self.fsdp_backend
        assert self.num_updates_per_batch == 1 and self.algorithm.beta == 0.004
        assert not self.update_planner.shuffle_updates
        assert not self.algorithm.recomputes_anchor
        if variant in ['V2', 'V3']:
            assert not any(getattr(module, 'fused_projections', False) for module in backend.model.modules())
            for key in ['forward_prefetch', 'copy_engine_all_gather', 'defer_grad_sync', 'use_torch_compile']:
                assert not getattr(backend.fsdp_cfg, key)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.set_float32_matmul_precision('highest')
        if variant == 'V3':
            assert not backend.fsdp_cfg.mixed_precision
            audit = audit_fp32(backend.model)
        else:
            audit = {'mode': 'original mixed-bf16 parameter layout'}
        state = torch.load(Path(batch_directory) / f'state_rank{rank}.pt', map_location='cpu', weights_only=False)
        assert state['rank'] == rank and state['world_size'] == 8 and state['rollout_id'] == 20
        assert_same(backend._checkup_before, state['parameters'])
        part = state['part']
        assert part.batch_size == 24
        for start in range(0,24,8):
            assert len({sid.rsplit('/',1)[0] for sid in part.sample_ids[start:start+8]}) == 1
        if variant == 'V3':
            part = map_tensors(part, lambda t: t.float() if t.is_floating_point() else t)
        self._align_track_inputs(part)
        backend.model.train()
        micros = tuple((i,min(i+micro,24)) for i in range(0,24,micro))
        saved_tgr = part.segment.policy_loss_weights.clone()
        saved_adv = part.advantages.clone()
        saved_old = part.segment.sde_logp.clone()
        part = self._prepare_for_training(part, plans=(micros,), order=None)
        assert torch.equal(saved_tgr, part.segment.policy_loss_weights)
        assert torch.equal(saved_adv, part.advantages) and torch.equal(saved_old, part.segment.sde_logp)
        self.algorithm._checkup_on_policy = True
        parameter_names = {canonical(n) for n,p in backend.model.named_parameters() if p.requires_grad}
        assert len(parameter_names) == 600
        for split in ['full','A','B']:
            directory = Path(output_directory) / variant / split
            directory.mkdir(parents=True, exist_ok=True)
            assert not (directory / 'DONE').exists(), 'refuse overwriting complete gradients'
            self.algorithm._checkup_split = split
            self.algorithm._checkup_ratio_rows = []
            self.algorithm._checkup_log_backward = True
            print(f'CHECKUP_SPLIT_BEGIN variant={variant} split={split} rank={rank} micro={micro}', flush=True)
            dumps = []

            def dump_only(*, max_grad_norm):
                gradients, metadata = {}, {}
                for raw_name, parameter in backend.model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    name = canonical(raw_name)
                    grad = parameter.grad
                    assert grad is not None, name
                    info = layout(grad, rank)
                    assert info == backend._checkup_metadata[name]
                    value = local_tensor(grad).detach().float().cpu().clone()
                    assert torch.isfinite(value).all(), name
                    gradients[name], metadata[name] = value, info
                assert gradients.keys() == parameter_names
                atomic_save(directory / f'rank{rank}.pt', gradients)
                dumps.append(metadata)
                backend.optimizer.zero_grad(set_to_none=True)
                return 0.0

            backend.optimizer_step = dump_only
            started = time.perf_counter()
            try:
                with torch.autocast('cuda', enabled=False):
                    result = self._run_update(part, micros=micros, order=None, training_progress=20/199)
            finally:
                del backend.optimizer_step
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            assert len(dumps) == 1
            rows = torch.cat(self.algorithm._checkup_ratio_rows, dim=0)
            assert rows.shape == (24, 5)
            atomic_save(directory / f'ratio_rank{rank}.pt', {'rows': rows, 'sample_ids': list(part.sample_ids)})
            verify = backend.checkup_verify()
            assert torch.equal(saved_adv, part.advantages) and torch.equal(saved_old, part.segment.sde_logp)
            atomic_json(directory / f'meta_rank{rank}.json', {'rank':rank, 'variant':variant, 'split':split,
                'micro':micro, 'seconds':seconds, 'unmasked_global_loss':result.loss, 'parameters':dumps[0],
                'optimizer_updates':0, 'lora_unchanged':verify['lora_unchanged'], 'precision_audit':dict(audit),
                'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
            dist.barrier()
            if rank == 0:
                write_full_meta(directory,variant,split,micro)
            dist.barrier()
            print(f'CHECKUP_SPLIT_DONE variant={variant} split={split} rank={rank} seconds={seconds:.1f}', flush=True)
        return {'rank':rank, 'variant':variant, 'optimizer_updates':0, 'lora_unchanged':True}
