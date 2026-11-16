"""隔离的诊断 backend/stack：只装载live LoRA，capture绝不反向或更新。"""

from pathlib import Path

import torch
import torch.distributed as dist

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack.single import TrainStack
from unirl.types.sample import Part
from unirl.utils.h3_checkup_data import (
    assert_same, atomic_json, atomic_save, canonical, layout, live_state, local_slice, local_tensor,
)


def forbidden_update(*args, **kwargs):
    raise RuntimeError('CHECKUP_FORBIDDEN_OPTIMIZER_OR_SCHEDULER_UPDATE')


class CheckupBackend(FSDPBackend):
    def optimizer_step(self, *, max_grad_norm):
        return forbidden_update()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def checkup_load_live(self, checkpoint, output):
        rank = dist.get_rank()
        state = live_state(checkpoint)
        parameters = {canonical(n): p for n, p in self.model.named_parameters() if p.requires_grad}
        assert parameters.keys() == state.keys(), (parameters.keys() - state.keys(), state.keys() - parameters.keys())
        self._checkup_before = {}
        metadata = {}
        with torch.no_grad():
            for name, parameter in parameters.items():
                info = layout(parameter, rank)
                expected = local_slice(state[name], info)
                local = local_tensor(parameter)
                assert local.dtype == torch.float32
                local.copy_(expected)
                self._checkup_before[name] = local.detach().cpu().clone()
                assert torch.equal(self._checkup_before[name], expected)
                metadata[name] = info
        self.optimizer.step = forbidden_update
        if self.scheduler is not None:
            self.scheduler.step = forbidden_update
        assert self.ema is None, 'capture/replay does not construct or restore EMA'
        self._checkup_metadata = metadata
        self._checkup_output = Path(output)
        atomic_json(Path(output) / f'load_rank{rank}.json', {'rank': rank, 'parameters': metadata,
                    'loaded_live_only': True, 'optimizer_state_entries': len(self.optimizer.state),
                    'optimizer_step_count': self._optimizer_step_count})
        self.checkup_verify()
        seen = False

        def first_forward(module, args, kwargs):
            nonlocal seen
            if not seen:
                seen = True
                print(f'CHECKUP_MODEL_FORWARD rank={rank} grad_enabled={torch.is_grad_enabled()}', flush=True)

        self.model.register_forward_pre_hook(first_forward, with_kwargs=True)
        return {'rank': rank, 'live_parameters': len(parameters)}

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def checkup_verify(self):
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                assert torch.equal(local_tensor(parameter).detach().cpu(), self._checkup_before[canonical(name)]), name
        assert self._optimizer_step_count == 0 and not self.optimizer.state
        return {'rank': dist.get_rank(), 'lora_unchanged': True, 'optimizer_updates': 0}


class CaptureStack(TrainStack):
    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def capture_part(self, part, directory):
        rank = dist.get_rank()
        assert part.batch_size == 24
        self._align_track_inputs(part)
        part = self._prepare_for_training(part, plans=(((0, 8), (8, 16), (16, 24)),), order=None)
        kept = Part(sample_ids=list(part.sample_ids), segment=part.segment, conditions=part.conditions,
                    rewards=part.rewards, advantages=part.advantages)
        kept = kept.to_device('cpu').clone()
        payload = {'schema': 1, 'part': kept, 'parameters': self.fsdp_backend._checkup_before,
                   'parameter_metadata': self.fsdp_backend._checkup_metadata, 'rank': rank,
                   'world_size': dist.get_world_size(), 'rollout_id': 20, 'optimizer_updates': 0}
        target = Path(directory) / f'state_rank{rank}.pt'
        atomic_save(target, payload)
        loaded = torch.load(target, map_location='cpu', weights_only=False)
        assert_same(payload, loaded)
        self.fsdp_backend.checkup_verify()
        dist.barrier()
        if rank == 0:
            (Path(directory) / 'RANKS_DONE').write_text('8 ranks, exact serialization roundtrip, zero updates\n')
        print(f'CHECKUP_CAPTURE_SAVED rank={rank} rows={part.batch_size}', flush=True)
        return {'rank': rank, 'rows': part.batch_size}
