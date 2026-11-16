"""本任务的 live LoRA、分片与序列化工具，不改变训练目标。"""

import dataclasses
import json
import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor


def canonical(name):
    return name.replace('_checkpoint_wrapped_module.', '').replace('_orig_mod.', '')


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def atomic_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def live_state(checkpoint):
    raw = torch.load(Path(checkpoint) / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
    assert raw['save_mode'] == 'adapter'
    live = {canonical(k): v for k, v in raw['policy_state_dict'].items()
            if k.endswith('.default.weight') and ('.lora_A.' in k or '.lora_B.' in k)}
    assert len(live) == 600, len(live)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in live.values())
    return live


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def layout(tensor, rank):
    shape = list(tensor.shape)
    if isinstance(tensor, DTensor):
        mesh = tensor.device_mesh.mesh.cpu()
        assert mesh.ndim == 1, mesh
        order = mesh.tolist()
        placements = [f'Shard({p.dim})' if p.is_shard() else 'Replicate()' for p in tensor.placements]
        assert len(placements) == 1
        placement = tensor.placements[0]
        if placement.is_shard():
            dim = placement.dim
            size, start = placement._local_shard_size_and_offset(shape[dim], len(order), order.index(rank))
            offset = [0] * len(shape)
            offset[dim] = start
            local_shape = shape.copy()
            local_shape[dim] = size
        else:
            assert placement.is_replicate(), placement
            offset, local_shape = [0] * len(shape), shape.copy()
    else:
        order = list(range(torch.distributed.get_world_size())) if torch.distributed.is_initialized() else [0]
        placements, offset, local_shape = ['Replicate()'], [0] * len(shape), shape.copy()
    assert list(local_tensor(tensor).shape) == local_shape
    return {'global_shape': shape, 'placements': placements, 'mesh_shape': [len(order)],
            'mesh_rank_order': order, 'offset': offset, 'local_shape': local_shape}


def local_slice(full, info):
    assert list(full.shape) == info['global_shape']
    return full[tuple(slice(offset, offset + size) for offset, size in zip(info['offset'], info['local_shape']))]


def split_factor(split, rank, world=8):
    assert split in ('full', 'A', 'B') and world == 8 and 0 <= rank < world
    if split == 'full':
        return 1.0
    return 2.0 if ((rank < world // 2) == (split == 'A')) else 0.0


def masked_loss(loss, split, rank):
    factor = split_factor(split, rank)
    return loss if factor == 1.0 else loss * factor


def map_tensors(obj, function):
    if isinstance(obj, torch.Tensor):
        return function(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **{f.name: map_tensors(getattr(obj, f.name), function)
                                          for f in dataclasses.fields(obj) if f.init})
    if isinstance(obj, dict):
        return {k: map_tensors(v, function) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_tensors(v, function) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_tensors(v, function) for v in obj)
    return obj


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.cpu(), b.cpu())
    elif dataclasses.is_dataclass(a):
        assert type(a) is type(b)
        for field in dataclasses.fields(a):
            assert_same(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b, (a, b)
