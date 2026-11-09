"""Opt-in H3 diagnostics; FSDP2 private API and measurement limits live in utils/README.md."""

import inspect
import json
import os
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import wraps
from pathlib import Path
from threading import Lock
from time import perf_counter

import torch

ALL_GATHER = os.environ.get("H3_DIAG_AG") == "1"
GRAD_DUMP_DIR = os.environ.get("H3_DIAG_GRAD_DUMP")
REWARDS_DIR = os.environ.get("H3_DIAG_REWARDS")
GRAD_AB_DIR = os.environ.get("H3_DIAG_GRAD_AB")
GRAD_AB_MICROS = tuple(int(size) for size in os.environ.get("H3_DIAG_GRAD_AB_MICROS", "1,8,8,4,2").split(","))
STATE_SAVE_DIR = os.environ.get("H3_DIAG_STATE_SAVE")
STATE_LOAD_DIR = os.environ.get("H3_DIAG_STATE_LOAD")
REPLAY_ONLY = os.environ.get("H3_DIAG_REPLAY_ONLY") == "1"
GRAD_AB_ONLY = os.environ.get("H3_DIAG_GRAD_AB_ONLY") == "1"
if REPLAY_ONLY and not STATE_LOAD_DIR:
    raise ValueError("H3_DIAG_REPLAY_ONLY=1 requires H3_DIAG_STATE_LOAD=<directory>.")
if GRAD_AB_ONLY and not GRAD_AB_DIR:
    raise ValueError("H3_DIAG_GRAD_AB_ONLY=1 requires H3_DIAG_GRAD_AB=<directory>.")


def dump_first_gradients(function):
    """Install a pre-clipping dump only when requested at process startup."""
    if not GRAD_DUMP_DIR:
        return function
    dumped = False

    @wraps(function)
    def wrapped(backend, *args, **kwargs):
        nonlocal dumped
        if not dumped:
            from torch.distributed.tensor import DTensor

            gradients = {}
            for name, parameter in backend.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    gradient = parameter.grad
                    if isinstance(gradient, DTensor):
                        gradient = gradient.to_local()
                    gradients[name] = gradient.detach().float().cpu()
            directory = Path(GRAD_DUMP_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            torch.save(gradients, directory / f"grad_rank{rank}.pt")
            dumped = True
        return function(backend, *args, **kwargs)

    return wrapped


def _local_gradients(backend):
    """Detached fp32 CPU copies of this rank's trainable gradient shards, keyed by parameter name."""
    from torch.distributed.tensor import DTensor

    gradients = {}
    for name, parameter in backend.model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            gradient = parameter.grad
            if isinstance(gradient, DTensor):
                gradient = gradient.to_local()
            gradients[name] = gradient.detach().float().cpu()
    return gradients


def compare_micro_gradients(function):
    """Before the first real update, recompute its synced gradient at each requested micro size without stepping."""
    if not GRAD_AB_DIR:
        return function
    done = False

    @wraps(function)
    def wrapped(stack, parts, **kwargs):
        nonlocal done
        if not done:
            done = True
            backend = stack.fsdp_backend
            configured = stack.micro_batch_size
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            for index, micro in enumerate(GRAD_AB_MICROS):
                directory = Path(GRAD_AB_DIR) / f"{index}_micro{micro}"
                directory.mkdir(parents=True, exist_ok=True)

                def dump(*, max_grad_norm, directory=directory):
                    torch.save(_local_gradients(backend), directory / f"grad_rank{rank}.pt")
                    backend.optimizer.zero_grad(set_to_none=True)
                    return 0.0

                stack.micro_batch_size = micro
                backend.optimizer_step = dump
                started = perf_counter()
                result = function(stack, parts, **kwargs)
                meta = {"micro": micro, "seconds": perf_counter() - started, "loss": result.loss}
                meta.update({key: value for key, value in result.metrics.items() if isinstance(value, (int, float))})
                (directory / f"meta_rank{rank}.json").write_text(json.dumps(meta) + "\n")
            del backend.optimizer_step
            stack.micro_batch_size = configured
            if GRAD_AB_ONLY:
                return result
        return function(stack, parts, **kwargs)

    return wrapped


def _trainable_shards(backend):
    """This rank's live trainable parameters as local shards, keyed by parameter name."""
    from unirl.distributed.local import local_view

    return {name: local_view(p) for name, p in backend.model.named_parameters() if p.requires_grad}


def _training_fields(part):
    """CPU copy of the Part fields train_track reads; decoded media and reward bookkeeping are left out."""
    kept = type(part)(
        sample_ids=part.sample_ids, segment=part.segment, conditions=part.conditions, advantages=part.advantages
    )
    return kept.to_device("cpu").clone()


def _cast_like(loaded, own, path):
    """``loaded`` with each floating tensor cast to the dtype of this process's tensor at the same path."""
    if isinstance(loaded, torch.Tensor):
        if not loaded.is_floating_point():
            return loaded
        if not (isinstance(own, torch.Tensor) and own.is_floating_point()):
            raise ValueError(f"H3_DIAG_STATE_LOAD: {path} has no floating tensor in this process's own Part.")
        return loaded.to(own.dtype)
    if isinstance(loaded, dict):
        own = own if isinstance(own, dict) else {}
        return {key: _cast_like(value, own.get(key), f"{path}[{key!r}]") for key, value in loaded.items()}
    if is_dataclass(loaded):
        for field in fields(loaded):
            value = _cast_like(getattr(loaded, field.name), getattr(own, field.name, None), f"{path}.{field.name}")
            setattr(loaded, field.name, value)
    return loaded


def exchange_first_state(function):
    """At each rank's first train_track call, save and/or load its training Parts and trainable local shards."""
    if not (STATE_SAVE_DIR or STATE_LOAD_DIR):
        return function
    done = False

    @wraps(function)
    def wrapped(stack, parts, **kwargs):
        nonlocal done
        if not done:
            done = True
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            window = parts if isinstance(parts, tuple) else (parts,)
            shards = _trainable_shards(stack.fsdp_backend)
            if STATE_SAVE_DIR:
                directory = Path(STATE_SAVE_DIR)
                directory.mkdir(parents=True, exist_ok=True)
                state = {
                    "parts": [_training_fields(part) for part in window],
                    "parameters": {name: shard.detach().to("cpu", copy=True) for name, shard in shards.items()},
                }
                torch.save(state, directory / f"state_rank{rank}.pt")
            if STATE_LOAD_DIR:
                state = torch.load(Path(STATE_LOAD_DIR) / f"state_rank{rank}.pt", weights_only=False)
                saved = state["parameters"]
                if {name: value.shape for name, value in saved.items()} != {n: s.shape for n, s in shards.items()}:
                    raise ValueError("H3_DIAG_STATE_LOAD: trainable parameter names or local shard shapes differ.")
                with torch.no_grad():
                    for name, shard in shards.items():
                        shard.copy_(saved[name])
                loaded = tuple(
                    _cast_like(part, own, f"parts[{index}]")
                    for index, (part, own) in enumerate(zip(state["parts"], window, strict=True))
                )
                parts = loaded if isinstance(parts, tuple) else loaded[0]
        return function(stack, parts, **kwargs)

    return wrapped


def replayed_first_part(ranks):
    """Driver: every rank's saved first-update Part joined in rank order; DP_SCATTER hands rank r its own rows back."""
    directory = Path(STATE_LOAD_DIR)
    found = len(list(directory.glob("state_rank*.pt")))
    if found != ranks:
        raise ValueError(f"H3_DIAG_REPLAY_ONLY: {directory} holds {found} state_rank*.pt files, not {ranks}.")
    parts = [torch.load(directory / f"state_rank{rank}.pt", weights_only=False)["parts"][0] for rank in range(ranks)]
    return type(parts[0]).concat(parts)


def dump_rewards(function):
    """Install driver reward export without changing request order or reward tensors."""
    if not REWARDS_DIR:
        return function
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        sample, _ = result
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        requests = arguments.arguments["samples"]
        roots = [root for request in requests for root in request.parts[0].sample_ids]
        prompt_indices = {root: index for index, root in enumerate(roots)}
        part = sample.parts[-1]
        rewards = dict(zip(part.sample_ids, part.rewards.detach().cpu().tolist()))
        rows = [
            {"sample_id": sid, "prompt_index": prompt_indices[root], "reward": rewards[sid]}
            for request in requests
            for sid, root in zip(request.parts[-1].sample_ids, request.root_group_ids(-1))
        ]
        directory = Path(REWARDS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rewards_rollout{arguments.arguments['rollout_id']}.json"
        path.write_text(json.dumps(rows) + "\n")
        return result

    return wrapped


def install_all_gather(record, take, aggregate, stages):
    """Instrument FSDP2 unshard/copy-out intervals; supported private fields are documented in README.md."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    active = []
    totals = {}
    lock = Lock()

    @contextmanager
    def measured_record(stage):
        entry = (stage, object())
        with record(stage):
            with lock:
                active.append(entry)
            try:
                yield
            finally:
                with lock:
                    active.remove(entry)

    def measure(function, *, launch):
        @wraps(function)
        def wrapped(group, *args, **kwargs):
            with lock:
                candidates = [stage for stage, _ in active if stage != "score"]
            if not candidates or not isinstance(group.mesh_info, FSDPMeshInfo):
                return function(group, *args, **kwargs)
            if group._all_gather_process_group.size() == 1:
                return function(group, *args, **kwargs)
            stage = candidates[-1]
            pending = group._all_gather_result is not None
            cuda = torch.cuda.is_available()
            if cuda:
                torch.cuda.synchronize()
            started = perf_counter()
            result = function(group, *args, **kwargs)
            if cuda:
                torch.cuda.synchronize()
            seconds = perf_counter() - started
            launched = launch and not pending and group._all_gather_result is not None
            if launched or (not launch and pending):
                with lock:
                    previous_seconds, previous_count = totals.get(stage, (0.0, 0))
                    totals[stage] = previous_seconds + seconds, previous_count + int(launched)
            return result

        return wrapped

    FSDPParamGroup.unshard = measure(FSDPParamGroup.unshard, launch=True)
    FSDPParamGroup.wait_for_unshard = measure(FSDPParamGroup.wait_for_unshard, launch=False)

    def measured_take():
        readings = take()
        with lock:
            for stage in stages:
                seconds, count = totals.get(stage, (0.0, 0))
                readings[stage].update(ag_seconds=seconds, ag_count=count)
            totals.clear()
        return readings

    def measured_aggregate(readings):
        metrics = aggregate(readings)
        for stage in stages:
            for source, suffix in (("ag_seconds", "ag_s"), ("ag_count", "ag_count")):
                metrics[f"stage/{stage}_{suffix}"] = sum(rank[stage][source] for rank in readings) / len(readings)
        return metrics

    return measured_record, measured_take, measured_aggregate
