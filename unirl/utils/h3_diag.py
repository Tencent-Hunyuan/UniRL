"""Opt-in H3 diagnostics; FSDP2 private API and measurement limits live in utils/README.md."""

import inspect
import json
import os
from contextlib import contextmanager
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
        return function(stack, parts, **kwargs)

    return wrapped


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
