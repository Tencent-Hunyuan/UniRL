"""Two-rank regression coverage for frozen teacher construction."""

import os
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from peft.utils import load_peft_weights

from unirl.train.lora import FrozenAdapters, _resolve_adapter_checkpoint, adapter_names, inject_lora


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


def _fault(mode, path):
    if mode == "missing":
        # Same configured path on every rank; this rank's mount does not have the directory.
        target = os.path.abspath(path)
        real_isdir = os.path.isdir

        def hidden(candidate):
            if os.path.abspath(candidate) == target:
                return False
            return real_isdir(candidate)

        return patch("os.path.isdir", side_effect=hidden)
    if mode == "read_weights":
        return patch("peft.utils.load_peft_weights", side_effect=OSError("rank-local weight read"))
    if mode == "read_config":
        return patch.object(LoraConfig, "from_pretrained", side_effect=OSError("rank-local config read"))
    if mode in {"partial", "different_weights"}:

        def altered_weights(*args, **kwargs):
            weights = load_peft_weights(*args, **kwargs)
            key = next(key for key in weights if "lora_B" in key)
            if mode == "partial":
                del weights[key]
            else:
                weights[key] = weights[key].clone()
                weights[key][0, 0] += 1
            return weights

        return patch("peft.utils.load_peft_weights", side_effect=altered_weights)
    if mode == "different_config":
        original = LoraConfig.from_pretrained

        def altered_config(*args, **kwargs):
            config = original(*args, **kwargs)
            config.lora_alpha += 1
            return config

        return patch.object(LoraConfig, "from_pretrained", side_effect=altered_config)
    return nullcontext()


def _raise_unless_agreed(label, error, errors, required, forbidden=()):
    if error is None or errors[0] != errors[1] or any(part not in error for part in required):
        raise AssertionError(f"{label}: {errors}")
    if any(part in error for part in forbidden):
        raise AssertionError(f"{label}: {error}")


def _capture_inject(model, specs, fault):
    with fault:
        try:
            FrozenAdapters.inject(model, specs)
        except RuntimeError as exc:
            return str(exc)
        return None


def _worker(rank, store, path):
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        checks = {
            "read_weights": (("Failed to read", "rank 1"), ()),
            "read_config": (("Failed to read", "rank 1"), ()),
            "missing": (("Failed to read", "rank 1", path, "FileNotFoundError"), ("adapter_config.json",)),
            "partial": (("Failed to inject", "rank 1"), ()),
            "different_weights": (("weights sha", "rank 1"), ()),
            "different_config": (("lora_alpha", "rank 1"), ()),
        }
        for mode, (required, forbidden) in checks.items():
            model = Tiny()
            inject_lora(model, rank=2, alpha=4, target_modules=("proj",))
            error = _capture_inject(
                model,
                [{"name": "teacher", "path": path}],
                _fault(mode, path) if rank == 1 else nullcontext(),
            )
            errors = [None, None]
            dist.all_gather_object(errors, error)
            _raise_unless_agreed(mode, error, errors, required, forbidden)

        model = Tiny()
        inject_lora(model, rank=2, alpha=4, target_modules=("proj",))
        error = _capture_inject(
            model,
            [{"name": "teacher", "path": path}],
            _fault("partial", path) if rank == 1 else nullcontext(),
        )
        errors = [None, None]
        dist.all_gather_object(errors, error)
        _raise_unless_agreed("retry-setup", error, errors, ("Failed to inject", "rank 1"))
        if "teacher" in adapter_names(model) or "teacher" in getattr(model, "peft_config", {}):
            raise AssertionError(f"rank {rank} kept a failed teacher: {adapter_names(model)}")
        leaked = [
            op
            for op in getattr(model, "_deferred_ops", [])
            if isinstance(op, partial) and getattr(op.func, "__name__", "") == "_load_frozen_adapter"
        ]
        if leaked:
            raise AssertionError(f"rank {rank} kept a deferred teacher load")
        sha = FrozenAdapters.inject(model, [{"name": "teacher", "path": path}]).shas["teacher"]
        shas = [None, None]
        dist.all_gather_object(shas, sha)
        if len(sha) != 64 or shas[0] != shas[1]:
            raise AssertionError(f"retry after partial inject: {shas}")

        model = Tiny()
        inject_lora(model, rank=2, alpha=4, target_modules=("proj",))
        own_name = "teacher" if rank == 0 else "other"
        error = _capture_inject(model, [{"name": own_name, "path": path}], nullcontext())
        errors = [None, None]
        dist.all_gather_object(errors, error)
        _raise_unless_agreed("spec-name", error, errors, ("Frozen adapter specs differ", "'teacher'", "'other'"))
        if adapter_names(model) != {"default"}:
            raise AssertionError(f"spec mismatch mutated rank {rank}: {adapter_names(model)}")

        specs = [{"name": "teacher", "path": path}] if rank == 0 else []
        error = _capture_inject(Tiny(), specs, nullcontext())
        errors = [None, None]
        dist.all_gather_object(errors, error)
        _raise_unless_agreed("spec-empty", error, errors, ("Frozen adapter specs differ", "rank 0:", "rank 1:"))

        if FrozenAdapters.inject(Tiny(), []).shas:
            raise AssertionError("empty teacher list produced a sha")

        model = Tiny()
        inject_lora(model, rank=2, alpha=4, target_modules=("proj",))
        sha = FrozenAdapters.inject(model, [{"name": "teacher", "path": path}]).shas["teacher"]
        shas = [None, None]
        dist.all_gather_object(shas, sha)
        if len(sha) != 64 or shas[0] != shas[1]:
            raise AssertionError(f"teacher weights differ: {shas}")
    finally:
        dist.destroy_process_group()


class FrozenTeacherRankAgreementTest(TestCase):
    def test_checkpoint_location_split(self):
        self.assertEqual(_resolve_adapter_checkpoint("org/repo"), ("org/repo", None))
        self.assertEqual(_resolve_adapter_checkpoint("org/repo/folder"), ("org/repo", "folder"))
        for missing in ("/no/such/frozen-teacher", "./no/such/frozen-teacher"):
            with self.subTest(missing=missing):
                with self.assertRaises(FileNotFoundError) as caught:
                    _resolve_adapter_checkpoint(missing)
                self.assertIn(missing, str(caught.exception))

    def test_missing_absolute_path_names_that_path(self):
        missing = "/tmp/unirl-missing-frozen-teacher"
        with self.assertRaises(RuntimeError) as caught:
            FrozenAdapters.inject(Tiny(), [{"name": "teacher", "path": missing}])
        message = str(caught.exception)
        self.assertIn(missing, message)
        self.assertIn("FileNotFoundError", message)
        self.assertNotIn("adapter_config.json", message)

    def test_rank_local_failures_and_success(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "teacher"
            config = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], task_type="FEATURE_EXTRACTION")
            teacher = get_peft_model(Tiny(), config)
            for name, param in teacher.named_parameters():
                if ".lora_B." in name:
                    param.data.fill_(0.25)
            teacher.save_pretrained(path)
            mp.spawn(_worker, args=((Path(directory) / "group").as_uri(), str(path)), nprocs=2)
