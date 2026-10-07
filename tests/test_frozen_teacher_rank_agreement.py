"""Two-rank regression coverage for frozen teacher construction."""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from peft.utils import load_peft_weights

from unirl.train.lora import FrozenAdapters, inject_lora


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


def _fault(mode):
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


def _worker(rank, store, path):
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        for mode in ("read_weights", "read_config", "missing", "partial", "different_weights", "different_config"):
            model = Tiny()
            inject_lora(model, rank=2, alpha=4, target_modules=("proj",))
            local_path = str(Path(path) / "not-mounted") if rank == 1 and mode == "missing" else path
            with _fault(mode) if rank == 1 else nullcontext():
                try:
                    FrozenAdapters.inject(model, [{"name": "teacher", "path": local_path}])
                except RuntimeError as exc:
                    error = str(exc)
                else:
                    error = None
            errors = [None, None]
            dist.all_gather_object(errors, error)
            if not error or errors[0] != errors[1] or not any(mark in error for mark in ("rank 1", "rank(s) [1]")):
                raise AssertionError(f"{mode}: {errors}")

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
