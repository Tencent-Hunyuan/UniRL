from __future__ import annotations

import json
import sys
from types import ModuleType

from unirl.rollout.engine.fastvideo.engine import _verify_checkpoint_unipc_spec
from unirl.sde.unipc import UniPCSpec


def _write_scheduler_config(path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "_class_name": "FlowUniPCMultistepScheduler",
                "solver_order": 2,
                "solver_type": "bh2",
                "lower_order_final": True,
            }
        ),
        encoding="utf-8",
    )


def test_verify_checkpoint_unipc_spec_reads_local_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "scheduler" / "scheduler_config.json"
    _write_scheduler_config(config_path)

    _verify_checkpoint_unipc_spec(str(tmp_path), UniPCSpec())


def test_verify_checkpoint_unipc_spec_downloads_huggingface_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "scheduler_config.json"
    _write_scheduler_config(config_path)
    calls = []
    hub = ModuleType("huggingface_hub")

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(config_path)

    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    _verify_checkpoint_unipc_spec("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", UniPCSpec())

    assert calls == [
        {
            "repo_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "filename": "scheduler/scheduler_config.json",
        }
    ]
