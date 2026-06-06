"""Typed port reservation, boot-kwarg assembly, and CPU-importability (G3/G5)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from unirl.rollout.engine.vllm_omni_v2.backends.native import _assemble_omni_kwargs
from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniPorts, VLLMOmniV2EngineConfig


def test_reserve_yields_one_valid_port():
    ports = VLLMOmniPorts.reserve()
    assert 1 <= ports.master_port <= 65535


def test_from_ports_roundtrip_and_validation():
    ports = VLLMOmniPorts.from_ports([30100])
    assert ports.master_port == 30100
    with pytest.raises(ValueError, match="TCP port"):
        VLLMOmniPorts(master_port=0)
    with pytest.raises(ValueError, match="expects 1 ports"):
        VLLMOmniPorts.from_ports([30100, 30200])


def test_assemble_omni_kwargs_layering():
    """Dedicated keys ride ON TOP of the omni_extra escape hatch; the hatch
    still overrides the timeout/mode defaults (v1 parity)."""
    cfg = VLLMOmniV2EngineConfig(
        model_path="/x",
        modality="sd3_t2i",
        omni_extra={"stage_init_timeout": 99, "enable_sleep_mode": False, "master_port": 1},
    )
    ports = VLLMOmniPorts(master_port=30100)
    intent = cfg.server_intent(model_config=None, ports=ports, extra={"stage_yaml": "x.yaml", "mode": "text-to-image"})
    kw = _assemble_omni_kwargs(intent)
    assert kw["stage_init_timeout"] == 99  # escape hatch beats defaults
    assert kw["init_timeout"] == 1800  # untouched default survives
    assert kw["mode"] == "text-to-image"  # adapter mode survives
    assert kw["enable_sleep_mode"] is True  # dedicated key beats the hatch
    assert kw["master_port"] == 30100  # ports doctrine: untouchable


def test_assemble_omni_kwargs_omits_disabled_keys():
    """enable_sleep_mode absent when off (upstream YAML defaults apply);
    master_port absent without a reserved set — matching v1's conditional
    injection semantics."""
    cfg = VLLMOmniV2EngineConfig(model_path="/x", modality="sd3_t2i", enable_sleep_mode=False)
    intent = cfg.server_intent(model_config=None, ports=None, extra={"stage_yaml": "x.yaml"})
    kw = _assemble_omni_kwargs(intent)
    assert "enable_sleep_mode" not in kw
    assert "master_port" not in kw


def test_package_imports_without_runtime():
    """The whole driver-side package must import with vllm-omni absent —
    the runtime import is confined to the seam impl's boot/verbs (and the
    worker-side role-8 modules, loaded only by qualname in the worker)."""
    code = """
import importlib, sys
for blocked in ("vllm", "vllm_omni", "sglang"):
    sys.modules[blocked] = None  # force ImportError on any eager import
mods = [
    "unirl.rollout.engine.vllm_omni_v2",
    "unirl.rollout.engine.vllm_omni_v2.config",
    "unirl.rollout.engine.vllm_omni_v2.engine",
    "unirl.rollout.engine.vllm_omni_v2.weight_sync",
    "unirl.rollout.engine.vllm_omni_v2.backends",
    "unirl.rollout.engine.vllm_omni_v2.backends.base",
    "unirl.rollout.engine.vllm_omni_v2.backends.native",
    "unirl.rollout.engine.vllm_omni_v2.adapters",
    "unirl.rollout.engine.vllm_omni_v2.utils",
    "unirl.rollout.engine.vllm_omni_v2.patches",
    "unirl.rollout.engine.vllm_omni_v2.worker",
    "unirl.rollout.engine.vllm_omni_v2.pipelines",
]
for m in mods:
    importlib.import_module(m)
print("OK")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
