"""Typed port reservation + the CPU-importability contract (G3/G5)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniPorts


def test_reserve_yields_two_distinct_valid_ports():
    ports = VLLMOmniPorts.reserve()
    assert ports.stage0_master_port != ports.stage1_master_port
    for p in (ports.stage0_master_port, ports.stage1_master_port):
        assert 1 <= p <= 65535


def test_from_ports_roundtrip_and_validation():
    ports = VLLMOmniPorts.from_ports([30100, 30200])
    assert (ports.stage0_master_port, ports.stage1_master_port) == (30100, 30200)
    with pytest.raises(ValueError, match="distinct"):
        VLLMOmniPorts(stage0_master_port=30100, stage1_master_port=30100)
    with pytest.raises(ValueError, match="TCP port"):
        VLLMOmniPorts(stage0_master_port=0, stage1_master_port=30100)


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
