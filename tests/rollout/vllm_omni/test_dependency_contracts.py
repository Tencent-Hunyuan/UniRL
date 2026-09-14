from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_stable_vllm_omni_dependency_contract() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    extras = project["project"]["optional-dependencies"]
    assert "vllm==0.28.0 ; sys_platform == 'linux'" in extras["vllm"]
    assert "vllm-omni==0.28.0 ; sys_platform == 'linux'" in extras["vllm"]
    assert "transformers==5.12.1 ; sys_platform == 'linux'" in extras["vllm"]
    assert "kernels>=0.12,<0.13 ; sys_platform == 'linux'" in extras["sglang"]

    overrides = project["tool"]["uv"]["override-dependencies"]
    assert "diffusers==0.40.0" in overrides
    assert "tokenizers>=0.22,<0.23" in overrides
    assert all("kernels" not in requirement for requirement in overrides)
