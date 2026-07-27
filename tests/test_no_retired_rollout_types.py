"""Guard the Sample migration against executable uses of retired rollout types."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "unirl"
RETIRED_NAMES = {"RolloutReq", "RolloutResp", "RolloutTrack", "texts_from_req"}
RETIRED_MODULES = {
    "unirl.types.rollout_req",
    "unirl.types.rollout_resp",
    "unirl.types.prompts",
}


def test_no_executable_retired_rollout_types() -> None:
    offenders = []
    for source in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in RETIRED_MODULES:
                offenders.append(f"{source.relative_to(REPO_ROOT)}:{node.lineno}: import from {node.module}")
            elif isinstance(node, ast.Name) and node.id in RETIRED_NAMES:
                offenders.append(f"{source.relative_to(REPO_ROOT)}:{node.lineno}: name {node.id}")
    assert offenders == []
