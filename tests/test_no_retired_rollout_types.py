"""Keep production rollout boundaries on Sample/Part."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "unirl"
RETIRED_MODULES = {
    "unirl.types.prompts",
    "unirl.types.rollout_req",
    "unirl.types.rollout_resp",
}
RETIRED_NAMES = {"RolloutInputs", "RolloutReq", "RolloutResp", "RolloutTrack"}


def test_python_sources_do_not_use_retired_rollout_carriers() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in RETIRED_MODULES:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: import from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in RETIRED_MODULES:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.Name) and node.id in RETIRED_NAMES:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.id}")

    assert not violations, "Retired rollout carriers found:\n" + "\n".join(violations)


def test_retired_rollout_modules_remain_absent() -> None:
    for filename in ("prompts.py", "rollout_req.py", "rollout_resp.py"):
        assert not (SOURCE_ROOT / "types" / filename).exists()
