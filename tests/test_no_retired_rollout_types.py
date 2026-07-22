"""Guard the Sample/Part rollout boundary against legacy carrier regressions."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from unirl.types.sample import Part

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "unirl"
RETIRED_MODULES = {"unirl.types.rollout_req", "unirl.types.rollout_resp", "unirl.types.prompts"}
RETIRED_NAMES = {"RolloutReq", "RolloutResp", "RolloutTrack", "RolloutInputs"}
RETIRED_ROLLOUT_ASSEMBLY_NAMES = {
    "_gen_part_index_for_track",
    "assemble_sample",
    "build_segments",
    "decoded_for_track",
    "segments_for_track",
    "track_name",
}
PART_FIELDS = {field.name for field in fields(Part)}


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


def test_retired_rollout_modules_are_absent() -> None:
    assert not (SOURCE_ROOT / "types" / "rollout_req.py").exists()
    assert not (SOURCE_ROOT / "types" / "rollout_resp.py").exists()
    assert not (SOURCE_ROOT / "types" / "prompts.py").exists()


def test_rollout_adapters_do_not_rebuild_named_response_tracks() -> None:
    """Keep adapter output assembly on typed Part fills, including keyword names."""
    roots = [
        SOURCE_ROOT / "rollout" / "engine" / "vllm_omni",
        SOURCE_ROOT / "rollout" / "engine" / "sglang" / "adapters",
        SOURCE_ROOT / "rollout" / "engine" / "sglang_diffusion" / "adapters",
    ]
    violations: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = [
                    getattr(node, "id", None),
                    getattr(node, "attr", None),
                    getattr(node, "arg", None),
                    getattr(node, "name", None),
                ]
                if isinstance(node, ast.keyword):
                    names.append(node.arg)
                for name in names:
                    if name in RETIRED_ROLLOUT_ASSEMBLY_NAMES:
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")

    assert not violations, "Legacy named response-track assembly found:\n" + "\n".join(violations)


def test_part_copy_calls_name_real_part_fields() -> None:
    """Catch semantic carrier typos that a retired-type name scan cannot see."""
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            func_name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if func_name != "_part_with_field":
                continue
            field_arg = node.args[1]
            if not isinstance(field_arg, ast.Constant) or not isinstance(field_arg.value, str):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: _part_with_field field must be a string literal"
                )
            elif field_arg.value not in PART_FIELDS:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: unknown Part field {field_arg.value!r}"
                )

    assert not violations, "Invalid Part copy calls found:\n" + "\n".join(violations)
