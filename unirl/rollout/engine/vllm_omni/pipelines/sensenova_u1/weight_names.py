"""SenseNova full-weight sync name validation without runtime imports."""

from __future__ import annotations

from collections.abc import Iterable, Set

STACKED_WEIGHT_MAPPINGS = (
    (".qkv_proj_mot_gen", ".q_proj_mot_gen"),
    (".qkv_proj_mot_gen", ".k_proj_mot_gen"),
    (".qkv_proj_mot_gen", ".v_proj_mot_gen"),
    (".qkv_proj", ".q_proj"),
    (".qkv_proj", ".k_proj"),
    (".qkv_proj", ".v_proj"),
    (".gate_up_proj", ".gate_proj"),
    (".gate_up_proj", ".up_proj"),
)


def missing_weight_sync_names(names: Iterable[str], parameter_names: Set[str]) -> list[str]:
    """Return names neither direct-loadable nor covered by a fused mapping."""
    missing: list[str] = []
    for name in names:
        if name in parameter_names:
            continue
        mapped = next(
            (
                name.replace(source_name, target_name)
                for target_name, source_name in STACKED_WEIGHT_MAPPINGS
                if source_name in name and name.replace(source_name, target_name) in parameter_names
            ),
            None,
        )
        if mapped is None:
            missing.append(name)
    return missing


__all__ = ["STACKED_WEIGHT_MAPPINGS", "missing_weight_sync_names"]
