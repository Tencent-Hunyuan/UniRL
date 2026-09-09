"""Named provider profiles for the parity experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ParityProfile(str, Enum):
    PUBLIC_REFERENCE = "public_reference"


@dataclass(frozen=True)
class ProfileConfig:
    name: ParityProfile
    providers: Mapping[str, str]


_PUBLIC = ProfileConfig(
    name=ParityProfile.PUBLIC_REFERENCE,
    providers={
        "norm": "vllm_bi",
        "norm_backward": "torch",
        "reductions": "vllm_bi",
        "dense": "vllm_bi",
        "grouped": "vllm_bi_loop",
        "gate": "vllm_bi",
        "router_softmax": "vllm_bi",
        "moe_combine": "torch",
    },
)


def resolve_profile(value: str | ParityProfile | None = None) -> ProfileConfig:
    raw = value or os.environ.get("UNIRL_PARITY_PROFILE", ParityProfile.PUBLIC_REFERENCE.value)
    if not isinstance(raw, ParityProfile):
        ParityProfile(str(raw))
    return _PUBLIC


__all__ = [
    "ParityProfile",
    "ProfileConfig",
    "resolve_profile",
]
