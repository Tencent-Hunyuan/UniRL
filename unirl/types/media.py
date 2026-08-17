"""Typed media references for multimodal rollout inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from unirl.distributed.tensor.batch import Batch, concat_field

SUPPORTED_MEDIA_MODALITIES = frozenset({"image", "audio", "video", "action"})
SUPPORTED_MEDIA_ROLES = frozenset({"prompt", "condition", "target"})


def normalize_media_uri(uri: object, *, context: str = "MediaRef") -> str:
    """Normalize a media URI and reject unsupported object-store schemes."""
    if not isinstance(uri, str):
        raise TypeError(f"{context}: uri must be a str, got {type(uri).__name__}.")
    normalized = uri.strip()
    if not normalized:
        raise ValueError(f"{context}: uri must be a non-empty string.")
    if normalized.startswith(("s3://", "gs://")):
        raise ValueError(
            f"{context}: object-store URI {normalized!r} is not supported; "
            "materialize to a local path or HTTP(S) URL first."
        )
    return normalized


@dataclass(frozen=True)
class MediaRef:
    """A lightweight reference to one per-sample media item."""

    modality: str
    role: str
    uri: str

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str):
            raise TypeError(f"MediaRef.modality must be a str, got {type(self.modality).__name__}.")
        if not isinstance(self.role, str):
            raise TypeError(f"MediaRef.role must be a str, got {type(self.role).__name__}.")
        modality = self.modality.strip().lower()
        role = self.role.strip().lower()
        if not modality or not role:
            raise ValueError("MediaRef modality and role must be non-empty strings.")
        if modality not in SUPPORTED_MEDIA_MODALITIES:
            raise ValueError(
                f"MediaRef modality must be one of {sorted(SUPPORTED_MEDIA_MODALITIES)}, got {modality!r}."
            )
        if role not in SUPPORTED_MEDIA_ROLES:
            raise ValueError(f"MediaRef role must be one of {sorted(SUPPORTED_MEDIA_ROLES)}, got {role!r}.")
        uri = normalize_media_uri(self.uri, context="MediaRef")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "uri", uri)


@dataclass
class MediaRefs(Batch):
    """Batch-aligned sparse URI media inputs."""

    rows: List[List[MediaRef]] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: List[List[MediaRef]] = []
        for row, refs in enumerate(self.rows):
            if not isinstance(refs, (list, tuple)):
                raise TypeError(f"MediaRefs.rows[{row}] must be a list of MediaRef values.")
            values = list(refs)
            invalid = [type(ref).__name__ for ref in values if not isinstance(ref, MediaRef)]
            if invalid:
                raise TypeError(f"MediaRefs.rows[{row}] contains non-MediaRef values: {invalid}.")
            normalized.append(values)
        self.rows = normalized

    @classmethod
    def from_rows(cls, rows: List[List[MediaRef]]) -> "MediaRefs":
        return cls(rows=[list(row) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)


__all__ = [
    "MediaRef",
    "MediaRefs",
    "SUPPORTED_MEDIA_MODALITIES",
    "SUPPORTED_MEDIA_ROLES",
    "normalize_media_uri",
]
