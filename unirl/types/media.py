"""Typed media references for multimodal rollout inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from unirl.distributed.tensor.batch import Batch, concat_field


@dataclass(frozen=True)
class MediaRef:
    """A lightweight reference to one per-sample media input.

    The reference is intentionally small and serializable. Actual media loading
    happens on the actor/sampler side so the driver does not move large tensors
    through Ray.
    """

    modality: str
    role: str
    uri: str

    def __post_init__(self) -> None:
        modality = str(self.modality).strip().lower()
        role = str(self.role).strip().lower()
        uri = str(self.uri).strip()
        if not modality or not role or not uri:
            raise ValueError("MediaRef modality, role, and uri must be non-empty strings.")
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "uri", uri)


@dataclass
class MediaRefs(Batch):
    """Batch-aligned sparse URI media inputs.

    ``rows[i]`` is the ordered list of media references for sample ``i``. An
    empty row represents a text-only sample. Keeping sparsity inside each row
    preserves the normal rectangular :class:`Batch` contract, so CONCAT
    selection, slicing, transport, and rollout fan-out remain row-aligned.
    """

    rows: List[List[MediaRef]] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: List[List[MediaRef]] = []
        for row, refs in enumerate(self.rows):
            if not isinstance(refs, (list, tuple)):
                raise TypeError(f"MediaRefs.rows[{row}] must be a list of MediaRef values.")
            values = list(refs)
            invalid = [type(ref).__name__ for ref in values if not isinstance(ref, MediaRef)]
            if invalid:
                raise TypeError(
                    f"MediaRefs.rows[{row}] contains non-MediaRef values: {invalid}."
                )
            normalized.append(values)
        self.rows = normalized

    @classmethod
    def from_rows(cls, rows: List[List[MediaRef]]) -> "MediaRefs":
        return cls(rows=[list(row) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)


__all__ = ["MediaRef", "MediaRefs"]
