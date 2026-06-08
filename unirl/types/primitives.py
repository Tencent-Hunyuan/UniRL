"""Multimodal input/output primitives.

Per-sample types (``Text``, ``Image``, ``Video``, ``Audio``) are plain
dataclasses used at the user-facing boundary — input construction and
per-sample iteration in reward functions.

Batch types (``Texts``, ``Images``, ``Videos``, ``Audios``) are
``Batch`` SoA containers used in storage and transport. Round-trip
helpers (``from_list`` / ``to_list``) bridge between the two forms.

Tier in the four-tier pipeline:
    Primitive → (encode/embed) → Condition → (diffuse/autoregress) → Segment → (decode) → Primitive

Batching contract for varlen primitives (Videos, Audios)
--------------------------------------------------------
A batched primitive whose tensor data is varlen along dim 0 (frames packed
across all samples for ``Videos``; samples packed across all examples for
``Audios``) MUST declare that tensor with ``FieldKind.PACKED`` — never
``FieldKind.CONCAT``. ``CONCAT`` semantically means "dim 0 is the sample
axis"; for packed-along-time/length data dim 0 is the packed sequence axis
instead, and the framework needs ``_packed_cu_seqlens`` metadata to know
how to ``concat`` / ``select`` / ``slice`` such instances per-sample.

Per the ``Batch`` protocol contract (see
:class:`unirl.distributed.tensor.batch.Batch`), ``_packed_cu_seqlens`` is a
framework-managed hidden attribute:

- Construct via :meth:`Batch.pack` (or a thin wrapper like ``from_list``
  that delegates to ``pack``) with ``Sequence[Tensor]`` per packed field —
  the framework computes and attaches the cu_seqlens.
- Read via the inherited :attr:`Batch.cu_seqlens` property. Each batched
  primitive may also expose a domain alias (``Videos.cu_frames`` /
  ``Audios.cu_samples``) for readability at call sites — both point to
  the same framework-managed tensor.
- Never declare an explicit ``cu_*`` dataclass field. That breaks the
  framework's auto-propagation: ``concat`` / ``select`` / ``slice``
  rebuild ``_packed_cu_seqlens`` on the output instance, but they won't
  rebuild a user-declared field. The two values drift, and per-sample
  slicing becomes incorrect.

These rules generalize to any future ragged-along-dim-0 primitive (e.g.
``PointClouds`` with varlen point counts). Image-like primitives where
dim 0 IS the sample axis (``Images.pixels: [B, C, H, W]``) keep using
``FieldKind.CONCAT`` as before — that's the rectangular case the
framework's default machinery already handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import PIL.Image
import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field

# ---------------------------------------------------------------------------
# Per-sample primitives
# ---------------------------------------------------------------------------


@dataclass
class Text:
    """A text sample."""

    text: str

    @classmethod
    def from_str(cls, s: str) -> "Text":
        return cls(text=s)

    def to_str(self) -> str:
        return self.text


@dataclass
class Embedding:
    embedding: torch.Tensor


@dataclass
class Image:
    """A single image as a ``[C, H, W]`` tensor with values in ``[0, 1]``."""

    pixels: torch.Tensor

    def to_pil(self) -> PIL.Image.Image:
        from torchvision.transforms.functional import to_pil_image

        return to_pil_image(self.pixels.clamp(0.0, 1.0))


@dataclass
class Video:
    """A video as a ``[T, C, H, W]`` tensor with values in ``[0, 1]``."""

    frames: torch.Tensor

    def to_pils(self) -> List[PIL.Image.Image]:
        from torchvision.transforms.functional import to_pil_image

        return [to_pil_image(frame.clamp(0.0, 1.0)) for frame in self.frames]


@dataclass
class Audio:
    """A single audio sample as a ``[L]`` or ``[C, L]`` waveform tensor."""

    waveform: torch.Tensor


@dataclass
class TextAndImage:
    text: Text
    image: Image


@dataclass
class TextAndVideo:
    text: Text
    video: Video


# ---------------------------------------------------------------------------
# Batch (packed) primitives
# ---------------------------------------------------------------------------


@dataclass
class Texts(Batch):
    """Batch text samples — list of strings, batch dim is ``len(texts)``."""

    texts: List[str] = concat_field(default_factory=list)

    @classmethod
    def from_list(cls, items: List[Text]) -> "Texts":
        return cls(texts=[t.text for t in items])

    def to_list(self) -> List[Text]:
        return [Text(text=t) for t in self.texts]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class Images(Batch):
    """Batch images packed as a single ``[B, C, H, W]`` tensor.

    Assumes uniform shape within the batch.
    """

    pixels: torch.Tensor = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_list(cls, items: List[Image]) -> "Images":
        if not items:
            raise ValueError("Cannot build Images from an empty list")
        pixels_list = [img.pixels for img in items]
        # Handle variable-size images (e.g. VLM training) by padding to max size
        if len(set(p.shape for p in pixels_list)) != 1:
            max_h = max(p.shape[-2] for p in pixels_list)
            max_w = max(p.shape[-1] for p in pixels_list)
            padded = []
            for p in pixels_list:
                if p.shape[-2] == max_h and p.shape[-1] == max_w:
                    padded.append(p)
                else:
                    c, h, w = p.shape
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded_p = torch.nn.functional.pad(p, (0, pad_w, 0, pad_h), mode="constant", value=0)
                    padded.append(padded_p)
            pixels_list = padded
        stacked = torch.stack(pixels_list, dim=0)
        return cls(pixels=stacked)

    def to_list(self) -> List[Image]:
        return [Image(pixels=self.pixels[i]) for i in range(self.pixels.shape[0])]

    def __len__(self) -> int:
        return int(self.pixels.shape[0]) if self.pixels is not None else 0


@dataclass
class Videos(Batch):
    """Batch videos with ragged time dim, packed varlen along T.

    ``frames`` is concatenated along T for all samples: ``[total_T, C, H, W]``.
    Per-sample boundaries live on the framework-managed ``cu_seqlens``
    (exposed by the inherited :attr:`Batch.cu_seqlens` property and the
    domain alias :attr:`cu_frames`). Sample ``i``'s frames are
    ``frames[cu_frames[i]:cu_frames[i+1]]``. ``cu_frames[B]`` equals
    ``total_T``.

    Construct via :meth:`from_list` (or :meth:`Batch.pack` directly),
    not by passing pre-packed tensors to ``__init__`` — the constructor
    path doesn't compute cu_seqlens. ``concat`` / ``select`` / ``slice``
    operate per-sample and rebuild ``_packed_cu_seqlens`` on the output;
    see module docstring for the protocol contract.
    """

    frames: torch.Tensor = field(kind=FieldKind.PACKED, default=None)

    @property
    def cu_frames(self) -> Optional[torch.Tensor]:
        """Per-sample cumulative frame offsets — alias for :attr:`cu_seqlens`.

        Same shape and meaning as the old explicit ``cu_frames`` field;
        kept as a property so call sites that read ``videos.cu_frames``
        keep working. The underlying tensor is framework-managed (never
        set by user code; rebuilt by ``concat`` / ``select`` / ``slice``).
        """
        return self.cu_seqlens

    @classmethod
    def from_list(cls, items: List[Video]) -> "Videos":
        if not items:
            raise ValueError("Cannot build Videos from an empty list")
        # Delegate to ``Batch.pack`` so the framework computes and
        # attaches ``_packed_cu_seqlens``. ``pack`` ``torch.cat``s the
        # per-sample frames along dim 0 internally.
        return cls.pack(frames=[v.frames for v in items])

    def to_list(self) -> List[Video]:
        cu = self.cu_seqlens
        if cu is None or self.frames is None:
            return []
        return [Video(frames=self.frames[int(cu[i]) : int(cu[i + 1])]) for i in range(int(cu.shape[0]) - 1)]

    def __len__(self) -> int:
        cu = self.cu_seqlens
        return int(cu.shape[0]) - 1 if cu is not None else 0


@dataclass
class Audios(Batch):
    """Batch audio with ragged length dim, packed varlen along L.

    ``waveform`` is concatenated along L for all samples:
    ``[total_L, C]`` (or ``[total_L]``). Per-sample boundaries live on
    the framework-managed ``cu_seqlens`` (exposed by the inherited
    :attr:`Batch.cu_seqlens` property and the domain alias
    :attr:`cu_samples`).

    Construct via :meth:`from_list` (or :meth:`Batch.pack` directly).
    See module docstring for the varlen-primitive protocol contract.
    """

    waveform: torch.Tensor = field(kind=FieldKind.PACKED, default=None)

    @property
    def cu_samples(self) -> Optional[torch.Tensor]:
        """Per-sample cumulative sample offsets — alias for :attr:`cu_seqlens`."""
        return self.cu_seqlens

    @classmethod
    def from_list(cls, items: List[Audio]) -> "Audios":
        if not items:
            raise ValueError("Cannot build Audios from an empty list")
        # Delegate to ``Batch.pack`` so the framework computes and
        # attaches ``_packed_cu_seqlens``.
        return cls.pack(waveform=[a.waveform for a in items])

    def to_list(self) -> List[Audio]:
        cu = self.cu_seqlens
        if cu is None or self.waveform is None:
            return []
        return [Audio(waveform=self.waveform[int(cu[i]) : int(cu[i + 1])]) for i in range(int(cu.shape[0]) - 1)]

    def __len__(self) -> int:
        cu = self.cu_seqlens
        return int(cu.shape[0]) - 1 if cu is not None else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cumsum(values: List[int]) -> List[int]:
    out: List[int] = []
    total = 0
    for v in values:
        total += int(v)
        out.append(total)
    return out


__all__ = [
    "Audio",
    "Audios",
    "Embedding",
    "Image",
    "Images",
    "Text",
    "TextAndImage",
    "TextAndVideo",
    "Texts",
    "Video",
    "Videos",
]
