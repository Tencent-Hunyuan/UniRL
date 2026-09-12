"""Multimodal input/output primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import PIL.Image
import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field, packed_field
from unirl.types.media import MediaRefs


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
    """Image batch with LLM-style packed storage for arbitrary CHW layouts."""

    packed_pixels: torch.Tensor = packed_field(default=None)
    image_shapes: torch.Tensor = concat_field(default=None)

    @property
    def cu_pixels(self) -> Optional[torch.Tensor]:
        return self.cu_seqlens

    @classmethod
    def from_dense(cls, pixels: torch.Tensor) -> "Images":
        if pixels is None or pixels.ndim != 4:
            raise ValueError(
                f"Images.from_dense expects pixels [B, C, H, W], got {None if pixels is None else tuple(pixels.shape)}"
            )
        batch_size, channels, height, width = (int(dim) for dim in pixels.shape)
        sample_size = channels * height * width
        image_shapes = torch.tensor([[channels, height, width]] * batch_size, dtype=torch.long)
        instance = cls(packed_pixels=pixels.reshape(-1), image_shapes=image_shapes)
        object.__setattr__(
            instance,
            "_packed_cu_seqlens",
            torch.arange(batch_size + 1, dtype=torch.long) * sample_size,
        )
        return instance

    @classmethod
    def from_list(cls, items: List[Image]) -> "Images":
        if not items:
            raise ValueError("Cannot build Images from an empty list")
        pixels_list = [img.pixels for img in items]
        if any(p is None or p.ndim != 3 for p in pixels_list):
            bad = [None if p is None else tuple(p.shape) for p in pixels_list]
            raise ValueError(f"Images.from_list expects per-sample pixels [C, H, W], got {bad}")
        channels = {int(p.shape[0]) for p in pixels_list}
        if len(channels) != 1:
            raise ValueError(f"Images.from_list requires a consistent channel count, got {sorted(channels)}")
        shapes = [tuple(p.shape) for p in pixels_list]
        if len(set(shapes)) == 1:
            return cls.from_dense(torch.stack(pixels_list, dim=0))
        image_shapes = torch.tensor(shapes, dtype=torch.long)
        return cls.pack(
            packed_pixels=[pixels.reshape(-1) for pixels in pixels_list],
            image_shapes=image_shapes,
        )

    def to_list(self) -> List[Image]:
        cu = self.cu_pixels
        if cu is None or self.packed_pixels is None or self.image_shapes is None:
            return []
        images: List[Image] = []
        for index in range(int(cu.shape[0]) - 1):
            shape = tuple(int(v) for v in self.image_shapes[index].tolist())
            flat = self.packed_pixels[int(cu[index]) : int(cu[index + 1])]
            expected = shape[0] * shape[1] * shape[2]
            if int(flat.numel()) != expected:
                raise ValueError(
                    f"Images sample {index} has packed length {int(flat.numel())}, "
                    f"expected {expected} for shape {shape}"
                )
            images.append(Image(pixels=flat.view(shape)))
        return images

    def to_dense(self) -> torch.Tensor:
        """Return a uniform ``[B,C,H,W]`` view sharing packed-pixel storage."""
        if self.packed_pixels is None or self.image_shapes is None or self.cu_pixels is None or len(self) == 0:
            raise ValueError("Images.to_dense requires a non-empty materialized batch")
        shape_rows = [tuple(int(v) for v in row.tolist()) for row in self.image_shapes]
        shapes = set(shape_rows)
        if len(shapes) != 1:
            raise ValueError(f"Images.to_dense requires uniform shapes, got {sorted(shapes)}")
        shape = shape_rows[0]
        expected = shape[0] * shape[1] * shape[2]
        if self.lengths is None or any(int(length) != expected for length in self.lengths):
            raise ValueError(f"Images.to_dense packed lengths do not match shape {shape}")
        return self.packed_pixels.view(len(self), *shape)

    def to_pils(self) -> List[PIL.Image.Image]:
        """Per-sample PIL conversion — batch counterpart of :meth:`Image.to_pil`."""
        return [img.to_pil() for img in self.to_list()]

    def __len__(self) -> int:
        cu = self.cu_pixels
        return int(cu.shape[0]) - 1 if cu is not None else 0


@dataclass
class ImageSet(Batch):
    """One logical row's ordered image attachments, backed by an inner ``Images`` batch."""

    images: Optional[Images] = concat_field(default=None)
    metadata: List[Dict[str, Any]] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        count = len(self.images) if self.images is not None else 0
        if not self.metadata and count:
            self.metadata = [{} for _ in range(count)]
        if len(self.metadata) != count:
            raise ValueError(f"ImageSet metadata count {len(self.metadata)} != image count {count}")

    @classmethod
    def from_images(
        cls,
        images: Images,
        *,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "ImageSet":
        values = [{} for _ in range(len(images))] if metadata is None else [dict(item) for item in metadata]
        return cls(images=images, metadata=values)

    @classmethod
    def from_list(
        cls,
        items: Sequence[Image],
        *,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "ImageSet":
        values = list(items)
        if not values:
            return cls(metadata=[] if metadata is None else [dict(item) for item in metadata])
        return cls.from_images(Images.from_list(values), metadata=metadata)

    def to_list(self) -> List[Image]:
        return [] if self.images is None else self.images.to_list()

    def to_pils(self) -> List[PIL.Image.Image]:
        return [] if self.images is None else self.images.to_pils()

    def __len__(self) -> int:
        return 0 if self.images is None else len(self.images)


@dataclass
class ImageSets(Batch):
    """Batch-aligned ordered image sets with variable attachment counts per row."""

    rows: List[ImageSet] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        invalid = [type(row).__name__ for row in self.rows if not isinstance(row, ImageSet)]
        if invalid:
            raise TypeError(f"ImageSets rows must contain ImageSet values, got {invalid}")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Union[ImageSet, Images, Sequence[Image]]],
    ) -> "ImageSets":
        normalized: List[ImageSet] = []
        for row in rows:
            if isinstance(row, ImageSet):
                normalized.append(row)
            elif isinstance(row, Images):
                normalized.append(ImageSet.from_images(row))
            else:
                normalized.append(ImageSet.from_list(list(row)))
        return cls(rows=normalized)

    @classmethod
    def from_images(cls, images: Images) -> "ImageSets":
        return cls(rows=[ImageSet.from_images(images.slice(index, index + 1)) for index in range(len(images))])

    @property
    def counts(self) -> torch.Tensor:
        return torch.tensor([len(row) for row in self.rows], dtype=torch.long)

    def require_exactly_one(self, *, context: str = "ImageSets") -> Images:
        if not self.rows:
            raise ValueError(f"{context} requires a non-empty image-set batch")
        bad = [index for index, row in enumerate(self.rows) if len(row) != 1]
        if bad:
            counts = sorted({len(row) for row in self.rows})
            raise ValueError(f"{context} requires exactly one image per row; counts={counts}, first bad row={bad[0]}")
        return Images.concat([row.images for row in self.rows if row.images is not None])

    def primary_images(self, *, context: str = "ImageSets") -> Images:
        if not self.rows:
            raise ValueError(f"{context} requires a non-empty image-set batch")
        missing = [index for index, row in enumerate(self.rows) if not row]
        if missing:
            raise ValueError(f"{context} requires at least one image per row; first empty row={missing[0]}")
        return Images.concat([row.images.slice(0, 1) for row in self.rows if row.images is not None])

    def to_slots(self, *, context: str = "ImageSets") -> List[Images]:
        counts = sorted({len(row) for row in self.rows})
        if not counts or counts == [0]:
            return []
        if len(counts) != 1:
            raise ValueError(f"{context} requires uniform image counts for dense slots; counts={counts}")
        return [
            Images.concat([row.images.slice(slot, slot + 1) for row in self.rows if row.images is not None])
            for slot in range(counts[0])
        ]

    def flatten(self) -> tuple[Optional[Images], torch.Tensor]:
        populated = [row.images for row in self.rows if row.images is not None]
        owners = torch.tensor(
            [row for row, image_set in enumerate(self.rows) for _ in range(len(image_set))],
            dtype=torch.long,
        )
        return (Images.concat(populated) if populated else None), owners

    def to_pil_rows(self) -> List[List[PIL.Image.Image]]:
        return [row.to_pils() for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


ImagePrimitive = Union[Images, ImageSets]


def as_image_sets(value: ImagePrimitive) -> ImageSets:
    """Normalize legacy singleton-row ``Images`` into the multi-image row contract."""
    return value if isinstance(value, ImageSets) else ImageSets.from_images(value)


def require_single_images(value: ImagePrimitive, *, context: str) -> Images:
    """Return one image per row, rejecting multi-image or empty rows."""
    return value if isinstance(value, Images) else value.require_exactly_one(context=context)


@dataclass
class Videos(Batch):
    """Batch videos with ragged time dim, packed varlen along T."""

    frames: torch.Tensor = field(kind=FieldKind.PACKED, default=None)

    @property
    def cu_frames(self) -> Optional[torch.Tensor]:
        """Per-sample cumulative frame offsets — alias for :attr:`cu_seqlens`."""
        return self.cu_seqlens

    @classmethod
    def from_list(cls, items: List[Video]) -> "Videos":
        if not items:
            raise ValueError("Cannot build Videos from an empty list")
        frames_list = [v.frames for v in items]
        if any(frames is None or frames.ndim != 4 for frames in frames_list):
            bad = [None if frames is None else tuple(frames.shape) for frames in frames_list]
            raise ValueError(f"Videos.from_list expects per-sample frames [T, C, H, W], got {bad}")
        channels = {int(frames.shape[1]) for frames in frames_list}
        if len(channels) != 1:
            raise ValueError(f"Videos.from_list requires a consistent channel count, got {sorted(channels)}")
        if len({tuple(frames.shape[1:]) for frames in frames_list}) != 1:
            max_h = max(int(frames.shape[-2]) for frames in frames_list)
            max_w = max(int(frames.shape[-1]) for frames in frames_list)
            resized = []
            for frames in frames_list:
                if int(frames.shape[-2]) != max_h or int(frames.shape[-1]) != max_w:
                    frames = torch.nn.functional.interpolate(
                        frames.float(), size=(max_h, max_w), mode="bilinear", align_corners=False
                    ).to(frames.dtype)
                resized.append(frames)
            frames_list = resized
        return cls.pack(frames=frames_list)

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
    """Batch audio with ragged length dim, packed varlen along L."""

    waveform: torch.Tensor = field(kind=FieldKind.PACKED, default=None)

    @property
    def cu_samples(self) -> Optional[torch.Tensor]:
        """Per-sample cumulative sample offsets — alias for :attr:`cu_seqlens`."""
        return self.cu_seqlens

    @classmethod
    def from_list(cls, items: List[Audio]) -> "Audios":
        if not items:
            raise ValueError("Cannot build Audios from an empty list")
        return cls.pack(waveform=[a.waveform for a in items])

    def to_list(self) -> List[Audio]:
        cu = self.cu_seqlens
        if cu is None or self.waveform is None:
            return []
        return [Audio(waveform=self.waveform[int(cu[i]) : int(cu[i + 1])]) for i in range(int(cu.shape[0]) - 1)]

    def __len__(self) -> int:
        cu = self.cu_seqlens
        return int(cu.shape[0]) - 1 if cu is not None else 0


def _cumsum(values: List[int]) -> List[int]:
    out: List[int] = []
    total = 0
    for v in values:
        total += int(v)
        out.append(total)
    return out


PrimitiveValue = Union[Texts, ImagePrimitive, Videos, Audios, MediaRefs]


def primitive_modality_key(prim: PrimitiveValue) -> str:
    """Map a batched primitive to its modality slot key."""
    if isinstance(prim, Texts):
        return "text"
    if isinstance(prim, Images):
        return "image"
    if isinstance(prim, ImageSets):
        return "image"
    if isinstance(prim, Videos):
        return "video"
    if isinstance(prim, Audios):
        return "audio"
    if isinstance(prim, MediaRefs):
        return "media"
    raise TypeError(f"primitive_modality_key: unknown primitive type {type(prim).__name__!r}")


__all__ = [
    "Audio",
    "Audios",
    "Embedding",
    "Image",
    "ImagePrimitive",
    "Images",
    "ImageSet",
    "ImageSets",
    "MediaRefs",
    "Text",
    "TextAndImage",
    "TextAndVideo",
    "Texts",
    "PrimitiveValue",
    "Video",
    "Videos",
    "as_image_sets",
    "primitive_modality_key",
    "require_single_images",
]
