"""BagelDiffusionConditions — per-sample KV-cache contexts for the Bagel stage.

Bagel differs from SD3 / HunyuanImage3: its conditioning is not a dense encoded
tensor but a set of prebuilt KV-cache contexts produced by running the prompt
through the und path. The trainside (A1) diffusion stage delegates sampling /
replay to the vendored ``generate_image`` / ``_forward_flow``, which consume these
contexts directly, so this container holds them as opaque, **per-sample**, in-process
values (trainside only — they are not transported across worker pools).

Three contexts mirror flow_grpo's inferencer, one entry per sample (Bagel is
``bs=1`` per ``_forward_flow`` call, so a batch is a *list* of per-sample
contexts, not a stacked tensor):

- ``gen_contexts[i]``      : prompt context (conditional branch) — required
- ``cfg_text_contexts[i]`` : unconditional context (text-CFG branch)
- ``cfg_img_contexts[i]``  : image-CFG context
- ``image_shapes[i]``      : (H, W) for sample i

These are ``concat_field`` lists so :meth:`RolloutTrack.slice` / ``concat`` /
``select`` (which the train stack drives per micro-batch) re-index them per sample
exactly like SD3's tensor conditions — the framework's list-field machinery
(``_slice_value`` / ``_concat_value``) handles arbitrary objects (each opaque
context is one list element).

``Condition`` subclass so it is a valid ``RolloutTrack.conditions`` dict value;
``to_dict`` emits it under a single ``"bagel"`` key, ``from_dict`` reads it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.tensor.batch import concat_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class BagelDiffusionConditions(Condition):
    """Per-sample opaque conditioning (KV contexts + image shape) for Bagel."""

    modality: ClassVar[Modality] = Modality.IMAGE

    gen_contexts: List[Any] = concat_field(default_factory=list)
    cfg_text_contexts: List[Any] = concat_field(default_factory=list)
    cfg_img_contexts: List[Any] = concat_field(default_factory=list)
    image_shapes: List[Tuple[int, int]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.gen_contexts)

    @classmethod
    def for_sample(
        cls,
        *,
        gen_context: Any,
        image_shape: Tuple[int, int],
        cfg_text_context: Optional[Any] = None,
        cfg_img_context: Optional[Any] = None,
    ) -> "BagelDiffusionConditions":
        """Build a single-sample conditions (1-element lists).

        The pipeline calls this per prompt, then concatenates the per-sample
        instances into the batched track conditions.
        """
        if gen_context is None:
            raise ValueError("BagelDiffusionConditions.for_sample: gen_context is required.")
        if image_shape is None or len(image_shape) != 2:
            raise ValueError(
                f"BagelDiffusionConditions.for_sample: image_shape must be a (H, W) pair; got {image_shape!r}."
            )
        return cls(
            gen_contexts=[gen_context],
            cfg_text_contexts=[cfg_text_context],
            cfg_img_contexts=[cfg_img_context],
            image_shapes=[tuple(image_shape)],
        )

    def single(self) -> Tuple[Any, Any, Any, Tuple[int, int]]:
        """Return ``(gen, cfg_text, cfg_img, image_shape)`` for a 1-sample batch.

        The diffusion stage runs one prompt per ``_forward_flow`` call (navit
        ``bs=1``), so it consumes conditions one sample at a time (the train stack
        slices to ``micro_batch_size=1``). CFG contexts fall back to ``gen`` when
        absent (CFG-off). Raises if the batch isn't exactly one sample.
        """
        require(
            self.batch_size == 1,
            f"BagelDiffusionConditions.single: expected exactly 1 sample (navit bs=1; "
            f"set micro_batch_size=1), got {self.batch_size}.",
        )
        gen = self.gen_contexts[0]
        cfg_text = (
            self.cfg_text_contexts[0] if self.cfg_text_contexts and self.cfg_text_contexts[0] is not None else gen
        )
        cfg_img = self.cfg_img_contexts[0] if self.cfg_img_contexts and self.cfg_img_contexts[0] is not None else gen
        image_shape = tuple(self.image_shapes[0])
        return gen, cfg_text, cfg_img, image_shape

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BagelDiffusionConditions":
        """Read the conditions back from a ``RolloutTrack.conditions`` dict.

        Accepts the canonical ``{"bagel": <BagelDiffusionConditions>}`` shape that
        :meth:`to_dict` emits (already an instance, possibly sliced by the train
        stack). Raises otherwise.
        """
        bagel = d.get("bagel")
        if isinstance(bagel, cls):
            return bagel
        raise ValueError(
            "BagelDiffusionConditions.from_dict: expected a 'bagel' key holding a "
            f"BagelDiffusionConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Emit as a single ``"bagel"`` entry for ``RolloutTrack.conditions``.

        The whole container is one ``Condition`` dict value; the train stack's
        ``slice`` / ``concat`` re-index its per-sample lists alongside the segment.
        """
        return {"bagel": self}


__all__ = ["BagelDiffusionConditions"]
