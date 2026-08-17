"""BagelDiffusionConditions — per-sample KV-cache contexts for the Bagel stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.tensor.batch import concat_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class BagelARConditions(Condition):
    """Per-sample RAW prompt material for the Bagel AR (text-out) stage."""

    modality: ClassVar[Modality] = Modality.TEXT

    prompt_splits: List[List[Dict[str, Any]]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.prompt_splits)

    @classmethod
    def for_sample(cls, *, splits: List[Dict[str, Any]]) -> "BagelARConditions":
        """Build a single-sample conditions (1-element list) from ordered splits."""
        require(bool(splits), "BagelARConditions.for_sample: splits must be non-empty.")
        for sp in splits:
            require(
                isinstance(sp, dict) and sp.get("kind") in ("text", "vit"),
                f"BagelARConditions.for_sample: each split must be a dict with kind in ('text', 'vit'); got {sp!r}.",
            )
        return cls(prompt_splits=[list(splits)])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BagelARConditions":
        """Read the conditions back from a ``Part.conditions`` dict."""
        bagel_ar = d.get("bagel_ar")
        if isinstance(bagel_ar, cls):
            return bagel_ar
        raise ValueError(
            "BagelARConditions.from_dict: expected a 'bagel_ar' key holding a "
            f"BagelARConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Emit as a single ``"bagel_ar"`` entry for ``Part.conditions``."""
        return {"bagel_ar": self}


@dataclass
class BagelDiffusionConditions(Condition):
    """Per-sample opaque conditioning (KV contexts + image shape) for Bagel."""

    modality: ClassVar[Modality] = Modality.IMAGE

    gen_contexts: List[Any] = concat_field(default_factory=list)
    cfg_text_contexts: List[Any] = concat_field(default_factory=list)
    cfg_img_contexts: List[Any] = concat_field(default_factory=list)
    prompts: List[Any] = concat_field(default_factory=list)
    image_shapes: List[Tuple[int, int]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.gen_contexts) or len(self.prompts)

    def has_contexts(self) -> bool:
        """True when opaque KV contexts are present (trainside / colocate path)."""
        return bool(self.gen_contexts) and self.gen_contexts[0] is not None

    @classmethod
    def for_sample(
        cls,
        *,
        gen_context: Any,
        image_shape: Tuple[int, int],
        cfg_text_context: Optional[Any] = None,
        cfg_img_context: Optional[Any] = None,
        prompt: Optional[str] = None,
    ) -> "BagelDiffusionConditions":
        """Build a single-sample conditions (1-element lists)."""
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
            prompts=[prompt],
            image_shapes=[tuple(image_shape)],
        )

    def single(self) -> Tuple[Any, Any, Any, Tuple[int, int]]:
        """Return ``(gen, cfg_text, cfg_img, image_shape)`` for a 1-sample batch."""
        require(
            self.batch_size == 1,
            f"BagelDiffusionConditions.single: expected exactly 1 sample (navit bs=1; "
            f"set micro_batch_size=1), got {self.batch_size}.",
        )
        require(
            self.has_contexts(),
            "BagelDiffusionConditions.single: no opaque KV contexts present "
            "(deferred-prompt path). Rebuild from single_prompt() on a bundle first.",
        )
        gen = self.gen_contexts[0]
        cfg_text = (
            self.cfg_text_contexts[0] if self.cfg_text_contexts and self.cfg_text_contexts[0] is not None else gen
        )
        cfg_img = self.cfg_img_contexts[0] if self.cfg_img_contexts and self.cfg_img_contexts[0] is not None else gen
        image_shape = tuple(self.image_shapes[0])
        return gen, cfg_text, cfg_img, image_shape

    def single_prompt(self) -> Tuple[str, Tuple[int, int]]:
        """Return ``(prompt, image_shape)`` for a 1-sample deferred-prompt batch."""
        require(
            self.batch_size == 1,
            f"BagelDiffusionConditions.single_prompt: expected exactly 1 sample "
            f"(navit bs=1; set micro_batch_size=1), got {self.batch_size}.",
        )
        require(
            bool(self.prompts) and self.prompts[0] is not None,
            "BagelDiffusionConditions.single_prompt: no prompt present; the rollout "
            "adapter must ship prompts for the deferred-rebuild path.",
        )
        image_shape = tuple(self.image_shapes[0])
        return str(self.prompts[0]), image_shape

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BagelDiffusionConditions":
        """Read the conditions back from a ``Part.conditions`` dict."""
        bagel = d.get("bagel")
        if isinstance(bagel, cls):
            return bagel
        raise ValueError(
            "BagelDiffusionConditions.from_dict: expected a 'bagel' key holding a "
            f"BagelDiffusionConditions instance; got keys {sorted(d.keys())}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Emit as a single ``"bagel"`` entry for ``Part.conditions``."""
        return {"bagel": self}


__all__ = ["BagelARConditions", "BagelDiffusionConditions"]
