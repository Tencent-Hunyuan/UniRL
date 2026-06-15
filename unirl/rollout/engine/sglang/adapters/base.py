"""Driver-side ``RolloutReq``↔``RolloutResp`` conversion: the adapter ABC + registry.

A thin top ABC (registry + boilerplate) over the per-shape base adapter
(:mod:`text` — both registered families emit the packed-text ``"ar"`` track) that
holds the conversion logic as overridable methods. The VLM adapter overrides only
the steps that differ and self-registers by ``model_family`` key. Selected once
at engine construction via :func:`get_adapter`.

Pure: never imports SGLang — adapters consume the seam's ``RawResult`` protocol,
not the runtime. The tokenizer/processor are *injected* by the engine (loading
them is I/O the engine owns; tests pass stubs), so ``build_inputs`` /
``build_response`` stay exercisable with canned data.

``build_inputs`` returns a :class:`PreparedInputs` rather than a bare payload
list: the response side needs encode-time artifacts (the prompt token ids the
server saw; the VLM processor encodings), so the engine threads the prepared
object through to ``build_response`` instead of the adapter keeping per-call
state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.sglang.backends import RawResult
from unirl.rollout.engine.sglang.utils import ResolvedSampling
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, type["ModelAdapter"]] = {}


def register_adapter(key: str):
    """Class decorator: register an adapter under its ``model_family`` key."""

    def deco(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        require(
            key not in _REGISTRY,
            f"adapter key {key!r} already registered by {_REGISTRY.get(key)!r}",
        )
        _REGISTRY[key] = cls
        cls.model_family = key
        return cls

    return deco


def get_adapter(key: str) -> type["ModelAdapter"]:
    """Look up the adapter class for a ``model_family`` key."""
    require(
        key in _REGISTRY,
        f"unknown model_family {key!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[key]


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
# The build_inputs → build_response thread
# --------------------------------------------------------------------------- #


@dataclass
class MMEncoding:
    """One VLM sample's multimodal input for the SRT rollout.

    - ``image`` (PIL): base64'd into the ``/generate`` ``image_data`` so the
      server actually attends the image.
    - ``text``: the chat-templated string with a SINGLE image placeholder —
      sent to SRT, whose processor re-expands it. (Sending the pre-expanded
      ``input_ids`` + ``image_data`` instead makes SRT return 500.)
    - ``input_ids``: the processor's EXPANDED id sequence — stored as the replay
      prompt so rollout and replay teacher-force over the identical token stream.
    - ``pixel_values`` / ``image_grid_thw``: attached to the response conditions
      so the replay teacher-forces over the IDENTICAL multimodal input.
    """

    image: Any = None
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    pixel_values: Any = None
    image_grid_thw: Any = None


@dataclass
class PreparedInputs:
    """One ``generate`` call's prepared driver-side state.

    ``wire`` holds the ready-to-POST per-prompt ``/generate`` payloads (the
    engine stamps ``lora_path`` onto them when an adapter is active — the
    adapter stays unaware of weight sync). ``prompt_token_ids`` is what the
    server saw per prompt, replicated per sibling into the response's prompt
    condition. ``mm`` carries the VLM encodings (``None`` for text).
    """

    wire: List[Dict[str, Any]] = field(default_factory=list)
    prompt_token_ids: List[List[int]] = field(default_factory=list)
    resolved_n: int = 1
    mm: Optional[List[MMEncoding]] = None


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #


class ModelAdapter(ABC):
    """Thin ABC: registry key + boilerplate defaults + the two conversion seams.

    The conversion *logic* lives on the per-shape base adapter
    (:class:`~.text.TextLMAdapter`); this ABC only declares the boilerplate
    every adapter shares and the two abstract methods the engine drives.
    """

    model_family: str = ""

    def __init__(
        self,
        config: Any,
        model_config: Any = None,
        *,
        tokenizer: Any,
        processor: Any = None,
    ) -> None:
        self.cfg = config
        self.model_config = model_config
        self._tokenizer = tokenizer
        self._processor = processor
        self.validate()

    # ---- model-specific ServerArgs extras (override hook; default none) ----
    def boot_kwargs(self) -> Dict[str, Any]:
        """Extra SGLang ServerArgs intent a model needs beyond the generic set.

        The generic server kwargs are derived in ``config.server_intent``;
        recipes own the multimodal/LoRA/attention knobs via ``engine_kwargs``
        (parity with the predecessor — nothing is auto-added here).
        """
        return {}

    # ---- validation ----
    def validate(self) -> None:
        require(
            bool(getattr(self.cfg, "pretrained_model_ckpt_path", "")),
            f"{type(self).__name__} requires config.pretrained_model_ckpt_path",
        )
        require(
            self._tokenizer is not None,
            f"{type(self).__name__} requires a tokenizer",
        )

    # ---- tokenizer-derived helpers ----
    def pad_token_id(self) -> int:
        # External boundary: HF tokenizers are duck-typed (pad/eos optional).
        pad = getattr(self._tokenizer, "pad_token_id", None) or getattr(self._tokenizer, "eos_token_id", None)
        return int(pad) if pad is not None else 0

    # ---- the two conversion seams the engine drives ----
    @abstractmethod
    def build_inputs(self, req: RolloutReq, *, sampling: ResolvedSampling) -> PreparedInputs:
        """Translate a ``RolloutReq`` into per-prompt SRT ``/generate`` payloads."""

    @abstractmethod
    def build_response(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> RolloutResp:
        """Translate the seam's results back into a typed ``RolloutResp``."""


__all__ = [
    "MMEncoding",
    "ModelAdapter",
    "PreparedInputs",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
