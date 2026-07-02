"""The backend seam contract — the ``Backend`` protocol + the wire types.

Every ``sglang_diffusion`` collaborator reaches the runtime through this protocol;
the real implementations live beside it (``native.py`` — in-process ``DiffGenerator``
/ ZMQ scheduler client; an HTTP-server impl would land as ``http.py``). This module
holds no runtime code at all, so it is trivially CPU-importable.

**No RL types cross this seam.** ``generate`` takes a plain ``dict`` of SGLang
sampling kwargs and returns ``list[RawResult]`` (a structural view of SGLang's
``GenerationResult``); the engine core + adapters do the ``RolloutReq``↔``RolloutResp``
translation. Implementations absorb their transport asymmetries (in-process tensors
vs. HTTP-serialized payloads) behind these signatures.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL.Image import Image as PILImage

#: A text-conditioning field on the wire: one fused tensor or a per-encoder list
#: (e.g. SD3's CLIP-L / CLIP-G / T5). ``utils.fuse_encoder_outputs`` reduces it.
EncoderOutputs = Union["torch.Tensor", Sequence["torch.Tensor"], None]

#: A decoded ``samples`` payload — SGLang returns bare tensors / numpy arrays /
#: PIL images / ``(video, audio)`` 2-tuples without a typed wrapper.
#: ``utils.decode_sample`` canonicalizes it.
MediaPayload = Union["torch.Tensor", "np.ndarray", "PILImage", tuple, list, None]


class RawResult(Protocol):
    """Structural view of SGLang's ``GenerationResult`` — the wire fields this
    engine consumes. Implementations return objects satisfying this protocol
    structurally (the native impl passes ``GenerationResult`` through; an HTTP impl
    deserializes into the same shape; test fakes stand in), which keeps
    adapters/utils SGLang-free while still naming the contract.

    Population follows the request: ``trajectory_latents`` /
    ``trajectory_timesteps`` / ``prompt_embeds`` are always requested;
    ``trajectory_log_probs`` only in native-logprob mode; the ``negative_*``
    triple only under CFG.
    """

    #: Dense denoising trajectory, ``[B, T+1, ...]`` (T+1: initial latents prepended).
    trajectory_latents: Optional["torch.Tensor"]
    #: Echo of the σ schedule actually used, ``[T+1]`` (the verifier also
    #: tolerates array-likes / raw ``sigma * num_train_timesteps`` scales).
    trajectory_timesteps: Optional["torch.Tensor"]
    #: Per-SDE-transition log-probs ``[B, S]``; native-logprob mode only.
    trajectory_log_probs: Optional["torch.Tensor"]
    samples: MediaPayload
    prompt_embeds: EncoderOutputs
    pooled_prompt_embeds: EncoderOutputs
    encoder_attention_mask: EncoderOutputs
    negative_prompt_embeds: EncoderOutputs
    neg_pooled_prompt_embeds: EncoderOutputs
    #: Mask paired with ``negative_prompt_embeds`` — variable-length encoders
    #: (Qwen-VL) require it for mask-consuming replay conditioning.
    negative_attention_mask: EncoderOutputs


@runtime_checkable
class Backend(Protocol):
    """The seam every ``sglang_diffusion`` collaborator reaches the runtime through."""

    # generation
    def generate(self, sampling_kwargs: Dict[str, Any]) -> List[RawResult]: ...
    def prepare_latent_shape(self, *, height: int, width: int, num_frames: int, batch_size: int) -> tuple: ...
    # memory / lifecycle / health
    def release_memory(self, *, tags: Sequence[str], cpu_backup_tags: Optional[Sequence[str]] = None) -> None: ...
    def resume_memory(self, *, tags: Sequence[str]) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
    # weight-sync verbs (runtime payload types stay inside the impls)
    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None: ...
    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None: ...
    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: List[str],
        flush_cache: bool,
    ) -> None: ...
    def destroy_weights_group(self, *, group_name: str) -> None: ...
    def set_lora(
        self,
        *,
        lora_nickname: str,
        lora_tensors: Dict[str, Any],
        target: str = "all",
        strength: float = 1.0,
        lora_alpha: Optional[float] = None,
    ) -> None: ...
    def weights_checksum(self, *, module_names: List[str]) -> dict: ...

    # update_from_ipc is intentionally absent — SGLang has no IPC receiver.


__all__ = ["Backend", "RawResult", "EncoderOutputs", "MediaPayload"]
