"""The backend seam contract — the ``Backend`` protocol + the wire types.

Every ``vllm_omni`` collaborator reaches the vllm-omni runtime through this
protocol; the real implementation lives beside it (``native.py`` — the in-process
``Omni`` orchestrator). This module holds no runtime code at all, so it is
trivially CPU-importable.

**No RL types cross this seam.** ``generate`` takes :class:`GenerateCall`\\ s —
plain prompt dicts + :class:`StageSampling` intent (kind + kwargs; the impl
constructs the real ``vllm.SamplingParams`` / ``OmniDiffusionSamplingParams``
objects) — and returns per-request-grouped lists of :class:`OmniRawResult`
(a structural view of vllm-omni's ``OmniRequestOutput``). The engine core +
adapters do the ``RolloutReq``↔``RolloutResp`` translation.

The seam absorbs the transport asymmetries: ``Omni.generate``'s flat output
list is grouped back to per-request order by the ``"{i}_{uuid}"`` request-id
prefix (the impl owns that parsing), LoRA activation objects
(``OmniLoRARequest`` attach + the HI3 ``lora_request`` generate kwarg) are
constructed inside the impl, and AR prompt tokenization (vllm-omni's
``build_prompt_tokens``) is exposed as the :meth:`Backend.tokenize_prompt`
verb so adapters never import the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

#: ``StageSampling.kind`` values: ``"ar"`` builds ``vllm.SamplingParams``;
#: ``"diffusion"`` builds ``OmniDiffusionSamplingParams`` (and is the LoRA
#: attach point when an adapter is active).
STAGE_KIND_AR = "ar"
STAGE_KIND_DIFFUSION = "diffusion"


@dataclass(frozen=True)
class StageSampling:
    """Sampling-params intent for one stage — kind + plain ctor kwargs.

    Adapters build these instead of the runtime's params objects (which would
    require importing vllm / vllm-omni); the impl maps ``kind`` to the real
    class. ``kwargs`` may carry tensors (e.g. ``extra_args.initial_noise_batch``)
    — vllm-omni routes ``extra_args`` to the worker preserving tensor values.
    """

    kind: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (STAGE_KIND_AR, STAGE_KIND_DIFFUSION):
            raise ValueError(
                f"StageSampling.kind must be {STAGE_KIND_AR!r} or {STAGE_KIND_DIFFUSION!r}; got {self.kind!r}"
            )


@dataclass(frozen=True)
class GenerateCall:
    """One ``Omni.generate`` invocation — prompts + per-stage sampling intent.

    Adapters return a list of these from ``build_inputs``: normally one call
    carrying the whole batch; ``dit_recaption`` returns N single-prompt calls so
    each image gets its own sampling seed (see the adapter for why seeds can't
    ride a shared per-stage params object).

    ``group_by_request_id`` selects how the impl groups the call's flat output
    back to per-request lists: ``True`` (default) parses the ``"{i}_{uuid}"``
    request-id prefix; ``False`` treats the whole flat list as the single
    request's group (only valid for single-prompt calls — preserves the v1
    ``dit_recaption`` per-prompt path byte-for-byte).
    """

    prompts: List[Any]
    sampling: List[StageSampling]
    group_by_request_id: bool = True

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("GenerateCall.prompts must be non-empty")
        if not self.sampling:
            raise ValueError("GenerateCall.sampling must be non-empty")
        if not self.group_by_request_id and len(self.prompts) != 1:
            raise ValueError(
                "GenerateCall.group_by_request_id=False is only valid for "
                f"single-prompt calls; got {len(self.prompts)} prompts"
            )


class OmniRawResult(Protocol):
    """Structural view of vllm-omni's ``OmniRequestOutput`` — the wire fields
    this engine consumes. The native impl passes ``OmniRequestOutput`` through
    (it satisfies this protocol structurally); test fakes (``SimpleNamespace``
    with the fields) stand in. Adapters/utils annotate against it and stay
    vllm-omni-free.

    Population by stage kind:

    - Every output: ``request_id`` (``"{i}_{uuid}"``; the impl consumes it for
      grouping), ``stage_id``, ``final_output_type`` (``"text"`` for the AR
      stage, ``"image"`` / ``"video"`` for the final DiT stage).
    - AR stage (``final_output_type == "text"``): ``request_output`` — the
      nested vLLM ``RequestOutput`` (``.outputs[0].token_ids`` / ``.logprobs``
      / ``.text``) — and ``prompt_token_ids`` (the sample's true, un-padded
      prompt; vLLM runs prompts per-request with no batch padding).
    - DiT stage (``"image"`` / ``"video"``): ``images`` (PIL list; per-prompt
      frame list for video), ``trajectory_latents`` ``[1, T+1, ...]`` (dense —
      every step recorded), ``trajectory_timesteps`` ``[T+1]`` (the field name
      reads "timesteps" but the RL pipeline subclass overwrites its contents
      with the true [0, 1] σ schedule), ``trajectory_log_probs`` ``[1, K]``
      (K = SDE-gated step count; 0 for NFT/forward-process), and
      ``custom_output`` — the dataclass-routed capture dict that survives the
      worker IPC boundary. Documented keys: ``"fused_mm_capture"`` (HI3
      ``prepare_inputs_for_generation`` capture), ``"text_capture"`` (SD3 /
      HV1.5 ``encode_prompt`` capture), ``"sde_step_indices"`` (the SDE-gated
      step ids echoed by the scheduler). Missing capture is a fatal
      misconfiguration the *adapter* raises on — the seam passes the dict
      through structurally.
    """

    request_id: str
    stage_id: Optional[int]
    final_output_type: Optional[str]
    request_output: Optional[Any]
    prompt_token_ids: Optional[Sequence[int]]
    images: Optional[Sequence[Any]]
    trajectory_latents: Optional["torch.Tensor"]
    trajectory_timesteps: Optional["torch.Tensor"]
    trajectory_log_probs: Optional["torch.Tensor"]
    custom_output: Optional[dict]


@runtime_checkable
class Backend(Protocol):
    """The seam every ``vllm_omni`` collaborator reaches the runtime through."""

    # generation
    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]: ...
    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]: ...
    # stage topology
    def num_stages(self) -> int: ...
    def tp_per_stage(self) -> Dict[int, int]: ...
    # memory / lifecycle / health
    def sleep_task(self) -> None: ...
    def wake_task(self) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
    # weight-sync verbs — per-stage collective_rpc fan-out lives INSIDE the impl;
    # runtime payload shaping (MultiprocessingSerializer / torch.save byte copy,
    # PEFT-envelope wrapping, remove-then-add ordering) stays there too.
    def update_from_ipc(
        self,
        *,
        peft_config: Optional[dict],
        base_sync_done: bool,
        use_shm: bool,
        replica_rank: Optional[int],
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
        target_modules: Optional[List[str]],
        flush_cache: bool,
    ) -> None: ...
    def destroy_weights_group(self, *, group_name: str) -> None: ...
    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None: ...
    def set_lora_handle(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None: ...
    def set_lora_copy(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None: ...
    def param_checksums(self, *, names: List[str]) -> dict: ...
    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict: ...


__all__ = [
    "Backend",
    "GenerateCall",
    "OmniRawResult",
    "StageSampling",
    "STAGE_KIND_AR",
    "STAGE_KIND_DIFFUSION",
]
