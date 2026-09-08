"""The backend seam contract — the ``Backend`` protocol + the wire types."""

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

STAGE_KIND_AR = "ar"
STAGE_KIND_DIFFUSION = "diffusion"


@dataclass(frozen=True)
class StageSampling:
    """Sampling-params intent for one stage — kind + plain ctor kwargs."""

    kind: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (STAGE_KIND_AR, STAGE_KIND_DIFFUSION):
            raise ValueError(
                f"StageSampling.kind must be {STAGE_KIND_AR!r} or {STAGE_KIND_DIFFUSION!r}; got {self.kind!r}"
            )


@dataclass(frozen=True)
class GenerateCall:
    """One ``Omni.generate`` invocation — prompts + per-stage sampling intent."""

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
    """Wire view of vllm-omni output: ``trajectory_latents [1, T+1, ...]``, ``trajectory_log_probs [1, K]``."""

    request_id: str
    stage_id: Optional[int]
    final_output_type: Optional[str]
    request_output: Optional[Any]
    prompt_token_ids: Optional[Sequence[int]]
    images: Optional[Sequence[Any]]
    trajectory_latents: Optional["torch.Tensor"]
    trajectory_timesteps: Optional["torch.Tensor"]
    trajectory_log_probs: Optional["torch.Tensor"]
    multimodal_output: Optional[dict]


@runtime_checkable
class Backend(Protocol):
    """The seam every ``vllm_omni`` collaborator reaches the runtime through."""

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]: ...
    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]: ...
    def num_stages(self) -> int: ...
    def tp_per_stage(self) -> Dict[int, int]: ...
    def sleep_task(self) -> None: ...
    def wake_task(self) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
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
