"""The native ``Backend`` impl — the in-process ``Omni`` orchestrator."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pprint import pformat
from typing import Any, Dict, List, Mapping, Optional, Sequence

from unirl.rollout.engine.vllm_omni.backends.base import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.utils.graceful_shutdown import terminate_descendants

logger = logging.getLogger(__name__)

_ENGINE_PROC_PREFIX = "VLLM::"


def _import_omni_runtime() -> Dict[str, Any]:
    """Lazy import of the vllm-omni runtime types. Imported once per process."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams as VLLMSamplingParams
    from vllm_omni.diffusion.data import OmniSleepTask, OmniWakeTask
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

    return {
        "AutoTokenizer": AutoTokenizer,
        "Omni": Omni,
        "OmniDiffusionSamplingParams": OmniDiffusionSamplingParams,
        "OmniLoRARequest": OmniLoRARequest,
        "OmniSleepTask": OmniSleepTask,
        "OmniWakeTask": OmniWakeTask,
        "VLLMSamplingParams": VLLMSamplingParams,
    }


def _resolve_stage_yaml(name: str) -> str:
    """Return the absolute path of the local stage-config YAML asset."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "stage_configs", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"_resolve_stage_yaml: YAML not found at {path}")
    return path


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Mapping/OmegaConf/attr-tolerant getter (``None`` coerces to default)."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    value = getter(key, default) if callable(getter) else getattr(cfg, key, default)
    return default if value is None else value


def _tp_from_stage_configs(stage_configs: Sequence[Any]) -> Dict[int, int]:
    """Extract ``{stage_id: tensor_parallel_size}`` from the runtime's configs."""
    tp_map: Dict[int, int] = {}
    for entry in stage_configs:
        sid = int(_cfg_get(entry, "stage_id", len(tp_map)))
        ea = _cfg_get(entry, "engine_args", {})
        tp = _cfg_get(ea, "tensor_parallel_size")
        if tp is None:
            tp = _cfg_get(_cfg_get(ea, "parallel_config", {}), "tensor_parallel_size")
        tp_map[sid] = int(tp) if tp is not None else 1
    return tp_map


def _assemble_omni_kwargs(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Spell the boot intent into ``Omni`` ctor kwargs."""
    omni_kwargs = dict(intent.get("omni_kwargs") or {})
    if intent.get("enable_sleep_mode"):
        omni_kwargs["enable_sleep_mode"] = True
    return omni_kwargs


@contextmanager
def _master_port_env(port: Optional[int]):
    """Pin ``MASTER_PORT`` to the reserved engine port for the duration of ``Omni()``."""
    if port is None:
        yield
        return
    saved = os.environ.get("MASTER_PORT")
    os.environ["MASTER_PORT"] = str(int(port))
    logger.info("VLLM-Omni boot: MASTER_PORT=%s (was %s) for engine construction", port, saved)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = saved


class VLLMOmniBackend:
    """The native ``Backend`` impl over the ``Omni`` orchestrator."""

    def __init__(
        self,
        omni: Any,
        runtime: Dict[str, Any],
        *,
        tokenizer: Optional[Any],
        tp_per_stage: Dict[int, int],
    ) -> None:
        self._omni: Optional[Any] = omni
        self._rt = runtime
        self._tokenizer = tokenizer
        self._tp_per_stage = dict(tp_per_stage)

    @classmethod
    def boot(cls, intent: Dict[str, Any]) -> "VLLMOmniBackend":
        """Spell the intent into ``Omni`` ctor kwargs and spawn."""
        from unirl.rollout.engine.vllm_omni.patches import install as install_patches

        install_patches()

        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        rt = _import_omni_runtime()

        if intent.get("clear_cuda_visible"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        import fcntl

        # Release the trainer CUDA cache before spawning the engine process.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - belt and braces; never block a boot
            pass

        yaml_path = _resolve_stage_yaml(str(intent["stage_yaml"]))
        omni_kwargs = _assemble_omni_kwargs(intent)
        ports = intent.get("ports")
        boot_master_port = int(ports.master_port) if ports is not None else None
        logger.info(
            "VLLM-Omni boot intent (before engine startup):\n%s",
            pformat(
                {
                    **intent,
                    "stage_yaml_path": yaml_path,
                    "assembled_omni_kwargs": omni_kwargs,
                    "boot_master_port": boot_master_port,
                },
                sort_dicts=True,
            ),
        )
        serialize = os.environ.get("DIFFRL_OMNI_BOOT_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_omni_boot.lock", "a+") if serialize else None
        omni = None
        try:
            try:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                with _master_port_env(boot_master_port):
                    omni = rt["Omni"](
                        model=str(intent["model_path"]),
                        stage_configs_path=yaml_path,
                        **omni_kwargs,
                    )
            finally:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()

            try:
                from omegaconf import OmegaConf

                resolved_stage_configs = OmegaConf.to_container(
                    OmegaConf.create(omni.stage_configs),
                    resolve=True,
                )
            except Exception:  # noqa: BLE001 - config logging must never block boot
                resolved_stage_configs = omni.stage_configs
            logger.info(
                "VLLM-Omni resolved runtime stage configs (after all overrides):\n%s",
                pformat(resolved_stage_configs, sort_dicts=True),
            )

            tokenizer = None
            if intent.get("needs_driver_tokenizer"):
                tokenizer = rt["AutoTokenizer"].from_pretrained(str(intent["model_path"]), trust_remote_code=True)

            return cls(
                omni,
                rt,
                tokenizer=tokenizer,
                tp_per_stage=_tp_from_stage_configs(omni.stage_configs),
            )
        except BaseException:
            logger.exception("VLLM-Omni boot failed; tearing down any engine processes")
            if omni is not None:
                try:
                    close = getattr(omni, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    logger.exception("Failed to close the half-booted vLLM-Omni engine")
            terminate_descendants(os.getpid(), name_prefix=_ENGINE_PROC_PREFIX)
            raise

    def _require_omni(self) -> Any:
        if self._omni is None:
            raise RuntimeError("VLLMOmniBackend: engine not initialized (shut down?)")
        return self._omni

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]:
        """Run each call through ``Omni.generate`` and group per request."""
        omni = self._require_omni()
        groups: List[List[OmniRawResult]] = []
        for call in calls:
            sp_list = [self._build_sampling_params(s, attach_lora=attach_lora) for s in call.sampling]
            generate_kwargs: Dict[str, Any] = {"use_tqdm": False}
            if attach_lora and ar_lora_passthrough:
                generate_kwargs["lora_request"] = self._lora_request()
            flat = list(omni.generate(call.prompts, sp_list, **generate_kwargs))
            if call.group_by_request_id:
                groups.extend(_group_by_request(flat, len(call.prompts)))
            else:
                groups.append(flat)
        return groups

    def _build_sampling_params(self, sampling: StageSampling, *, attach_lora: bool) -> Any:
        if sampling.kind == STAGE_KIND_AR:
            return self._rt["VLLMSamplingParams"](**sampling.kwargs)
        sp = self._rt["OmniDiffusionSamplingParams"](**sampling.kwargs)
        if attach_lora:
            sp.lora_request = self._lora_request()
            sp.lora_scale = 1.0
        return sp

    def _lora_request(self) -> Any:
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )

        return self._rt["OmniLoRARequest"](
            lora_name=DIFFRL_LORA_NAME,
            lora_int_id=int(DIFFRL_LORA_INT_ID),
            lora_path=DIFFRL_LORA_PATH,
        )

    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]:
        """HI3 prompt tokens via vllm-omni's ``build_prompt_tokens``."""
        if self._tokenizer is None:
            raise RuntimeError(
                "VLLMOmniBackend.tokenize_prompt: no driver tokenizer loaded "
                "(boot intent did not set needs_driver_tokenizer)."
            )
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
            build_prompt_tokens,
        )

        # vllm-omni 0.27 returns a PromptTokensResult here, not a bare id list.
        return list(build_prompt_tokens(text, self._tokenizer, task=task, sys_type=sys_type).token_ids)

    def num_stages(self) -> int:
        return int(self._require_omni().engine.num_stages)

    def tp_per_stage(self) -> Dict[int, int]:
        return dict(self._tp_per_stage)

    def _stage_ids(self) -> List[int]:
        return list(range(self.num_stages()))

    @staticmethod
    def _require_ack_success(action: str, stage_id: int, task_id: str, acks: object) -> None:
        # Worker handlers catch their own exceptions and answer
        # ``OmniACK(status="ERROR")`` instead of raising, so a discarded return
        # value turns a failed sleep/wake into a silent partial transition — the
        # exact state the engine's transition latch exists to refuse.
        success_count = 0

        def ack_field(ack: object, name: str, default: object = None) -> object:
            return ack.get(name, default) if isinstance(ack, Mapping) else getattr(ack, name, default)

        def ack_error(ack: object) -> object:
            return ack_field(ack, "error_msg", ack_field(ack, "error"))

        def validate_result(result: object) -> None:
            nonlocal success_count
            if result is None:
                # Non-reporting worker ranks legitimately return None; at least
                # one explicit rank-0 SUCCESS ACK is still required below.
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    validate_result(item)
                return

            status = ack_field(result, "status")
            if status is None:
                raise RuntimeError(f"vllm-omni {action} returned no ACK status for stage {stage_id}: result={result!r}")
            if status != "SUCCESS":
                raise RuntimeError(
                    f"vllm-omni {action} failed on stage {stage_id}: worker rank "
                    f"{ack_field(result, 'rank', '?')} answered status={status!r} "
                    f"error={ack_error(result)!r}"
                )
            ack_stage_id = ack_field(result, "stage_id")
            ack_task_id = ack_field(result, "task_id")
            if ack_stage_id is None or int(ack_stage_id) != stage_id:
                raise RuntimeError(f"vllm-omni {action} ACK stage mismatch: expected {stage_id}, got {ack_stage_id!r}")
            if ack_task_id is None or str(ack_task_id) != task_id:
                raise RuntimeError(
                    f"vllm-omni {action} ACK task mismatch on stage {stage_id}: "
                    f"expected {task_id!r}, got {ack_task_id!r}"
                )
            success_count += 1

        validate_result(acks)
        if success_count == 0:
            raise RuntimeError(f"vllm-omni {action} returned no successful ACK for stage {stage_id}")

    def sleep_task(self) -> None:
        """Fan ``handle_sleep_task`` to every stage's workers (level 1)."""
        import uuid

        omni = self._require_omni()
        for sid in self._stage_ids():
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_sleep_task",
                args=(self._rt["OmniSleepTask"](level=1, task_id=task_id),),
                stage_ids=[int(sid)],
            )
            self._require_ack_success("sleep", int(sid), task_id, acks)

    def wake_task(self) -> None:
        """Fan ``handle_wake_task`` to every stage's workers + sync CUDA."""
        import uuid

        import torch

        omni = self._require_omni()
        for sid in self._stage_ids():
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_wake_task",
                args=(self._rt["OmniWakeTask"](tags=None, task_id=task_id),),
                stage_ids=[int(sid)],
            )
            self._require_ack_success("wake", int(sid), task_id, acks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def ping(self) -> bool:
        return self._omni is not None

    def shutdown(self) -> None:
        if self._omni is not None:
            try:
                close = getattr(self._omni, "close", None)
                if callable(close):
                    close()
            finally:
                self._omni = None

        reaped = terminate_descendants(os.getpid(), name_prefix=_ENGINE_PROC_PREFIX)
        if reaped:
            logger.warning("Reaped %d engine process(es) that outlived the vLLM-Omni shutdown", reaped)

    def update_from_ipc(
        self,
        *,
        peft_config: Optional[dict],
        base_sync_done: bool,
        use_shm: bool,
        replica_rank: Optional[int],
    ) -> None:
        """Fan a bucketed CUDA-IPC state-dict update out to per-stage workers."""
        omni = self._require_omni()
        kwargs = {
            "peft_config": peft_config,
            "base_sync_done": base_sync_done,
            "use_shm": use_shm,
            "replica_rank": replica_rank,
        }
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="update_weights_from_ipc",
                args=(),
                kwargs={**kwargs, "stage_id": int(sid)},
                stage_ids=[int(sid)],
            )

    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        omni = self._require_omni()
        kwargs = {
            "master_address": str(master_address),
            "master_port": int(master_port),
            "rank_offset": int(rank_offset),
            "world_size": int(world_size),
            "group_name": str(group_name),
            "backend": str(backend),
        }
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="init_weights_update_group",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]],
        flush_cache: bool,
    ) -> None:
        omni = self._require_omni()
        kwargs = {
            "names": list(names),
            "dtypes": list(dtypes),
            "shapes": [list(s) for s in shapes],
            "group_name": str(group_name),
            "target_modules": list(target_modules) if target_modules else None,
            "flush_cache": bool(flush_cache),
        }
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="update_weights_from_distributed",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    def destroy_weights_group(self, *, group_name: str) -> None:
        if self._omni is None:
            return
        for sid in self._stage_ids():
            self._omni.engine.collective_rpc(
                method="destroy_weights_update_group",
                args=(),
                kwargs={"group_name": str(group_name)},
                stage_ids=[int(sid)],
            )

    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None:
        """Fan a SGLang-shape tensor payload to per-stage workers."""
        omni = self._require_omni()
        kwargs = {
            "serialized_named_tensors": list(serialized_named_tensors),
            "target_modules": list(target_modules) if target_modules else None,
            "load_format": load_format,
            "flush_cache": bool(flush_cache),
        }
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="update_weights_from_tensor",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    def set_lora_handle(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None:
        """Zero-copy LoRA push via ``MultiprocessingSerializer`` shm handles."""
        import torch

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )

        omni = self._require_omni()
        lora_tensors = self._wrap_peft_envelope(lora_tensors)
        self._remove_existing_lora(int(DIFFRL_LORA_INT_ID))

        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            MultiprocessingSerializer,
        )

        for sid in self._stage_ids():
            cloned = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
            serialized = MultiprocessingSerializer.serialize(cloned, output_str=True)
            omni.engine.collective_rpc(
                method="set_lora_from_tensor_dict",
                args=(
                    str(adapter_name) or DIFFRL_LORA_NAME,
                    int(DIFFRL_LORA_INT_ID),
                    DIFFRL_LORA_PATH,
                    dict(peft_config or {}),
                    serialized,
                ),
                stage_ids=[int(sid)],
            )

    def set_lora_copy(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None:
        """Byte-copy LoRA push (``torch.save`` + base64) — TP>1-broadcast-safe."""
        import base64
        import io

        import torch

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )

        omni = self._require_omni()
        lora_tensors = self._wrap_peft_envelope(lora_tensors)
        self._remove_existing_lora(int(DIFFRL_LORA_INT_ID))

        cpu_tensors = {
            name: t.detach().to("cpu") if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
        }
        buf = io.BytesIO()
        torch.save(cpu_tensors, buf)
        serialized = base64.b64encode(buf.getvalue()).decode("ascii")

        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="set_lora_from_tensor_dict_copy",
                args=(
                    str(adapter_name) or DIFFRL_LORA_NAME,
                    int(DIFFRL_LORA_INT_ID),
                    DIFFRL_LORA_PATH,
                    dict(peft_config or {}),
                    serialized,
                ),
                stage_ids=[int(sid)],
            )

    @staticmethod
    def _wrap_peft_envelope(lora_tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap canonical wire keys in the PEFT envelope vllm-omni expects."""
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        first_key = next(iter(lora_tensors), "")
        if lora_tensors and not first_key.startswith("base_model.model."):
            return adapt_lora_for_vllm(lora_tensors)
        return lora_tensors

    def _remove_existing_lora(self, adapter_id: int) -> None:
        """Drop the existing adapter on every stage before re-adding."""
        omni = self._require_omni()
        for sid in self._stage_ids():
            try:
                omni.engine.collective_rpc(
                    method="remove_lora",
                    args=(int(adapter_id),),
                    stage_ids=[int(sid)],
                )
            except Exception:
                pass

    def param_checksums(self, *, names: List[str]) -> dict:
        """Fan ``_diffrl_loaded_param_checksums`` across stages and ranks."""
        omni = self._require_omni()
        out: dict = {}
        for sid in self._stage_ids():
            results = omni.engine.collective_rpc(
                method="_diffrl_loaded_param_checksums",
                args=(list(names),),
                stage_ids=[int(sid)],
            )
            out[int(sid)] = results[0] if isinstance(results, list) and results else results
        return out

    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict:
        """Fan ``_diffrl_loaded_lora_checksums`` across stages and ranks."""
        omni = self._require_omni()
        out: dict = {}
        for sid in self._stage_ids():
            results = omni.engine.collective_rpc(
                method="_diffrl_loaded_lora_checksums",
                args=(int(adapter_id), list(names) if names else None),
                stage_ids=[int(sid)],
            )
            out[int(sid)] = results[0] if isinstance(results, list) and results else results
        return out


def _group_by_request(flat_outputs: Sequence[Any], n: int) -> List[List[Any]]:
    """Group ``Omni.generate``'s flat output list into per-request lists."""
    grouped: List[List[Any]] = [[] for _ in range(n)]
    for out in flat_outputs:
        rid = getattr(out, "request_id", "") or ""
        if "_" in rid:
            idx_part = rid.split("_", 1)[0]
            try:
                idx = int(idx_part)
            except ValueError:
                continue
            if 0 <= idx < n:
                grouped[idx].append(out)
    return grouped


__all__ = ["VLLMOmniBackend"]
