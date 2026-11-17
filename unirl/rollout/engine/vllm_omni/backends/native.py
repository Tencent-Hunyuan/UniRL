"""The native ``Backend`` impl — the in-process ``Omni`` orchestrator."""

from __future__ import annotations

import logging
import os
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import torch

from unirl.rollout.engine.vllm_omni.backends.base import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
)
from unirl.utils.graceful_shutdown import terminate_descendants

logger = logging.getLogger(__name__)

_ENGINE_PROC_PREFIX = "VLLM::"


class VLLMOmniBackend:
    """The native ``Backend`` impl over the ``Omni`` orchestrator."""

    def __init__(
        self,
        omni: Any,
        runtime: Dict[str, Any],
        *,
        tokenizer: Optional[Any],
    ) -> None:
        self.omni: Optional[Any] = omni
        self.rt = runtime
        self.tokenizer = tokenizer

    @classmethod
    def boot(cls, intent: Dict[str, Any]) -> "VLLMOmniBackend":
        """Spell the intent into ``Omni`` ctor kwargs and spawn."""
        import multiprocessing as mp

        import torch

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # vLLM disables cuDNN SDPA process-wide at import. Preserve the
        # colocated train actor's state; spawned workers initialize their own.
        cudnn_sdp_enabled = torch.backends.cuda.cudnn_sdp_enabled() if torch.cuda.is_available() else None
        try:
            from unirl.rollout.engine.vllm_omni.patches import install as install_patches
            from unirl.rollout.engine.vllm_omni.plugin import register_unirl_runtime

            install_patches()
            from transformers import AutoTokenizer
            from vllm import SamplingParams as VLLMSamplingParams
            from vllm_omni.diffusion.data import OmniSleepTask, OmniWakeTask
            from vllm_omni.entrypoints.omni import Omni
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
            from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

            rt = {
                "AutoTokenizer": AutoTokenizer,
                "Omni": Omni,
                "OmniDiffusionSamplingParams": OmniDiffusionSamplingParams,
                "OmniLoRARequest": OmniLoRARequest,
                "OmniSleepTask": OmniSleepTask,
                "OmniWakeTask": OmniWakeTask,
                "VLLMSamplingParams": VLLMSamplingParams,
            }
            register_unirl_runtime()
        finally:
            if cudnn_sdp_enabled is not None:
                torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_enabled)

        if intent.get("clear_cuda_visible"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        import fcntl

        # Release the trainer CUDA cache before spawning the engine process.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - belt and braces; never block a boot
            pass

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        deploy_config_path = os.path.join(here, "deploy_configs", intent["deploy_config"])
        if not os.path.exists(deploy_config_path):
            raise FileNotFoundError(f"VLLMOmniBackend.boot: YAML not found at {deploy_config_path}")

        omni_kwargs = dict(intent["omni_kwargs"])
        if intent.get("enable_sleep_mode"):
            omni_kwargs["enable_sleep_mode"] = True
        ports = intent.get("ports")
        boot_master_port = ports.master_port if ports is not None else None
        logger.info(
            "VLLM-Omni boot intent (before engine startup):\n%s",
            pformat(
                {
                    **intent,
                    "deploy_config_path": deploy_config_path,
                    "assembled_omni_kwargs": omni_kwargs,
                    "boot_master_port": boot_master_port,
                },
                sort_dicts=True,
            ),
        )
        serialize = os.environ.get("UNIRL_OMNI_BOOT_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/unirl_omni_boot.lock", "a+") if serialize else None
        omni = None
        try:
            try:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                saved_master_port = os.environ.get("MASTER_PORT")
                if boot_master_port is not None:
                    os.environ["MASTER_PORT"] = str(boot_master_port)
                    logger.info(
                        "VLLM-Omni boot: MASTER_PORT=%s (was %s) for engine construction",
                        boot_master_port,
                        saved_master_port,
                    )
                try:
                    omni = rt["Omni"](
                        model=intent["model_path"],
                        deploy_config=deploy_config_path,
                        **omni_kwargs,
                    )
                finally:
                    if boot_master_port is not None:
                        if saved_master_port is None:
                            os.environ.pop("MASTER_PORT", None)
                        else:
                            os.environ["MASTER_PORT"] = saved_master_port
            finally:
                if lock_file is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()

            stage_configs = omni.engine.stage_configs
            try:
                from omegaconf import OmegaConf

                resolved_stage_configs = OmegaConf.to_container(
                    OmegaConf.create(stage_configs),
                    resolve=True,
                )
            except Exception:  # noqa: BLE001 - config logging must never block boot
                resolved_stage_configs = stage_configs
            logger.info(
                "VLLM-Omni resolved runtime stage configs (after all overrides):\n%s",
                pformat(resolved_stage_configs, sort_dicts=True),
            )

            tokenizer = None
            if intent.get("needs_driver_tokenizer"):
                tokenizer = rt["AutoTokenizer"].from_pretrained(intent["model_path"], trust_remote_code=True)

            return cls(
                omni,
                rt,
                tokenizer=tokenizer,
            )
        except BaseException:
            logger.exception("VLLM-Omni boot failed; tearing down any engine processes")
            if omni is not None:
                try:
                    omni.close()
                except Exception:
                    logger.exception("Failed to close the half-booted vLLM-Omni engine")
            terminate_descendants(os.getpid(), name_prefix=_ENGINE_PROC_PREFIX)
            raise

    def require_omni(self) -> Any:
        if self.omni is None:
            raise RuntimeError("VLLMOmniBackend: engine not initialized (shut down?)")
        return self.omni

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]:
        """Run each call through ``Omni.generate`` and group per request."""
        omni = self.require_omni()
        lora_req = None
        if attach_lora:
            from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
                UNIRL_LORA_INT_ID,
                UNIRL_LORA_NAME,
                UNIRL_LORA_PATH,
            )

            lora_req = self.rt["OmniLoRARequest"](
                lora_name=UNIRL_LORA_NAME,
                lora_int_id=UNIRL_LORA_INT_ID,
                lora_path=UNIRL_LORA_PATH,
            )

        groups: List[List[OmniRawResult]] = []
        for call in calls:
            sp_list = []
            for sampling in call.sampling:
                if sampling.kind == STAGE_KIND_AR:
                    sp_list.append(self.rt["VLLMSamplingParams"](**sampling.kwargs))
                else:
                    sp = self.rt["OmniDiffusionSamplingParams"](**sampling.kwargs)
                    if lora_req is not None:
                        sp.lora_request = lora_req
                        sp.lora_scale = 1.0
                    sp_list.append(sp)
            generate_kwargs: Dict[str, Any] = {"use_tqdm": False}
            if lora_req is not None and ar_lora_passthrough:
                generate_kwargs["lora_request"] = lora_req
            flat = list(omni.generate(call.prompts, sp_list, **generate_kwargs))
            if call.group_by_request_id:
                n = len(call.prompts)
                grouped: List[List[OmniRawResult]] = [[] for _ in range(n)]
                for out in flat:
                    rid = out.request_id
                    try:
                        idx_part, suffix = rid.split("_", 1)
                        if not suffix:
                            raise ValueError("empty request-id suffix")
                        idx = int(idx_part)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(f"vllm-omni returned an invalid request_id: {rid!r}") from exc
                    if not 0 <= idx < n:
                        raise RuntimeError(f"vllm-omni request_id index {idx} is outside the prompt batch of size {n}")
                    grouped[idx].append(out)
                groups.extend(grouped)
            else:
                groups.append(flat)
        return groups

    def tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]:
        """HI3 prompt tokens via vllm-omni's ``build_prompt_tokens``."""
        if self.tokenizer is None:
            raise RuntimeError(
                "VLLMOmniBackend.tokenize_prompt: no driver tokenizer loaded "
                "(boot intent did not set needs_driver_tokenizer)."
            )
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
            build_prompt_tokens,
        )

        # vLLM-Omni returns a PromptTokensResult here, not a bare id list.
        return list(build_prompt_tokens(text, self.tokenizer, task=task, sys_type=sys_type).token_ids)

    def num_stages(self) -> int:
        return int(self.require_omni().engine.num_stages)

    def tp_per_stage(self) -> Dict[int, int]:
        tp_map: Dict[int, int] = {}
        for entry in self.require_omni().engine.stage_configs:
            engine_args = entry.engine_args
            tp = engine_args.get("tensor_parallel_size")
            if tp is None:
                parallel_config = engine_args.get("parallel_config")
                if parallel_config is not None:
                    tp = parallel_config.get("tensor_parallel_size")
            tp_map[int(entry.stage_id)] = int(tp) if tp is not None else 1
        return tp_map

    def collective_rpc(
        self,
        stage_id: int,
        method: str,
        *,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        allow_false: bool = False,
    ) -> List[Any]:
        results = self.require_omni().engine.collective_rpc(
            method=method,
            args=args,
            kwargs=kwargs,
            stage_ids=[stage_id],
        )

        def validate(result: object) -> None:
            if result is None:
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    validate(item)
                return
            if isinstance(result, bool):
                if not result and not allow_false:
                    raise RuntimeError(f"vllm-omni {method} returned False on stage {stage_id}")
                return
            if not isinstance(result, Mapping):
                raise RuntimeError(f"vllm-omni {method} returned an unexpected result on stage {stage_id}: {result!r}")

            status = result.get("status")
            error = result.get("error")
            if result.get("supported") is False or error or (status is not None and status != "SUCCESS"):
                reason = result.get("reason") or result.get("error_msg") or error or status
                raise RuntimeError(f"vllm-omni {method} failed on stage {stage_id}: {reason!r}")

        validate(results)
        return results

    @staticmethod
    def require_ack_success(action: str, stage_id: int, task_id: str, acks: object) -> None:
        # Worker handlers catch their own exceptions and answer
        # ``OmniACK(status="ERROR")`` instead of raising, so a discarded return
        # value turns a failed sleep/wake into a silent partial transition — the
        # exact state the engine's transition latch exists to refuse.
        success_count = 0

        def validate_result(result: Any) -> None:
            nonlocal success_count
            if result is None:
                # Non-reporting worker ranks legitimately return None; at least
                # one explicit rank-0 SUCCESS ACK is still required below.
                return
            if isinstance(result, (list, tuple)):
                for item in result:
                    validate_result(item)
                return

            if isinstance(result, Mapping):
                # Subprocess diffusion stages serialize OmniACK via
                # OmniMsgpackEncoder (asdict) without reconstructing the
                # dataclass on decode, and StagePool error replies arrive as
                # ``{"supported": False, "error": ...}``.
                status = result.get("status")
                if status is None:
                    raise RuntimeError(
                        f"vllm-omni {action} returned no ACK status for stage {stage_id}: result={result!r}"
                    )
                rank = result["rank"]
                error_msg = result["error_msg"]
                ack_stage_id = result["stage_id"]
                ack_task_id = result["task_id"]
            else:
                status = result.status
                if status is None:
                    raise RuntimeError(
                        f"vllm-omni {action} returned no ACK status for stage {stage_id}: result={result!r}"
                    )
                rank = result.rank
                error_msg = result.error_msg
                ack_stage_id = result.stage_id
                ack_task_id = result.task_id

            if status != "SUCCESS":
                raise RuntimeError(
                    f"vllm-omni {action} failed on stage {stage_id}: worker rank "
                    f"{rank} answered status={status!r} "
                    f"error={error_msg!r}"
                )
            if ack_stage_id != stage_id:
                raise RuntimeError(f"vllm-omni {action} ACK stage mismatch: expected {stage_id}, got {ack_stage_id!r}")
            if ack_task_id != task_id:
                raise RuntimeError(
                    f"vllm-omni {action} ACK task mismatch on stage {stage_id}: "
                    f"expected {task_id!r}, got {ack_task_id!r}"
                )
            success_count += 1

        validate_result(acks)
        if success_count == 0:
            raise RuntimeError(f"vllm-omni {action} returned no successful ACK for stage {stage_id}")

    def sleep_task(self) -> None:
        """Sleep AR stages through EngineCore and diffusion stages through worker RPC."""
        import uuid

        omni = self.require_omni()
        for sid in range(self.num_stages()):
            if omni.engine.get_stage_metadata(sid).stage_type != "diffusion":
                self.collective_rpc(sid, "sleep", args=(1, "abort"))
                continue
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_sleep_task",
                args=(self.rt["OmniSleepTask"](level=1, task_id=task_id),),
                stage_ids=[sid],
            )
            self.require_ack_success("sleep", sid, task_id, acks)

    def wake_task(self) -> None:
        """Wake AR stages through EngineCore and diffusion stages through worker RPC."""
        import uuid

        import torch

        omni = self.require_omni()
        for sid in range(self.num_stages()):
            if omni.engine.get_stage_metadata(sid).stage_type != "diffusion":
                # ``None`` wakes every tag, including UniRL's MoE workspace tag.
                self.collective_rpc(sid, "wake_up", kwargs={"tags": None})
                continue
            task_id = str(uuid.uuid4())
            acks = omni.engine.collective_rpc(
                method="handle_wake_task",
                args=(self.rt["OmniWakeTask"](tags=None, task_id=task_id),),
                stage_ids=[sid],
            )
            self.require_ack_success("wake", sid, task_id, acks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def ping(self) -> bool:
        omni = self.omni
        if omni is None:
            return False
        try:
            return omni.is_running
        except Exception:
            logger.exception("vLLM-Omni health check failed")
            return False

    def shutdown(self) -> None:
        if self.omni is not None:
            try:
                self.omni.close()
            finally:
                self.omni = None

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
        kwargs = {
            "peft_config": peft_config,
            "base_sync_done": base_sync_done,
            "use_shm": use_shm,
            "replica_rank": replica_rank,
        }
        for sid in range(self.num_stages()):
            self.collective_rpc(
                sid,
                "update_weights_from_ipc",
                kwargs={**kwargs, "stage_id": sid},
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
        kwargs = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": rank_offset,
            "world_size": world_size,
            "group_name": group_name,
            "backend": backend,
        }
        for sid in range(self.num_stages()):
            self.collective_rpc(
                sid,
                "init_weights_update_group",
                kwargs=kwargs,
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
        kwargs = {
            "names": list(names),
            "dtypes": list(dtypes),
            "shapes": [list(s) for s in shapes],
            "group_name": group_name,
            "target_modules": list(target_modules) if target_modules else None,
            "flush_cache": flush_cache,
        }
        for sid in range(self.num_stages()):
            self.collective_rpc(
                sid,
                "update_weights_from_distributed",
                kwargs=kwargs,
            )

    def destroy_weights_group(self, *, group_name: str) -> None:
        if self.omni is None:
            return
        for sid in range(self.num_stages()):
            self.collective_rpc(
                sid,
                "destroy_weights_update_group",
                kwargs={"group_name": group_name},
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
        kwargs = {
            "serialized_named_tensors": list(serialized_named_tensors),
            "target_modules": list(target_modules) if target_modules else None,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        for sid in range(self.num_stages()):
            self.collective_rpc(
                sid,
                "update_weights_from_tensor",
                kwargs=kwargs,
            )

    def set_lora_handle(
        self,
        *,
        lora_tensors: Dict[str, torch.Tensor],
        peft_config: Optional[dict],
    ) -> None:
        """Zero-copy LoRA push via ``MultiprocessingSerializer`` shm handles."""
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            UNIRL_LORA_INT_ID,
            UNIRL_LORA_NAME,
            UNIRL_LORA_PATH,
        )
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            MultiprocessingSerializer,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        lora_tensors = adapt_lora_for_vllm(lora_tensors)
        peft_dict = {} if peft_config is None else dict(peft_config)

        for sid in range(self.num_stages()):
            self.collective_rpc(sid, "remove_lora", args=(UNIRL_LORA_INT_ID,), allow_false=True)
            cloned = {name: t.detach().clone() for name, t in lora_tensors.items()}
            serialized = MultiprocessingSerializer.serialize(cloned, output_str=True)
            self.collective_rpc(
                sid,
                "set_lora_from_tensor_dict",
                args=(
                    UNIRL_LORA_NAME,
                    UNIRL_LORA_INT_ID,
                    UNIRL_LORA_PATH,
                    peft_dict,
                    serialized,
                ),
            )

    def set_lora_copy(
        self,
        *,
        lora_tensors: Dict[str, torch.Tensor],
        peft_config: Optional[dict],
    ) -> None:
        """Byte-copy LoRA push (``torch.save`` + base64) — TP>1-broadcast-safe."""
        import base64
        import io

        import torch

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            UNIRL_LORA_INT_ID,
            UNIRL_LORA_NAME,
            UNIRL_LORA_PATH,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        lora_tensors = adapt_lora_for_vllm(lora_tensors)
        peft_dict = {} if peft_config is None else dict(peft_config)

        cpu_tensors = {name: t.detach().to("cpu") for name, t in lora_tensors.items()}
        buf = io.BytesIO()
        torch.save(cpu_tensors, buf)
        serialized = base64.b64encode(buf.getvalue()).decode("ascii")

        for sid in range(self.num_stages()):
            self.collective_rpc(sid, "remove_lora", args=(UNIRL_LORA_INT_ID,), allow_false=True)
            self.collective_rpc(
                sid,
                "set_lora_from_tensor_dict_copy",
                args=(
                    UNIRL_LORA_NAME,
                    UNIRL_LORA_INT_ID,
                    UNIRL_LORA_PATH,
                    peft_dict,
                    serialized,
                ),
            )

    def param_checksums(self, *, names: List[str]) -> dict:
        """Fan ``_unirl_loaded_param_checksums`` across stages and ranks."""
        out: dict = {}
        for sid in range(self.num_stages()):
            results = self.collective_rpc(
                sid,
                "_unirl_loaded_param_checksums",
                args=(list(names),),
            )
            out[sid] = results[0] if isinstance(results, list) and results else results
        return out

    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict:
        """Fan ``_unirl_loaded_lora_checksums`` across stages and ranks."""
        out: dict = {}
        for sid in range(self.num_stages()):
            results = self.collective_rpc(
                sid,
                "_unirl_loaded_lora_checksums",
                args=(adapter_id, list(names) if names else None),
            )
            out[sid] = results[0] if isinstance(results, list) and results else results
        return out


__all__ = ["VLLMOmniBackend"]
