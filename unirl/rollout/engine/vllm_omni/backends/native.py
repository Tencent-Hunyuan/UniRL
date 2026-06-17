"""The native ``Backend`` impl — the in-process ``Omni`` orchestrator.

The ONLY module that imports the vllm-omni runtime or does I/O — boot included.
Because every runtime import is lazy (inside :meth:`VLLMOmniBackend.boot` and the
verbs), the module imports on CPU and the rest of the package is exercisable
without a GPU.

Boot sequence (load-bearing order — see each step's note):

1. ``patches.install()`` — vllm/vllm-omni monkey-patches, including the
   ``mp.Process`` wrap that re-installs them inside every spawn child.
2. ``mp.set_start_method("spawn", force=True)`` — before any Omni mp object
   exists (fork-context SemLocks can't cross into spawn workers).
3. ``CUDA_VISIBLE_DEVICES`` pop when the adapter's boot intent asks for it
   (HI3 multi-GPU stages; the documented last-resort env override — vllm-omni
   reads CVD for per-stage device pinning and has no arg for it).
4. ``Omni(...)`` with the PRISTINE packaged stage YAML + ctor kwargs —
   ``enable_sleep_mode`` / ``master_port`` ride the runtime's own override
   channel (the ``base_engine_args`` merge + the dedicated sleep-mode
   injection in ``AsyncOmniEngine._resolve_stage_configs``), so no YAML
   rewrite, no temp file. Then the driver-side ``AutoTokenizer`` when the
   modality needs it; ``tp_per_stage`` reads back from the runtime's own
   merged ``omni.stage_configs``.

This replaces v1's ``base + rank*200 + idx*50`` port math and ``RANK``-env
fallback with one reserved master-port base riding ``Omni(master_port=...)``;
each stage settles its own port from that base (pinned v0.20.0:
``base + random(0, 100)`` then a +37 bind-check scan; ≥ v0.21.0rc2: honored
verbatim, scan only on collision — and note env ``MASTER_PORT`` then takes
precedence over the kwarg).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from unirl.rollout.engine.vllm_omni.backends.base import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)

logger = logging.getLogger(__name__)


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


def _resolve_stage_yaml(name: str, source: str) -> str:
    """Return the absolute path of the stage-config YAML asset.

    Local YAMLs ship in ``stage_configs/`` next to this package. Upstream
    YAMLs (the AR-only modalities) are looked up under
    ``<vllm_omni_project>/vllm_omni/model_executor/stage_configs/`` — the
    upstream loader requires an absolute path and raises on bare names.
    """
    if source == "local":
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "stage_configs", name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"_resolve_stage_yaml: local YAML not found at {path}")
        return path
    if source == "upstream":
        import vllm_omni  # runtime import — sanctioned here only

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(vllm_omni.__file__)))
        path = os.path.join(project_root, "vllm_omni", "model_executor", "stage_configs", name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"_resolve_stage_yaml: upstream YAML {name!r} not found at {path}. vllm-omni may have moved the file."
            )
        return path
    raise ValueError(f"_resolve_stage_yaml: unknown source {source!r} (expected 'local' or 'upstream')")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Mapping/OmegaConf/attr-tolerant getter (``None`` coerces to default)."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    value = getter(key, default) if callable(getter) else getattr(cfg, key, default)
    return default if value is None else value


def _tp_from_stage_configs(stage_configs: Sequence[Any]) -> Dict[int, int]:
    """Extract ``{stage_id: tensor_parallel_size}`` from the runtime's configs.

    Reads ``omni.stage_configs`` post-boot — the runtime's own merged
    (OmegaConf) per-stage configs, so this is the authoritative parse rather
    than a re-read of the YAML asset. LLM stages store
    ``engine_args.tensor_parallel_size``; diffusion stages store
    ``engine_args.parallel_config.tensor_parallel_size``. Falls back to 1
    when neither key is present.
    """
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
    """Spell the boot intent into ``Omni`` ctor kwargs.

    ``intent["omni_kwargs"]`` arrives pre-layered by ``config.server_intent``
    (timeouts < adapter ``mode`` < the ``omni_extra`` escape hatch). The
    dedicated keys go ON TOP — the escape hatch must not override ports or
    the sleep gate:

    - ``enable_sleep_mode=True`` (only when the intent asks): vllm-omni's
      ``AsyncOmniEngine._resolve_stage_configs`` injects it into every stage
      whose YAML doesn't define it (none of ours do), gating the
      ``CuMemAllocator`` pool ``worker.sleep()`` needs. The runtime first
      logs a spurious "top-level engine args are ignored: enable_sleep_mode"
      warning (the strip filter sees a vllm ``EngineArgs`` field) — the
      dedicated injection block applies it regardless; verified at the
      v0.20.0 pin and upstream main.
    - ``master_port`` (only when ports were reserved): merged into every
      stage's ``engine_args`` via the loader's ``base_engine_args`` channel;
      each stage settles its own port from this base (see
      :class:`VLLMOmniPorts`).
    """
    omni_kwargs = dict(intent.get("omni_kwargs") or {})
    if intent.get("enable_sleep_mode"):
        omni_kwargs["enable_sleep_mode"] = True
    ports = intent.get("ports")
    if ports is not None:
        omni_kwargs["master_port"] = int(ports.master_port)
    return omni_kwargs


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

    # ------------------------------------------------------------------ #
    # Boot — the only place the runtime import / spawn / env override live
    # ------------------------------------------------------------------ #

    @classmethod
    def boot(cls, intent: Dict[str, Any]) -> "VLLMOmniBackend":
        """Spell the intent into ``Omni`` ctor kwargs and spawn.

        ``intent`` is the dict from ``config.server_intent`` (adapter boot
        extras + the reserved port base already overlaid). The stage YAML is
        passed PRISTINE — ``enable_sleep_mode`` / ``master_port`` ride the
        runtime's own ctor-kwarg override channel (see
        :func:`_assemble_omni_kwargs`), so there is no YAML rewrite and no
        temp file. See the module docstring for the load-bearing boot order.
        """
        # 1. Patches first: install() wraps mp.Process so spawn children
        #    re-run the hijack at startup — the primary mechanism for
        #    propagating patches across the spawn boundary.
        from unirl.rollout.engine.vllm_omni.patches import install as install_patches

        install_patches()

        # 2. Spawn start method before any Omni mp object exists.
        import multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        rt = _import_omni_runtime()

        # 3. Scoped-env last resort: HI3 multi-GPU stages need to see ALL
        #    physical GPUs so vllm-omni can pin each stage to its yaml
        #    ``runtime.devices``. Permanent pop, matching v1 (a restore-after-
        #    boot is a post-parity follow-up; see _HI3 colocate landmine note
        #    in the adapter).
        if intent.get("clear_cuda_visible"):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # 4. Spawn Omni off the pristine YAML asset + the assembled kwargs.
        #
        # Node-local boot serialization: colocated replicas (8 per node) each
        # spawn worker subprocesses that hold ~20 GiB anon RSS while
        # materializing weights (safetensors -> dtype-cast staging). Eight
        # simultaneous boots burst past the pod's k8s memcg limit and the
        # kernel OOM-kills raylet/python (LIN-382 qwen probe, 2026-06-07:
        # "Memory cgroup out of memory: Killed process ... anon-rss:
        # 20216496kB", raylet SIGKILL -> ActorUnavailableError). An exclusive
        # flock makes the heavy-load window single-file per node — boots take
        # N * t_load instead of dying; it also narrows the master-port settle
        # TOCTOU window as a side effect. Disable via
        # DIFFRL_OMNI_BOOT_SERIALIZE=0 (e.g. single-replica smokes).
        import fcntl

        # Return the host process's reserved-but-unallocated CUDA pool to the
        # driver before spawning the engine. boot() runs inside the trainer's
        # ray actor: the colocate flow full-loads the model per rank before
        # FSDP shards it, leaving ~35-40 GiB reserved in THIS process's torch
        # caching allocator — memory the engine SUBPROCESS cannot see or use
        # (LIN-382 qwen probe-c: engine model 53.7 GiB + dummy run hit "116
        # MiB free" on a 95 GiB GPU with the trainer's pool holding the rest).
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - belt and braces; never block a boot
            pass

        yaml_path = _resolve_stage_yaml(str(intent["stage_yaml"]), str(intent.get("stage_yaml_source", "local")))
        serialize = os.environ.get("DIFFRL_OMNI_BOOT_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_omni_boot.lock", "a+") if serialize else None
        try:
            if lock_file is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            omni = rt["Omni"](
                model=str(intent["model_path"]),
                stage_configs_path=yaml_path,
                **_assemble_omni_kwargs(intent),
            )
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

        # Driver-side tokenizer for AR prompt-token construction (workers
        # reload their own from the model path). Pure-DiT modalities skip it.
        tokenizer = None
        if intent.get("needs_driver_tokenizer"):
            tokenizer = rt["AutoTokenizer"].from_pretrained(str(intent["model_path"]), trust_remote_code=True)

        return cls(
            omni,
            rt,
            tokenizer=tokenizer,
            # The runtime's own merged per-stage configs — authoritative,
            # no YAML re-read.
            tp_per_stage=_tp_from_stage_configs(omni.stage_configs),
        )

    def _require_omni(self) -> Any:
        if self._omni is None:
            raise RuntimeError("VLLMOmniBackend: engine not initialized (shut down?)")
        return self._omni

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        calls: Sequence[GenerateCall],
        *,
        attach_lora: bool = False,
        ar_lora_passthrough: bool = False,
    ) -> List[List[OmniRawResult]]:
        """Run each call through ``Omni.generate`` and group per request.

        ``attach_lora`` activates the pre-loaded adapter: an ``OmniLoRARequest``
        is patched onto every diffusion-stage params object (the DiT worker
        resets the active adapter to ``None`` per forward without it), and —
        when ``ar_lora_passthrough`` (HI3 AR-prelude modalities; requires the
        ``lora_request`` passthrough patch) — also passed as a top-level
        ``Omni.generate`` kwarg for the AR stage's input processor.
        """
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
                # Single-prompt call: its flat list IS the per-request group
                # (the v1 dit_recaption per-prompt path, byte-for-byte).
                groups.append(flat)
        return groups

    def _build_sampling_params(self, sampling: StageSampling, *, attach_lora: bool) -> Any:
        if sampling.kind == STAGE_KIND_AR:
            # ``logprobs=1`` rides in the kwargs (the adapter sets it) so vLLM
            # emits per-token logp on the sampled token.
            return self._rt["VLLMSamplingParams"](**sampling.kwargs)
        sp = self._rt["OmniDiffusionSamplingParams"](**sampling.kwargs)
        if attach_lora:
            # Without the attach, vllm-omni's DiT worker resets the active
            # adapter to None on every forward and the loaded adapter would
            # silently never run on the rollout pass.
            sp.lora_request = self._lora_request()
            # ``lora_scale`` is read alongside lora_request in
            # ``diffusion_worker.execute_model``; 1.0 = apply as-loaded.
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
        """HI3 prompt tokens via vllm-omni's ``build_prompt_tokens``.

        Tokenizes segment-by-segment to match HF ``apply_chat_template``
        byte-for-byte; needs the driver-side tokenizer the boot intent
        requested (``needs_driver_tokenizer``).
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "VLLMOmniBackend.tokenize_prompt: no driver tokenizer loaded "
                "(boot intent did not set needs_driver_tokenizer)."
            )
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
            build_prompt_tokens,
        )

        return build_prompt_tokens(text, self._tokenizer, task=task, sys_type=sys_type)

    # ------------------------------------------------------------------ #
    # Stage topology
    # ------------------------------------------------------------------ #

    def num_stages(self) -> int:
        return int(self._require_omni().engine.num_stages)

    def tp_per_stage(self) -> Dict[int, int]:
        return dict(self._tp_per_stage)

    def _stage_ids(self) -> List[int]:
        return list(range(self.num_stages()))

    # ------------------------------------------------------------------ #
    # Memory / lifecycle / health
    # ------------------------------------------------------------------ #

    def sleep_task(self) -> None:
        """Fan ``handle_sleep_task`` to every stage's workers (level 2)."""
        import uuid

        omni = self._require_omni()
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="handle_sleep_task",
                args=(self._rt["OmniSleepTask"](level=1, task_id=str(uuid.uuid4())),),
                stage_ids=[int(sid)],
            )

    def wake_task(self) -> None:
        """Fan ``handle_wake_task`` to every stage's workers + sync CUDA."""
        import uuid

        import torch

        omni = self._require_omni()
        for sid in self._stage_ids():
            omni.engine.collective_rpc(
                method="handle_wake_task",
                args=(self._rt["OmniWakeTask"](tags=None, task_id=str(uuid.uuid4())),),
                stage_ids=[int(sid)],
            )
        if torch.cuda.is_available():
            # Mirrors AsyncOmni.wake_up's synchronize(); ensures pool
            # restoration is visible before the next generate.
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

    # ------------------------------------------------------------------ #
    # Weight-sync verbs — per-stage collective_rpc fan-out lives here
    # ------------------------------------------------------------------ #

    def update_from_ipc(
        self,
        *,
        peft_config: Optional[dict],
        base_sync_done: bool,
        use_shm: bool,
        replica_rank: Optional[int],
    ) -> None:
        """Fan a bucketed CUDA-IPC state-dict update out to per-stage workers.

        ``replica_rank`` (optional) overrides the worker-side
        ``replica_rank_from_env()`` so colocated engines on one node use
        distinct ZMQ socket paths; ``None`` preserves the env-based behavior.
        """
        omni = self._require_omni()
        kwargs = {
            "peft_config": peft_config,
            "base_sync_done": base_sync_done,
            "use_shm": use_shm,
            "replica_rank": replica_rank,
        }
        for sid in self._stage_ids():
            # stage_id rides the kwargs so the worker extension's zmq-handle
            # computation sees it.
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
        """Fan a SGLang-shape tensor payload to per-stage workers.

        Pure dispatcher: each worker receives the full list and picks
        ``[self.local_rank]`` in its receive-side handler.
        """
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

    # ------------------------------------------------------------------ #
    # LoRA tensor bag — two genuinely different transports
    # ------------------------------------------------------------------ #

    def set_lora_handle(
        self,
        *,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        peft_config: Optional[dict],
    ) -> None:
        """Zero-copy LoRA push via ``MultiprocessingSerializer`` shm handles.

        Per-stage re-serialisation *with cloned tensors*: under the
        ``file_system`` sharing strategy each tensor's storage gets a named
        ``/dev/shm`` file that's unlinked once the consuming workers refcount
        it to zero — re-serialising the same storage hands the next stage a
        stale handle. Single-consumer-per-stage only (the ``file_descriptor``
        one-shot fd pops after the FIRST consumer; TP>1 stages need
        :meth:`set_lora_copy`).
        """
        import torch

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )

        omni = self._require_omni()
        lora_tensors = self._wrap_peft_envelope(lora_tensors)
        self._remove_existing_lora(int(DIFFRL_LORA_INT_ID))

        # Pass primitive fields, not an ``OmniTensorLoRARequest``: vllm's
        # msgspec wire encoder doesn't recognise our Struct subclass and
        # decodes it positionally as a list. The inner tensors can't survive
        # the msgpack wire either — encode via the vendored serializer; the
        # worker mixin deserialises and rebuilds the struct locally.
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
        """Byte-copy LoRA push (``torch.save`` + base64) — TP>1-broadcast-safe.

        A single ``collective_rpc`` broadcasts the same blob to every TP worker
        of a stage; ``torch.save`` bytes have no shared resource, so each
        worker ``torch.load``\\ s its own copy and the fan-out is unbounded
        (unlike the one-shot fd handle in :meth:`set_lora_handle`). LoRA is
        tiny, so copying per rank is free.
        """
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
        """Wrap canonical wire keys in the PEFT envelope vllm-omni expects.

        Senders ship ``<pipeline_prefix><module>.lora_A.weight``; vllm-omni's
        ``PEFTHelper`` expects ``base_model.model.<...>``. Idempotent check on
        the first key.
        """
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        first_key = next(iter(lora_tensors), "")
        if lora_tensors and not first_key.startswith("base_model.model."):
            return adapt_lora_for_vllm(lora_tensors)
        return lora_tensors

    def _remove_existing_lora(self, adapter_id: int) -> None:
        """Drop the existing adapter on every stage before re-adding.

        Matches the receive-side ``_diffrl_load_bucket`` ordering.
        ``remove_lora`` raises when the id isn't loaded; fine on the first
        call — subsequent failures still flow up via the add below.
        """
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

    # ------------------------------------------------------------------ #
    # Post-load value-correctness read-back
    # ------------------------------------------------------------------ #

    def param_checksums(self, *, names: List[str]) -> dict:
        """Fan ``_diffrl_loaded_param_checksums`` across stages and ranks.

        Returns ``{stage_id: [rank0_dict, rank1_dict, ...]}``.
        """
        omni = self._require_omni()
        out: dict = {}
        for sid in self._stage_ids():
            results = omni.engine.collective_rpc(
                method="_diffrl_loaded_param_checksums",
                args=(list(names),),
                stage_ids=[int(sid)],
            )
            # collective_rpc returns ``[stage_results]`` where stage_results
            # is ``[rank0, rank1, ...]`` — strip the outer list.
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
    """Group ``Omni.generate``'s flat output list into per-request lists.

    ``Omni._run_generation`` builds ``request_ids = [f"{i}_{uuid4()}" for i in
    range(B)]`` (one per prompt); each request contributes one output per
    final stage (2 for t2i/it2i after the Stage-0 ``final_output`` flip, 1
    otherwise). The mapping back to request index is the ``i_`` prefix; if the
    orchestrator's ordering invariant changes upstream, the per-group counts
    won't match downstream expectations and the adapter raises — better than
    silently misaligning.
    """
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
