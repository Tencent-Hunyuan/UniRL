"""Top-level vLLM-Omni rollout engine for HunyuanImage-3.

Single class, ``VLLMOmniRolloutEngine``, with one public method
``generate(req: RolloutReq) → RolloutResp``. The engine commits to a
modality at construction time (``cfg.modality``) and stands up one
``Omni`` instance with the matching stage config:

- ``"t2i"``  → ``stage_configs/hunyuan_image3_t2i_rl.yaml``  (AR + DiT, our pipeline subclass on Stage 1)
- ``"it2i"`` → ``stage_configs/hunyuan_image3_it2i_rl.yaml`` (AR + DiT, Stage 0 final_output flipped)
- ``"i2t"``  → upstream ``hunyuan_image3_i2t.yaml``           (AR only)
- ``"t2t"``  → upstream ``hunyuan_image3_t2t.yaml``           (AR only)

HI3's ~150GB of weights mean only one modality per process, so eager-
single is both simpler and the only realistic option.

Lifecycle: one-shot ``__init__``. The actor passes ``device``,
``strategy``, ``rank``, ``model_config`` as kwargs at construction time
and the engine completes everything (Omni boot, tokenizer load, spawn
start method). There is no separate ``initialize(device)`` step.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import tempfile
from typing import Any, Optional

import torch
import yaml

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig
from unirl.rollout.engine.vllm_omni.request import _to_omni_per_stage
from unirl.rollout.engine.vllm_omni.response import (
    _to_rollout_resp,
    group_by_request,
)
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

# Local YAMLs ship in stage_configs/ alongside this module. AR-only
# modalities use upstream YAMLs which Omni resolves by name from its
# own stage_configs directory.
_LOCAL_YAML = {
    "t2i": "hunyuan_image3_t2i_rl.yaml",
    "it2i": "hunyuan_image3_it2i_rl.yaml",
    "sd35_t2i": "sd35_t2i_rl.yaml",
    # Single-stage pure-DiT, HunyuanVideo-1.5 (text → video). Analogous to
    # sd35_t2i: no AR prelude, no driver-side tokenizer.
    "t2v": "hunyuan_video15_t2v_rl.yaml",
    # Two-engine v2 trainer (trainer/unified_model.py): AR-only think_recaption engine +
    # standalone DiT engine that eats an externally-injected recaption.
    "ar_recaption": "hunyuan_image3_ar_recaption_rl.yaml",
    "dit_recaption": "hunyuan_image3_dit_recaption_rl.yaml",
}
_UPSTREAM_YAML = {
    "i2t": "hunyuan_image3_i2t.yaml",
    "t2t": "hunyuan_image3_t2t.yaml",
}

logger = logging.getLogger(__name__)

# Modalities that require ``patches/vllm_omni/0001-omni-py-lora_request-
# passthrough.patch`` on the rollout pod. The patch adds the
# ``lora_request`` kwarg to ``Omni.generate()``'s signature so we can
# forward it into the AR-prelude stage's input processor (PR #105).
#
# Empirically every current AR-prelude modality is a HunyuanImage-3
# variant — they share both the patch dependency AND the
# ``hunyuan_image3_*.yaml`` stage config family. Named after the
# model family (not the underlying ``stage_type != "diffusion"``
# mechanism) so future additions force an explicit decision: a new
# non-HI3 model that happens to have an AR prelude must be added here
# deliberately, and adding it implies "this model also requires the
# pod patch" — not a silent extension.
#
# Pure-DiT modalities (currently only ``sd35_t2i``; future SD3/Qwen-
# Image/WAN/etc.) are NOT in this set: their stage-0
# ``stage_type: diffusion`` causes vllm-omni's
# ``_build_add_request_message`` to discard the ``lora_request`` kwarg
# unconditionally, so passing it is dead weight that would only force
# the pod-patch requirement on top of stock upstream vllm-omni for
# zero functional gain.
#
# DELETE-WHEN: vllm-omni upstreams the ``lora_request`` kwarg to
#   ``Omni.generate()`` (i.e. ``patches/vllm_omni/0001-omni-py-lora_
#   request-passthrough.patch`` is no longer needed). At that point:
#     1. drop this ``_HI3_MODALITIES`` constant + its docstring
#     2. drop the ``if ... in _HI3_MODALITIES`` gate below and pass
#        ``lora_request`` unconditionally inside the existing
#        ``if _lora_req_for_generate is not None:`` block
#     3. drop ``patches/vllm_omni/0001-*`` and the apply step in
#        ``patches/README.md``
#   Alternative trigger: HI3-family modalities retire AND no other
#   model adopts the AR-prelude LoRA path — same 3 deletions apply.
# ``ar_recaption`` is the AR-only think_recaption stage (it HAS an AR prelude
# stage, so it needs the lora_request passthrough patch). ``dit_recaption`` is
# pure-DiT (single diffusion stage) like sd35_t2i and is intentionally NOT here.
_HI3_MODALITIES = frozenset({"t2i", "it2i", "i2t", "t2t", "ar_recaption"})

# Modalities whose request carries diffusion params and therefore needs a σ
# schedule pinned (``ensure_req_sigmas``). AR-only modalities (``i2t`` / ``t2t``
# / the two-engine ``ar_recaption``) carry ``ARSamplingParams`` with NO
# diffusion sub-block, so ``ensure_req_sigmas`` would raise on them — gate it.
_DIT_BEARING_MODALITIES = frozenset({"t2i", "it2i", "sd35_t2i", "dit_recaption", "t2v"})

# HI3 modalities whose stage config requests tensor-parallel across multiple
# physical GPUs (TP4 on the AR and/or DiT stage). Ray restricts each DevicePool
# worker's ``CUDA_VISIBLE_DEVICES`` to its single reserved GPU, so vLLM-Omni's
# ``set_stage_devices`` sees only 1 device and the TP4 stage cannot start
# ("requested logical devices ['0','1','2','3'], but only 1 device available").
# For these modalities we clear ``CUDA_VISIBLE_DEVICES`` before constructing
# ``Omni`` so the engine sees all physical GPUs and vLLM-Omni pins each stage to
# its yaml ``runtime.devices`` (AR→0-3, DiT→4-7). NOT applied to ``sd35_t2i``
# (TP1, correctly pinned to its single reserved GPU — clearing would break the
# SD3 colocate data-parallel path). The engine MUST therefore be wired as a
# single multi-GPU actor (one worker), not replicated per device.
#
# ⚠️ COLOCATE LANDMINE: this anchor-on-one-worker + clear-CUDA_VISIBLE pattern is
# safe ONLY when no training side shares the GPUs (e.g. the rollout-only boot
# smoke). Under colocate training the engine anchored at worker-0 actually drives
# physical GPUs 0-3, while DevicePool's workers 1/2/3 nominally own cards 1/2/3 —
# so FSDP ranks 1/2/3 land on the SAME physical cards as the AR engine's TP
# children → guaranteed OOM (the 42+51>95GB pitfall). For colocate the rollout
# engines need a REAL GPU partition: reserve their cards out of the training pool
# (e.g. a dedicated num_gpus=4 actor), NOT anchor+clear. Do not copy this into
# the trainer.
_HI3_MULTI_GPU_MODALITIES = frozenset({"t2i", "it2i", "i2t", "t2t", "ar_recaption", "dit_recaption"})

# Per-rank port base for deterministic master_port assignment.
# Stride of 200 gives headroom above vllm-omni's settle_port retry loop
# (which scans up to 37 ports from master_port) and the random(0,100) offset.
_VLLM_OMNI_PORT_BASE = 30200
_VLLM_OMNI_PORT_STRIDE = 200


def seed_from_sample_id(sample_id: str) -> int:
    """Deterministic 31-bit diffusion seed for one image, keyed by sample_id.

    The M images of a recaption MUST draw distinct noise (else the diffusion
    GRPO advantage is identically 0 — the whole group collapses to the same
    reward). We cannot vary the seed through ``OmniDiffusionSamplingParams``
    because vllm-omni's ``resolve_sampling_params_list`` requires exactly one
    sampling-params object PER STAGE (not per prompt) and the inline diffusion
    client shares that single object across every prompt of a ``generate()``
    call — ``OmniDiffusionRequest.__post_init__`` then assigns a random seed
    only on the FIRST request (when ``seed`` is None) and the mutated object
    poisons all the rest with that same seed. So ``generate()`` issues one call
    per prompt with its own seed set HERE, derived from the unique sample_id
    (e.g. ``p0/a0/i3``) so it is globally distinct AND reproducible (a fixed
    sample_id always maps to the same noise — useful for debugging; GRPO replay
    itself reuses the stored ``trajectory_latents`` rather than re-sampling).
    ``< 2**31`` matches the range vllm-omni's own random-seed fallback uses.
    """
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _resolve_yaml_path(modality: str) -> str:
    """Return absolute path to the stage-config YAML for ``modality``.

    Local YAMLs ship in ``stage_configs/`` next to this module. Upstream
    YAMLs are looked up under
    ``<vllm_omni_project>/vllm_omni/model_executor/stage_configs/``.
    The upstream loader at ``vllm_omni/entrypoints/utils.py:585`` requires
    an absolute path and raises ``FileNotFoundError`` on bare names.
    """
    if modality in _LOCAL_YAML:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "stage_configs", _LOCAL_YAML[modality])
    if modality in _UPSTREAM_YAML:
        import vllm_omni  # local import: only needed at engine-construction time

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(vllm_omni.__file__)))
        candidate = os.path.join(
            project_root,
            "vllm_omni",
            "model_executor",
            "stage_configs",
            _UPSTREAM_YAML[modality],
        )
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"_resolve_yaml_path: upstream YAML {_UPSTREAM_YAML[modality]!r} "
                f"not found at {candidate}. vllm-omni may have moved the file."
            )
        return candidate
    raise ValueError(
        f"_resolve_yaml_path: unknown modality {modality!r}. Choose from {list(_LOCAL_YAML) + list(_UPSTREAM_YAML)}."
    )


def _inject_master_port(yaml_path: str, master_port: int) -> str:
    """Write a copy of ``yaml_path`` with ``master_port`` set on every stage's
    ``engine_args`` and return the temp path.

    vllm-omni's ``StageRuntimeData.__post_init__`` uses
    ``(self.master_port or 30005) + random.randint(0, 100)`` when
    ``master_port`` is None. Multiple concurrent actors draw from the same
    small range and collide. Injecting a deterministic per-rank value
    eliminates the race. Caller owns cleanup of the returned file.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for idx, entry in enumerate(data.get("stage_args", [])):
        ea = entry.setdefault("engine_args", {})
        ea["master_port"] = master_port + idx * 50
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False)
        return tmp.name


def _inject_enable_sleep_mode(yaml_path: str) -> str:
    """Write a copy of ``yaml_path`` with ``enable_sleep_mode: True`` set on
    every stage's ``engine_args`` and return the temp path.

    vllm-omni's ``CuMemAllocator`` GPU memory pool is gated on this per-stage
    ``OmniEngineArgs`` flag at construction time; without it,
    ``worker.sleep()`` raises. Caller owns cleanup of the returned file.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for entry in data.get("stage_args", []):
        ea = entry.setdefault("engine_args", {})
        ea["enable_sleep_mode"] = True
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False)
        return tmp.name


def _parse_tp_per_stage(yaml_path: str) -> dict:
    """Extract ``{stage_id: tensor_parallel_size}`` from a stage-config YAML.

    LLM stages store ``engine_args.tensor_parallel_size``.
    Diffusion stages store ``engine_args.parallel_config.tensor_parallel_size``.
    Falls back to 1 when neither key is present.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    tp_map: dict = {}
    for entry in data.get("stage_args", []):
        sid = int(entry.get("stage_id", len(tp_map)))
        ea = entry.get("engine_args", {})
        tp = ea.get("tensor_parallel_size", None)
        if tp is None:
            pc = ea.get("parallel_config", {})
            tp = pc.get("tensor_parallel_size", None)
        tp_map[sid] = int(tp) if tp is not None else 1
    return tp_map


class VLLMOmniRolloutEngine(BaseRolloutEngine):
    """Single entrypoint for prototype-side rollout against vLLM-Omni."""

    _component_name = "vllm_omni"

    def __init__(
        self,
        config: VLLMOmniEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Any = None,
    ) -> None:
        # Install vllm / vllm-omni monkey-patches in the driver process before
        # any worker subprocess is spawned. ``hijack()`` first calls
        # ``wrap_mp_process_for_children()``, which hooks ``mp.Process.__init__``
        # so every spawn-spawned worker target also runs ``hijack()`` at startup
        # — the primary mechanism for propagating patches across the spawn
        # boundary. The same ``hijack()`` is invoked again from the worker
        # extension's ``BucketedIPCReceiveMixin.__new__`` as defensive backup;
        # both calls are idempotent. Mirrors the LIN-210 sglang patch entry
        # point in ``samplers/sglang/engine.py``.
        from unirl.rollout.engine.vllm_omni.vllm_patches import VLLMOmniHijack

        VLLMOmniHijack.hijack()

        # vllm-omni's MultiprocDiffusionExecutor calls
        # ``mp.set_start_method("spawn", force=True)`` after constructing
        # multiprocessing.Event/Pipe objects in the parent process. The
        # default start method is "fork" on Linux, so those objects'
        # SemLocks bind to a fork context — passing them into a
        # spawn-context worker raises "A SemLock created in a fork
        # context is being shared with a process in a spawn context."
        # Switching to spawn before instantiating Omni avoids the
        # mismatch. Keep this until upstream fixes the ordering;
        # validated experimentally per the plan's pre-build verification.
        import multiprocessing as mp

        from transformers import AutoTokenizer
        from vllm_omni.entrypoints.omni import Omni

        self.cfg = config
        # `device`, `strategy`, `rank`, `model_config` are accepted to match
        # the engine construction contract `Engine(config=..., device=...,
        # strategy=..., rank=..., model_config=...)`. The engine carries them as
        # attributes for subclass / extension use; the synchronous Omni
        # entrypoint does not consume them directly.
        self.device = device
        self.strategy = strategy
        self.rank = rank
        self.model_config = model_config

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Tokenizer is loaded once on the driver side for prompt-token
        # construction; it doesn't ride into the worker (workers reload
        # from the model path).
        # SD3.5 has no top-level tokenizer (only subfolder CLIP-L /
        # CLIP-G / T5 tokenizers under text_encoder_*) — and its single-
        # stage diffusion path doesn't call build_prompt_tokens anyway,
        # so skip the AutoTokenizer load entirely for that modality.
        # HunyuanVideo-1.5 (t2v) is the same shape: tokenizers live in
        # tokenizer/ + tokenizer_2/ subfolders, the worker loads them
        # internally, and the driver-side translator needs none.
        if self.cfg.modality in ("sd35_t2i", "t2v"):
            self._tokenizer = None
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path, trust_remote_code=True)

        yaml_path = _resolve_yaml_path(self.cfg.modality)
        # Cache TP-per-stage from the original YAML before any injection
        # rewrites. Used by ``tp_per_stage()`` so the IPC weight-sync handler
        # can skip orphan ranks that have no matching receiver.
        self._tp_per_stage: dict = _parse_tp_per_stage(yaml_path)
        # Programmatic injection so user-shipped YAMLs stay clean. Disabling
        # ``cfg.enable_sleep_mode`` falls back to the upstream YAML defaults
        # (no CuMemAllocator pool — sleep() would raise).
        self._sleep_yaml_tmp: Optional[str] = None
        if self.cfg.enable_sleep_mode:
            self._sleep_yaml_tmp = _inject_enable_sleep_mode(yaml_path)
            yaml_path = self._sleep_yaml_tmp
        # rank_info isn't available at __init__ (set in setup() later); fall
        # back to the DevicePool-provided ``RANK`` env (device id, unique per
        # node) to avoid vllm-omni's narrow random fallback colliding across
        # actors. Mirrors sglang/engine.py:117-128.
        port_rank = self.rank
        if port_rank is None:
            env_rank = os.environ.get("RANK")
            port_rank = int(env_rank) if env_rank is not None and env_rank.isdigit() else 0
        base_port = _VLLM_OMNI_PORT_BASE + int(port_rank) * _VLLM_OMNI_PORT_STRIDE
        new_tmp = _inject_master_port(yaml_path, base_port)
        if self._sleep_yaml_tmp:
            os.unlink(self._sleep_yaml_tmp)
        self._sleep_yaml_tmp = new_tmp
        yaml_path = new_tmp
        self._is_offloaded: bool = False

        # Multi-GPU HI3 stages need to see ALL physical GPUs so vLLM-Omni can pin
        # each stage to its yaml ``runtime.devices`` (AR→0-3, DiT→4-7). Ray pins
        # this worker's CUDA_VISIBLE_DEVICES to its single reserved GPU; clear it
        # so Omni's per-stage device assignment works. Safe only because this
        # engine is wired as a single multi-GPU actor (see _HI3_MULTI_GPU_MODALITIES).
        if self.cfg.modality in _HI3_MULTI_GPU_MODALITIES:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        omni_kwargs: dict = dict(
            model=self.cfg.model_path,
            stage_configs_path=yaml_path,
            # HI3 weights are ~150GB; loading from cephfs over the
            # network easily blows past the 300s default. Allow up to
            # 20 min per stage. Caller can override via cfg.omni_extra.
            stage_init_timeout=1200,
            # AsyncOmniEngine has its own orchestrator startup_timeout
            # (init_timeout in the kwarg) which defaults to 600s. With
            # cephfs load + 8-GPU spawn-mode init this is too tight for
            # HI3; bump to 30 min so the orchestrator has room to come up
            # before stage_init_timeout takes over per-stage.
            init_timeout=1800,
        )
        if self.cfg.modality in ("t2i", "it2i", "sd35_t2i", "dit_recaption"):
            omni_kwargs["mode"] = "text-to-image"
        omni_kwargs.update(self.cfg.omni_extra)
        self._omni: Optional[Any] = Omni(**omni_kwargs)

        # LoRA activation state. ``set_lora_from_tensors`` flips this to True
        # after the worker-side ``add_lora`` returns. ``generate`` then
        # attaches a per-request ``OmniLoRARequest`` to the DiT sampling
        # params — without that attachment vllm-omni's DiT worker resets
        # the active adapter to ``None`` on every forward
        # (``vllm-omni/.../diffusion/worker/diffusion_worker.py:256-262``
        # + ``.../diffusion/lora/manager.py:213-226``), so the loaded
        # adapter would silently never run on the rollout pass.
        self._lora_loaded: bool = False

        # σ schedule policy — loaded once from the pretrained ckpt dir's
        # JSONs (scheduler/transformer/vae configs). ``ensure_req_sigmas``
        # consumes it in ``generate`` to pin ``req.sigmas`` before the
        # request crosses the wire to the vllm-omni worker subprocess.
        # The worker's pipeline subclass forwards
        # ``OmniDiffusionSamplingParams.sigmas`` into
        # ``scheduler.set_timesteps(sigmas=...)`` and echoes back the
        # actual schedule used; the response handler asserts a match.
        # ``model_config.shift`` is the single SOT for the σ schedule.
        # Silent fallback to 3.0 would mis-shift Wan (5.0) / Flux (1.0) /
        # HunyuanVideo (1.0) etc., so fail fast if the model config doesn't
        # carry it.
        if not hasattr(model_config, "shift"):
            raise RuntimeError(
                f"VLLMOmniRolloutEngine requires model_config.shift; got "
                f"{type(model_config).__name__}. Use a registered model preset "
                f"(e.g. ``sd3``, ``wan21``, ``wan22``, ``hunyuan_image3``)."
            )
        # Mirror the trainside engine's "Pipeline owns its dynamic-shift
        # posture" hook here by checking the model config for an explicit
        # use_dynamic_shifting + overrides declaration. Generic — works for
        # any model config that declares it. Falls back to the original
        # path-only behavior (silent static for HF-repo-id paths) for model
        # configs that don't set it.
        #
        # TODO: when Qwen-Image / future dynamic-shift models gain
        # vllm_omni recipes, ensure the model config carries the explicit
        # dynamic_overrides dict.
        require_dynamic = bool(getattr(model_config, "use_dynamic_shifting", False))
        dynamic_overrides = getattr(model_config, "dynamic_shift_overrides", None)
        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            self.cfg.model_path,
            shift=float(model_config.shift),
            require_dynamic=require_dynamic,
            dynamic_overrides=dynamic_overrides,
        )

        # LoRA state storage for sleep/wake recovery. When sleep discards
        # adapters, we need to re-load them on wake. Store the adapter details
        # so we can call set_lora_from_tensors again after wake.
        self._last_lora_name: Optional[str] = None
        self._last_lora_tensors: Optional[dict] = None
        self._last_peft_config: Optional[dict] = None

    # ------------------------------------------------------------------
    # BaseRolloutEngine — generation
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        # SD3.5's single-stage path doesn't use a driver-side tokenizer;
        # t2v (HunyuanVideo-1.5) is the same. Other modalities require it
        # for build_prompt_tokens.
        if self.cfg.modality not in ("sd35_t2i", "t2v") and self._tokenizer is None:
            raise RuntimeError("VLLMOmniRolloutEngine: tokenizer not initialized")
        self._validate_request(req)
        # Main-repo SSOT for σ: pin once via the shared helper. Translator
        # forwards via ``OmniDiffusionSamplingParams.sigmas``; response
        # handler asserts the worker echoed back what we sent. AR-only
        # modalities have no diffusion params, so skip (ensure_req_sigmas
        # would raise on a missing diffusion sub-block).
        if self.cfg.modality in _DIT_BEARING_MODALITIES:
            ensure_req_sigmas(req, self.schedule_policy)
        prompts, sampling_params_list = _to_omni_per_stage(
            req,
            self.cfg,
            modality=self.cfg.modality,
            tokenizer=self._tokenizer,
        )
        # Activate the loaded LoRA adapter on rollout.
        #
        # DiT-stage activation (covers ALL modalities including pure-DiT
        # sd35_t2i): lora_request is patched onto each
        # OmniDiffusionSamplingParams via _attach_lora_request. The DiT
        # worker reads sp.lora_request inside execute_model and routes
        # to the LoRA manager (see vllm_omni/diffusion/worker/
        # diffusion_worker.py:258).
        #
        # AR-stage activation (HI3 family ONLY): lora_request is also
        # passed as a top-level kwarg to Omni.generate(), which
        # vllm-omni's _build_add_request_message forwards into
        # input_processor.process_inputs() for the non-diffusion stage 0
        # (the AR prelude). This branch is gated on ``self.cfg.modality
        # in _HI3_MODALITIES`` — see the constant's docstring for why
        # SD3 / pure-DiT modalities are intentionally NOT included.
        # Pod-local patch (patches/vllm_omni/0001-*) is REQUIRED only
        # for the HI3 family.
        _lora_req_for_generate = None
        if self._lora_loaded:
            self._attach_lora_request(sampling_params_list)  # DiT stage

            from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

            from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
                DIFFRL_LORA_INT_ID,
                DIFFRL_LORA_NAME,
                DIFFRL_LORA_PATH,
            )

            _lora_req_for_generate = OmniLoRARequest(
                lora_name=DIFFRL_LORA_NAME,
                lora_int_id=int(DIFFRL_LORA_INT_ID),
                lora_path=DIFFRL_LORA_PATH,
            )

        # ``Omni.generate`` returns a flat list across all final stages;
        # we group back to per-request before translating.
        generate_kwargs: dict = {"use_tqdm": False}
        if _lora_req_for_generate is not None and self.cfg.modality in _HI3_MODALITIES:
            generate_kwargs["lora_request"] = _lora_req_for_generate

        # dit_recaption: one generate() per prompt so each image gets its own
        # seed (see seed_from_sample_id — a single shared sampling-params object
        # would make every image draw identical noise). Each single-prompt call
        # yields exactly that request's final output(s), so its flat list IS the
        # per-request group; no group_by_request needed.
        if self.cfg.modality == "dit_recaption":
            per_request: list = []
            recipe_gids = list(req.init_noise_group_ids or [])
            for idx, (sample_id, prompt) in enumerate(zip(req.sample_ids, prompts)):
                sp_for_prompt = copy.deepcopy(sampling_params_list)
                seed = seed_from_sample_id(sample_id)
                gid = recipe_gids[idx] if idx < len(recipe_gids) else None
                for sp in sp_for_prompt:
                    if hasattr(sp, "seed"):
                        sp.seed = seed
                    # Each single-prompt generate runs with batch_size=1 in the
                    # worker, so ship ONLY this sample's x_T recipe gid. The deepcopy
                    # carries the whole batch's gids; left untouched, the worker's
                    # NoiseRecipe.for_batch(1) would slice gids[:1] and hand gids[0]
                    # to EVERY image — collapsing all per-rollout x_T to the first
                    # sample's noise (GRPO group-diversity loss). Mirrors the
                    # per-sample seed override above.
                    ea = getattr(sp, "extra_args", None)
                    if gid is not None and isinstance(ea, dict) and ea.get("init_noise_group_ids"):
                        ea["init_noise_group_ids"] = [str(gid)]
                per_request.append(list(self._omni.generate([prompt], sp_for_prompt, **generate_kwargs)))
            return _to_rollout_resp(req, per_request, modality=self.cfg.modality)

        flat_outputs = list(self._omni.generate(prompts, sampling_params_list, **generate_kwargs))
        per_request = group_by_request(flat_outputs, len(req.sample_ids))
        return _to_rollout_resp(req, per_request, modality=self.cfg.modality)

    def _attach_lora_request(self, sampling_params_list: list) -> None:
        """Patch ``OmniDiffusionSamplingParams.lora_request`` on every DiT-stage
        params object in the list so the worker activates our pre-loaded adapter.

        Scope: DiT-stage activation only (covers SD3 single-stage today).
        The HI3 AR-stage path uses ``vllm.SamplingParams`` whose stock
        upstream form does NOT carry a ``lora_request`` field; ``generate``
        rejects the HI3+LoRA combo up front (see the modality check there),
        so this helper never runs into the AR stage in practice. If HI3 AR
        LoRA gets wired in the future, it should NOT go through this
        helper — AR LoRA activation in vllm is per-prompt via
        ``LoRARequest`` attached at ``add_request`` time.

        Resolving ``OmniDiffusionSamplingParams`` / ``OmniLoRARequest`` is
        deferred to call time because their import tree requires ``vllm``
        (only available in the rollout worker process).
        """
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

        from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )

        lora_req = OmniLoRARequest(
            lora_name=DIFFRL_LORA_NAME,
            lora_int_id=int(DIFFRL_LORA_INT_ID),
            lora_path=DIFFRL_LORA_PATH,
        )
        for sp in sampling_params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.lora_request = lora_req
                # lora_scale field is read alongside lora_request in
                # ``diffusion_worker.execute_model``; default 1.0 means
                # "apply the adapter as-loaded" (no external rescale).
                sp.lora_scale = 1.0

    # ------------------------------------------------------------------
    # BaseRolloutEngine — lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._omni is not None:
            try:
                close = getattr(self._omni, "close", None)
                if callable(close):
                    close()
            finally:
                self._omni = None
        if self._sleep_yaml_tmp is not None:
            try:
                os.unlink(self._sleep_yaml_tmp)
            except FileNotFoundError:
                pass
            self._sleep_yaml_tmp = None

    def health_check(self) -> bool:
        if self._omni is None:
            return False
        if self.cfg.modality in ("sd35_t2i", "t2v"):
            return True
        return self._tokenizer is not None

    # ------------------------------------------------------------------
    # BaseRolloutEngine — runtime offload (vllm-omni level-2 sleep)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Fan ``handle_sleep_task`` to every stage's workers (level 2)."""
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        if self._is_offloaded:
            return
        import uuid

        from vllm_omni.diffusion.data import OmniSleepTask

        omni_stage_ids = list(range(int(self._omni.engine.num_stages)))
        for sid in omni_stage_ids:
            self._omni.engine.collective_rpc(
                method="handle_sleep_task",
                args=(OmniSleepTask(level=1, task_id=str(uuid.uuid4())),),
                stage_ids=[int(sid)],
            )
        self._is_offloaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Fan ``handle_wake_task`` to every stage's workers."""
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        if not self._is_offloaded:
            return
        import uuid

        from vllm_omni.diffusion.data import OmniWakeTask

        stage_ids_internal = list(range(int(self._omni.engine.num_stages)))
        for sid in stage_ids_internal:
            self._omni.engine.collective_rpc(
                method="handle_wake_task",
                args=(OmniWakeTask(tags=None, task_id=str(uuid.uuid4())),),
                stage_ids=[int(sid)],
            )
        if torch.cuda.is_available():
            # Mirrors AsyncOmni.wake_up's current_omni_platform.synchronize();
            # ensures pool restoration is visible before the next generate.
            torch.cuda.synchronize()

        # Re-load LoRA after wake if it was previously loaded.
        # sleep(level=1) preserves base weights but LoRA adapters may be
        # discarded. Re-sending them after wake ensures rollout uses the
        # correct adapted model.
        #
        # Fail-fast on reload failure: if ``_lora_loaded`` was true before
        # sleep, the rollout MUST have an active LoRA adapter after wake
        # — otherwise subsequent ``generate`` calls silently run the base
        # model, producing rollouts that disagree with the trainer's
        # adapter-modulated forward. The downstream effect is invisible
        # in metrics until the GRPO ratio drifts so far that PPO clip
        # fraction blows up. Better to surface the failure here and let
        # the training loop crash than ship silent base-model rollouts.
        if self._lora_loaded and self._last_lora_tensors is not None:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                "[LoRA-WAKE] Re-loading LoRA after sleep/wake. adapter_name=%s",
                self._last_lora_name,
            )
            try:
                # HI3 two-engine stages are TP>1, so re-load via the byte-copy
                # transport (a zero-copy handle crashes ranks 2..N — see
                # set_lora_from_tensors_copy). SD3 / single-GPU stays on handle.
                if self.cfg.modality in ("ar_recaption", "dit_recaption"):
                    self.set_lora_from_tensors_copy(
                        adapter_name=self._last_lora_name,
                        lora_tensors=self._last_lora_tensors,
                        peft_config=self._last_peft_config,
                    )
                else:
                    self.set_lora_from_tensors(
                        adapter_name=self._last_lora_name,
                        lora_tensors=self._last_lora_tensors,
                        peft_config=self._last_peft_config,
                    )
                logger.info("[LoRA-WAKE] LoRA re-loaded successfully.")
            except Exception as exc:
                # Mark LoRA as no-longer-active and KEEP the engine in
                # offloaded state so the next ``generate`` also raises
                # (defense-in-depth: if a caller swallows the wake_up
                # exception, the subsequent ``generate`` invariant
                # ``if self._is_offloaded: raise`` catches it).
                self._lora_loaded = False
                self._is_offloaded = True
                raise RuntimeError(
                    f"[LoRA-WAKE] Failed to re-load LoRA adapter "
                    f"{self._last_lora_name!r} after sleep/wake; refusing "
                    f"to continue serving because rollout would silently "
                    f"run the base model, drifting old/new log-probs and "
                    f"the GRPO ratio. Original error: {exc!r}"
                ) from exc

        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return bool(self._is_offloaded)

    # ------------------------------------------------------------------
    # Stage topology
    # ------------------------------------------------------------------

    def tp_per_stage(self) -> dict:
        """Return ``{stage_id: tensor_parallel_size}`` for each stage.

        Parsed once from the stage-config YAML at construction time. The
        IPC weight-sync handler needs this to skip orphan train ranks that
        exceed a given stage's TP size (e.g. FSDP DP=8 vs dual-stage TP=4+4).
        """
        return dict(self._tp_per_stage)

    # ------------------------------------------------------------------
    # BaseRolloutEngine — weight sync (bucketed CUDA-IPC)
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
        track_prefix: str = "",
    ) -> None:
        """Fan a state-dict update out to the per-stage worker subprocesses.

        ``replica_rank`` (optional) overrides the worker-side
        ``replica_rank_from_env()`` so colocated engines on one node use
        distinct ZMQ socket paths; ``None`` preserves the env-based v1 behavior.
        """
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")

        stage_ids = list(range(int(self._omni.engine.num_stages)))

        kwargs = {
            "peft_config": peft_config,
            "base_sync_done": base_sync_done,
            "use_shm": use_shm,
            "replica_rank": replica_rank,
        }
        for sid in stage_ids:
            # Pass the stage_id positionally so the worker extension's
            # ``update_weights_from_ipc`` receives it in its zmq-handle
            # computation.
            self._omni.engine.collective_rpc(
                method="update_weights_from_ipc",
                args=(),
                kwargs={**kwargs, "stage_id": int(sid)},
                stage_ids=[int(sid)],
            )
        # Phase-2 LoRA sync (peft_config + base_sync_done) has registered
        # the adapter on every worker. Mirror what ``set_lora_from_tensors``
        # does on the NCCL/Tensor fallback (line ~607): flip ``_lora_loaded``
        # so the next ``generate`` attaches a lora_request to the DiT
        # sampling params. Without this, the IPC path leaves ``_lora_loaded``
        # False, ``generate`` skips ``_attach_lora_request``, and the DiT
        # worker's per-request ``set_active_adapter(None)`` deactivates the
        # adapter we just synced — rollout silently runs base weights.
        if peft_config and base_sync_done:
            self._lora_loaded = True

    # ------------------------------------------------------------------
    # BaseRolloutEngine — weight sync (NCCL broadcast)
    # ------------------------------------------------------------------

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        """Bring up an NCCL group on the requested stages' workers."""
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        kwargs = {
            "master_address": str(master_address),
            "master_port": int(master_port),
            "rank_offset": int(rank_offset),
            "world_size": int(world_size),
            "group_name": str(group_name),
            "backend": str(backend),
        }
        for sid in stage_ids:
            self._omni.engine.collective_rpc(
                method="init_weights_update_group",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    def update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        target_modules: Optional[list[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive a bucket of named tensors via NCCL on the requested stages."""
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        kwargs = {
            "names": list(names),
            "dtypes": list(dtypes),
            "shapes": [list(s) for s in shapes],
            "group_name": str(group_name),
            "target_modules": list(target_modules) if target_modules else None,
            "flush_cache": bool(flush_cache),
        }
        for sid in stage_ids:
            self._omni.engine.collective_rpc(
                method="update_weights_from_distributed",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        if self._omni is None:
            return
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        for sid in stage_ids:
            self._omni.engine.collective_rpc(
                method="destroy_weights_update_group",
                args=(),
                kwargs={"group_name": str(group_name)},
                stage_ids=[int(sid)],
            )

    # ------------------------------------------------------------------
    # BaseRolloutEngine — weight sync (LoRA tensor bag)
    # ------------------------------------------------------------------

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict,
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap a LoRA adapter from in-memory tensors on each stage.

        Builds an ``OmniTensorLoRARequest`` (the hijack-aware request type)
        and dispatches ``add_lora`` to each stage's workers via
        ``collective_rpc``. Requires the per-stage worker-extension class
        installed in the stage YAML to have run ``VLLMOmniHijack.hijack()``
        in its ``__new__`` — the bucketed-IPC mixin does this.
        """
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")

        from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        # Senders (nccl.py, tensor.py) ship canonical-format keys:
        #   ``<pipeline_prefix><module>.lora_A.weight``
        # vllm-omni's PEFTHelper expects the PEFT envelope:
        #   ``base_model.model.<pipeline_prefix><module>.lora_A.weight``
        # Wrap if not already wrapped (idempotent check on first key).
        first_key = next(iter(lora_tensors), "")
        if lora_tensors and not first_key.startswith("base_model.model."):
            lora_tensors = adapt_lora_for_vllm(lora_tensors)

        stage_ids = list(range(int(self._omni.engine.num_stages)))

        # Store LoRA state for re-loading after sleep/wake cycles.
        self._last_lora_name = adapter_name
        if isinstance(lora_tensors, dict):
            self._last_lora_tensors = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
        else:
            self._last_lora_tensors = lora_tensors
        self._last_peft_config = dict(peft_config or {})

        # Drop existing adapter first if any — matches the receive-side
        # ``_diffrl_load_bucket`` ordering for IPC + LoRA mode.
        for sid in stage_ids:
            try:
                self._omni.engine.collective_rpc(
                    method="remove_lora",
                    args=(int(DIFFRL_LORA_INT_ID),),
                    stage_ids=[int(sid)],
                )
            except Exception:
                # ``remove_lora`` raises when the id isn't loaded; fine on
                # the first call. Subsequent failures still flow up via
                # the add_lora call below.
                logger.debug(
                    "Ignoring failure while removing existing LoRA adapter from stage %s before reload.",
                    sid,
                    exc_info=True,
                )

        # Pass primitive fields, not an ``OmniTensorLoRARequest``: vllm's
        # wire encoder (msgspec) doesn't recognise our Struct subclass and
        # decodes it positionally as a list, breaking ``.lora_int_id``
        # access on the worker. The inner tensors also can't survive the
        # msgpack wire — encode them via SGLang's
        # ``MultiprocessingSerializer`` (same as B.2 tensor-payload).
        # The mixin deserialises and rebuilds the struct locally.
        #
        # Re-serialise per stage *with cloned tensors*: under
        # ``file_system`` sharing strategy each tensor's storage gets a
        # named ``/dev/shm`` file that's unlinked once the consuming
        # workers refcount it down to zero. Re-serialising the same
        # underlying storage hands the next stage a stale handle pointing
        # at the already-unlinked file. Cloning into fresh storages
        # forces new shm files per stage.
        from unirl.rollout.engine.vllm_omni.weight_sync.sgl_compat import (
            MultiprocessingSerializer,
        )

        for sid in stage_ids:
            cloned = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
            lora_tensors_serialized = MultiprocessingSerializer.serialize(cloned, output_str=True)
            self._omni.engine.collective_rpc(
                method="set_lora_from_tensor_dict",
                args=(
                    str(adapter_name) or DIFFRL_LORA_NAME,
                    int(DIFFRL_LORA_INT_ID),
                    DIFFRL_LORA_PATH,
                    dict(peft_config or {}),
                    lora_tensors_serialized,
                ),
                stage_ids=[int(sid)],
            )
        # Adapter is now resident on every requested stage's lora_manager;
        # subsequent ``generate`` calls must attach an OmniLoRARequest to
        # the per-stage sampling params so the worker's
        # ``set_active_adapter`` finds it (otherwise vllm-omni defaults to
        # ``None`` → deactivate, and the rollout silently runs base model).
        self._lora_loaded = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: dict,
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Cross-process LoRA push for the HI3 two-engine trainer (byte copy).

        # DELETE-WHEN: the vLLM-Omni LoRA handle transport is TP>1-broadcast-safe
        #   (e.g. file_system sharing, or a per-rank re-register monkey-patch).
        #   Then the caller can drop ``copy=True`` and use
        #   :meth:`set_lora_from_tensors`, and this byte-copy fork (+ its worker
        #   mate ``set_lora_from_tensor_dict_copy``) is dead. Only caller:
        #   ``RemoteLoraWeightSync`` with ``copy=True`` (weight_sync/lora.py).

        Same effect as :meth:`set_lora_from_tensors`, but the transport is a
        *data copy* instead of a zero-copy shared handle. The cross-process sender
        (``RemoteLoraWeightSync`` with ``copy=True``) invokes this from train
        rank 0 via ``Worker.call``: HI3 anchors its AR / DiT engines on separate
        workers (disjoint GPU partition), so the LoRA sync can't reach them as
        same-worker siblings.

        Why a byte copy and not the SD3 ``MultiprocessingSerializer`` handle:
        that handle uses the ``file_descriptor`` strategy, whose one-shot fd
        ``resource_sharer`` pops after the FIRST consumer. A single
        ``collective_rpc`` broadcasts the same blob to every TP worker of a
        stage, so for the HI3 TP>1 stages ranks 2..N would raise
        ``KeyError`` / ``EOFError``. ``torch.save`` bytes (base64-wrapped for the
        msgpack wire) have no shared resource — each worker ``torch.load``s its
        own independent copy, so the fan-out is unbounded. LoRA is tiny (tens of
        MB), so copying per rank is free.
        """
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")

        from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
            DIFFRL_LORA_NAME,
            DIFFRL_LORA_PATH,
        )
        from unirl.utils.peft_merge import adapt_lora_for_vllm

        first_key = next(iter(lora_tensors), "")
        if lora_tensors and not first_key.startswith("base_model.model."):
            lora_tensors = adapt_lora_for_vllm(lora_tensors)

        stage_ids = list(range(int(self._omni.engine.num_stages)))

        # Store LoRA state for re-loading after sleep/wake cycles.
        self._last_lora_name = adapter_name
        if isinstance(lora_tensors, dict):
            self._last_lora_tensors = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
        else:
            self._last_lora_tensors = lora_tensors
        self._last_peft_config = dict(peft_config or {})

        # Drop the existing adapter first (matches the receive-side ordering).
        for sid in stage_ids:
            try:
                self._omni.engine.collective_rpc(
                    method="remove_lora",
                    args=(int(DIFFRL_LORA_INT_ID),),
                    stage_ids=[int(sid)],
                )
            except Exception:
                logger.debug(
                    "Ignoring failure while removing existing LoRA adapter from stage %s before reload.",
                    sid,
                    exc_info=True,
                )

        # Byte copy (NOT a zero-copy handle): serialise once, fan out unbounded.
        import base64 as _base64
        import io as _io

        _cpu_tensors = {
            name: t.detach().to("cpu") if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
        }
        _buf = _io.BytesIO()
        torch.save(_cpu_tensors, _buf)
        lora_tensors_serialized = _base64.b64encode(_buf.getvalue()).decode("ascii")

        for sid in stage_ids:
            self._omni.engine.collective_rpc(
                method="set_lora_from_tensor_dict_copy",
                args=(
                    str(adapter_name) or DIFFRL_LORA_NAME,
                    int(DIFFRL_LORA_INT_ID),
                    DIFFRL_LORA_PATH,
                    dict(peft_config or {}),
                    lora_tensors_serialized,
                ),
                stage_ids=[int(sid)],
            )
        self._lora_loaded = True

    # ------------------------------------------------------------------
    # BaseRolloutEngine — weight sync (SGLang-shape one-bag tensor payload)
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        target_modules: Optional[list[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Fan a SGLang-shape tensor-payload update out to per-stage workers.

        Pure dispatcher: each worker receives the full
        ``serialized_named_tensors`` list and picks ``[self.local_rank]``
        in its receive-side handler.
        """
        del track_prefix
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        kwargs = {
            "serialized_named_tensors": list(serialized_named_tensors),
            "target_modules": list(target_modules) if target_modules else None,
            "load_format": load_format,
            "flush_cache": bool(flush_cache),
        }
        for sid in stage_ids:
            self._omni.engine.collective_rpc(
                method="update_weights_from_tensor",
                args=(),
                kwargs=kwargs,
                stage_ids=[int(sid)],
            )

    # ------------------------------------------------------------------
    # Post-load value-correctness — read-back hashes for trainer-side
    # comparison. Worker-side counterparts are
    # ``BucketedIPCReceiveMixin._diffrl_loaded_param_checksums`` and
    # ``_diffrl_loaded_lora_checksums``; trainer-side helpers live in
    # ``weight_sync.checksum``.
    # ------------------------------------------------------------------

    def loaded_param_checksums(
        self,
        *,
        names: list[str],
    ) -> dict:
        """Fan ``_diffrl_loaded_param_checksums`` across stages and ranks.

        Returns ``{stage_id: [rank0_dict, rank1_dict, ...]}`` where each
        ``rankN_dict`` is ``{name: short_sha256_hex}`` for the worker's
        loaded parameters. Caller compares against trainer-side
        :func:`weight_sync.checksum.compute_param_checksums`.

        For TP-flat target names (e.g. layer norms, the smoke test's
        current surface) every rank's hash should equal the trainer's
        hash. TP-sharded names need an external all-gather — see the
        worker-side method's docstring.
        """
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        out: dict = {}
        for sid in stage_ids:
            results = self._omni.engine.collective_rpc(
                method="_diffrl_loaded_param_checksums",
                args=(list(names),),
                stage_ids=[int(sid)],
            )
            # ``collective_rpc`` returns ``[stage_results]`` where
            # ``stage_results`` is ``[rank0, rank1, ...]``. Strip the
            # outer list so ``out[sid]`` is the per-rank list directly.
            out[int(sid)] = results[0] if isinstance(results, list) and results else results
        return out

    def loaded_lora_checksums(
        self,
        *,
        adapter_id: int,
        names: Optional[list[str]] = None,
    ) -> dict:
        """Fan ``_diffrl_loaded_lora_checksums`` across stages and ranks.

        Returns ``{stage_id: [rank0_dict, rank1_dict, ...]}`` where each
        ``rankN_dict`` is ``{layer_name: {field: hex}}``.

        ``lora.optimize()`` has run on the worker side, so ``lora_b``
        hashes are post-scaling — the trainer must apply the same
        ``alpha / r`` scaling before hashing for equality.
        :func:`weight_sync.checksum.compute_lora_checksums_post_optimize`
        handles that.
        """
        if self._omni is None:
            raise RuntimeError("VLLMOmniRolloutEngine: engine not initialized")
        stage_ids = list(range(int(self._omni.engine.num_stages)))
        out: dict = {}
        for sid in stage_ids:
            results = self._omni.engine.collective_rpc(
                method="_diffrl_loaded_lora_checksums",
                args=(int(adapter_id), list(names) if names else None),
                stage_ids=[int(sid)],
            )
            out[int(sid)] = results[0] if isinstance(results, list) and results else results
        return out

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    def _validate_request(self, req: RolloutReq) -> None:
        has_image = req.primitives.get("image") is not None
        m = self.cfg.modality
        if m in ("t2i", "sd35_t2i", "t2v") and has_image:
            raise ValueError(
                f"VLLMOmniRolloutEngine: modality={m!r} rejects image-bearing "
                "requests; use an image-conditioned modality instead."
            )
        if m == "t2t" and has_image:
            raise ValueError(
                "VLLMOmniRolloutEngine: modality='t2t' rejects image-bearing requests; use modality='i2t' instead."
            )
        if m in ("it2i", "i2t") and not has_image:
            raise ValueError(f"VLLMOmniRolloutEngine: modality={m!r} requires req.primitives['image'].")


__all__ = ["VLLMOmniRolloutEngine"]
