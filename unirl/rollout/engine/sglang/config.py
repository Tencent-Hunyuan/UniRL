"""SGLang rollout-engine configuration (new ``BaseRolloutEngine`` protocol).

Consumed by ``SGLangRolloutEngine``; the rollout actor constructs the engine
with ``config=<this>`` + ``device`` / ``strategy`` / ``rank`` / ``model_config``
(as a composed-engine child, via :meth:`SGLangEngineConfig.make_engine`).

Differences vs the legacy config:

- ``model_family: Literal["sd3", "flux2_klein"]`` replaces the legacy
  substring-based ``_infer_model_type`` heuristic. Explicit field.
- ``populate_conditions: bool`` opts the response translator into packing
  ``RolloutResp.tracks["image"].conditions["text"]`` (+ ``"negative_text"`` when CFG is on)
  from SGLang's emitted prompt embeddings.
- ``init_same_noise: bool`` gates the engine-internal noise fallback that
  shares Gaussian noise across same-group samples when the caller did not
  pre-ship one via ``req.request_conditions["initial_latents"]``.
- The engine always best-effort emits SGLang's ``trajectory_log_probs`` onto
  ``LatentSegment.sde_logp`` (degrading to ``None`` when the build doesn't
  return them). Whether those native log-probs are *used* or recomputed is a
  training-layer decision (``algorithm.old_logp_source``), not an engine flag.
- The legacy ``verify_weight_checksum`` flag is gone. Checksums are now an
  on-demand query via ``SGLangRolloutEngine.loaded_param_checksums(names)``
  (vllm-omni-shape return).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Optional, Tuple

from omegaconf import SI

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig

# Per-rank SGLang port layout for co-located actors on a single node. Each
# rank reserves a stride-sized slice starting at ``base + rank * stride``:
# slot 0 is port, slot 11 is scheduler_port, slot 23 is master_port.
_SGLANG_PORT_BASE = 33000
_SGLANG_PORT_STRIDE = 100


# Only families with a verified end-to-end SGLang rollout path (LIN-365).
# "flux" (FLUX.1) / "mochi" were removed: never implemented anywhere in UniRL,
# and upstream sglang has no mochi pipeline at all. "hunyuan_video" is deferred;
# to commission it, inject ``LoRAPipeline`` into ``HunyuanVideoPipeline.__bases__``
# (the fork's entire hunyuan-lora feature was that one line — generalize
# ``_patches/patch_sd3_lora_pipeline``), add a Videos primitive to the response
# translator (video samples are currently dropped, see response.py TODO), then
# re-add the family string here.
_VALID_MODEL_FAMILIES = ("sd3", "flux2_klein")


@dataclass
class SGLangEngineConfig(BaseEngineConfig):
    """Configuration for the SGLang rollout-side inference engine."""

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

        return SGLangRolloutEngine(config=self, **deps)

    # --- Sampling (live interpolation back to top-level cfg.sampling) ---
    # Snapshot of the top-level ``cfg.sampling`` interpolation. Keep this as
    # ``Any`` because ``OmegaConf.resolve`` may write the resolved DictConfig
    # back into this field, and runtime sampling is carried by
    # ``RolloutReq.sampling_params`` instead.
    sampling: Any = dc_field(default_factory=lambda: SI("${sampling}"))

    # --- Required: model family for trainer-side typed-condition reconstruction ---
    model_family: str = "sd3"

    # --- Conditions packing ---
    populate_conditions: bool = True

    # --- Engine-internal noise fallback (only used when caller did not pre-ship
    # ``req.request_conditions["initial_latents"]``) ---
    init_same_noise: bool = False

    # --- Parallelism & GPU ---
    num_gpus: int = 1
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None

    # --- SGLang engine behaviour ---
    local_mode: bool = True
    disable_autocast: bool = False

    # --- Forward chunking ---
    # Intra-call chunk size for the DiffGenerator forward path. When set and a
    # request exceeds it, ``generate`` slices the request into ``forward_batch_size``
    # -sample sub-batches, runs one SGLang forward per chunk, and concatenates —
    # bounding DiT-forward / VAE-decode peak memory (SGLang's memory-saver pool
    # reabsorbs freed space, so the per-forward activation is otherwise bounded
    # only by the whole batch). ``None`` = whole batch in a single forward.
    forward_batch_size: Optional[int] = None

    # --- Weight sync ---
    target_modules: Optional[Tuple[str, ...]] = None

    # --- LoRA ---
    lora_merge_mode: Optional[str] = None

    # --- SGLang network ---
    host: Optional[str] = None
    port: Optional[int] = None
    scheduler_port: Optional[int] = None

    # --- Escape hatch for rare / advanced ServerArgs overrides ---
    engine_kwargs: Optional[Dict[str, Any]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.model_family = str(self.model_family or "").strip().lower()
        require(
            self.model_family in _VALID_MODEL_FAMILIES,
            f"SGLangEngineConfig.model_family must be one of {set(_VALID_MODEL_FAMILIES)}; got {self.model_family!r}",
        )
        require(
            self.num_gpus >= 1,
            f"SGLangEngineConfig.num_gpus must be >= 1; got {self.num_gpus!r}",
        )
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"SGLangEngineConfig.tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.sp_degree is None or self.sp_degree >= 1,
            f"SGLangEngineConfig.sp_degree must be >= 1 when set; got {self.sp_degree!r}",
        )
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"SGLangEngineConfig.forward_batch_size must be >= 1 when set; got {self.forward_batch_size!r}",
        )
        require(
            self.local_mode or (self.host is not None and self.scheduler_port is not None),
            f"SGLangEngineConfig: remote mode (local_mode=False) requires host and "
            f"scheduler_port; got host={self.host!r}, scheduler_port={self.scheduler_port!r}",
        )

    # ------------------------------------------------------------------
    # SGLang ServerArgs construction (ported verbatim from legacy)
    # ------------------------------------------------------------------

    def build_server_kwargs(
        self,
        server_args_cls: Any,
        *,
        model_config: Any,
    ) -> Dict[str, Any]:
        """Build a kwargs dict suitable for ``ServerArgs.from_kwargs()``.

        Priority (highest → lowest):
        1. Typed fields on this config (when set / non-None).
        2. ``engine_kwargs`` entries whose key is a valid ServerArgs field.

        ``model_config`` is duck-typed: any object exposing
        ``.pretrained_model_ckpt_path`` works. Registered PipelineConfigs
        (``SD3PipelineConfig``, ``WAN21PipelineConfig``, etc.) all carry
        it as a top-level field; they intentionally do NOT inherit from
        a common base because each pipeline owns its own config schema.
        """
        allowed_keys = {f.name for f in dataclasses.fields(server_args_cls)}
        result: Dict[str, Any] = {}

        for key, value in (self.engine_kwargs or {}).items():
            if key in allowed_keys:
                result[key] = value

        if model_config.pretrained_model_ckpt_path:
            result["model_path"] = model_config.pretrained_model_ckpt_path

        result["num_gpus"] = self.num_gpus

        if self.tp_size is not None:
            result["tp_size"] = int(self.tp_size)
        if self.sp_degree is not None and "sp_degree" in allowed_keys:
            result["sp_degree"] = int(self.sp_degree)

        if "disable_autocast" in allowed_keys:
            result["disable_autocast"] = bool(self.disable_autocast)

        if self.lora_merge_mode is not None and "lora_merge_mode" in allowed_keys:
            result["lora_merge_mode"] = self.lora_merge_mode
        elif model_config.use_lora and "lora_merge_mode" in allowed_keys:
            result.setdefault("lora_merge_mode", "online")

        if (
            model_config.use_lora
            and model_config.lora_target_modules is not None
            and "lora_target_modules" in allowed_keys
        ):
            result["lora_target_modules"] = list(model_config.lora_target_modules)

        if self.host is not None:
            result["host"] = str(self.host)
        if self.port is not None:
            result["port"] = int(self.port)
        if self.scheduler_port is not None:
            result["scheduler_port"] = int(self.scheduler_port)

        return result

    def with_sglang_ports(self, rank: int) -> "SGLangEngineConfig":
        """Return a new config with SGLang ports filled from *rank* (local mode only)."""
        if not self.local_mode:
            return self

        actor_base = _SGLANG_PORT_BASE + int(rank) * _SGLANG_PORT_STRIDE
        require(
            actor_base <= 65000,
            f"SGLang port range exceeded: base={_SGLANG_PORT_BASE}, stride={_SGLANG_PORT_STRIDE}, rank={rank}",
        )

        new_engine_kwargs = dict(self.engine_kwargs or {})
        new_engine_kwargs.setdefault("master_port", actor_base + 23)

        return dataclasses.replace(
            self,
            port=self.port if self.port is not None else actor_base,
            scheduler_port=(self.scheduler_port if self.scheduler_port is not None else actor_base + 11),
            engine_kwargs=new_engine_kwargs,
        )


__all__ = ["SGLangEngineConfig"]
