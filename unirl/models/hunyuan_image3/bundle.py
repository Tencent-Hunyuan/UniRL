"""HunyuanImage3Bundle — concrete weights+params holder for HunyuanImage 3.0.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules HunyuanImage3 ships with: 1× shared MoE transformer
(``HunyuanImage3ForCausalMM`` from the upstream ``hunyuan_image_3``
package), 1× SigLIP2 ViT vision tower, 1× 3D-VAE, 1× tokenizer,
1× scheduler. No LoRA injection, FSDP wrap, adapter switching, autocast
helpers, or weight-sync logic — those are lifecycle concerns owned
outside the bundle.

The shared backbone is a single ``nn.Module`` that operates in either
``mode="gen_text"`` (autoregressive) or ``mode="gen_image"`` (DiT
denoise) depending on the call site. The AR and Diffusion stages both
call into ``self.transformer`` directly with different ``mode=`` kwargs.

MoE expert parallelism is intentionally not exposed here — initial
integration runs at EP=1 (full backbone replicated). Add EP wiring when
the training stack grows EP support.

Use :meth:`HunyuanImage3Bundle.from_config` to load a small / single-
device HuggingFace checkpoint via ``trust_remote_code=True`` (the
upstream package ships the modeling code in the checkpoint repo). For
the real 80B ``tencent/HunyuanImage-3.0`` weights this path will OOM —
the training stack constructs the bundle manually with
``device_map="auto"``.

Chat-template input prep (tokenizer wrapper + rope helpers) lives on
``HunyuanImage3TextEmbedStage`` (``embed_for_ar`` / ``embed_for_gen_image``),
not here.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import HunyuanImage3PipelineConfig

logger = logging.getLogger(__name__)


class HunyuanImage3Bundle(Bundle):
    """HunyuanImage 3.0 bundle: shared MoE transformer + ViT + 3D-VAE + tokenizer + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        vit: Optional[nn.Module],
        tokenizer: Any,
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        mrope_section: Tuple[int, int, int] = (0, 32, 32),
        vae_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        # ``vae`` / ``vit`` are exposed as ``@property`` over ``self._vae`` /
        # ``self._vit`` so :meth:`materialize` can flip them after meta-init
        # construction (where they're initially ``None``). The backing
        # transformer always has ``transformer.vae`` / ``transformer.vision_model``
        # children — they may be on meta until materialized.
        self._vae = vae
        self._vit = vit
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dtype = dtype
        # Used by :meth:`materialize` so the loaded VAE lands at the same
        # dtype as ``from_config``'s eager path. Defaults to ``dtype`` when
        # ``vae_dtype`` isn't provided.
        self.vae_dtype = vae_dtype if vae_dtype is not None else dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.mrope_section = mrope_section

    @property
    def vae(self) -> Optional[nn.Module]:
        return self._vae

    @property
    def vit(self) -> Optional[nn.Module]:
        return self._vit

    @classmethod
    def from_config(cls, config: HunyuanImage3PipelineConfig) -> "HunyuanImage3Bundle":
        """Load all HunyuanImage3 components from a HuggingFace checkpoint.

        Loads ``HunyuanImage3ForCausalMM`` via
        ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
        — the ckpt's ``auto_map`` resolves the wrapper class that owns
        ``vae`` and ``vision_model``. (``AutoModel`` would land on the
        backbone-only ``HunyuanImage3Model``, which has neither.)

        ``.to(device)`` is unconditional here, which only works for small
        checkpoints. The 80B ``tencent/HunyuanImage-3.0`` weights need
        ``device_map="auto"`` — for that path, callers construct the
        bundle directly via ``__init__``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .compat import apply_hi3_transformers5_compat

        apply_hi3_transformers5_compat()
        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")

        # Shared MoE backbone — operates in both gen_text and gen_image modes.
        # Must be ``AutoModelForCausalLM`` so we get ``HunyuanImage3ForCausalMM``
        # (with .vae / .vision_model / ._tkwrapper); ``AutoModel`` returns
        # the inner ``HunyuanImage3Model`` (backbone only).
        transformer = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        # The 3D-VAE and SigLIP2 ViT are bundled inside the same checkpoint
        # repo as auxiliary modules — the upstream code attaches them on the
        # main model, so we expose direct references rather than reloading.
        vae = getattr(transformer, "vae", None) or getattr(transformer, "vae_model", None)
        if vae is None:
            raise RuntimeError(
                "HunyuanImage3Bundle.from_config: could not locate VAE on the "
                "loaded backbone. Expected attribute `vae` or `vae_model`. "
                "Verify the checkpoint at " + path + " is a HunyuanImage3 build."
            )
        vae = vae.to(device=device, dtype=vae_dtype).eval()
        vae.requires_grad_(False)

        # Real upstream attribute is ``vision_model`` (hunyuan.py:1729). The
        # other names are kept for forward-compat with older / forked weights.
        vit = (
            getattr(transformer, "vit", None)
            or getattr(transformer, "vision_tower", None)
            or getattr(transformer, "siglip", None)
            or getattr(transformer, "vision_model", None)
        )
        if vit is None:
            raise RuntimeError(
                "HunyuanImage3Bundle.from_config: could not locate ViT on the "
                "loaded backbone. Expected attribute `vit`, `vision_tower`, "
                "`siglip`, or `vision_model`. Verify the checkpoint at " + path + "."
            )
        vit = vit.to(device).eval()
        vit.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # Scheduler — upstream ships ``FlowMatchDiscreteScheduler`` inside
        # ``hunyuan_image_3.hunyuan_image_3_pipeline``. Try direct import; fall
        # back to a no-scheduler bundle (pipeline will use
        # ``sde.runtime.get_sigma_schedule`` directly).
        scheduler: Any = None
        try:
            from hunyuan_image_3.hunyuan_image_3_pipeline import (  # type: ignore[import-not-found]
                FlowMatchDiscreteScheduler,
            )

            scheduler = FlowMatchDiscreteScheduler.from_pretrained(path)
        except Exception:  # noqa: BLE001 — upstream may not be importable, fall back
            logger.debug(
                "Failed to load HunyuanImage3 scheduler from %s; falling back to None.",
                path,
                exc_info=True,
            )
            scheduler = None

        return cls(
            transformer=transformer,
            vae=vae,
            vit=vit,
            tokenizer=tokenizer,
            scheduler=scheduler,
            dtype=dtype,
            vae_dtype=vae_dtype,
            device=device,
            pretrained_path=path,
            mrope_section=tuple(config.mrope_section),
        )

    # ------------------------------------------------------------------
    # Meta-init constructor for large models (80B+).
    # ------------------------------------------------------------------

    @classmethod
    def from_meta_config(
        cls,
        config: HunyuanImage3PipelineConfig,
    ) -> "HunyuanImage3Bundle":
        """Build the bundle with the full ``HunyuanImage3ForCausalMM`` on
        meta-device. Returns immediately without any per-rank weight load.

        Used for the 80B path. Every parameter (decoder, lm_head, heads,
        vae, vit) stays on meta until :meth:`materialize` runs — that
        single call covers the FSDP-wrapped decoder via DCP plus all
        wrapper-level non-FSDP children, with vae / vit opt-in via
        ``with_aux``.
        """
        from accelerate import init_empty_weights
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GenerationConfig,
        )

        from .compat import apply_hi3_transformers5_compat

        apply_hi3_transformers5_compat()
        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")

        # 1. Cheap config load — just JSON.
        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        # 2. Build full model on meta. ``init_empty_weights`` puts every
        #    ``nn.Parameter`` allocation on the meta device, so no weight
        #    memory is allocated. Tokenizer wrapper still needs to be set up
        #    afterwards — that's metadata, not parameters.
        with init_empty_weights():
            transformer = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

        # ``from_config`` does NOT load ``generation_config.json`` (only
        # ``from_pretrained`` does). HI3 pipelines read custom fields like
        # ``sequence_template``, ``bot_task``, ``flow_shift`` off the
        # generation config to drive t2i / t2t / t2it routing — so we
        # explicitly load it here. Falls back silently if the file is
        # missing (the HF default GenerationConfig is fine for non-HI3
        # paths).
        try:
            transformer.generation_config = GenerationConfig.from_pretrained(path)
        except (OSError, ValueError):
            pass

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # Scheduler — same as ``from_config``; tiny, no meta concerns.
        scheduler: Any = None
        try:
            from hunyuan_image_3.hunyuan_image_3_pipeline import (  # type: ignore[import-not-found]
                FlowMatchDiscreteScheduler,
            )

            scheduler = FlowMatchDiscreteScheduler.from_pretrained(path)
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to load HunyuanImage3 scheduler from %s; falling back to None.",
                path,
                exc_info=True,
            )
            scheduler = None

        return cls(
            transformer=transformer,
            vae=None,
            vit=None,
            tokenizer=tokenizer,
            scheduler=scheduler,
            dtype=dtype,
            vae_dtype=vae_dtype,
            device=device,
            pretrained_path=path,
            mrope_section=tuple(config.mrope_section),
        )

    # ------------------------------------------------------------------
    # Materialization (single entry point, covers decoder + heads + opt-in aux)
    # ------------------------------------------------------------------

    # Wrapper-level diffusion-head modules: siblings of ``transformer.model``
    # (the FSDP-wrapped decoder) that are part of the diffusion forward path.
    # Always materialized — they're tiny vs. the decoder and required for
    # replay.
    _DECODER_HEAD_ATTRS = (
        "lm_head",
        "final_layer",
        "patch_embed",
        "time_embed",
        "time_embed_2",
        "timestep_emb",
        "vision_aligner",
    )

    def materialize(
        self,
        *,
        device: torch.device,
        with_aux: Sequence[str] = (),
    ) -> None:
        """Single-call materialization for the meta-init path.

        Allocates per-rank storage on ``device`` for everything in the
        materialization set + loads HF weights via DCP's
        ``set_model_state_dict`` — which transparently handles DTensor
        (the FSDP-wrapped decoder) and plain tensors (heads, opt-in
        vae/vit) in one collective. The materialization set is:

        - ``transformer.model`` (FSDP-wrapped decoder, always)
        - All wrapper-level diffusion heads listed in
          :attr:`_DECODER_HEAD_ATTRS` (always; tiny, required for replay)
        - ``transformer.vae`` if ``"vae" in with_aux``
        - ``transformer.vision_model`` if ``"vit" in with_aux``

        Idempotent: modules already materialized (full-load path or repeat
        call) skip the per-shard ``to_empty`` step.

        Pre-condition: phase 2 (FSDPPolicy construction → ``fully_shard``)
        has already run. Bundle's caller is responsible for the wrap order.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        aux_set = tuple(with_aux)
        for name in aux_set:
            if name not in {"vae", "vit"}:
                raise ValueError(
                    f"HunyuanImage3Bundle.materialize: unknown aux module {name!r}; expected 'vae' or 'vit'."
                )

        # Plan: (top-level attr name on transformer, module). Decoder is
        # always first. Heads + aux follow. Top-level attr names double as
        # state_dict prefixes (decoder -> "model.*"; lm_head -> "lm_head.*").
        plan: List[Tuple[str, nn.Module]] = []
        decoder = getattr(self.transformer, "model", None)
        if decoder is None or not isinstance(decoder, nn.Module):
            raise RuntimeError(
                "HunyuanImage3Bundle.materialize: transformer.model missing — "
                "checkpoint may not be a HunyuanImage3 build."
            )
        plan.append(("model", decoder))
        for attr in self._DECODER_HEAD_ATTRS:
            head = getattr(self.transformer, attr, None)
            if head is None or not isinstance(head, nn.Module):
                continue
            plan.append((attr, head))
        if "vae" in aux_set:
            vae = getattr(self.transformer, "vae", None)
            if vae is None or not isinstance(vae, nn.Module):
                raise RuntimeError(
                    "HunyuanImage3Bundle.materialize: with_aux='vae' but transformer.vae is missing on the wrapper."
                )
            plan.append(("vae", vae))
        if "vit" in aux_set:
            vit = getattr(self.transformer, "vision_model", None)
            if vit is None or not isinstance(vit, nn.Module):
                raise RuntimeError(
                    "HunyuanImage3Bundle.materialize: with_aux='vit' but "
                    "transformer.vision_model is missing on the wrapper."
                )
            plan.append(("vision_model", vit))

        # 1. Allocate per-rank storage for any module still on meta. The
        #    decoder's ``to_empty`` is FSDP-aware (per-shard cuda alloc);
        #    heads and aux are plain ``nn.Module.to_empty``. Already-
        #    materialized modules (full-load path or repeat call) are
        #    skipped — the meta check covers both DTensor shards and
        #    regular params.
        for _attr, module in plan:
            if _module_has_meta_param(module):
                module.to_empty(device=device)

        # 2. Build filtered state_dict on rank 0 only. Keys retain their
        #    wrapper-level namespace (e.g. ``model.layers.0.weight``,
        #    ``lm_head.weight``) — DCP matches them against
        #    ``self.transformer``'s parameters in a single pass.
        if _current_rank() == 0:
            prefixes = tuple(attr for attr, _ in plan)
            sd = _collect_filtered_state_dict(self.pretrained_path, prefixes=prefixes)

            # [Bug B fix] LoRA key rename: peft.inject_adapter_in_model wraps
            # q/k/v/o_proj.weight as q/k/v/o_proj.base_layer.weight. The ckpt
            # has original names (*.weight). With strict=False,
            # set_model_state_dict silently skips the mismatch → meta-allocated
            # NaN params stay. Fix: rename ckpt keys to LoRA-wrapped namespace.
            expected_names = {name for name, _ in self.transformer.named_parameters(remove_duplicate=False)}
            rename_map = {}
            for name in expected_names:
                if not name.endswith((".base_layer.weight", ".base_layer.bias")):
                    continue
                ck_key = name.replace(".base_layer.", ".")
                if ck_key in sd:
                    rename_map[ck_key] = name
            if rename_map:
                for old_k, new_k in rename_map.items():
                    sd[new_k] = sd.pop(old_k)
                print(
                    f"[Bug B fix] HunyuanImage3Bundle.materialize: "
                    f"renamed {len(rename_map)} ckpt keys to LoRA-wrapped "
                    f"base_layer namespace.",
                    flush=True,
                )
        else:
            sd = {}

        # 3. Single DCP load
        set_model_state_dict(
            self.transformer,
            sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )

        # [Bug B fix] Post-load validation: verify all LoRA base_layer
        # params are finite and not on meta.
        if _current_rank() == 0:
            _bl_checked, _bl_bad = 0, 0
            for name, p in self.transformer.named_parameters(remove_duplicate=False):
                if ".base_layer." not in name:
                    continue
                _bl_checked += 1
                if p.is_meta or not p.data.isfinite().all():
                    _bl_bad += 1
            if _bl_checked > 0:
                if _bl_bad > 0:
                    raise RuntimeError(
                        f"[Bug B fix] FATAL: {_bl_bad}/{_bl_checked} LoRA "
                        f"base_layer params are meta/non-finite after DCP load. "
                        f"LoRA key rename may have failed."
                    )
                print(
                    f"[Bug B fix] HunyuanImage3Bundle.materialize: "
                    f"verified {_bl_checked} LoRA base_layer params loaded "
                    f"finite ✓",
                    flush=True,
                )

        del sd

        # 4. Post-load casts + freeze + eval for aux modules. Heads stay
        #    in bundle.dtype (already correct from ``to_empty``); decoder
        #    stays in whatever FSDP set. Aux needs its target_dtype
        #    restored — ``to_empty`` preserves the meta tensor's dtype
        #    (which after the FSDP-policy pre-wrap homogenization is
        #    bundle.dtype = bf16 across the board, but vae must run in
        #    vae_dtype for numerical reasons).
        if "vae" in aux_set:
            vae_module = self.transformer.vae
            vae_module.to(dtype=self.vae_dtype).eval().requires_grad_(False)
            self._vae = vae_module
        if "vit" in aux_set:
            vit_module = self.transformer.vision_model
            vit_module.to(dtype=self.dtype).eval().requires_grad_(False)
            self._vit = vit_module


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing into children) is on
    the meta device. Used to gate per-shard ``to_empty`` calls."""
    for p in module.parameters(recurse=True):
        if p.is_meta:
            return True
    return False


def _current_rank() -> int:
    """Return the current torch.distributed rank, or 0 if not initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _collect_filtered_state_dict(
    pretrained_path: str,
    *,
    prefixes: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Stream the HF safetensors checkpoint, returning all keys whose
    top-level prefix matches one of ``prefixes`` (matched as
    ``prefix + "."``). Keys are returned at the wrapper-level namespace
    (no stripping) so they match against ``HunyuanImage3ForCausalMM``'s
    parameter names directly.

    Reads ``model.safetensors.index.json`` to find which shard files
    contain matching keys, then opens only those shards. Falls back to
    the single-file ``model.safetensors`` layout when no index is present.
    """
    import json

    from safetensors.torch import safe_open

    index_path = os.path.join(pretrained_path, "model.safetensors.index.json")
    single_path = os.path.join(pretrained_path, "model.safetensors")

    prefix_dots = tuple(p + "." for p in prefixes)

    def _matches(key: str) -> bool:
        return any(key.startswith(pd) for pd in prefix_dots)

    out: Dict[str, torch.Tensor] = {}

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map: Dict[str, str] = index.get("weight_map", {})
        # Group keys by shard file so we open each file at most once.
        files_to_keys: Dict[str, List[str]] = {}
        for key, fname in weight_map.items():
            if not _matches(key):
                continue
            files_to_keys.setdefault(fname, []).append(key)
        for fname, keys in files_to_keys.items():
            shard_path = os.path.join(pretrained_path, fname)
            with safe_open(shard_path, framework="pt") as f:
                for key in keys:
                    out[key] = f.get_tensor(key)
        return out

    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt") as f:
            for key in f.keys():
                if _matches(key):
                    out[key] = f.get_tensor(key)
        return out

    raise FileNotFoundError(
        f"Could not find HF safetensors index or single-file ckpt at "
        f"{pretrained_path}. Expected {index_path!r} or {single_path!r}."
    )


__all__ = ["HunyuanImage3Bundle"]
