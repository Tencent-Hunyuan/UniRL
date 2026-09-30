"""HunyuanImage3Bundle — concrete weights+params holder for HunyuanImage 3.0."""

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
        self._vae = vae
        self._vit = vit
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dtype = dtype
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

    def trainable_module(self) -> nn.Module:
        """The sharded trainable subtree the backend wraps: the bare decoder"""
        return self.transformer.model

    def prepare_for_expert_parallel(self) -> None:
        """Make the decoder expert-parallel-ready (backend hook; called only when"""
        from unirl.train.backend.veomni.ep.models.hi3 import replace_hunyuan_moe_with_fused

        n_swapped = replace_hunyuan_moe_with_fused(self.transformer.model)
        logger.info("expert-parallel: swapped %d HunyuanMoE layer(s) for FusedHunyuanMoE", n_swapped)
        self._ep_enabled = True

    @classmethod
    def from_config(cls, config: HunyuanImage3PipelineConfig) -> "HunyuanImage3Bundle":
        """Load all HunyuanImage3 components from a HuggingFace checkpoint."""
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

        transformer = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        vae = getattr(transformer, "vae", None) or getattr(transformer, "vae_model", None)
        if vae is None:
            raise RuntimeError(
                "HunyuanImage3Bundle.from_config: could not locate VAE on the "
                "loaded backbone. Expected attribute `vae` or `vae_model`. "
                "Verify the checkpoint at " + path + " is a HunyuanImage3 build."
            )
        vae = vae.to(device=device, dtype=vae_dtype).eval()
        vae.requires_grad_(False)

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

    @classmethod
    def from_meta_config(
        cls,
        config: HunyuanImage3PipelineConfig,
    ) -> "HunyuanImage3Bundle":
        """Build the bundle with the full ``HunyuanImage3ForCausalMM`` on"""
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

        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        with init_empty_weights():
            transformer = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

        try:
            transformer.generation_config = GenerationConfig.from_pretrained(path)
        except (OSError, ValueError):
            pass

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        transformer.model.init_weights = lambda: None  # type: ignore[method-assign]

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
        """Single-call materialization for the meta-init path."""
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        from unirl.models.types.post_materialize import canonical_param_name

        aux_set = tuple(with_aux)
        for name in aux_set:
            if name not in {"vae", "vit"}:
                raise ValueError(
                    f"HunyuanImage3Bundle.materialize: unknown aux module {name!r}; expected 'vae' or 'vit'."
                )

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

        for _attr, module in plan:
            if _module_has_meta_param(module):
                module.to_empty(device=device)

        if _current_rank() == 0:
            prefixes = tuple(attr for attr, _ in plan)
            sd = _collect_filtered_state_dict(self.pretrained_path, prefixes=prefixes)

            if getattr(self, "_ep_enabled", False):
                from unirl.train.backend.veomni.ep.models.hi3 import fuse_expert_state_dict

                sd = fuse_expert_state_dict(sd)

            # Remap LoRA base_layer keys before loading to avoid skipped parameters.
            # Canonical names: ``set_model_state_dict`` below matches against
            # ``state_dict()`` keys, which never carry the AC-wrapper segment
            # that ``named_parameters()`` exposes.
            expected_names = {
                canonical_param_name(name) for name, _ in self.transformer.named_parameters(remove_duplicate=False)
            }
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

        expert_sd = {}
        if getattr(self, "_ep_enabled", False):
            from unirl.train.backend.veomni.ep.models.hi3 import is_fused_expert_key

            expert_sd = {k: sd.pop(k) for k in list(sd) if is_fused_expert_key(k)}

        set_model_state_dict(
            self.transformer,
            sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )

        if getattr(self, "_ep_enabled", False):
            from unirl.train.backend.veomni.ep import load_ep_experts
            from unirl.train.backend.veomni.ep.models.hi3 import is_fused_expert_key

            n_exp = load_ep_experts(self.transformer, expert_sd, is_fused_expert_key)
            if n_exp == 0:
                raise RuntimeError(
                    "expert-parallel: load_ep_experts loaded 0 EP-sharded expert params — "
                    "expert weights would stay meta/uninitialized. Check is_fused_expert_key "
                    "against the checkpoint keys."
                )
            if _current_rank() == 0:
                logger.info("expert-parallel: loaded %d EP-sharded expert param(s)", n_exp)

            from unirl.train.backend.veomni.ep import register_unsharded_param_hooks

            n_hooked = register_unsharded_param_hooks(self.transformer)
            if n_hooked == 0:
                raise RuntimeError(
                    "expert-parallel: register_unsharded_param_hooks hooked 0 root params — "
                    "wte/ln_f/lm_head would hit mixed Tensor/DTensor at forward. Check the "
                    "hook targets against the model."
                )
            if _current_rank() == 0:
                logger.info("expert-parallel: hooked root params for direct all-gather: %d", n_hooked)

        if _current_rank() == 0:
            _bl_checked, _bl_bad, _bl_cpu_offloaded = 0, 0, 0
            for name, p in self.transformer.named_parameters(remove_duplicate=False):
                if ".base_layer." not in name:
                    continue
                _bl_checked += 1
                if p.is_meta:
                    _bl_bad += 1
                    continue
                # FSDP2's CPU offload splits DTensor storage mid-construction; the state-dict load already checked them.
                try:
                    is_finite = bool(p.data.isfinite().all())
                except RuntimeError as exc:
                    if "storage on different device" not in str(exc):
                        raise
                    _bl_cpu_offloaded += 1
                    continue
                if not is_finite:
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
                    f"verified {_bl_checked - _bl_cpu_offloaded} LoRA base_layer "
                    f"params loaded finite; deferred {_bl_cpu_offloaded} "
                    f"CPU-offloaded shard(s) ✓",
                    flush=True,
                )

        del sd

        if "vae" in aux_set:
            vae_module = self.transformer.vae
            vae_module.to(dtype=self.vae_dtype).eval().requires_grad_(False)
            self._vae = vae_module
        if "vit" in aux_set:
            vit_module = self.transformer.vision_model
            vit_module.to(dtype=self.dtype).eval().requires_grad_(False)
            self._vit = vit_module


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing into children) is on"""
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
    """Stream the HF safetensors checkpoint, returning all keys whose"""
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
