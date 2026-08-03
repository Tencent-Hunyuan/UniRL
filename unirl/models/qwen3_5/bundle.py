"""Qwen3_5Bundle — concrete weights+processor+tokenizer holder for Qwen3.5 VL.

Mirror of :class:`unirl.models.qwen_vl.QwenVLBundle` (vision tower freeze,
meta-init, AutoProcessor) but branches on ``model_type``:

* ``qwen3_5``      -> ``Qwen3_5ForConditionalGeneration``      (dense)
* ``qwen3_5_moe``  -> ``Qwen3_5MoeForConditionalGeneration``   (MoE)

Also applies the ``fast_pos_embed_interpolate`` device-fix patch (borrowed
from verl) on the vision tower, which is needed when
``meta_init_transformer=True`` + FSDP2 cpu_offload would otherwise leave
``self.pos_embed`` on CPU while ``grid_thw`` is on GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from packaging.version import Version

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer, capture_init_state
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3_5PipelineConfig

logger = logging.getLogger(__name__)


def _patch_fast_pos_embed_interpolate(visual_module: nn.Module) -> None:
    """Bind verl's ``fast_pos_embed_interpolate`` onto the vision tower.

    The upstream implementation reads ``self.pos_embed.weight.device`` for
    the output device; under FSDP2 cpu_offload ``self.pos_embed`` is still
    on CPU after materialization while the caller passes a CUDA ``grid_thw``,
    producing a device-mismatch crash. verl's fix takes the device from
    ``grid_thw`` instead. We bind the function as a method so ``self`` is
    the vision module.
    """
    import types

    def fast_pos_embed_interpolate(self, grid_thw):  # noqa: D401
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = grid_thw.device  # verl fix: device from grid_thw, not pos_embed

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    visual_module.fast_pos_embed_interpolate = types.MethodType(fast_pos_embed_interpolate, visual_module)


def _model_class_for(model_type: str):
    """Return the HF ``ForConditionalGeneration`` class for ``model_type``."""
    import transformers

    version = Version(transformers.__version__)
    if version < Version("5.0.0"):
        raise RuntimeError(
            f"Qwen3_5Bundle requires transformers >= 5.0 (got {transformers.__version__}); "
            "Qwen3.5 is natively supported only from 5.0 onward."
        )
    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    elif model_type == "qwen3_5_moe":
        from transformers import Qwen3_5MoeForConditionalGeneration as Cls
    else:
        raise ValueError(f"Qwen3_5Bundle: unexpected model_type {model_type!r}; expected 'qwen3_5' or 'qwen3_5_moe'.")
    return Cls


class Qwen3_5Bundle(Bundle):
    """Qwen3.5 bundle: VL transformer + processor + tokenizer."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: Qwen3_5PipelineConfig) -> "Qwen3_5Bundle":
        """Load the Qwen3.5 transformer + processor + tokenizer from a HF checkpoint."""
        from transformers import AutoConfig, AutoProcessor

        path = config.pretrained_model_ckpt_path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        hf_config = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
        model_type = hf_config.model_type
        ModelCls = _model_class_for(model_type)

        load_kwargs = {}
        if getattr(config, "attn_implementation", None):
            load_kwargs["attn_implementation"] = str(config.attn_implementation)

        meta_init_state = None
        if config.meta_init_transformer:
            if model_type == "qwen3_5_moe":
                # VeOmni owns the MoE meta build (EP plan, fused kernels) and
                # constructs on init_device="meta" itself, so it cannot route
                # through build_meta_init_transformer. Replicate that helper's
                # finalize inline instead: capture init-computed non-persistent
                # state (GDN conv/rope buffers absent from the checkpoint, else
                # to_empty clobbers them), cast (metadata-only on meta), and stamp
                # init_weights to a no-op so parallelize does not re-init after
                # to_empty. capture_init_state requires buffers real on CPU, which
                # VeOmni's meta build provides.
                from unirl.train.backend.veomni import _compat

                _compat.ensure_qwen3_5_moe_installed()
                from veomni.arguments import OpsImplementationConfig
                from veomni.models.auto import build_foundation_model

                dtype_name = {
                    torch.float16: "float16",
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }.get(dtype)
                if dtype_name is None:
                    raise ValueError(f"Qwen3_5Bundle: VeOmni does not support model dtype {dtype}")
                ops = OpsImplementationConfig(
                    attn_implementation=config.attn_implementation or "eager",
                    moe_implementation="fused_triton",
                    cross_entropy_loss_implementation="eager",
                    rms_norm_implementation="eager",
                    swiglu_mlp_implementation="eager",
                    rotary_pos_emb_implementation="eager",
                    rotary_pos_emb_vision_implementation="eager",
                    load_balancing_loss_implementation="eager",
                    rms_norm_gated_implementation="fla",
                    causal_conv1d_implementation="fla",
                    chunk_gated_delta_rule_implementation="fla",
                )
                transformer = build_foundation_model(
                    config_path=hf_config,
                    weights_path=None,
                    torch_dtype=dtype_name,
                    init_device="meta",
                    ops_implementation=ops,
                )
                meta_init_state = capture_init_state(transformer)
                transformer = transformer.to(dtype)
                transformer.init_weights = lambda: None
            else:
                transformer, meta_init_state = build_meta_init_transformer(lambda: ModelCls(hf_config), dtype=dtype)
        else:
            transformer = ModelCls.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=bool(config.trust_remote_code),
                **load_kwargs,
            ).to(device)

        # Patch the vision tower's fast_pos_embed_interpolate (verl borrow).
        # Only when meta_init is on — eager from_pretrained already has the
        # right devices and the upstream impl is fine.
        if config.meta_init_transformer:
            visual = getattr(transformer.model, "visual", None)
            if (
                visual is not None
                and type(visual).__module__.startswith("transformers.")
                and hasattr(visual, "pos_embed")
            ):
                _patch_fast_pos_embed_interpolate(visual)
                logger.info("Patched fast_pos_embed_interpolate on %s", type(visual).__name__)

        # Structural (sets requires_grad, no weight access); persists through
        # to_empty + load on both meta and eager builds.
        if config.freeze_vision_tower:
            visual = getattr(transformer.model, "visual", None)
            if visual is not None:
                visual.requires_grad_(False)
                logger.info(
                    "Froze vision tower (%s parameters).",
                    sum(1 for _ in visual.parameters()),
                )

        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning(
                    "Qwen3_5 transformer %s does not expose gradient_checkpointing_enable; "
                    "skipping use_gradient_checkpointing=True.",
                    type(transformer).__name__,
                )

        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = path
            # Ray-robust restore carrier for init-computed non-persistent state
            # (captured above before to_empty; the transformer is now on meta, so
            # recapturing here would fail). restore_init_state replays it in
            # load_trainable_weights after the sharded weight load.
            bundle._meta_init_state = meta_init_state
        return bundle

    def prepare_for_expert_parallel(self) -> None:
        """Validate that the meta-built Qwen3.5-MoE model exposes VeOmni's EP plan."""
        model_type = getattr(getattr(self.transformer, "config", None), "model_type", None)
        if model_type != "qwen3_5_moe":
            raise ValueError(
                f"Qwen3_5Bundle expert parallelism requires a qwen3_5_moe checkpoint; got model_type={model_type!r}."
            )
        get_parallel_plan = getattr(self.transformer, "get_parallel_plan", None)
        if not callable(get_parallel_plan):
            raise RuntimeError(
                "Qwen3_5Bundle expert parallelism requires the VeOmni-patched Qwen3.5-MoE "
                "model with get_parallel_plan(); set meta_init_transformer=true."
            )
        if get_parallel_plan() is None:
            raise RuntimeError("Qwen3_5Bundle: VeOmni get_parallel_plan() returned None.")


__all__ = ["Qwen3_5Bundle"]
