"""BagelBundle — weights+params holder for BAGEL-7B-MoT (gen-only T2I)."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .config import BagelPipelineConfig
from .vendor.data.data_utils import add_special_tokens
from .vendor.inferencer import InterleaveInferencer
from .vendor.modeling.autoencoder import load_ae
from .vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .vendor.modeling.qwen2 import Qwen2Tokenizer

BAGEL_FSDP_BLOCK_CLASS = "Qwen2MoTDecoderLayer"


class BagelBundle(Bundle):
    """BAGEL-7B-MoT bundle: MoT transformer + FLUX VAE + tokenizer + inferencer."""

    def __init__(
        self,
        *,
        model: Any,
        vae: Any,
        tokenizer: Any,
        new_token_ids: dict,
        vae_transform: Any,
        vit_transform: Any,
        inferencer: Any,
        dtype: torch.dtype,
        vae_dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        latent_patch_size: int,
        latent_channels: int,
        latent_downsample: int,
        config: Optional[BagelPipelineConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.transformer = model.language_model
        self.vae = vae
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.inferencer = inferencer
        self.dtype = dtype
        self.vae_dtype = vae_dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels
        self.latent_downsample = latent_downsample

    @classmethod
    def from_config(cls, config: BagelPipelineConfig) -> "BagelBundle":
        """Load BAGEL-7B-MoT (gen + optional und ViT) from a local checkpoint dir."""
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")

        model_dir = config.pretrained_model_ckpt_path

        llm_config = Qwen2Config.from_json_file(os.path.join(model_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vae_model, vae_config = load_ae(local_path=os.path.join(model_dir, "ae.safetensors"))

        vit_config = None
        if config.enable_vit:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_dir, "vit_config.json"))
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=config.enable_vit,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=config.latent_patch_size,
            max_latent_size=config.max_latent_size,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config) if config.enable_vit else None
            model = Bagel(language_model, vit_model, bagel_config)
            if config.enable_vit:
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_dir, "ema.safetensors"),
            device_map={"": str(device)},
            dtype=dtype,
            offload_buffers=False,
            force_hooks=True,
            offload_folder="/tmp/bagel_offload",
        ).eval()

        tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Image transforms (image-conditioned paths only; pure T2I never exercises
        # them, but the inferencer constructor requires both). The geometry lives in
        # rl_ops so the vllm_omni worker's it2i prefill builds the SAME pair — see
        # rl_ops.build_image_transforms for the sizes and the stride-14 rationale.
        vae_transform, vit_transform = rl_ops.build_image_transforms()

        vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
        vae_model.requires_grad_(False)
        model.requires_grad_(False)
        if not config.use_lora:
            # Unfreeze decoder blocks before FSDP wrapping so trainable and sharded parameter sets match.
            n_blocks = 0
            for module in model.language_model.modules():
                if type(module).__name__ == BAGEL_FSDP_BLOCK_CLASS:
                    module.requires_grad_(True)
                    n_blocks += 1
            if n_blocks == 0:
                raise RuntimeError(
                    f"BagelBundle.from_config: use_lora=False (full fine-tuning) but found no "
                    f"{BAGEL_FSDP_BLOCK_CLASS} blocks to unfreeze in language_model."
                )

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        return cls(
            model=model,
            vae=vae_model,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            inferencer=inferencer,
            dtype=dtype,
            vae_dtype=vae_dtype,
            device=device,
            pretrained_path=model_dir,
            latent_patch_size=int(model.latent_patch_size),
            latent_channels=int(model.latent_channel),
            latent_downsample=int(model.latent_downsample),
            config=config,
        )

    def trainable_module(self) -> "torch.nn.Module":
        """Return the MoT transformer — the FSDP wrap target / trainable root."""
        return self.transformer


__all__ = ["BAGEL_FSDP_BLOCK_CLASS", "BagelBundle"]
