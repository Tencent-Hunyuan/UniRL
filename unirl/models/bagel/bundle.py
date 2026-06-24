"""BagelBundle — weights+params holder for BAGEL-7B-MoT (gen-only T2I).

Container of the MoT transformer + FLUX-style VAE + tokenizer; the und ViT path is
disabled. LoRA/FSDP/autocast live in the train backend — ``from_config`` only loads.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import BagelPipelineConfig
from .vendor.data.data_utils import add_special_tokens
from .vendor.data.transforms import ImageTransform
from .vendor.inferencer import InterleaveInferencer
from .vendor.modeling.autoencoder import load_ae
from .vendor.modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
from .vendor.modeling.qwen2 import Qwen2Tokenizer

# FSDP wrap block class for the MoT decoder (recipe backend.block_class_names).
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
    ) -> None:
        super().__init__()
        self.model = model
        # The trainable MoT (where the *_moe_gen experts live); same object the
        # vendored generate_image / _forward_flow run on. Named transformer so
        # recipes can set backend.trainable_attr: transformer.
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
        """Load BAGEL-7B-MoT (gen-only) EMA weights from a local checkpoint directory.

        Loads + freezes only; the FSDP wrap and LoRA injection run later in the backend.
        """
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

        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=False,
            llm_config=llm_config,
            vit_config=None,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=config.latent_patch_size,
            max_latent_size=config.max_latent_size,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            model = Bagel(language_model, None, bagel_config)

        # force_hooks=True attaches accelerate AlignDevicesHooks so the vendored
        # inferencer (which builds packed index tensors on CPU) gets inputs auto-moved
        # to the model device; the FSDP path must remove these hooks before fully_shard.
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

        # Image transforms match flow_grpo (vae 512/256/8, vit 490/112/7); only the
        # inferencer constructor requires them — pure T2I never exercises them.
        vae_transform = ImageTransform(512, 256, 8)
        vit_transform = ImageTransform(490, 112, 7)

        vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
        vae_model.requires_grad_(False)
        # Freeze the whole MoT; the backend re-enables only the params it injects.
        model.requires_grad_(False)

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
        )

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (``model.language_model``) — FSDP wrap / trainable root."""
        return self.transformer


__all__ = ["BAGEL_FSDP_BLOCK_CLASS", "BagelBundle"]
