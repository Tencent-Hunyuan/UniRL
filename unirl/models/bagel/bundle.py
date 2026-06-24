"""BagelBundle — weights+params holder for BAGEL-7B-MoT (gen-only T2I).

Implements the empty :class:`Bundle` Protocol. Pure container of the modules
BAGEL ships with for text-to-image: one MoT transformer (``Bagel`` wrapping a
``Qwen2ForCausalLM`` whose ``Qwen2MoTDecoderLayer`` blocks hold both und and gen
experts) + one FLUX-style VAE + one tokenizer. The und ViT tower is opt-in
(``config.enable_vit`` → ``visual_und=True``): pure T2I needs only the gen
expert + VAE; image-INPUT tasks (it2i editing / i2t / it2t) need the ViT to
prefill image semantics into the KV context.

LoRA injection / FSDP wrap / autocast lifecycle are owned outside the bundle
(the train backend), so ``from_config`` only loads + freezes. The trainable
surface is ``model.language_model`` (the MoT, where the ``*_moe_gen`` experts
live); the FSDP block class is ``Qwen2MoTDecoderLayer``.

Construction mirrors flow_grpo's ``train_bagel.py`` setup so the vendored
``InterleaveInferencer`` / ``generate_image`` path the diffusion stage delegates
to behaves identically.
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
from .vendor.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
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
        # The trainable MoT (where the *_moe_gen experts live). Same object the
        # vendored generate_image / _forward_flow run on, so FSDP2 fully_shard
        # (in-place) on this reference shards the gen forward too. Named
        # ``transformer`` so recipes can set backend.trainable_attr: transformer.
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
        """Load BAGEL-7B-MoT (gen + optional und ViT) from a local checkpoint dir.

        Replicates flow_grpo/train_bagel.py:316-414 minus LoRA/optimizer (which
        the train backend owns). Loads the EMA weights via
        ``load_checkpoint_and_dispatch`` onto a single device; the FSDP wrap and
        LoRA injection run later in :class:`FSDPBackend`.

        Note: ``load_checkpoint_and_dispatch`` attaches accelerate device hooks.
        For the dedicated FSDP path (Phase 6) those may need removal via
        ``accelerate.hooks.remove_hook_from_module`` before ``fully_shard``; for
        the standalone bundle smoke they are harmless.
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

        vit_config = None
        if config.enable_vit:
            # Official inference setup (ByteDance-Seed/Bagel app.py @ vendored
            # commit a2fa77d): no RoPE, drop the last SigLIP layer; the checkpoint
            # stores the patch embedding as a Linear (converted below, pre-load).
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

        # force_hooks=True attaches accelerate AlignDevicesHooks (matching
        # flow_grpo/train_bagel.py) so the vendored inferencer / generate_image
        # path — which builds packed index tensors on CPU and calls submodule
        # forwards directly — has its inputs auto-moved to the model device.
        # Phase 6 (UniRL FSDP) must remove these hooks before fully_shard
        # (accelerate.hooks.remove_hook_from_module(model, recurse=True)).
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

        # Image transforms match flow_grpo (vae 512/256/8, vit 490/112/7). Only
        # used for image-conditioned paths; pure T2I never exercises them, but
        # the inferencer constructor requires both.
        vae_transform = ImageTransform(512, 256, 8)
        vit_transform = ImageTransform(490, 112, 7)

        vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
        vae_model.requires_grad_(False)
        # Freeze the whole MoT here; the backend re-enables only the LoRA (or
        # moe_gen) params it injects/unfreezes.
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
        """Return the MoT transformer — the FSDP wrap target / trainable root.

        ``model.language_model`` holds the ``Qwen2MoTDecoderLayer`` blocks whose
        ``*_moe_gen`` experts are the only trained params (via LoRA). The gen
        heads (``vae2llm`` / ``time_embedder`` / ``llm2vae`` / ``latent_pos_embed``)
        sit on the parent ``Bagel`` module and stay frozen in the LoRA setup.
        """
        return self.transformer


__all__ = ["BAGEL_FSDP_BLOCK_CLASS", "BagelBundle"]
