"""WAN21Bundle — concrete weights+params holder for WAN 2.1 T2V / I2V.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules WAN 2.1 ships with: 1× transformer (``WanTransformer3DModel``),
1× 3D VAE (``AutoencoderKLWan``), 1× UMT5/T5 text encoder + tokenizer.
Optional I2V vision tower (``CLIPVisionModel`` + image processor) loaded
only when ``transformer.config.image_dim > 0`` — see ``uses_clip_vision``.
No LoRA injection, FSDP wrap, adapter switching, autocast helpers, or
weight-sync logic — those are lifecycle concerns owned outside the
bundle.

No ``scheduler`` field here either: WAN sigma scheduling always goes
through :func:`unirl.sde.runtime.get_sigma_schedule` with the
config-side ``shift`` (matches legacy ``samplers/fsdp/wan_sampler.py``
convention). Bundles for models that DO use a diffusers scheduler (SD3
``retrieve_timesteps`` dynamic shift, HI3 ``set_timesteps``) carry a
``scheduler`` field; WAN doesn't, so we don't.

Use :meth:`WAN21Bundle.from_config` to load a HuggingFace checkpoint.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import WAN21PipelineConfig


class WAN21Bundle(Bundle):
    """WAN 2.1 T2V / I2V bundle: transformer + VAE + UMT5 text encoder
    (+ optional CLIP vision tower for I2V).
    """

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        max_sequence_length: int,
        vision_encoder: Optional[nn.Module] = None,
        image_processor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.max_sequence_length = max_sequence_length
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor

    @property
    def uses_clip_vision(self) -> bool:
        """True iff the bundle loaded a CLIP vision tower (I2V path).

        Pipelines / stages branch on this to decide whether to construct
        a :class:`WAN21CLIPVisionEncodeStage` and emit an
        ``ImageEmbedCondition``. T2V bundles set both
        ``vision_encoder`` / ``image_processor`` to ``None`` and the
        property is ``False``.
        """
        return self.vision_encoder is not None

    @classmethod
    def from_config(cls, config: WAN21PipelineConfig) -> "WAN21Bundle":
        """Load all WAN 2.1 components from a HuggingFace checkpoint."""
        try:
            from diffusers import AutoencoderKLWan, WanTransformer3DModel
        except ImportError:
            # Fallback for older diffusers: ``AutoModel`` does dynamic
            # dispatch on the checkpoint config. Matches the fallback in
            # legacy ``models/wan21.py``.
            from diffusers import AutoModel

            AutoencoderKLWan = AutoModel
            WanTransformer3DModel = AutoModel
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel
        except ImportError:
            from transformers import AutoTokenizer
            from transformers import T5EncoderModel as UMT5EncoderModel

        path = config.pretrained_model_ckpt_path
        vae_path = config.vae_ckpt_path or path
        te_path = config.text_encoder_ckpt_path or path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")
        te_raw = config.text_encoder_dtype if config.text_encoder_dtype is not None else config.model_precision
        te_dtype = parse_torch_dtype(te_raw, field_name="text_encoder_dtype")

        meta_init_state = None
        if config.meta_init_transformer:
            # Meta-init (FSDP / VeOmni load_sharded path): architecture only,
            # no per-rank weight allocation; the backend to_empty-materializes
            # and broadcast-loads from the stashed dir after sharding.
            # build_meta_init_transformer keeps WanRotaryPosEmbed's freqs_cos/
            # freqs_sin (non-persistent buffers, absent from the checkpoint and
            # init-computed) REAL and captures them; torch.device("meta") would
            # force them to meta too -> to_empty leaves them garbage, zeroing
            # self-attn to_q/to_k LoRA gradients. meta_init_state is stashed on
            # the bundle below as the Ray-robust restore carrier.
            transformer_config = WanTransformer3DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: WanTransformer3DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = WanTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype)
            # Dtype unification matters even though from_pretrained got
            # torch_dtype=dtype: diffusers leaves some parameters / buffers
            # (timestep embeddings, RoPE freqs, ...) in fp32, and FSDP's
            # _init_mp_dtypes asserts a uniform original-param dtype across
            # the wrapped module.
            transformer = transformer.to(device, dtype=dtype)

        vae: Optional[nn.Module] = None
        if config.load_vae:
            vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
            vae.requires_grad_(False)

        text_encoder = (
            UMT5EncoderModel.from_pretrained(te_path, subfolder="text_encoder", torch_dtype=te_dtype).to(device).eval()
        )
        text_encoder.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")

        # Optional CLIP vision tower for I2V: WAN 2.1 I2V checkpoints
        # declare ``image_dim > 0`` on the transformer config; T2V
        # checkpoints (and the WAN 2.2 family) leave it 0. Loading is
        # gated strictly on this signal — setting
        # ``image_encoder_ckpt_path`` against a ``image_dim == 0``
        # checkpoint is a config error (no silent fallback).
        image_dim = int(getattr(transformer.config, "image_dim", 0) or 0)
        vision_encoder: Optional[nn.Module] = None
        image_processor: Optional[Any] = None
        if image_dim > 0:
            try:
                from transformers import CLIPImageProcessor, CLIPVisionModel
            except ImportError as e:
                raise ImportError(
                    "WAN21Bundle.from_config: transformer declares image_dim>0 "
                    "(I2V checkpoint) but `transformers.CLIPVisionModel` / "
                    "`CLIPImageProcessor` is unavailable."
                ) from e
            ie_path = config.image_encoder_ckpt_path or path
            vision_encoder = (
                CLIPVisionModel.from_pretrained(ie_path, subfolder="image_encoder", torch_dtype=dtype).to(device).eval()
            )
            vision_encoder.requires_grad_(False)
            image_processor = CLIPImageProcessor.from_pretrained(ie_path, subfolder="image_processor")
        elif config.image_encoder_ckpt_path is not None:
            raise ValueError(
                "WAN21Bundle.from_config: image_encoder_ckpt_path="
                f"{config.image_encoder_ckpt_path!r} was set, but the transformer "
                "checkpoint declares image_dim=0 (T2V). Either point "
                "pretrained_model_ckpt_path at an I2V checkpoint or clear "
                "image_encoder_ckpt_path."
            )

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            max_sequence_length=int(config.max_sequence_length),
            vision_encoder=vision_encoder,
            image_processor=image_processor,
        )
        if config.meta_init_transformer:
            # Consumed by the backend's post-shard weight load.
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            # Ray-robust restore carrier for init-computed non-persistent state.
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["WAN21Bundle"]
