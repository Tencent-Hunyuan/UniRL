"""SenseNova-U1.5 model and tokenizer bundle."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import SenseNovaU1PipelineConfig
from .vendor.neo_unify import register as register_neo_unify
from .vendor.neo_unify import set_attn_backend
from .vendor.neo_unify.transformers_compat import pretrained_dtype_kwargs

logger = logging.getLogger(__name__)


def _set_generation_trainability(model: nn.Module, *, enabled: bool) -> int:
    """Freeze the shared/understanding path and optionally unfreeze the image branch."""
    model.requires_grad_(False)
    if not enabled:
        return 0

    trainable = 0
    for name, parameter in model.named_parameters():
        if name.startswith("fm_modules.") or "_mot_gen" in name:
            parameter.requires_grad_(True)
            trainable += parameter.numel()
    if trainable == 0:
        raise RuntimeError(
            "SenseNovaU1Bundle: no generation parameters matched `fm_modules.*` "
            "or `*_mot_gen`; the vendored model/checkpoint layout is incompatible."
        )
    return trainable


class SenseNovaU1TrainableModel(nn.Module):
    """FSDP-callable facade around NEOChatModel's inference-only helper API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # NEOChatModel.forward is deliberately unimplemented upstream. Keeping
        # helper calls behind this real forward is still mandatory: the wrapper's
        # root FSDP hooks gather shared embeddings and sibling fm_modules.
        self.model = model

    def forward(self, mode: str, **kwargs: Any) -> Any:
        if mode == "prefix":
            cache, _ = self.model._t2i_prefix_forward(
                kwargs["input_ids"],
                kwargs["indexes"],
                kwargs["attention_mask"],
            )
            return cache
        if mode == "predict_velocity":
            return self._predict_velocity(**kwargs)
        raise ValueError(f"SenseNovaU1TrainableModel.forward: unsupported mode {mode!r}.")

    def _predict_velocity(
        self,
        *,
        normalized_pixels: torch.Tensor,
        packed_pixels: torch.Tensor,
        image_indexes: torch.Tensor,
        prefix_cache: Any,
        data_time: torch.Tensor,
        image_shape: tuple[int, int],
        noise_scale: float,
        uncondition_image_indexes: Optional[torch.Tensor] = None,
        uncondition_prefix_cache: Any = None,
    ) -> Any:
        model = self.model
        height, width = (int(v) for v in image_shape)
        patch = int(model.patch_size)
        merge = int(1 / float(model.downsample_ratio))
        grid_h, grid_w = height // patch, width // patch
        image_tokens = (height // (patch * merge)) * (width // (patch * merge))
        device = normalized_pixels.device

        grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long, device=device)
        vision_patches = model.patchify(normalized_pixels, patch, channel_first=True)
        image_embeds = model.extract_feature(
            vision_patches.reshape(grid_h * grid_w, -1),
            gen_model=True,
            grid_hw=grid_hw,
        ).reshape(1, image_tokens, -1)

        timestep = data_time.expand(image_tokens)
        timestep_embeddings = model.fm_modules["timestep_embedder"](timestep).reshape(1, image_tokens, -1)
        if bool(model.add_noise_scale_embedding):
            noise_value = torch.full_like(timestep, float(noise_scale) / float(model.noise_scale_max_value))
            timestep_embeddings = timestep_embeddings + model.fm_modules["noise_scale_embedder"](noise_value).reshape(
                1, image_tokens, -1
            )
        image_embeds = image_embeds + timestep_embeddings

        condition_velocity = model._t2i_predict_v(
            image_embeds,
            image_indexes,
            {"full_attention": None},
            prefix_cache,
            data_time,
            packed_pixels,
            image_token_num=image_tokens,
            timestep_embeddings=timestep_embeddings,
            image_size=(width, height),
        )
        if uncondition_prefix_cache is None:
            return condition_velocity
        uncondition_velocity = model._t2i_predict_v(
            image_embeds,
            uncondition_image_indexes,
            {"full_attention": None},
            uncondition_prefix_cache,
            data_time,
            packed_pixels,
            image_token_num=image_tokens,
            timestep_embeddings=timestep_embeddings,
            image_size=(width, height),
        )
        return condition_velocity, uncondition_velocity


class SenseNovaU1Bundle(Bundle):
    """SenseNova-U1.5 NEO-Unify backbone plus its Qwen tokenizer."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        config: Optional[SenseNovaU1PipelineConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.transformer = SenseNovaU1TrainableModel(model)
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: SenseNovaU1PipelineConfig) -> "SenseNovaU1Bundle":
        """Load a local or Hub-format SenseNova-U1 checkpoint."""
        import fcntl

        serialize = os.environ.get("DIFFRL_MODEL_LOAD_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_model_load.lock", "a+") if serialize else None
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return cls._from_config_locked(config)
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

    @classmethod
    def _from_config_locked(cls, config: SenseNovaU1PipelineConfig) -> "SenseNovaU1Bundle":
        import transformers
        from packaging.version import Version
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        register_neo_unify()
        set_attn_backend(config.attention_backend)

        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        hf_config = AutoConfig.from_pretrained(path)
        if getattr(hf_config, "model_type", None) != "neo_chat":
            raise ValueError(
                f"SenseNovaU1Bundle expected model_type='neo_chat' at {path!r}, "
                f"got {getattr(hf_config, 'model_type', None)!r}."
            )
        if not bool(getattr(hf_config, "use_pixel_head", False)):
            raise ValueError(
                "SenseNovaU1Bundle currently supports the U1.5 pixel-head checkpoint; "
                "config.use_pixel_head must be true."
            )

        dtype_kwargs = pretrained_dtype_kwargs(dtype)
        if Version(transformers.__version__) < Version("4.57"):
            # Transformers 4.56 still forwards the newer `dtype` kwarg into the
            # model constructor instead of consuming it in from_pretrained.
            dtype_kwargs = {"torch_dtype": dtype}
        model = AutoModel.from_pretrained(
            path,
            config=hf_config,
            **dtype_kwargs,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path)

        trainable = _set_generation_trainability(
            model,
            enabled=bool(config.full_finetune_generation),
        )
        model.eval()
        logger.info(
            "Loaded SenseNova-U1 from %s with %d generation-path trainable parameters.",
            path,
            trainable,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            config=config,
        )

    def trainable_module(self) -> nn.Module:
        """Return the complete wrapper containing every generation-path module."""
        return self.transformer


__all__ = [
    "SenseNovaU1Bundle",
    "SenseNovaU1TrainableModel",
]
