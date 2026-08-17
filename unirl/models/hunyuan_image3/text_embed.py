"""HunyuanImage3TextEmbedStage — chat-template-driven input prep."""

from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.types.primitives import Texts

from .bundle import HunyuanImage3Bundle
from .compat import repair_hi3_tokenizer_backend
from .conditions import HunyuanImage3FusedMultimodalCondition


def _resolve_build_batch_2d_rope():
    """Locate the upstream ``build_batch_2d_rope`` rope helper."""
    for _name, _mod in sys.modules.items():
        if _name.startswith("transformers_modules.") and hasattr(_mod, "build_batch_2d_rope"):
            return _mod.build_batch_2d_rope
    raise RuntimeError(
        "HunyuanImage3TextEmbedStage: could not locate build_batch_2d_rope "
        "in any transformers_modules.* — was AutoModelForCausalLM.from_pretrained "
        "called with trust_remote_code=True before bundle construction?"
    )


def _resolve_get_system_prompt():
    """Locate the checkpoint's ``get_system_prompt``; FSDP2 can rebind ``__module__``."""
    for _name, _mod in sys.modules.items():
        if _name.startswith("transformers_modules.") and hasattr(_mod, "get_system_prompt"):
            return _mod.get_system_prompt
    raise RuntimeError(
        "HunyuanImage3TextEmbedStage: could not locate get_system_prompt "
        "in any transformers_modules.* — was the checkpoint loaded with "
        "trust_remote_code=True?"
    )


def _resolve_system_prompt_preset(sys_type: str, bot_task: str) -> Optional[str]:
    """Resolve a checkpoint system-prompt preset, stripped as upstream does."""
    resolved = _resolve_get_system_prompt()(str(sys_type), str(bot_task))
    if resolved is None:
        return None
    stripped = str(resolved).strip()
    return stripped or None


def _optional_output_tensor(output: Any, names: Tuple[str, ...], device: torch.device) -> Optional[torch.Tensor]:
    """First non-None attribute of ``output`` among ``names``, moved to"""
    for name in names:
        t = getattr(output, name, None)
        if t is not None:
            return t.to(device)
    return None


class HunyuanImage3TextEmbedStage:
    """HunyuanImage3 chat-template-driven input-prep stage (AR + diffusion)."""

    def __init__(
        self,
        bundle: HunyuanImage3Bundle,
        *,
        max_sequence_length: int = 1024,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = max_sequence_length

    def _apply_chat_template(
        self,
        *,
        mode: str,
        batch_prompt: Optional[List[str]],
        bot_task: str,
        cfg_factor: int,
        sequence_template: Optional[str] = None,
        batch_message_list: Optional[Any] = None,
        batch_gen_image_info: Optional[Any] = None,
        batch_system_prompt: Optional[List[str]] = None,
        batch_cot_text: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        batch_cond_image_info: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Run the upstream tokenizer wrapper; returns ``(output, sections)``."""
        transformer = self.bundle.transformer
        config = transformer.config
        gen_config = transformer.generation_config

        if getattr(transformer, "_tkwrapper", None) is None and getattr(transformer, "_tokenizer", None) is None:
            # Default omitted model_version; the tokenizer ignores its value.
            if not hasattr(config, "model_version"):
                config.model_version = "instruct"
            transformer.load_tokenizer(self.bundle.pretrained_path)
        tkw = getattr(transformer, "_tkwrapper", None) or getattr(transformer, "_tokenizer", None)
        repair_hi3_tokenizer_backend(tkw, self.bundle.pretrained_path)

        effective_sequence_template = (
            gen_config.sequence_template if sequence_template is None else str(sequence_template)
        )

        # Two text sections tokenize independently; direct-T2I pretrain needs one BPE pass.
        if (
            mode == "gen_image"
            and effective_sequence_template == "pretrain"
            and sequence_template is not None
            and bot_task == "image"
            and cfg_factor == 1
            and batch_message_list is None
            and batch_prompt is not None
            and batch_system_prompt is not None
            and batch_cot_text is None
            and batch_cond_image_info is None
        ):
            if len(batch_prompt) != len(batch_system_prompt):
                raise ValueError(
                    "pretrain direct-T2I batch_prompt and batch_system_prompt must align, "
                    f"got {len(batch_prompt)} and {len(batch_system_prompt)}"
                )
            if all(batch_system_prompt):
                batch_prompt = [
                    f"{system_prompt}{prompt}" for prompt, system_prompt in zip(batch_prompt, batch_system_prompt)
                ]
                batch_system_prompt = None

        _cond_kw = (
            "batch_cond_images"
            if "batch_cond_images" in inspect.signature(tkw.apply_chat_template).parameters
            else "batch_cond_image_info"
        )
        out = tkw.apply_chat_template(
            batch_prompt=batch_prompt,
            batch_message_list=batch_message_list,
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            max_length=max_length,
            bot_task=bot_task,
            image_base_size=config.image_base_size,
            sequence_template=effective_sequence_template,
            cfg_factor=cfg_factor,
            drop_think=gen_config.drop_think,
            **{_cond_kw: batch_cond_image_info},
        )
        return out["output"], out["sections"]

    def _fused_common(
        self,
        output: Any,
        sections: Any,
        *,
        rope_seq_len: Optional[int] = None,
    ) -> Tuple[
        torch.device,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Tensor prep shared by the AR and gen_image paths."""
        transformer = self.bundle.transformer
        config = transformer.config
        gen_config = transformer.generation_config

        device = transformer.model.wte.weight.device

        input_ids: torch.Tensor = output.tokens.to(device)
        n, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])

        rope_image_info = transformer.build_batch_rope_image_info(output, sections)
        build_batch_2d_rope = _resolve_build_batch_2d_rope()
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len if rope_seq_len is None else rope_seq_len,
            n_elem=config.attention_head_dim,
            device=device,
            base=config.rope_theta,
        )

        position_ids: torch.Tensor = torch.arange(0, seq_len, dtype=torch.long, device=device)[None].expand(n, -1)

        attention_mask: torch.Tensor = transformer._prepare_attention_mask_for_generation(
            input_ids,
            gen_config,
            model_kwargs={"tokenizer_output": output},
        ).to(device)

        cond_vit_image_mask = _optional_output_tensor(output, ("cond_vit_image_mask", "vit_image_mask"), device)

        # Stack RoPE so it travels as a per-sample CONCAT tensor.
        rope_cache = torch.stack([cos, sin], dim=1)
        return device, input_ids, attention_mask, position_ids, rope_cache, cond_vit_image_mask

    def embed_for_ar(
        self,
        p: Texts,
        *,
        bot_task: str = "auto",
        system_prompt: Optional[List[str]] = None,
        cot_text: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        batch_message_list: Optional[Any] = None,
        batch_cond_image_info: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build the unified-MM input tensors for ``mode="gen_text"``."""
        gen_config = self.bundle.transformer.generation_config

        prompts = list(p.texts) if batch_message_list is None else None

        output, sections = self._apply_chat_template(
            mode="gen_text",
            batch_prompt=prompts,
            bot_task=bot_task,
            cfg_factor=1,
            batch_message_list=batch_message_list,
            batch_gen_image_info=None,
            batch_system_prompt=system_prompt,
            batch_cot_text=cot_text,
            max_length=max_length,
            batch_cond_image_info=batch_cond_image_info,
        )

        prompt_len = int(output.tokens.shape[1])
        rope_seq_len = int(getattr(gen_config, "max_length", prompt_len))
        rope_seq_len = max(rope_seq_len, prompt_len)

        _device, input_ids, attention_mask, position_ids, rope_cache, cond_vit_image_mask = self._fused_common(
            output, sections, rope_seq_len=rope_seq_len
        )

        cond_vae_image_mask = _optional_output_tensor(output, ("cond_vae_image_mask", "vae_image_mask"), _device)
        cond_timestep_scatter_index = _optional_output_tensor(output, ("cond_timestep_scatter_index",), _device)

        prompt_lengths: Optional[torch.Tensor] = None
        real_pos = getattr(output, "real_pos", None)
        if real_pos is not None:
            rp = real_pos.to(device=_device, dtype=torch.long)
            if rp.dim() == 2:
                rp = rp[:, -1]
            prompt_lengths = rp.reshape(-1)

        fused = HunyuanImage3FusedMultimodalCondition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_cache=rope_cache,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vit_image_mask=cond_vit_image_mask,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            prompt_lengths=prompt_lengths,
        )
        return {"fused": fused, "tokenizer_output": output}

    def embed_for_gen_image(
        self,
        p: Texts,
        *,
        cfg: bool,
        height: int,
        width: int,
        bot_task: str = "image",
        cot_text: Optional[List[str]] = None,
        system_prompt: Optional[List[str]] = None,
        sys_type: Optional[str] = None,
        sequence_template: Optional[str] = None,
        batch_cond_image_info: Optional[List[List[Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the unified-MM input tensors for ``mode="gen_image"``."""
        transformer = self.bundle.transformer

        prompts = list(p.texts)
        if not prompts:
            raise ValueError("HunyuanImage3TextEmbedStage.embed_for_gen_image: prompts is empty")
        cfg_factor = 2 if cfg else 1
        if system_prompt is None and sys_type is not None:
            resolved_system_prompt = _resolve_system_prompt_preset(str(sys_type), bot_task)
            if resolved_system_prompt is not None:
                system_prompt = [resolved_system_prompt] * len(prompts)

        ip = transformer.image_processor
        if hasattr(ip, "build_image_info"):
            image_info = ip.build_image_info(f"{int(height)}x{int(width)}")
        elif hasattr(ip, "build_gen_image_info"):
            image_info = ip.build_gen_image_info(f"{int(height)}x{int(width)}")
        else:
            raise AttributeError(
                "HunyuanImage3 image_processor missing both 'build_image_info' and 'build_gen_image_info'."
            )
        batch_gen_image_info = [image_info] * len(prompts)

        output, sections = self._apply_chat_template(
            mode="gen_image",
            batch_prompt=prompts,
            bot_task=bot_task,
            cfg_factor=cfg_factor,
            sequence_template=sequence_template,
            batch_gen_image_info=batch_gen_image_info,
            batch_system_prompt=system_prompt,
            batch_cot_text=cot_text,
            batch_cond_image_info=batch_cond_image_info,
        )

        device, input_ids, attention_mask, position_ids, rope_cache, cond_vit_image_mask = self._fused_common(
            output, sections
        )

        gen_image_mask: torch.Tensor = output.gen_image_mask.to(device)
        gen_timestep_scatter_index: torch.Tensor = output.gen_timestep_scatter_index.to(device)

        cond_vae_image_mask = _optional_output_tensor(output, ("cond_vae_image_mask", "vae_image_mask"), device)
        cond_timestep_scatter_index = _optional_output_tensor(output, ("cond_timestep_scatter_index",), device)

        fused = HunyuanImage3FusedMultimodalCondition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_cache=rope_cache,
            gen_image_mask=gen_image_mask,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vit_image_mask=cond_vit_image_mask,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
        )
        return {"fused": fused, "tokenizer_output": output}


__all__ = ["HunyuanImage3TextEmbedStage"]
