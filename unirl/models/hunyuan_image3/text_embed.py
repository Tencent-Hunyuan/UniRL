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


class HunyuanImage3TextEmbedStage:
    """HunyuanImage3 chat-template-driven input-prep stage (AR + diffusion)."""

    def __init__(self, bundle: HunyuanImage3Bundle) -> None:
        self.bundle = bundle

    def _apply_chat_template(
        self,
        *,
        mode: str,
        batch_prompt: List[str],
        bot_task: str,
        cfg_factor: int,
        sequence_template: Optional[str] = None,
        batch_gen_image_info: Optional[Any] = None,
        batch_system_prompt: Optional[List[str]] = None,
        batch_cot_text: Optional[List[str]] = None,
        batch_cond_image_info: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """Run the upstream tokenizer wrapper and return ``(output, sections)`` (README ``## Gotchas``)."""
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

        # Two text sections tokenize independently; direct-T2I pretrain needs one BPE pass.
        if (
            mode == "gen_image"
            and sequence_template == "pretrain"
            and bot_task == "image"
            and cfg_factor == 1
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
            mode=mode,
            batch_gen_image_info=batch_gen_image_info,
            batch_system_prompt=batch_system_prompt,
            batch_cot_text=batch_cot_text,
            bot_task=bot_task,
            image_base_size=config.image_base_size,
            sequence_template=gen_config.sequence_template if sequence_template is None else sequence_template,
            cfg_factor=cfg_factor,
            drop_think=gen_config.drop_think,
            **{_cond_kw: batch_cond_image_info},
        )
        n_samples = len(batch_prompt)
        expected_rows = n_samples * cfg_factor
        rows = out["output"].tokens.shape[0]
        if rows != expected_rows:
            raise ValueError(
                f"HunyuanImage3TextEmbedStage: chat template returned {rows} row(s) for "
                f"{n_samples} prompt(s) at cfg_factor={cfg_factor}, expected {expected_rows}; "
                "the checkpoint tokenizer dropped samples (issue #520)."
            )
        return out["output"], out["sections"]

    def _build_fused(
        self,
        output: Any,
        sections: Any,
        *,
        rope_seq_len: Optional[int] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        gen_image_mask: Optional[torch.Tensor] = None,
        gen_timestep_scatter_index: Optional[torch.Tensor] = None,
    ) -> HunyuanImage3FusedMultimodalCondition:
        """Build the ``HunyuanImage3FusedMultimodalCondition`` shared by AR and ``gen_image``."""
        transformer = self.bundle.transformer
        config = transformer.config
        gen_config = transformer.generation_config

        device = transformer.model.wte.weight.device

        input_ids: torch.Tensor = output.tokens.to(device)
        n, seq_len = input_ids.shape

        upstream_mod = sys.modules[type(transformer).__module__]
        to_device = upstream_mod.to_device
        rope_image_info = transformer.build_batch_rope_image_info(output, sections)
        cos, sin = upstream_mod.build_batch_2d_rope(
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

        vae_mask = getattr(output, "vae_image_mask", None)
        vit_mask = getattr(output, "vit_image_mask", None)
        return HunyuanImage3FusedMultimodalCondition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            # Stack RoPE so it travels as a per-sample CONCAT tensor.
            rope_cache=torch.stack([cos, sin], dim=1),
            gen_image_mask=to_device(gen_image_mask, device),
            gen_timestep_scatter_index=to_device(gen_timestep_scatter_index, device),
            cond_vae_image_mask=to_device(
                getattr(output, "cond_vae_image_mask", None) if vae_mask is None else vae_mask, device
            ),
            cond_vit_image_mask=to_device(
                getattr(output, "cond_vit_image_mask", None) if vit_mask is None else vit_mask, device
            ),
            cond_timestep_scatter_index=to_device(output.cond_timestep_scatter_index, device),
            prompt_lengths=to_device(prompt_lengths, device),
        )

    def embed_for_ar(
        self,
        p: Texts,
        *,
        bot_task: str = "auto",
        system_prompt: Optional[List[str]] = None,
        cot_text: Optional[List[str]] = None,
        batch_cond_image_info: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build the unified-MM input tensors for ``mode="gen_text"``."""
        gen_config = self.bundle.transformer.generation_config

        output, sections = self._apply_chat_template(
            mode="gen_text",
            batch_prompt=p.texts,
            bot_task=bot_task,
            cfg_factor=1,
            batch_gen_image_info=None,
            batch_system_prompt=system_prompt,
            batch_cot_text=cot_text,
            batch_cond_image_info=batch_cond_image_info,
        )

        fused = self._build_fused(
            output,
            sections,
            rope_seq_len=max(gen_config.max_length, output.tokens.shape[1]),
            prompt_lengths=output.real_pos.reshape(-1),
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

        prompts = p.texts
        if not prompts:
            raise ValueError("HunyuanImage3TextEmbedStage.embed_for_gen_image: prompts is empty")
        cfg_factor = 2 if cfg else 1
        if system_prompt is None and sys_type is not None:
            resolved = sys.modules[type(transformer).__module__].get_system_prompt(sys_type, bot_task)
            if resolved is not None and (stripped := resolved.strip()):
                system_prompt = [stripped] * len(prompts)

        ip = transformer.image_processor
        image_size = f"{height}x{width}"
        if hasattr(ip, "build_image_info"):
            image_info = ip.build_image_info(image_size)
        elif hasattr(ip, "build_gen_image_info"):
            image_info = ip.build_gen_image_info(image_size)
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

        fused = self._build_fused(
            output,
            sections,
            gen_image_mask=output.gen_image_mask,
            gen_timestep_scatter_index=output.gen_timestep_scatter_index,
        )
        return {"fused": fused, "tokenizer_output": output}


__all__ = ["HunyuanImage3TextEmbedStage"]
