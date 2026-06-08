"""HunyuanImage3TextEmbedStage — chat-template-driven AR input prep.

Wraps the upstream ``_tkwrapper.apply_chat_template`` to build the
unified-multimodal input tensors for ``mode="gen_text"`` (the AR path).
``embed_for_ar`` is the canonical entry point: it produces ``input_ids``,
the 4D ``attention_mask``, ``position_ids``, mRoPE rope tables, and the
optional ``cond_vit_image_mask`` for i2t / it2i prefix passes.

HunyuanImage 3.0 has no separate text encoder — it's a unified-vocab MoE
model where text tokens share the same embedding table as image-vocab
tokens. The chat-template wrapper is what makes ``bot_task ∈ {auto,
image, think, recaption, think_recaption, img_ratio}`` produce visibly
different generations: the wrapper splices in ``<bot_task>`` markers
and / or ``<boi> <img_ratio_X> <img> <timestep> <eoi>`` blocks per the
selected preset (see vllm-omni ``prompt_utils.py:23-31``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.types.primitives import Texts

from .bundle import HunyuanImage3Bundle
from .conditions import HunyuanImage3FusedMultimodalCondition


class HunyuanImage3TextEmbedStage:
    """HunyuanImage3 chat-template-driven AR input-prep stage."""

    def __init__(
        self,
        bundle: HunyuanImage3Bundle,
        *,
        max_sequence_length: int = 1024,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = max_sequence_length

    # ------------------------------------------------------------------
    # Chat-template-driven input prep — canonical AR entry point.
    # ------------------------------------------------------------------

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
        """Build the unified-MM input tensors for ``mode="gen_text"``.

        Mirrors the prefill input-prep half of upstream
        ``HunyuanImage3ForCausalMM._generate(mode="gen_text")``: runs
        ``_tkwrapper.apply_chat_template(mode="gen_text")`` to splice the
        prompt into the chat template under the selected ``bot_task``
        preset, then derives the 4D causal+image-bidirectional
        ``attention_mask``, the ``[B, L]`` ``position_ids``, and the
        per-position mRoPE rope tables ``(cos, sin)``.

        Args:
            p: Texts primitive carrying B prompt strings.
            bot_task: Chat-template flag — one of ``{"auto", "image",
                "think", "recaption", "think_recaption", "img_ratio"}``.
                Drives stop-token selection downstream and (for ``think``
                / ``recaption``) splices an extra reasoning marker.
            system_prompt: Optional per-sample system prompts (length B).
            cot_text: Optional per-sample chain-of-thought primer
                (length B).
            max_length: Cap on the templated sequence length passed to
                the wrapper (None = wrapper default).
            batch_message_list: Optional per-sample message-list shape
                (used by i2t / it2i to embed ``<img>`` markers from
                pre-encoded image info). Mutually exclusive with the
                bare ``p.texts`` prompt path.
            batch_cond_image_info: Optional per-sample list of
                ``JointImageInfo`` for cond-image marker insertion. Pre-
                computed by ``HunyuanImage3VitEncodeStage.encode_for_cond_vit``
                and passed straight through to the chat-template wrapper
                so the resulting ``input_ids`` / ``cond_vit_image_mask``
                pin the right slots.

        Returns:
            Dict with the following keys (let ``B = len(p.texts)``,
            ``L = output.tokens.shape[1]``, ``D = head_dim``):

                fused           : HunyuanImage3FusedMultimodalCondition
                                  carries input_ids ``[B, L] long``,
                                  attention_mask ``[B, 1, L, L] bool``,
                                  position_ids ``[B, L] long``,
                                  rope_cache ``(cos, sin)`` each ``[B, L, D] float``,
                                  cond_vit_image_mask ``[B, L] bool`` (i2t / it2i;
                                  ``None`` for t2t).
                tokenizer_output: opaque upstream apply_chat_template output (carries
                                  ``real_pos`` etc. for the prefill
                                  ``_update_model_kwargs_for_generation`` hook).
        """
        import sys

        bundle = self.bundle
        transformer = bundle.transformer
        config = transformer.config
        gen_config = transformer.generation_config

        prompts = list(p.texts) if batch_message_list is None else None
        batch_size = (
            len(prompts) if prompts is not None else len(batch_message_list)  # type: ignore[arg-type]
        )

        # Ensure the tokenizer wrapper is loaded -- AutoModelForCausalLM
        # leaves ``_tkwrapper`` as None until ``load_tokenizer`` is called.
        if getattr(transformer, "_tkwrapper", None) is None:
            transformer.load_tokenizer(bundle.tokenizer)

        out = transformer._tkwrapper.apply_chat_template(
            batch_prompt=prompts,
            batch_message_list=batch_message_list,
            mode="gen_text",
            batch_gen_image_info=None,
            batch_cond_image_info=batch_cond_image_info,
            batch_system_prompt=system_prompt,
            batch_cot_text=cot_text,
            max_length=max_length,
            bot_task=bot_task,
            image_base_size=config.image_base_size,
            sequence_template=gen_config.sequence_template,
            cfg_factor=1,
            drop_think=gen_config.drop_think,
        )
        output, sections = out["output"], out["sections"]

        # Anchor every tensor to the ``wte`` device — under ``device_map="auto"``
        # this is typically cuda:0; HF hooks shuttle activations downstream.
        device = transformer.model.wte.weight.device

        input_ids: torch.Tensor = output.tokens.to(device)  # [B, L] long
        prompt_len: int = int(input_ids.shape[1])

        # mRoPE rope tables. Upstream (hunyuan.py:2306-2310) sizes the rope
        # to ``generation_config.max_length`` for ``mode="gen_text"`` so
        # decode steps' position_ids (which advance past the prompt) stay
        # in range. ``rope_image_info`` is empty for every sample in
        # gen_text -- there are no <img> sections.
        rope_seq_len = int(getattr(gen_config, "max_length", prompt_len))
        rope_seq_len = max(rope_seq_len, prompt_len)
        rope_image_info = transformer.build_batch_rope_image_info(output, sections)
        upstream_mod = sys.modules[type(transformer).__module__]
        build_batch_2d_rope = upstream_mod.build_batch_2d_rope
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=rope_seq_len,
            n_elem=config.attention_head_dim,
            device=device,
            base=config.rope_theta,
        )  # ([B, max_length, D], [B, max_length, D])

        position_ids: torch.Tensor = torch.arange(0, prompt_len, dtype=torch.long, device=device)[None].expand(
            batch_size, -1
        )  # [B, prompt_len] long

        attention_mask: torch.Tensor = transformer._prepare_attention_mask_for_generation(
            input_ids,
            gen_config,
            model_kwargs={"tokenizer_output": output},
        ).to(device)  # [B, 1, L, L] bool

        # When the wrapper saw cond images, ``output`` carries the
        # ``cond_vit_image_mask`` pin-points where ``<img>`` tokens land
        # in the input_ids. The unified-multimodal forward consumes this
        # to scatter ViT patch embeds into ``inputs_embeds`` via
        # ``instantiate_vit_image_tokens``.
        cond_vit_image_mask = getattr(output, "cond_vit_image_mask", None)
        if cond_vit_image_mask is not None:
            cond_vit_image_mask = cond_vit_image_mask.to(device)

        fused = HunyuanImage3FusedMultimodalCondition(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_cache=(cos, sin),
            cond_vit_image_mask=cond_vit_image_mask,
        )
        return {"fused": fused, "tokenizer_output": output}


__all__ = ["HunyuanImage3TextEmbedStage"]
