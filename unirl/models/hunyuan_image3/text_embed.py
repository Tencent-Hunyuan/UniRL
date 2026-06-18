"""HunyuanImage3TextEmbedStage — chat-template-driven input prep.

Wraps the upstream ``_tkwrapper.apply_chat_template`` to build the
unified-multimodal input tensors for both backbone modes:
:meth:`embed_for_ar` (``mode="gen_text"``, the t2t / i2t AR path) and
:meth:`embed_for_gen_image` (``mode="gen_image"``, the t2i / it2i
diffusion path). Each produces ``input_ids``, the 4D
causal+image-bidirectional ``attention_mask``, ``position_ids``, mRoPE
rope tables, and the mode's scatter masks.

HunyuanImage 3.0 has no separate text encoder — it's a unified-vocab MoE
model where text tokens share the same embedding table as image-vocab
tokens. The chat-template wrapper is what makes ``bot_task ∈ {auto,
image, think, recaption, think_recaption, img_ratio}`` produce visibly
different generations: the wrapper splices in ``<bot_task>`` markers
and / or ``<boi> <img_ratio_X> <img> <timestep> <eoi>`` blocks per the
selected preset (see vllm-omni ``prompt_utils.py:23-31``).
"""

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
    """Locate the upstream ``build_batch_2d_rope`` rope helper.

    FSDP2's ``fully_shard`` rebinds ``type(transformer).__module__`` to
    ``torch.distributed.fsdp._fully_shard._fully_shard``, so we can't
    index ``sys.modules`` by that key. Walk ``sys.modules`` for the
    original trust_remote_code module that owns ``build_batch_2d_rope``.
    (Caveat, same as before this helper was shared: with two different
    HI3 checkpoints loaded in one process the walk picks the first hit.)
    """
    for _name, _mod in sys.modules.items():
        if _name.startswith("transformers_modules.") and hasattr(_mod, "build_batch_2d_rope"):
            return _mod.build_batch_2d_rope
    raise RuntimeError(
        "HunyuanImage3TextEmbedStage: could not locate build_batch_2d_rope "
        "in any transformers_modules.* — was AutoModelForCausalLM.from_pretrained "
        "called with trust_remote_code=True before bundle construction?"
    )


def _optional_output_tensor(output: Any, names: Tuple[str, ...], device: torch.device) -> Optional[torch.Tensor]:
    """First non-None attribute of ``output`` among ``names``, moved to
    ``device``; None if absent. Attr names differ across HI3 checkpoint
    snapshots (base: ``cond_vit_image_mask`` / ``cond_vae_image_mask``;
    Instruct: ``vit_image_mask`` / ``vae_image_mask``)."""
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

    # ------------------------------------------------------------------
    # Shared input-prep internals.
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        *,
        mode: str,
        batch_prompt: Optional[List[str]],
        bot_task: str,
        cfg_factor: int,
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

        # The wrapper around the HF tokenizer (which knows how to splice in
        # <boi>, <eoi>, <img>, <timestep>, <img_ratio_*> markers) is lazily
        # populated upstream — ``load_tokenizer`` must be called explicitly
        # after ``from_pretrained``. It resolves its arg as a path
        # (from_pretrained), so pass the checkpoint path, not the tokenizer
        # object. Newer (Instruct) snapshots expose the wrapper as
        # ``_tokenizer``; older ones auto-populate ``_tkwrapper``.
        if getattr(transformer, "_tkwrapper", None) is None and getattr(transformer, "_tokenizer", None) is None:
            transformer.load_tokenizer(self.bundle.pretrained_path)
        tkw = getattr(transformer, "_tkwrapper", None) or getattr(transformer, "_tokenizer", None)
        # transformers 5.x loads HunyuanImage3TokenizerFast's Rust backend
        # char-level (pre_tokenizer/decoder=None) -> char-level, space-less
        # prompts -> char-by-char generation. Re-attach the correct BPE backend
        # from tokenizer.json. Idempotent (no-op once repaired). See compat.py.
        repair_hi3_tokenizer_backend(tkw, self.bundle.pretrained_path)

        # Cond-image kwarg name differs by checkpoint snapshot (base:
        # batch_cond_image_info, Instruct: batch_cond_images).
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
            sequence_template=gen_config.sequence_template,
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
        Tuple[torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Tensor prep shared by the AR and gen_image paths.

        Returns ``(device, input_ids, attention_mask, position_ids,
        rope_cache, cond_vit_image_mask)``. ``rope_seq_len=None`` sizes
        the rope tables to the templated sequence length (gen_image
        semantics); the AR path passes ``generation_config.max_length``
        so decode steps' position_ids stay in range.
        """
        transformer = self.bundle.transformer
        config = transformer.config
        gen_config = transformer.generation_config

        # Anchor every tensor to the ``wte`` device — under ``device_map="auto"``
        # this is typically cuda:0; HF hooks shuttle activations downstream.
        device = transformer.model.wte.weight.device

        input_ids: torch.Tensor = output.tokens.to(device)  # [N, L] long
        n, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])

        # mRoPE rope tables: (cos, sin), each [N, rope_seq_len, head_dim] float.
        rope_image_info = transformer.build_batch_rope_image_info(output, sections)
        build_batch_2d_rope = _resolve_build_batch_2d_rope()
        cos, sin = build_batch_2d_rope(
            image_infos=rope_image_info,
            seq_len=seq_len if rope_seq_len is None else rope_seq_len,
            n_elem=config.attention_head_dim,
            device=device,
            base=config.rope_theta,
        )

        # Position ids share across batch via expand to save memory: [N, L] long.
        position_ids: torch.Tensor = torch.arange(0, seq_len, dtype=torch.long, device=device)[None].expand(n, -1)

        # 4D causal+image-bidirectional attention mask: [N, 1, L, L] bool.
        attention_mask: torch.Tensor = transformer._prepare_attention_mask_for_generation(
            input_ids,
            gen_config,
            model_kwargs={"tokenizer_output": output},
        ).to(device)

        # When the wrapper saw cond images, ``output`` carries the mask that
        # pins where <img> tokens land in input_ids. The unified-MM forward
        # consumes this to scatter ViT patch embeds into ``inputs_embeds``
        # via ``instantiate_vit_image_tokens``.
        cond_vit_image_mask = _optional_output_tensor(output, ("cond_vit_image_mask", "vit_image_mask"), device)

        return device, input_ids, attention_mask, position_ids, (cos, sin), cond_vit_image_mask

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

        # Upstream (hunyuan.py:2306-2310) sizes the rope to
        # ``generation_config.max_length`` for ``mode="gen_text"`` so decode
        # steps' position_ids (which advance past the prompt) stay in range.
        # ``rope_image_info`` is empty for every sample in gen_text -- there
        # are no <img> sections.
        prompt_len = int(output.tokens.shape[1])
        rope_seq_len = int(getattr(gen_config, "max_length", prompt_len))
        rope_seq_len = max(rope_seq_len, prompt_len)

        _device, input_ids, attention_mask, position_ids, rope_cache, cond_vit_image_mask = self._fused_common(
            output, sections, rope_seq_len=rope_seq_len
        )

        # HI3-Instruct cond images are dual-encoded: the wrapper also splices
        # VAE <img> slots + a cond <timestep> token (i2t/it2i). Pin them so the
        # AR forward can scatter the VAE latents (else those slots stay bare
        # <img> embeddings → garbage comprehension). None for t2t (no cond image).
        cond_vae_image_mask = _optional_output_tensor(output, ("cond_vae_image_mask", "vae_image_mask"), _device)
        cond_timestep_scatter_index = _optional_output_tensor(output, ("cond_timestep_scatter_index",), _device)

        # Per-sample TRUE prompt length for the right-padded batch. The upstream
        # tokenizer right-pads a mixed-length batch to ``max_len`` and records the
        # real end in ``real_pos`` (one-past-last-valid → == prompt length, the
        # same quantity the rollout prefill reads at ``real_pos - 1``). Carry it on
        # ``fused`` so trainside ``replay`` slices off the right-pad — the same
        # ``prompt_lengths`` contract the two-engine adapter (adapters/hi3.py)
        # fills. Without it, ``replay`` would fall back to the padded length and
        # compute per-token logp at pad-shifted positions for short samples →
        # silent GRPO ratio error.
        prompt_lengths: Optional[torch.Tensor] = None
        real_pos = getattr(output, "real_pos", None)
        if real_pos is not None:
            rp = real_pos.to(device=_device, dtype=torch.long)
            if rp.dim() == 2:
                rp = rp[:, -1]
            prompt_lengths = rp.reshape(-1)  # [B]

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

    # ------------------------------------------------------------------
    # Chat-template-driven input prep — t2i / it2i diffusion entry point.
    # ------------------------------------------------------------------

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
        batch_cond_image_info: Optional[List[List[Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the unified-MM input tensors for ``mode="gen_image"``.

        Mirrors the input-prep half of ``HunyuanImage3ForCausalMM._generate``
        (upstream ``hunyuan.py`` ~lines 2200–2380): runs the tokenizer
        wrapper to splice prompt + ``<boi>`` + ``<img_ratio_X>`` + ``<img>``
        block + ``<timestep>`` + ``<eoi>`` into ``input_ids``, builds the
        4D causal+image-bidirectional ``attention_mask``, the per-token
        ``position_ids``, and the per-position rope tables ``(cos, sin)``.

        KV cache is intentionally NOT built — the unirl per-step
        kernel calls ``transformer(..., past_key_values=None,
        use_cache=False, first_step=True)`` every diffusion step.

        Args:
            p: Texts primitive carrying B prompt strings.
            cfg:
                Classifier-free-guidance flag; callers derive it from
                ``guidance_scale > 1.0``. When True, all returned tensors
                are batched ``[cond, uncond]`` along axis 0 (cond first,
                matching upstream ``HunyuanImage3Text2ImagePipeline.__call__``
                at ``hunyuan_image_3_pipeline.py:830``). The unconditional
                branch is built internally by the wrapper as a
                ``<cfg>``-token sequence — HunyuanImage3 never consumes
                negative-prompt text.
            height, width:
                Target image size in pixels. Snapped to the closest
                preset ratio by the upstream ``image_processor``.
            bot_task:
                Chat-template flag selecting the AR-prefix preset baked
                into ``input_ids``. One of ``{"image", "auto", "think",
                "recaption", "think_recaption", "img_ratio"}``. Default
                ``"image"`` matches vllm-omni's ``t2i_vanilla`` preset
                (no prefix marker). ``"think"`` / ``"recaption"`` insert
                a static ``<think>`` / ``<recaption>`` marker after the
                ``Assistant:`` system prompt — the model treats this as
                a reasoning-mode signal during the diffusion forward.
                Per vllm-omni ``prompt_utils.py:23-31``.
            cot_text:
                Optional per-sample chain-of-thought text (length B,
                NOT B*cfg — the wrapper duplicates internally and drops
                the CoT to ``<cfg>`` tokens on the uncond branch).
                Entries should carry literal ``<think>…</think>`` /
                ``<recaption>…</recaption>`` tag pairs so the wrapper's
                section parsing works (t2ti's AR phase produces these).
            system_prompt:
                Optional per-sample system prompts (length B), spliced
                ahead of the user prompt — t2ti passes the same resolved
                system prompt used for its AR phase.
            batch_cond_image_info:
                Optional per-sample list of ``JointImageInfo`` for
                cond-image marker insertion (it2i).

        Returns:
            Dict with the following keys:
                fused           : HunyuanImage3FusedMultimodalCondition
                                  carries input_ids ``[N, L] long``,
                                  attention_mask ``[N, 1, L, L] bool``,
                                  position_ids ``[N, L] long``,
                                  rope_cache ``(cos, sin)`` ``([N, L, D], [N, L, D]) float``,
                                  gen_image_mask ``[N, L] bool``,
                                  gen_timestep_scatter_index ``[N, K] long``,
                                  cond_vae_image_mask / cond_vit_image_mask /
                                  cond_timestep_scatter_index (``None`` for vanilla t2i;
                                  set when ``batch_cond_image_info`` is passed).
                tokenizer_output: opaque upstream apply_chat_template output (used by
                                  the KV-cache path's first ``_update_model_kwargs``
                                  call to gather down).

            where ``N = len(p.texts) * (2 if cfg else 1)`` and
            ``L = output.tokens.shape[1]``.

        All tensors live on the embedding-layer device of the bundle's
        transformer (under ``device_map="auto"`` this is typically cuda:0).
        """
        transformer = self.bundle.transformer

        prompts = list(p.texts)
        if not prompts:
            raise ValueError("HunyuanImage3TextEmbedStage.embed_for_gen_image: prompts is empty")
        cfg_factor = 2 if cfg else 1

        # Image info from explicit (h, w). Upstream's image_processor
        # snaps to the closest preset ratio. The method name differs
        # across HI3 checkpoints: Base ships ``build_image_info``,
        # Instruct ships ``build_gen_image_info`` (same semantics, two
        # default kwargs we don't need).
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

        # Tokenize + splice in special markers (<boi>, <img>, <timestep>,
        # <eoi>, ratio, plus cond-image <img> blocks for it2i). With
        # cfg_factor=2, the wrapper internally duplicates the prompt slot
        # for the unconditional branch (cond first).
        output, sections = self._apply_chat_template(
            mode="gen_image",
            batch_prompt=prompts,
            bot_task=bot_task,
            cfg_factor=cfg_factor,
            batch_gen_image_info=batch_gen_image_info,
            batch_system_prompt=system_prompt,
            batch_cot_text=cot_text,
            batch_cond_image_info=batch_cond_image_info,
        )

        device, input_ids, attention_mask, position_ids, rope_cache, cond_vit_image_mask = self._fused_common(
            output, sections
        )

        # gen_image_mask: [N, L] bool — positions of generated-image patches.
        gen_image_mask: torch.Tensor = output.gen_image_mask.to(device)
        # gen_timestep_scatter_index: [N, K] long (K is small, index of <timestep> tokens)
        gen_timestep_scatter_index: torch.Tensor = output.gen_timestep_scatter_index.to(device)

        # When ``batch_cond_image_info`` was passed, the wrapper emits
        # cond-image position pin-points and the cond-timestep scatter
        # index. ``None`` for vanilla t2i.
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
        # Opaque tokenizer wrapper output. Carries the slice info the
        # KV-cache path's first ``_update_model_kwargs_for_generation``
        # call needs to gather position_ids / attention_mask /
        # gen_timestep_scatter_index down from full-L to the L' changed
        # slice (timestep + image tokens) for steps 1..T-1.
        return {"fused": fused, "tokenizer_output": output}


__all__ = ["HunyuanImage3TextEmbedStage"]
