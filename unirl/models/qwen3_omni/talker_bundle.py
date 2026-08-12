"""Standalone Qwen3-Omni Talker bundle for direct TTS.

Unlike the Thinker bundle, this path never constructs or executes the full
Thinker. It keeps only the frozen Thinker token embedding table needed by the
text-only direct-TTS prefix, the trainable Talker (including MTP), and the
frozen Code2Wav decoder.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class FrozenThinkerEmbeddingProvider(nn.Module):
    """The only Thinker component direct TTS needs: its input embedding table."""

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        """Compatibility with the full Thinker's embedding accessor."""
        return self.embedding


def _checkpoint_shards(weights_dir: str) -> list[str]:
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"Qwen3OmniTalkerBundle: checkpoint directory not found: {weights_dir!r}"
        )
    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"Qwen3OmniTalkerBundle: no *.safetensors files under {weights_dir!r}"
        )
    return shards


def _iter_prefixed_checkpoint_tensors(
    weights_dir: str,
    *,
    key_prefix: str,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield only ``key_prefix`` tensors, one at a time, with the prefix stripped."""
    from safetensors import safe_open

    matched = 0
    for shard in _checkpoint_shards(weights_dir):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                if not source_key.startswith(key_prefix):
                    continue
                target_key = source_key[len(key_prefix) :]
                if not target_key:
                    continue
                matched += 1
                yield target_key, handle.get_tensor(source_key)
    if matched == 0:
        raise ValueError(
            f"Qwen3OmniTalkerBundle: no checkpoint tensors matched prefix "
            f"{key_prefix!r} under {weights_dir!r}"
        )


def _stream_load_prefixed_module(
    module: nn.Module,
    weights_dir: str,
    *,
    key_prefix: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Materialize a meta module from one composite-checkpoint subtree.

    Tensors are mmap-read and assigned one at a time, so loading the frozen
    embedding provider or Code2Wav never materializes the full Omni state dict.
    """
    from accelerate.utils.modeling import set_module_tensor_to_device

    expected = set(module.state_dict().keys())
    loaded: set[str] = set()
    unexpected: list[str] = []
    for target_key, value in _iter_prefixed_checkpoint_tensors(
        weights_dir,
        key_prefix=key_prefix,
    ):
        if target_key not in expected:
            unexpected.append(target_key)
            continue
        target_dtype = dtype if value.is_floating_point() else value.dtype
        set_module_tensor_to_device(
            module,
            target_key,
            device,
            value=value,
            dtype=target_dtype,
        )
        loaded.add(target_key)

    meta_params = [name for name, param in module.named_parameters() if param.is_meta]
    if any(".experts.gate_up_proj" in name or ".experts.down_proj" in name for name in meta_params):
        # HF stores Qwen3-Omni experts per expert/projection, while the runtime
        # owns stacked 3-D parameters. The streaming pass above intentionally
        # skipped those source-only keys; reconstruct only the still-meta fused
        # destinations before validating materialization.
        from unirl.train.backend.sharded_load import (
            _fuse_hf_expert_keys,
            _read_safetensors_dir,
        )

        expert_state = _read_safetensors_dir(weights_dir, key_prefix=key_prefix)
        expert_state = _fuse_hf_expert_keys(expert_state, module)
        for name in list(meta_params):
            value = expert_state.get(name)
            if not isinstance(value, torch.Tensor):
                continue
            set_module_tensor_to_device(
                module,
                name,
                device,
                value=value,
                dtype=dtype if value.is_floating_point() else value.dtype,
            )
            loaded.add(name)
        del expert_state

    # HF may omit a tied destination (for Talker, codec_head) from the
    # checkpoint. Re-establish ties before checking for unmaterialized params.
    if getattr(module, "_tied_weights_keys", None) and hasattr(module, "tie_weights"):
        module.tie_weights()

    meta_params = [name for name, param in module.named_parameters() if param.is_meta]
    if meta_params:
        raise RuntimeError(
            f"Qwen3OmniTalkerBundle: {len(meta_params)} parameter(s) remained on "
            f"meta after loading prefix {key_prefix!r}: {meta_params[:8]}"
        )
    if not loaded:
        raise RuntimeError(
            f"Qwen3OmniTalkerBundle: prefix {key_prefix!r} did not load any "
            f"state-dict keys for {type(module).__name__}"
        )
    if unexpected:
        logger.debug(
            "Ignored %d checkpoint key(s) below %s not owned by %s: %s",
            len(unexpected),
            key_prefix,
            type(module).__name__,
            unexpected[:6],
        )

    # Parameters are already on ``device``; this moves init-computed,
    # non-persistent buffers (RoPE/Code2Wav offsets) without a second model copy.
    module.to(device=device)


def _build_frozen_embedding_provider(
    full_config: Any,
    *,
    weights_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> FrozenThinkerEmbeddingProvider:
    from accelerate import init_empty_weights

    text_config = full_config.thinker_config.text_config
    with init_empty_weights(include_buffers=False):
        embedding = nn.Embedding(
            int(text_config.vocab_size),
            int(text_config.hidden_size),
            getattr(text_config, "pad_token_id", None),
        )
    _stream_load_prefixed_module(
        embedding,
        weights_dir,
        key_prefix="thinker.model.embed_tokens.",
        device=device,
        dtype=dtype,
    )
    provider = FrozenThinkerEmbeddingProvider(embedding)
    provider.eval()
    return provider


class Qwen3OmniTalkerBundle(Bundle):
    """Direct-TTS bundle: frozen input embeddings + Talker/MTP + Code2Wav."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        input_embedding_provider: FrozenThinkerEmbeddingProvider,
        code2wav: nn.Module,
        processor: Any,
        tokenizer: Any,
        config: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        default_speaker: str = "Ethan",
        tts_system_instruction: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.input_embedding_provider = input_embedding_provider
        self._code2wav = code2wav
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.enable_talker = True
        self.default_speaker = str(default_speaker)
        self.tts_system_instruction = tts_system_instruction

    @property
    def talker(self) -> nn.Module:
        return self.transformer

    @property
    def code2wav(self) -> nn.Module:
        return self._code2wav

    @property
    def thinker(self) -> FrozenThinkerEmbeddingProvider:
        """Compatibility accessor; this is not a full Thinker model."""
        return self.input_embedding_provider

    @property
    def omni(self) -> "Qwen3OmniTalkerBundle":
        """Legacy compatibility facade for callers that only access Omni parts."""
        return self

    @classmethod
    def from_config(cls, config: Any) -> "Qwen3OmniTalkerBundle":
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeCode2Wav,
            Qwen3OmniMoeTalkerForConditionalGeneration,
        )

        path = config.pretrained_model_ckpt_path
        tokenizer_path = getattr(config, "tokenizer_ckpt_path", None) or path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        full_config = AutoConfig.from_pretrained(
            path,
            trust_remote_code=bool(getattr(config, "trust_remote_code", True)),
        )
        if not getattr(full_config, "enable_audio_output", False):
            raise ValueError(
                "Qwen3OmniTalkerBundle requires a full Omni checkpoint with "
                "enable_audio_output=true"
            )
        if getattr(config, "attn_implementation", None):
            full_config.talker_config._attn_implementation = str(config.attn_implementation)

        talker_factory = lambda: Qwen3OmniMoeTalkerForConditionalGeneration(
            full_config.talker_config
        )
        meta_init = bool(getattr(config, "meta_init_transformer", False))
        if meta_init:
            talker, meta_init_state = build_meta_init_transformer(
                talker_factory,
                dtype=dtype,
            )
        else:
            with init_empty_weights(include_buffers=False):
                talker = talker_factory()
            _stream_load_prefixed_module(
                talker,
                path,
                key_prefix="talker.",
                device=device,
                dtype=dtype,
            )

        if getattr(config, "use_gradient_checkpointing", False):
            if hasattr(talker, "gradient_checkpointing_enable"):
                talker.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                logger.warning(
                    "Talker %s does not expose gradient_checkpointing_enable; skipping.",
                    type(talker).__name__,
                )

        freeze_mtp = bool(getattr(config, "freeze_mtp", False))
        freeze_talker_embeddings = bool(getattr(config, "freeze_talker_embeddings", True))
        phase1_layer0_lora_only = bool(getattr(config, "phase1_layer0_lora_only", False))
        if phase1_layer0_lora_only:
            # Freeze before PEFT injection.  Newly injected layer-0 LoRA tensors
            # become trainable; a post-injection guard in unirl.train.lora
            # re-freezes excluded subtrees and verifies the invariant.
            talker.requires_grad_(False)
            setattr(talker, "_unirl_phase1_layer0_lora_only", True)
        if freeze_mtp:
            talker.code_predictor.requires_grad_(False)
            setattr(talker, "_unirl_freeze_mtp_after_lora", True)
        if freeze_talker_embeddings:
            get_embeddings = getattr(talker, "get_input_embeddings", None)
            if not callable(get_embeddings):
                if phase1_layer0_lora_only:
                    raise TypeError("Phase-1 Talker must expose get_input_embeddings()")
                logger.warning("Talker %s has no input embedding accessor; cannot freeze it.", type(talker).__name__)
            else:
                get_embeddings().requires_grad_(False)
                setattr(talker, "_unirl_freeze_embeddings_after_lora", True)

        embedding_provider = _build_frozen_embedding_provider(
            full_config,
            weights_dir=path,
            device=device,
            dtype=dtype,
        )
        with init_empty_weights(include_buffers=False):
            code2wav = Qwen3OmniMoeCode2Wav(full_config.code2wav_config)
        _stream_load_prefixed_module(
            code2wav,
            path,
            key_prefix="code2wav.",
            device=device,
            dtype=dtype,
        )
        if bool(getattr(config, "freeze_code2wav", True)):
            code2wav.requires_grad_(False)
            code2wav.eval()

        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(getattr(config, "trust_remote_code", True)),
        )
        if tokenizer_path != path:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=bool(getattr(config, "trust_remote_code", True)),
            )
            if hasattr(processor, "tokenizer"):
                processor.tokenizer = tokenizer
        else:
            tokenizer = getattr(processor, "tokenizer", None) or processor
        if (
            getattr(tokenizer, "pad_token", None) is None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=talker,
            input_embedding_provider=embedding_provider,
            code2wav=code2wav,
            processor=processor,
            tokenizer=tokenizer,
            config=full_config,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            default_speaker=str(getattr(config, "default_speaker", "Ethan")),
            tts_system_instruction=getattr(config, "tts_system_instruction", None),
        )
        if meta_init:
            # Pattern B: the backend wraps/shards the standalone Talker, then
            # mmap-reads only ``talker.*`` tensors from the full Omni checkpoint.
            bundle._transformer_weights_path = path
            bundle._transformer_weights_key_prefix = "talker."
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = [
    "FrozenThinkerEmbeddingProvider",
    "Qwen3OmniTalkerBundle",
]
