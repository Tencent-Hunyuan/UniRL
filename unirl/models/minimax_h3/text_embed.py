"""MiniMax-H3 text embedding stage -- Qwen3-VL layer-50 hidden states."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import tempfile
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from functools import cache
from typing import TYPE_CHECKING, Any, Iterator, List, Tuple

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts
from unirl.utils.run_id import resolve_run_id

from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

logger = logging.getLogger(__name__)
_EMBED_SYNC_TIMEOUT = timedelta(minutes=30)
# One entry is [1, tokens, hidden] in the encoder's own dtype — a few MB for a
# typical prompt, so this bounds the CPU-resident cache at a few hundred MB.
_PROMPT_CACHE_SIZE = 64


def truncate_minimax_h3_text_encoder(text_encoder: torch.nn.Module) -> torch.nn.Module:
    """Cut the Qwen3-VL decoder back to the layer MiniMax-H3 conditions on."""
    model = text_encoder.model
    decoder = getattr(model, "language_model", model)
    num_layers = len(decoder.layers)
    require(
        num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
        f"truncate_minimax_h3_text_encoder: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
        f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
        f"the loaded conditioner has {num_layers}.",
    )
    # hidden_states[i] is decoder layer i's raw output, except that transformers
    # swaps the LAST entry for the normed last_hidden_state. Dropping the final
    # norm with the tail makes last_hidden_state exactly the old hidden_states[50].
    del decoder.layers[MINIMAX_H3_TEXT_ENCODER_LAYER:]
    decoder.norm = torch.nn.Identity()
    return text_encoder


def load_minimax_h3_conditioner(
    path: str, dtype: torch.dtype, *, nested: bool = True
) -> Tuple[torch.nn.Module, Any, Any]:
    """Load the frozen, truncated Qwen3-VL conditioner with its processor and tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    def subfolder(name: str) -> str | None:
        if nested:
            return name
        return name if os.path.isdir(os.path.join(path, name)) else None

    # H3 reads an intermediate hidden state, so this must be the full LM class,
    # not a truncated encoder; the layers past that state are dropped after load.
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        path, subfolder=subfolder("text_encoder"), torch_dtype=dtype
    )
    text_encoder = truncate_minimax_h3_text_encoder(text_encoder).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(path, subfolder=subfolder("processor"))
    tokenizer = AutoTokenizer.from_pretrained(path, subfolder=subfolder("tokenizer"))
    return text_encoder, processor, tokenizer


@torch.no_grad()
def encode_minimax_h3_prompt(
    *,
    text_encoder: torch.nn.Module,
    tokenizer: Any,
    processor: Any,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """Return layer-50 hidden states ``[1, L, D]`` from a truncated conditioner."""
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    # Qwen3-VL lays its 3D rotary positions out per modality run, read off the
    # token type ids the processor derives (0 text, 1 image, 2 video).
    # Text-only here, but the conditioner still wants them.
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )
    return outputs.last_hidden_state


@cache
def _residency_lock_path() -> str:
    """Return a per-run, per-user node-local conditioner lock path."""
    digest = hashlib.sha256(f"{os.getuid()}:{resolve_run_id()}".encode()).hexdigest()[:16]
    # Left behind on purpose: the file is empty and node-local, and unlinking it
    # would race the sibling ranks still holding the lock through it.
    return os.path.join(tempfile.gettempdir(), f"unirl_minimax_h3_onload_{digest}.lock")


@contextmanager
def _serialize_node_residency() -> Iterator[None]:
    """Admit one rank at a time to the conditioner residency window on this node."""
    if os.environ.get("UNIRL_MINIMAX_H3_ONLOAD_SERIALIZE", "1") == "0":
        yield
        return

    with open(_residency_lock_path(), "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class MiniMaxH3TextEmbedStage:
    """Encode prompts into the conditioning MiniMax-H3 was trained on."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.dtype = bundle.dtype
        self.device = bundle.device
        self.vae = bundle.vae
        self.audio_vae = bundle.audio_vae
        self._store = bundle.text_embed_store
        self._onload_for_embed = bundle.text_encoder_onload_for_embed
        # The encoder is frozen and prompt-only. Trainside forward_batch_size=1
        # invokes pipeline.generate once per sibling sample, so cache on CPU to
        # avoid repeating a 32B conditioner forward for the same prompt.
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._embedding_sync_group = None

    @property
    def _encoder_device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def _pack_embeds(self, embeds: List[torch.Tensor]) -> TextEmbedCondition:
        lengths = {int(tensor.shape[1]) for tensor in embeds}
        require(
            len(lengths) == 1,
            f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed sequence "
            f"geometry must be identical across the batch (LatentSegment stores latents in a CONCAT field), so a "
            f"mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
        )
        stacked = torch.cat(embeds, dim=0)
        return TextEmbedCondition(
            embeds=stacked,
            attn_mask=torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device),
        )

    @torch.no_grad()
    def embed(self, texts: Texts) -> TextEmbedCondition:
        """Encode one batch of prompts into a ``TextEmbedCondition``."""
        # ``Texts.texts`` is the raw list[str]; ``Texts.to_list()`` returns
        # list[Text] dataclass wrappers, which the tokenizer rejects. Same
        # accessor ltx2 and wan21 use.
        prompts: List[str] = list(texts.texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")
        if self._store is not None:
            return self._pack_embeds(
                [self._store.get(prompt).unsqueeze(0).to(device=self.device, dtype=self.dtype) for prompt in prompts]
            )
        require(
            self.text_encoder is not None,
            "MiniMaxH3TextEmbedStage: text_encoder is not loaded and text_embed_cache_path is unset.",
        )

        self._ensure_embedding_sync_group()
        unique_prompts = list(dict.fromkeys(prompts))
        resolved = {prompt: self._cache[prompt] for prompt in unique_prompts if prompt in self._cache}
        missing = [prompt for prompt in unique_prompts if prompt not in resolved]
        started = time.perf_counter()
        if missing:
            with self._embedding_residency():
                encoder_device = self._encoder_device
                for prompt in missing:
                    cached = encode_minimax_h3_prompt(
                        text_encoder=self.text_encoder,
                        tokenizer=self.tokenizer,
                        processor=self.processor,
                        prompt=prompt,
                        device=encoder_device,
                    ).to("cpu")
                    resolved[prompt] = cached
                    self._cache[prompt] = cached
                    if len(self._cache) > _PROMPT_CACHE_SIZE:
                        self._cache.popitem(last=False)
        for prompt in unique_prompts:
            if prompt in self._cache:
                self._cache.move_to_end(prompt)
        self._synchronize_embedding_ranks()
        logger.debug(
            "MiniMaxH3 text embeds: prompts=%d cache_hits=%d misses=%d onload=%s elapsed_s=%.3f",
            len(prompts),
            len(unique_prompts) - len(missing),
            len(missing),
            self._onload_for_embed,
            time.perf_counter() - started,
        )
        return self._pack_embeds([resolved[prompt].to(device=self.device, dtype=self.dtype) for prompt in prompts])

    def _ensure_embedding_sync_group(self) -> None:
        """Build the barrier group while the ranks are still in lockstep."""
        if self._embedding_sync_group is not None or not self._onload_for_embed:
            return
        import torch.distributed as dist

        # new_group is itself a collective over the default group, so it cannot
        # wait until a rank has already spent minutes in an uncached conditioner
        # forward. The barrier spans the whole world rather than one shard group:
        # an onload stalls a whole node, and under HSDP the replica all-reduce
        # crosses nodes just as the shard reduce-scatter stays inside one.
        # Its own group, not the memoized init_gloo_group(): an async DCP save
        # hands that one to a background thread and keeps it, and two threads
        # driving one process group concurrently is undefined.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self._embedding_sync_group = dist.new_group(backend="gloo")

    def _synchronize_embedding_ranks(self) -> None:
        """Keep delayed conditioner loads out of the next FSDP NCCL collective."""
        if self._embedding_sync_group is None:
            return
        import torch.distributed as dist

        dist.monitored_barrier(
            group=self._embedding_sync_group,
            timeout=_EMBED_SYNC_TIMEOUT,
            wait_all_ranks=True,
        )

    @contextmanager
    def _embedding_residency(self) -> Iterator[None]:
        """Temporarily exchange GPU-resident VAEs for the frozen text encoder."""
        target_device = torch.device(self.device)
        encoder_device = self._encoder_device
        if not self._onload_for_embed or target_device.type != "cuda" or encoder_device.type != "cpu":
            yield
            return

        # Eight DP workers share one node's host-memory channels. Concurrently
        # paging and copying eight 64 GB encoders made every H2D transfer
        # hour-scale; serialize the residency window per node while allowing
        # the two nodes to proceed independently.
        # Each restore is registered BEFORE its move: ``Module.to`` walks
        # parameters in place, so an OOM part-way through the 64 GB onload
        # leaves the conditioner half-resident and still has to be undone.
        with _serialize_node_residency(), ExitStack() as cleanup:
            for module in (self.vae, self.audio_vae):
                device = next(module.parameters()).device
                if device.type == "cuda":
                    cleanup.callback(module.to, device)
                    module.to("cpu")
            torch.cuda.empty_cache()
            cleanup.callback(torch.cuda.empty_cache)
            cleanup.callback(self.text_encoder.to, "cpu")
            self.text_encoder.to(target_device)
            yield


__all__ = [
    "MiniMaxH3TextEmbedStage",
    "encode_minimax_h3_prompt",
    "load_minimax_h3_conditioner",
]
