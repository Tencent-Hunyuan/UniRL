"""MiniMax-H3 text embedding stage -- Qwen3-VL layer-50 hidden states."""

from __future__ import annotations

import fcntl
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, List

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

logger = logging.getLogger(__name__)


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
        self._onload_for_embed = bool(bundle.text_encoder_onload_for_embed)
        # The encoder is frozen and prompt-only. Trainside forward_batch_size=1
        # invokes pipeline.generate once per sibling sample, so cache on CPU to
        # avoid repeating a 32B conditioner forward for the same prompt.
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_size = 64

    @property
    def _encoder_device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @property
    def _decoder(self):
        """The decoder stack that owns ``.layers``."""
        model = self.text_encoder.model
        return getattr(model, "language_model", model)

    @torch.no_grad()
    def embed(self, texts: Texts) -> TextEmbedCondition:
        """Encode one batch of prompts into a ``TextEmbedCondition``."""
        # ``Texts.texts`` is the raw list[str]; ``Texts.to_list()`` returns
        # list[Text] dataclass wrappers, which the tokenizer rejects. Same
        # accessor ltx2 and wan21 use.
        prompts: List[str] = list(texts.texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")

        num_layers = len(self._decoder.layers)
        require(
            num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
            f"MiniMaxH3TextEmbedStage: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
            f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"the loaded conditioner has {num_layers}.",
        )

        unique_prompts = list(dict.fromkeys(prompts))
        resolved = {prompt: self._cache[prompt] for prompt in unique_prompts if prompt in self._cache}
        missing = [prompt for prompt in unique_prompts if prompt not in resolved]
        started = time.perf_counter()
        if missing:
            with self._embedding_residency():
                encoder_device = self._encoder_device
                for prompt in missing:
                    cached = self._encode_prompt(prompt, encoder_device).detach().to("cpu").contiguous()
                    resolved[prompt] = cached
                    self._cache[prompt] = cached
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        for prompt in unique_prompts:
            if prompt in self._cache:
                self._cache.move_to_end(prompt)
        logger.info(
            "MiniMaxH3 text embeds: prompts=%d cache_hits=%d misses=%d onload=%s elapsed_s=%.3f",
            len(prompts),
            len(unique_prompts) - len(missing),
            len(missing),
            self._onload_for_embed,
            time.perf_counter() - started,
        )
        embeds = [resolved[prompt].to(device=self.device, dtype=self.dtype) for prompt in prompts]

        lengths = {int(e.shape[1]) for e in embeds}
        require(
            len(lengths) == 1,
            f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed sequence "
            f"geometry must be identical across the batch (LatentSegment stores latents in a CONCAT field), so a "
            f"mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
        )
        text_embeds = torch.cat(embeds, dim=0)
        return TextEmbedCondition(
            embeds=text_embeds,
            attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
        )

    def _encode_prompt(self, prompt: str, encoder_device: torch.device) -> torch.Tensor:
        """Run the frozen Qwen3-VL conditioner once for one prompt."""
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=encoder_device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, read
        # off the token type ids the processor derives (0 text, 1 image,
        # 2 video). Text-only here, but the conditioner still wants them.
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=encoder_device
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        return outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]

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
        lock_started = time.perf_counter()
        with open("/tmp/unirl_minimax_h3_text_encoder_onload.lock", "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logger.info("MiniMaxH3 conditioner residency lock acquired after %.3fs", time.perf_counter() - lock_started)
            vae_devices = [(module, next(module.parameters()).device) for module in (self.vae, self.audio_vae)]
            try:
                for module, device in vae_devices:
                    if device.type == "cuda":
                        module.to("cpu")
                torch.cuda.empty_cache()
                self.text_encoder.to(target_device)
                yield
            finally:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()
                for module, device in vae_devices:
                    if device.type == "cuda":
                        module.to(device)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = ["MiniMaxH3TextEmbedStage"]
