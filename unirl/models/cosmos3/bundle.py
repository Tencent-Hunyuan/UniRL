"""Cosmos3 SFT checkpoint bundle and freeze policy."""

from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any, Tuple

import torch

from unirl.models.cosmos3.config import Cosmos3SFTConfig
from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

# Understanding (AR/text) parameter set; trailing ``\.`` anchors are load-bearing (README # Gotchas).
_UND_PARAM_PATTERNS = (
    r"^embed_tokens\.",
    r"^lm_head\.",
    r"^norm\.",
    r"^layers\.\d+\.self_attn\.(to_q|to_k|to_v|to_out|norm_q|norm_k)\.",
    r"^layers\.\d+\.mlp\.",
    r"^layers\.\d+\.(input_layernorm|post_attention_layernorm)\.",
)
_UND_PARAM_RE = re.compile("|".join(f"(?:{p})" for p in _UND_PARAM_PATTERNS))


def is_understanding_param(name: str) -> bool:
    """Return whether a Cosmos3 transformer parameter belongs to the understanding stream."""
    return _UND_PARAM_RE.match(name) is not None


def _import_diffusers_classes():
    try:
        from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer, UniPCMultistepScheduler
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
    except ImportError as exc:  # pragma: no cover - version guard
        raise ImportError(
            "Cosmos3 support requires diffusers>=0.39 (Cosmos3OmniTransformer/"
            "Cosmos3OmniPipeline first shipped in 0.39.0)."
        ) from exc
    return Cosmos3OmniTransformer, AutoencoderKLWan, UniPCMultistepScheduler, Cosmos3OmniPipeline


def _require_signature(fn: Any, expected: Tuple[str, ...], what: str) -> None:
    """Fingerprint a patched diffusers callable so upstream drift fails at patch time (README # Gotchas)."""
    actual = tuple(inspect.signature(fn).parameters)
    if actual != expected:
        raise RuntimeError(
            f"Cosmos3 {what} drifted from the diffusers-0.39 surface the frozen patch copies: "
            f"expected parameters {expected}, got {actual}. Re-verify the patch against the "
            "installed diffusers before bumping the pin."
        )


def _patch_rotary_emb_contiguous(rotary_emb: torch.nn.Module) -> None:
    """Replace Cosmos3's zero-stride mRoPE batched GEMM with an equivalent broadcast multiply."""
    if getattr(rotary_emb, "_unirl_contiguous_rope", False):
        return
    _require_signature(
        type(rotary_emb).forward, ("self", "position_ids", "device", "dtype"), f"{type(rotary_emb).__name__}.forward"
    )
    for attr in ("inv_freq", "apply_interleaved_mrope", "rope_axes_dim"):
        if not hasattr(rotary_emb, attr):
            raise RuntimeError(f"Cosmos3 rotary patch relies on rotary_emb.{attr}, missing on installed diffusers.")

    def forward(position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = rotary_emb.inv_freq
        if hasattr(inv_freq, "full_tensor"):
            inv_freq = inv_freq.full_tensor()
        inv_freq = inv_freq.detach().float().contiguous().to(device)
        position_ids = position_ids.float().contiguous().to(device)
        freqs = position_ids.unsqueeze(-1) * inv_freq
        freqs = rotary_emb.apply_interleaved_mrope(freqs, rotary_emb.rope_axes_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    rotary_emb.forward = forward
    rotary_emb._unirl_contiguous_rope = True
    logger.info("Cosmos3Bundle: patched rotary_emb to contiguous broadcast mRoPE")


def _needs_h20_cu128_workaround() -> bool:
    """Whether this process is on the H20 stack with broken batched cuBLAS."""
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        return False
    if not torch.__version__.startswith("2.10.0"):
        return False
    return "H20" in torch.cuda.get_device_name(torch.cuda.current_device()).upper()


def _patch_frozen_dtensor_linears(transformer: torch.nn.Module) -> None:
    """Materialize frozen FSDP2 DTensors before fp32 ``F.linear`` on the affected H20 runtime."""
    patched = 0
    for module in transformer.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.requires_grad:
            continue
        if getattr(module, "_unirl_frozen_full_tensor", False):
            continue

        def make_forward(mod: torch.nn.Linear):
            def forward(x: torch.Tensor):
                w = mod.weight
                if hasattr(w, "full_tensor"):
                    w = w.full_tensor()
                b = mod.bias
                if b is not None and hasattr(b, "full_tensor"):
                    b = b.full_tensor()
                # H20+cu128 rejects every bf16 cublasGemmEx; fp32 sgemm works (README # Precision & loading).
                return torch.nn.functional.linear(
                    x.float(),
                    w.float(),
                    None if b is None else b.float(),
                ).to(dtype=x.dtype)

            return forward

        module.forward = make_forward(module)
        module._unirl_frozen_full_tensor = True
        patched += 1
    if patched:
        logger.info(
            "Cosmos3Bundle: patched %d Linear(s) to full_tensor() when frozen+DTensor",
            patched,
        )


def _mm_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute ``(..., K) @ (K, N) + (N,)`` through one 2-D GEMM."""
    out_shape = x.shape[:-1] + (weight.shape[1],)
    y = x.reshape(-1, x.shape[-1]) @ weight
    return (y + bias).reshape(out_shape)


def _patch_h20_strided_gemm(transformer: torch.nn.Module) -> None:
    """Bypass broken strided-batched cuBLAS paths on H20 with torch 2.10/cu128."""
    import diffusers.models.transformers.transformer_cosmos3 as cosmos3_mod

    cls = cosmos3_mod.DomainAwareLinear
    if not getattr(cls, "_unirl_safe_bmm", False):
        _require_signature(cls.forward, ("self", "x", "domain_id"), "DomainAwareLinear.forward")

        def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
            if domain_id.ndim == 0:
                domain_id = domain_id.unsqueeze(0)
            domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
            if x.shape[0] != domain_id.shape[0]:
                raise ValueError(
                    "Cosmos3 action domain_id batch size must match action tokens: "
                    f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
                )
            if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
                raise ValueError(
                    f"Cosmos3 action domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}."
                )
            weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
            bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
            if x.ndim not in (2, 3):
                raise ValueError(f"Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")
            if int(domain_id.min()) == int(domain_id.max()):
                return _mm_2d(x, weight[0], bias[0])
            return torch.stack(
                [_mm_2d(x[i], weight[i], bias[i]) for i in range(x.shape[0])],
                dim=0,
            )

        cls.forward = forward
        cls._unirl_safe_bmm = True
        logger.info("Cosmos3Bundle: patched DomainAwareLinear to 2-D GEMM (H20 cuBLAS)")

    orig = cosmos3_mod.dispatch_attention_fn
    if not getattr(orig, "_unirl_no_math_sdpa", False):
        _require_signature(
            orig,
            (
                "query",
                "key",
                "value",
                "attn_mask",
                "dropout_p",
                "is_causal",
                "scale",
                "enable_gqa",
                "attention_kwargs",
                "backend",
                "parallel_config",
            ),
            "dispatch_attention_fn",
        )
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:  # pragma: no cover
            sdpa_kernel = None
            SDPBackend = None

        def wrapped(query, key, value, *args, **kwargs):
            q = query.contiguous()
            k = key.contiguous()
            v = value.contiguous()
            # Repeat kv heads: fused kernels need equal q/kv head counts here (README # Gotchas).
            q_heads = q.shape[-2]
            kv_heads = k.shape[-2]
            if q_heads != kv_heads:
                if q_heads % kv_heads != 0:
                    raise ValueError(f"Cosmos3 GQA: q heads {q_heads} not divisible by kv heads {kv_heads}")
                nrep = q_heads // kv_heads
                k = k.repeat_interleave(nrep, dim=-2).contiguous()
                v = v.repeat_interleave(nrep, dim=-2).contiguous()
                kwargs = dict(kwargs)
                kwargs["enable_gqa"] = False
            if sdpa_kernel is None:
                return orig(q, k, v, *args, **kwargs)
            if q.dtype == torch.float32:
                backends = SDPBackend.EFFICIENT_ATTENTION
            else:
                backends = [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ]
            with sdpa_kernel(backends):
                return orig(q, k, v, *args, **kwargs)

        wrapped._unirl_no_math_sdpa = True
        cosmos3_mod.dispatch_attention_fn = wrapped
        logger.info("Cosmos3Bundle: patched attention dispatch to contiguous non-MATH SDPA")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(True)


class Cosmos3Bundle(Bundle):
    """Weights and frozen helpers for Cosmos3 SFT."""

    def __init__(self, *, transformer, vae, text_tokenizer, scheduler, config: Cosmos3SFTConfig) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.scheduler = scheduler
        self.config = config
        self.dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        self.device = torch.device(config.device)
        self.pretrained_path = config.pretrained_model_ckpt_path

    @classmethod
    def from_config(cls, config: Cosmos3SFTConfig) -> "Cosmos3Bundle":
        Cosmos3OmniTransformer, AutoencoderKLWan, UniPCMultistepScheduler, _ = _import_diffusers_classes()
        path = config.pretrained_model_ckpt_path
        device = torch.device(config.device)
        master_dtype = parse_torch_dtype(config.master_precision, field_name="master_precision")
        vae_dtype = parse_torch_dtype(config.vae_precision, field_name="vae_precision")

        # Uniform master-dtype storage (fp32 storage + bf16 FSDP compute: README # Gotchas).
        meta_init_state = None
        transformer_weights_root = path
        if config.meta_init_transformer:
            if not os.path.isdir(path):
                # sharded_load reads local *.safetensors only; resolve Hub IDs (wan21 VAE precedent).
                from huggingface_hub import snapshot_download

                transformer_weights_root = snapshot_download(repo_id=path, allow_patterns=["transformer/*"])
            transformer_config = Cosmos3OmniTransformer.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: Cosmos3OmniTransformer.from_config(transformer_config), dtype=master_dtype
            )
        else:
            transformer = Cosmos3OmniTransformer.from_pretrained(
                path,
                subfolder="transformer",
                torch_dtype=master_dtype,
            )
            transformer = transformer.to(device=device, dtype=master_dtype)
        if _needs_h20_cu128_workaround():
            logger.warning(
                "Cosmos3Bundle: enabling H20 torch-2.10/cu128 cuBLAS workarounds; "
                "upgrade the runtime to remove this compatibility path"
            )
            _patch_rotary_emb_contiguous(transformer.rotary_emb)
            _patch_h20_strided_gemm(transformer)
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=vae_dtype).to(device)
        vae.requires_grad_(False)
        vae.eval()

        from transformers import AutoTokenizer

        text_tokenizer = AutoTokenizer.from_pretrained(path, subfolder="text_tokenizer")
        scheduler = UniPCMultistepScheduler.from_pretrained(path, subfolder="scheduler")

        if config.freeze_understanding:
            frozen = trainable = 0
            for name, param in transformer.named_parameters():
                if is_understanding_param(name):
                    param.requires_grad_(False)
                    frozen += param.numel()
                else:
                    trainable += param.numel()
            logger.info(
                "Cosmos3Bundle: froze und stream (%.2fB params); gen stream trainable (%.2fB params)",
                frozen / 1e9,
                trainable / 1e9,
            )

        if _needs_h20_cu128_workaround():
            _patch_frozen_dtensor_linears(transformer)

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_tokenizer=text_tokenizer,
            scheduler=scheduler,
            config=config,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = os.path.join(transformer_weights_root, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle

    @property
    def flow_shift(self) -> float:
        """Effective flow shift: config override, else the checkpoint scheduler's."""
        if self.config.flow_shift is not None:
            return float(self.config.flow_shift)
        return float(getattr(self.scheduler.config, "flow_shift", 5.0))


__all__ = ["Cosmos3Bundle", "is_understanding_param"]
