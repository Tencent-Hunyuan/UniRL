"""Utilities for merging or extracting PEFT/LoRA weights."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.distributed.tensor import DTensor, Replicate

_PEFT_PREFIX = "base_model.model."


def _strip_peft_prefix(name: str) -> str:
    return name.removeprefix(_PEFT_PREFIX)


def _to_full_tensor(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Materialize DTensor parameters into regular tensors on CUDA.

    ``dtype`` (optional) casts floating tensors BEFORE the DTensor
    redistribute, so the all-gather moves wire-width bytes (e.g. bf16)
    instead of master-width (fp32). ``None`` keeps the tensor's own dtype.
    """
    tensor = tensor.cuda()
    if dtype is not None and tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if isinstance(tensor, DTensor):
        tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim).to_local()
    return tensor


def merged_state_dict(
    model: torch.nn.Module,
    adapter_name: str = "default",
    dtype: torch.dtype | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` pairs with LoRA deltas folded into base weights.

    ``lm_head.weight`` is skipped when ``tie_word_embeddings=True``: SGLang
    aliases it to ``model.embed_tokens.weight`` and rejects an explicit update.

    ``dtype`` (optional) is the wire dtype: yielded tensors are cast to it.
    The LoRA fold itself always runs at master width — only its output is
    cast — so the merge numerics are unchanged by the wire dtype.
    """
    skip_lm_head = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))

    def _cast(t: torch.Tensor) -> torch.Tensor:
        if dtype is not None and t.is_floating_point() and t.dtype != dtype:
            return t.to(dtype)
        return t

    if not hasattr(model, "peft_config"):
        for name, param in model.state_dict().items():
            if skip_lm_head and name == "lm_head.weight":
                continue
            yield (name, _to_full_tensor(param, dtype))
        return

    peft_cfg = model.peft_config[adapter_name]
    scaling = peft_cfg.lora_alpha / peft_cfg.r
    state_dict = model.state_dict()

    lora_groups: dict[str, dict[str, str]] = {}
    regular_keys: list[str] = []

    for raw_name in state_dict:
        name = _strip_peft_prefix(raw_name)

        if ".base_layer." in name:
            original = name.replace(".base_layer.", ".")
            lora_groups.setdefault(original, {})["base"] = raw_name
        elif ".lora_A." in name:
            prefix, adapter_suffix = name.split(".lora_A.", 1)
            adapter, *rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                original = prefix + "." + rest[0] if rest else prefix
                lora_groups.setdefault(original, {})["lora_A"] = raw_name
        elif ".lora_B." in name:
            prefix, adapter_suffix = name.split(".lora_B.", 1)
            adapter, *rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                original = prefix + "." + rest[0] if rest else prefix
                lora_groups.setdefault(original, {})["lora_B"] = raw_name
        else:
            regular_keys.append(raw_name)

    for original_name, group in lora_groups.items():
        if "base" not in group:
            continue
        if skip_lm_head and original_name == "lm_head.weight":
            continue
        # Merge inputs stay master-width (no ``dtype`` here): pre-casting them
        # to the wire dtype would round the LoRA update away before the fold.
        base = _to_full_tensor(state_dict[group["base"]])
        if "lora_A" in group and "lora_B" in group:
            lora_a = _to_full_tensor(state_dict[group["lora_A"]])
            lora_b = _to_full_tensor(state_dict[group["lora_B"]])
            # Merge in fp32: bf16 base + bf16 delta rounds the LoRA update away.
            merged = (base.float() + (lora_b.float() @ lora_a.float()) * scaling).to(base.dtype)
            yield (original_name, _cast(merged))
        else:
            yield (original_name, _cast(base))

    for raw_name in regular_keys:
        stripped = _strip_peft_prefix(raw_name)
        if skip_lm_head and stripped == "lm_head.weight":
            continue
        yield (stripped, _to_full_tensor(state_dict[raw_name], dtype))


def raw_state_dict(
    model: torch.nn.Module,
    adapter_name: str = "default",
    dtype: torch.dtype | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield base and LoRA weights separately, matching rollout-engine naming.

    ``dtype`` (optional) is the wire dtype: floating tensors are cast to it
    shard-side in :func:`_to_full_tensor`, before the DTensor all-gather.
    """
    if not hasattr(model, "peft_config"):
        for name, param in model.state_dict().items():
            yield (name, _to_full_tensor(param, dtype))
        return

    state_dict = model.state_dict()

    base_names: dict[str, str] = {}
    lora_a_keys: dict[str, str] = {}
    lora_b_keys: dict[str, str] = {}
    regular_keys: list[str] = []

    for raw_name in state_dict:
        name = _strip_peft_prefix(raw_name)

        if ".base_layer." in name:
            base_names[raw_name] = name
        elif ".lora_A." in name:
            prefix, adapter_suffix = name.split(".lora_A.", 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                lora_a_keys[prefix] = raw_name
        elif ".lora_B." in name:
            prefix, adapter_suffix = name.split(".lora_B.", 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter == adapter_name:
                lora_b_keys[prefix] = raw_name
        else:
            regular_keys.append(raw_name)

    for raw_name, stripped_name in base_names.items():
        yield (stripped_name.replace(".base_layer.", "."), _to_full_tensor(state_dict[raw_name], dtype))
    for prefix, raw_name in lora_a_keys.items():
        yield (prefix + ".lora_A", _to_full_tensor(state_dict[raw_name], dtype))
    for prefix, raw_name in lora_b_keys.items():
        yield (prefix + ".lora_B", _to_full_tensor(state_dict[raw_name], dtype))
    for raw_name in regular_keys:
        yield (_strip_peft_prefix(raw_name), _to_full_tensor(state_dict[raw_name], dtype))


def extract_lora_tensors(
    model: torch.nn.Module,
    *,
    param_prefix: str = "",
    adapter_name: str = "default",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Extract LoRA tensors in canonical wire format.

    Canonical format: ``<pipeline_prefix><module>.lora_A.weight`` and
    ``<pipeline_prefix><module>.lora_B.weight`` — PEFT envelope
    (``base_model.model.``) and per-adapter name stripped; pipeline prefix
    retained.  Downstream receivers convert to their engine-specific format:
    :func:`adapt_lora_for_vllm` re-adds the envelope for vllm-omni;
    :func:`adapt_lora_for_sglang` strips the prefix and injects ``.alpha``
    for SGLang.

    ``dtype`` (optional) is the wire dtype: floating LoRA tensors are cast to it
    shard-side in :func:`_to_full_tensor`, BEFORE the DTensor all-gather. This is
    load-bearing under ``master_dtype=fp32`` (the reward-collapse fix): the
    trainable LoRA params live in fp32, but the rollout engine's vLLM punica
    kernel hard-asserts bf16/fp16 — so the caller passes the FSDP compute dtype
    (``backend.weight_sync_dtype``) and the all-gather also moves half the bytes.
    ``None`` keeps each tensor's own dtype (the prior all-bf16-master behavior).
    """
    result: dict[str, torch.Tensor] = {}
    prefix = str(param_prefix or "")
    for raw_name, param in model.state_dict().items():
        name = _strip_peft_prefix(raw_name)
        for marker, suffix in ((".lora_A.", "lora_A"), (".lora_B.", "lora_B")):
            if marker not in name:
                continue
            head, adapter_suffix = name.split(marker, 1)
            adapter, *_rest = adapter_suffix.split(".", 1)
            if adapter != adapter_name:
                break
            out_name = f"{prefix}{head}.{suffix}.weight"
            result[out_name] = _to_full_tensor(param, dtype).detach().cpu()
            break

    # Defensive dtype check: vllm punica kernel hard-asserts inputs.dtype in
    # {fp16, bf16}. Catch fp32 LoRA here in trainer (cheap) rather than
    # crashing ~20min later in rollout. With ``dtype`` passed (the normal path)
    # this never fires; it backstops a caller that forgot to thread the wire
    # dtype while running master_dtype=fp32.
    _bad_dtype = [
        (k, v.dtype) for k, v in result.items() if ".lora_" in k and v.dtype not in (torch.bfloat16, torch.float16)
    ]
    if _bad_dtype:
        sample = ", ".join(f"{k}={dt}" for k, dt in _bad_dtype[:3])
        raise RuntimeError(
            f"extract_lora_tensors: {len(_bad_dtype)} LoRA tensor(s) have "
            f"unsupported dtype for vllm punica kernel (expected bf16/fp16). "
            f"Sample: [{sample}]. Pass dtype=backend.weight_sync_dtype (or check FSDP "
            f"MixedPrecisionPolicy.param_dtype)."
        )

    return result


def adapt_lora_for_vllm(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Wrap canonical-format LoRA keys in the vllm-omni PEFT envelope.

    Canonical → vllm-omni format::

        <pipeline_prefix><module>.lora_A.weight
        → base_model.model.<pipeline_prefix><module>.lora_A.weight

    This is the receiver-side adapter for
    :class:`~unirl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine`.
    """
    return {f"{_PEFT_PREFIX}{k}": v for k, v in tensors.items()}


def adapt_lora_for_sglang(
    tensors: dict[str, torch.Tensor],
    *,
    pipeline_prefix: str = "",
    peft_config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Convert canonical-format LoRA tensors to SGLang's native key format.

    Canonical → SGLang native::

        <pipeline_prefix><module>.lora_A.weight
        → <module>.lora_A.weight
        + <module>.alpha            ← injected from peft_config["lora_alpha"]

    SGLang's ``_apply_lora_to_layers`` keys its ``lora_layers`` dict by
    ``named_modules()`` of ``self.modules["transformer"]`` — i.e. starting
    *inside* the transformer — so layer keys are bare module names without
    any pipeline prefix.  The ``.alpha`` key is required so SGLang computes
    ``scale = lora_alpha / r`` correctly; without it SGLang falls back to
    ``inferred_alpha = inferred_rank`` → scale = 1.0 (wrong for alpha ≠ rank).

    Args:
        tensors: Canonical-format output of :func:`extract_lora_tensors`.
        pipeline_prefix: The pipeline-level prefix to strip, e.g.
            ``"transformer."`` for SD3/WAN/HV15/Qwen or ``"model."`` for
            HunyuanImage3.  Read from
            ``model_config.weight_sync_param_name_prefix`` at the call site.
        peft_config: PEFT config dict; provides ``lora_alpha`` for injecting
            ``.alpha`` keys.
    """
    prefix = str(pipeline_prefix or "")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        if prefix and key.startswith(prefix):
            key = key[len(prefix) :]
        result[key] = tensor

    if peft_config:
        lora_alpha = peft_config.get("lora_alpha")
        if lora_alpha is not None:
            layer_bases: set[str] = set()
            for k in result:
                for suf in (".lora_A.weight", ".lora_A"):
                    if k.endswith(suf):
                        layer_bases.add(k[: -len(suf)])
                        break
            alpha_tensor = torch.tensor(float(lora_alpha))
            for base in layer_bases:
                alpha_key = f"{base}.alpha"
                if alpha_key not in result:
                    result[alpha_key] = alpha_tensor

    return result
