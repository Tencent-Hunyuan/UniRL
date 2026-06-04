"""Shared tensor-fingerprint helpers for trainer ↔ rollout-worker
value-correctness checks.

Both sides of the weight-sync handshake hash via :func:`fingerprint_tensor`
so trainer-computed expected hashes are bit-comparable to worker-computed
post-load hashes. Single source of truth — defining the formula here once
removes the risk of silent drift between sender and receiver.

Used by:

- worker-side ``BucketedIPCReceiveMixin._diffrl_loaded_param_checksums``
  and ``_diffrl_loaded_lora_checksums``: hash the model parameters / LoRA
  layer weights immediately after ``load_weights`` / ``add_lora`` to
  expose what actually landed.
- trainer-side ``compute_param_checksums`` / ``compute_lora_checksums_post_optimize``
  helpers: hash the bytes that were *intended* to land, including any
  value transformations the load path is known to apply (e.g. LoRA
  ``lora.optimize()`` scales ``lora_b`` by ``alpha / r``).

Equality on every (stage, rank, name) is the value-correctness criterion
for weight-sync transports.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Tuple, Union

import torch


def fingerprint_tensor(t: torch.Tensor) -> str:
    """Return a 16-char hex SHA-256 prefix over ``(dtype, shape, all bytes)``.

    Costlier than the head/tail-only fingerprint used by the structural
    ``_diffrl_param_checksums``, but leaves no room for middle-byte
    corruption to slip past. Cheap for the inputs we hash here:

    - LoRA tensors are at most a few MB.
    - The smoke test's full-weight targets are TP-flat layer norms (KB).

    The 16-char prefix balances readable log lines against collision
    risk: 64 bits of SHA-256 output is overwhelmingly more than enough
    for the per-test target counts (≤ 32 tensors per sub-test).
    """
    data = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(data.dtype).encode())
    h.update(str(tuple(data.shape)).encode())
    # ``view(torch.uint8)`` requires contiguous storage (already done
    # above) and a dtype with itemsize-aligned numel — true for every
    # torch dtype we hash here. ``flatten()`` is a no-op view; numpy
    # ``.tobytes()`` does the actual byte materialisation.
    h.update(data.view(torch.uint8).flatten().numpy().tobytes())
    return h.hexdigest()[:16]


def compute_param_checksums(
    named_tensors: Union[
        Dict[str, torch.Tensor],
        Iterable[Tuple[str, torch.Tensor]],
    ],
) -> Dict[str, str]:
    """Hash each tensor in a name→tensor dict (or ``[(name, tensor), ...]``).

    Used by the smoke test's full-weight sub-tests (B.1 / B.2 / B.3) to
    capture the trainer-side expected hash of the synthetic state-dict
    *before* it goes on the wire.
    """
    items = named_tensors.items() if isinstance(named_tensors, dict) else named_tensors
    return {name: fingerprint_tensor(t) for name, t in items}


def _is_lora_b_name(name: str) -> bool:
    """Heuristic: PEFT names ``lora_B`` matrices with the ``.lora_B.``
    substring (or trailing ``.lora_B.weight``).

    ``lora.optimize()`` scales these by ``alpha / r``; ``lora_A`` and
    biases are unchanged. We mirror that scaling on the trainer side
    so trainer-hash == worker-hash after ``add_lora`` runs.
    """
    return ".lora_B." in name or name.endswith(".lora_B.weight")


def compute_lora_checksums_post_optimize(
    lora_tensors: Dict[str, torch.Tensor],
    peft_config: Dict,
) -> Dict[str, str]:
    """Hash each LoRA tensor as it would appear after the worker's
    ``LoRAModel.from_lora_tensors`` runs ``lora.optimize()``.

    ``optimize()`` does ``lora_b *= alpha / r`` and leaves ``lora_a``
    (and biases) untouched. We detect ``lora_b`` matrices by name and
    apply the same scaling here so the trainer-side expected hash is
    bit-comparable to the worker's post-load read-back.

    Reads ``r`` and ``lora_alpha`` from the supplied ``peft_config``
    (matching the keys the smoke test ships and the ones PEFTHelper
    consumes downstream).
    """
    r = float(peft_config.get("r", peft_config.get("rank", 8)))
    alpha = float(peft_config.get("lora_alpha", peft_config.get("alpha", r)))
    scale = alpha / r if r else 1.0
    out: Dict[str, str] = {}
    for name, t in lora_tensors.items():
        scaled = t * scale if _is_lora_b_name(name) else t
        out[name] = fingerprint_tensor(scaled)
    return out


__all__ = [
    "fingerprint_tensor",
    "compute_param_checksums",
    "compute_lora_checksums_post_optimize",
]
