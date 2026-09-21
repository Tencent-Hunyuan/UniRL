"""Collective-safe exact log-probability parity gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

_STATUS_SIZE = 8


@dataclass(frozen=True)
class _LocalStatus:
    shape_ok: bool
    dtype_ok: bool
    finite_ok: bool
    mismatch_count: int
    max_absdiff_fp32: float
    token_count: int
    k3_sum: float
    k3_max: float
    diagnostic: Dict[str, Any]

    def values(self) -> list[float]:
        max_absdiff = self.max_absdiff_fp32
        if not math.isfinite(max_absdiff):
            max_absdiff = math.inf
        k3_sum = self.k3_sum
        if not math.isfinite(k3_sum):
            k3_sum = math.inf
        k3_max = self.k3_max
        if not math.isfinite(k3_max):
            k3_max = math.inf
        return [
            float(self.shape_ok),
            float(self.dtype_ok),
            float(self.finite_ok),
            float(self.mismatch_count),
            max_absdiff,
            float(self.token_count),
            k3_sum,
            k3_max,
        ]


def local_gate_status(
    new_logp: Optional[torch.Tensor],
    rollout_logp: Optional[torch.Tensor],
    *,
    local_error: Optional[str] = None,
) -> _LocalStatus:
    """Build local status without raising before every rank reaches the collective."""
    diagnostic: Dict[str, Any] = {}
    token_count = 0
    try:
        new_shape = tuple(new_logp.shape) if isinstance(new_logp, torch.Tensor) else None
        rollout_shape = tuple(rollout_logp.shape) if isinstance(rollout_logp, torch.Tensor) else None
        token_count = int(new_logp.numel()) if isinstance(new_logp, torch.Tensor) else 0
        diagnostic.update(
            {
                "new_shape": new_shape,
                "rollout_shape": rollout_shape,
                "new_dtype": str(new_logp.dtype) if isinstance(new_logp, torch.Tensor) else None,
                "rollout_dtype": str(rollout_logp.dtype) if isinstance(rollout_logp, torch.Tensor) else None,
                "token_count": token_count,
            }
        )
        if local_error is not None:
            diagnostic["error"] = str(local_error)
            return _LocalStatus(False, False, False, 0, math.inf, token_count, math.inf, math.inf, diagnostic)
        if not isinstance(new_logp, torch.Tensor) or not isinstance(rollout_logp, torch.Tensor):
            diagnostic["error"] = "new_logp and rollout_logp must both be tensors"
            return _LocalStatus(False, False, False, 0, math.inf, token_count, math.inf, math.inf, diagnostic)

        dtype_ok = new_logp.dtype == torch.float32 and rollout_logp.dtype == torch.float32
        diagnostic["dtype_ok"] = dtype_ok
        if not dtype_ok:
            diagnostic["error"] = "exact parity requires replay and the unquantized rollout anchor to both be FP32"
            return _LocalStatus(False, False, False, 0, math.inf, token_count, math.inf, math.inf, diagnostic)

        new_fp32 = new_logp.detach()
        rollout_fp32 = rollout_logp.detach().to(device=new_fp32.device)
        new_finite = bool(torch.isfinite(new_fp32).all().item())
        rollout_finite = bool(torch.isfinite(rollout_fp32).all().item())
        shape_ok = new_fp32.shape == rollout_fp32.shape
        diagnostic.update(
            {
                "new_finite": new_finite,
                "rollout_finite": rollout_finite,
                "shape_ok": shape_ok,
            }
        )
        if not shape_ok:
            return _LocalStatus(
                False,
                dtype_ok,
                new_finite and rollout_finite,
                0,
                math.inf,
                token_count,
                math.inf,
                math.inf,
                diagnostic,
            )

        mismatch = new_fp32 != rollout_fp32
        mismatch_count = int(mismatch.sum().item())
        mismatch_indices = torch.nonzero(mismatch.reshape(-1), as_tuple=False).reshape(-1)
        first_mismatch = int(mismatch_indices[0].item()) if mismatch_indices.numel() else None
        if token_count:
            absdiff = (new_fp32 - rollout_fp32).abs()
            max_absdiff = float(absdiff.max().item())
            log_r = (new_fp32 - rollout_fp32).clamp(min=-20.0, max=20.0)
            k3 = torch.expm1(log_r) - log_r
            k3_sum = float(k3.sum().item())
            k3_max = float(k3.max().item())
        else:
            max_absdiff = 0.0
            k3_sum = 0.0
            k3_max = 0.0
        diagnostic.update(
            {
                "mismatch_count": mismatch_count,
                "first_mismatch": first_mismatch,
                "max_absdiff_fp32": max_absdiff,
                "first_new_fp32": (
                    float(new_fp32.reshape(-1)[first_mismatch].item()) if first_mismatch is not None else None
                ),
                "first_rollout_fp32": (
                    float(rollout_fp32.reshape(-1)[first_mismatch].item()) if first_mismatch is not None else None
                ),
            }
        )
        return _LocalStatus(
            True,
            dtype_ok,
            new_finite and rollout_finite,
            mismatch_count,
            max_absdiff,
            token_count,
            k3_sum,
            k3_max,
            diagnostic,
        )
    except Exception as error:
        diagnostic["error"] = f"{type(error).__name__}: {error}"
        return _LocalStatus(False, False, False, 0, math.inf, token_count, math.inf, math.inf, diagnostic)


def _collective_device(new_logp: Optional[torch.Tensor]) -> torch.device:
    if isinstance(new_logp, torch.Tensor) and new_logp.is_cuda:
        return new_logp.device
    backend = str(dist.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def collective_fail_closed(
    local: _LocalStatus,
    *,
    collective_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Collect one fixed status row per rank and make every rank fail together."""
    rows = [local.values()]
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        device = _collective_device(collective_tensor)
        status = torch.tensor(local.values(), dtype=torch.float64, device=device)
        gathered = [torch.empty(_STATUS_SIZE, dtype=status.dtype, device=device) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, status)
        rows = torch.stack(gathered).cpu().tolist()

    shape_ok = all(bool(row[0]) for row in rows)
    dtype_ok = all(bool(row[1]) for row in rows)
    finite_ok = all(bool(row[2]) for row in rows)
    mismatch_count = int(sum(row[3] for row in rows))
    max_absdiff = max(row[4] for row in rows)
    token_count = int(sum(row[5] for row in rows))
    k3_sum = sum(row[6] for row in rows)
    k3_max = max(row[7] for row in rows)
    local_token_counts = [int(row[5]) for row in rows]
    mixed_empty = any(count == 0 for count in local_token_counts) and any(count > 0 for count in local_token_counts)
    failed = not shape_ok or not dtype_ok or not finite_ok or mismatch_count != 0 or mixed_empty

    if failed:
        local_diagnostic = dict(local.diagnostic)
        local_diagnostic["rank"] = dist.get_rank() if distributed else 0
        diagnostics = [local_diagnostic]
        if distributed:
            diagnostics = [None] * dist.get_world_size()
            dist.all_gather_object(diagnostics, local_diagnostic)
        raise RuntimeError(
            "train/inference exact parity gate failed: "
            f"shape_ok={shape_ok} dtype_ok={dtype_ok} finite_ok={finite_ok} "
            f"mismatch_count={mismatch_count} max_absdiff_fp32={max_absdiff!r} "
            f"token_count={token_count} local_token_counts={local_token_counts} "
            f"diagnostics={diagnostics!r}"
        )

    return {
        "token_count": token_count,
        "per_rank_token_count": local_token_counts,
        "torch_equal_fp32": True,
        "mismatch_count": mismatch_count,
        "max_absdiff_fp32": max_absdiff,
        "k3_mean": k3_sum / token_count if token_count else 0.0,
        "k3_max": k3_max,
    }


def exact_parity_gate(
    new_logp: Optional[torch.Tensor],
    rollout_logp: Optional[torch.Tensor],
    *,
    local_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build local status, synchronize it once, and fail closed before backward."""
    local = local_gate_status(new_logp, rollout_logp, local_error=local_error)
    return collective_fail_closed(local, collective_tensor=new_logp)


__all__ = ["collective_fail_closed", "exact_parity_gate", "local_gate_status"]
