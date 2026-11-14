"""Install vLLM public batch-invariant reductions as CUDA ATen providers."""

from __future__ import annotations

import inspect

from ..compat import require_symbol
from ..registry import SymbolResult, symbol_result

_LIBRARY = None
_NATIVE_CUDA_OPS = (
    "aten::_log_softmax",
    "aten::softmax",
    "aten::_softmax",
    "aten::mean.dim",
)


def _cuda_registration(torch, operator: str) -> str | None:
    return next(
        (line for line in torch._C._dispatch_dump_table(operator).splitlines() if line.startswith("CUDA:")),
        None,
    )


def preflight_reduction_patch(*, strict: bool) -> None:
    import torch

    require_symbol(
        "vllm.v1.worker.gpu.sample.logprob",
        "compute_token_logprobs",
        parameters=(("logits", "<required>"), ("token_ids", "<required>")),
        origin=("vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs"),
        strict=strict,
    )
    if strict:
        conflicts = {
            operator: registration
            for operator in _NATIVE_CUDA_OPS
            if (registration := _cuda_registration(torch, operator)) is not None and "/pytorch/" not in registration
        }
        if conflicts:
            raise RuntimeError(f"unexpected pre-existing CUDA reduction providers: {conflicts}")


def install_reduction_patch() -> tuple[SymbolResult, ...]:
    global _LIBRARY
    if _LIBRARY is not None:
        raise RuntimeError("reduction parity patch installed twice")
    import torch

    from .providers import log_softmax, mean, softmax

    before_registrations = {operator: _cuda_registration(torch, operator) for operator in _NATIVE_CUDA_OPS}
    import vllm.v1.worker.gpu.sample.logprob as sample_logprob

    original_compute_token_logprobs = sample_logprob.compute_token_logprobs

    def aten_log_softmax(input, dim, half_to_float):
        return log_softmax(input.float() if half_to_float else input, dim=dim)

    library = torch.library.Library("aten", "IMPL")
    library.impl("aten::_log_softmax", aten_log_softmax, "CUDA")
    library.impl("aten::softmax", softmax, "CUDA")

    def aten_softmax(input, dim, half_to_float):
        return softmax(input.float() if half_to_float else input, dim=dim)

    library.impl("aten::_softmax", aten_softmax, "CUDA")
    library.impl("aten::mean.dim", mean, "CUDA")

    def compute_token_logprobs(logits, token_ids):
        return log_softmax(logits.float(), dim=-1).gather(
            -1,
            token_ids.to(torch.int64),
        )

    sample_logprob.compute_token_logprobs = compute_token_logprobs
    _LIBRARY = library
    registered = (
        ("aten::_log_softmax[CUDA]", aten_log_softmax, "aten::_log_softmax"),
        ("aten::softmax[CUDA]", softmax, "aten::softmax"),
        ("aten::_softmax[CUDA]", aten_softmax, "aten::_softmax"),
        ("aten::mean.dim[CUDA]", mean, "aten::mean.dim"),
    )
    results = []
    for symbol, provider, operator in registered:
        registration = _cuda_registration(torch, operator)
        results.append(
            SymbolResult(
                symbol=symbol,
                before_provider=before_registrations[operator] or "<missing>",
                before_signature=None,
                before_identity=None,
                after_provider=registration or "<missing>",
                after_signature=str(inspect.signature(provider)),
                after_identity=id(provider),
                verified=(
                    registration is not None
                    and "unirl_train_inference_parity_vllm/common/reductions.py" in registration
                ),
            )
        )
    if not all(result.verified for result in results):
        raise RuntimeError(f"failed to verify CUDA reduction providers: {results}")
    results.append(
        symbol_result(
            "vllm.v1.worker.gpu.sample.logprob.compute_token_logprobs",
            compute_token_logprobs,
            before=original_compute_token_logprobs,
            actual=sample_logprob.compute_token_logprobs,
        )
    )
    return tuple(results)


__all__ = ["install_reduction_patch", "preflight_reduction_patch"]
