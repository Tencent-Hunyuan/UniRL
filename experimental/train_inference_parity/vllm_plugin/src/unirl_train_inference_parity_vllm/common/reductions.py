"""Install vLLM public batch-invariant reductions as CUDA ATen providers."""

from __future__ import annotations

_LIBRARY = None


def install_reduction_patch() -> None:
    global _LIBRARY
    if _LIBRARY is not None:
        return
    import torch

    from .providers import log_softmax, mean, softmax

    def aten_log_softmax(input, dim, half_to_float):
        return log_softmax(input.float() if half_to_float else input, dim=dim)

    library = torch.library.Library("aten", "IMPL")
    library.impl("aten::_log_softmax", aten_log_softmax, "CUDA")
    library.impl("aten::softmax", softmax, "CUDA")
    library.impl("aten::_softmax", softmax, "CUDA")
    library.impl("aten::mean.dim", mean, "CUDA")
    try:
        import vllm.v1.worker.gpu.sample.logprob as sample_logprob

        def compute_token_logprobs(logits, token_ids):
            return log_softmax(logits.float(), dim=-1).gather(
                -1,
                token_ids.to(torch.int64),
            )

        sample_logprob.compute_token_logprobs = compute_token_logprobs
    except ImportError:
        pass
    _LIBRARY = library


__all__ = ["install_reduction_patch"]
