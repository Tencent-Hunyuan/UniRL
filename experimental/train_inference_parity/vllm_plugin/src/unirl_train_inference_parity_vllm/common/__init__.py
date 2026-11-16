"""Model-independent vLLM parity patches."""

from ..registry import PatchResult
from .attention import install_attention_contract, preflight_attention_contract
from .moe_combine import moe_combine
from .norm import install_norm_patch, preflight_norm_patch
from .precision import install_precision_contract, preflight_precision_contract
from .providers import preflight_providers
from .reductions import install_reduction_patch, preflight_reduction_patch


def preflight_common(strict: bool) -> None:
    preflight_providers(strict=strict)
    preflight_precision_contract(strict=strict)
    preflight_reduction_patch(strict=strict)
    preflight_norm_patch(strict=strict)
    preflight_attention_contract(strict=strict)


def install_common(*, strict: bool) -> PatchResult:
    if not strict:
        raise ValueError("the public-reference common installer requires strict=True")
    symbols = (
        *install_precision_contract(),
        *install_reduction_patch(),
        *install_norm_patch(),
        *install_attention_contract(),
    )
    return PatchResult(name="common", symbols=symbols)


__all__ = [
    "install_common",
    "moe_combine",
    "preflight_common",
]
