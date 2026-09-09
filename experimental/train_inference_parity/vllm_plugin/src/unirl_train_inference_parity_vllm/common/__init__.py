"""Model-independent vLLM parity patches."""

from .attention import install_attention_contract
from .norm import install_norm_patch
from .precision import install_precision_contract
from .reductions import install_reduction_patch


def install_common() -> None:
    install_precision_contract()
    install_reduction_patch()
    install_norm_patch()
    install_attention_contract()


__all__ = ["install_common"]
