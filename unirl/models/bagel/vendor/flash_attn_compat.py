"""Optional flash-attn compatibility for vendored BAGEL inference attention."""

try:
    from flash_attn import flash_attn_varlen_func
except Exception:
    from unirl.models.bagel.sdpa_varlen import flash_attn_varlen_func

__all__ = ["flash_attn_varlen_func"]
