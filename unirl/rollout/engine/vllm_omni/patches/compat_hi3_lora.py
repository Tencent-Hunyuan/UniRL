"""Runtime patch making ``vllm.model_executor.utils.get_moe_expert_mapping``
tolerate HI3's 2-tuple ``get_expert_mapping`` return shape.

vllm 0.20 expects the model's ``get_expert_mapping`` to return a flat
``list[tuple[str, str, int, str]]``. HI3 (vllm-omni's
``HunyuanImage3ForCausalMM``) returns a 2-tuple
``(expert_params_mapping, expert_weights_remapping)`` so its own
``load_weights`` can use the second element. The mismatch causes
``vllm/lora/utils.py:process_packed_modules_mapping`` to do::

    for _, weight_name, _, _ in moe_packed_mapping  # iterates (list, dict)

and trip ``ValueError: too many values to unpack (expected 4)`` at boot
when ``enable_lora=True``.

Workaround: patch ``vllm.model_executor.utils.get_moe_expert_mapping``
so that if a model returns ``(seq, dict)`` we hand vllm just ``seq``.
Idempotent via a sentinel attribute. Required only when AR-stage
``enable_lora`` is on; harmless otherwise.
"""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Patch ``get_moe_expert_mapping`` everywhere it's used so HI3's
    2-tuple return is unwrapped to the flat list vllm 0.20 expects.

    ``vllm/lora/utils.py`` does ``from vllm.model_executor.utils import
    get_moe_expert_mapping``, which captures a local reference at import
    time. Patching only the source module misses the cached reference in
    consumers, so we patch every known consumer's local symbol too.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from vllm.model_executor import utils as vllm_mu
    except ImportError:
        # vllm not installed in this process — nothing to patch (the import
        # is the worker's, not the smoke test's).
        return

    original = getattr(vllm_mu, "get_moe_expert_mapping", None)
    if original is None:
        _INSTALLED = True
        return
    if getattr(original, "_diffrl_hi3_unwrap", False):
        _INSTALLED = True
        return

    def _patched(model, _orig=original):
        result = _orig(model)
        # HI3 shape: ``(list_of_4tuples, dict)``. Anything else passes
        # through untouched.
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and not isinstance(result[0], (str, int, float))
            and hasattr(result[0], "__iter__")
            and isinstance(result[1], dict)
        ):
            return result[0]
        return result

    _patched._diffrl_hi3_unwrap = True  # type: ignore[attr-defined]
    vllm_mu.get_moe_expert_mapping = _patched

    # Replace the cached local reference inside ``vllm/lora/utils.py``
    # (and any other consumer module already loaded). New consumers that
    # haven't imported the symbol yet will pick up the patched ``vllm_mu``
    # version directly.
    try:
        from vllm.lora import utils as vllm_lora_utils

        if hasattr(vllm_lora_utils, "get_moe_expert_mapping"):
            vllm_lora_utils.get_moe_expert_mapping = _patched
    except ImportError:
        pass

    _INSTALLED = True


install()


__all__ = ["install"]
