"""Add ``TorchMemorySaverAdapter.is_available()`` to stock-upstream srt.

The fork added a 5-line ``is_available()`` staticmethod to
``srt/utils/torch_memory_saver_adapter.py`` (the only fork edit to ``srt`` that
matters; the ``common.py`` one-liner is cosmetic whitespace). ``MemorySaverHandler``
and the sleep/wake server-arg auto-enable path consult it to decide whether the
torch-memory-saver backend is importable. Idempotent; no-op if upstream already
defines it.
"""

from __future__ import annotations


def patch_srt() -> None:
    import sglang.srt.utils.torch_memory_saver_adapter as tmsa

    if hasattr(tmsa.TorchMemorySaverAdapter, "is_available"):
        return

    @staticmethod
    def is_available() -> bool:
        """Whether torch-memory-saver was imported successfully."""
        return tmsa.import_error is None

    tmsa.TorchMemorySaverAdapter.is_available = is_available
