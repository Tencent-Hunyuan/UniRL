"""Add ``TorchMemorySaverAdapter.is_available()`` to stock-upstream srt."""

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
