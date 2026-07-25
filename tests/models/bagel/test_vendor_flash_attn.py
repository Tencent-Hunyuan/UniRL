import importlib
import importlib.util
import sys


def test_flash_attn_fallback_stub_has_a_valid_spec(monkeypatch) -> None:
    vendor = importlib.import_module("unirl.models.bagel.vendor")
    real_import_module = importlib.import_module

    def import_without_flash_attn(name, package=None):
        if name == "flash_attn":
            raise ModuleNotFoundError("No module named 'flash_attn'", name="flash_attn")
        return real_import_module(name, package)

    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    monkeypatch.setattr(importlib, "import_module", import_without_flash_attn)
    importlib.reload(vendor)

    stub = sys.modules["flash_attn"]
    assert stub.__spec__ is not None
    assert importlib.util.find_spec("flash_attn") is stub.__spec__
    assert callable(stub.flash_attn_varlen_func)
