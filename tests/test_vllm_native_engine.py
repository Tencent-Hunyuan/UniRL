from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class _DType:
    itemsize: int = 1


@dataclass(frozen=True)
class _ParamMeta:
    name: str
    dtype: _DType
    shape: tuple[int, ...]


class _TrainerEngine:
    pass


def _load_module(monkeypatch: pytest.MonkeyPatch):
    weight_transfer = types.ModuleType("vllm.distributed.weight_transfer")
    weight_transfer.ParamMeta = _ParamMeta
    weight_transfer.VLLMWeightSyncClient = object
    weight_transfer.WeightSource = object
    ipc_engine = types.ModuleType("vllm.distributed.weight_transfer.ipc_engine")
    ipc_engine.IPCTrainerWeightTransferEngine = _TrainerEngine

    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.distributed": types.ModuleType("vllm.distributed"),
        "vllm.distributed.weight_transfer": weight_transfer,
        "vllm.distributed.weight_transfer.ipc_engine": ipc_engine,
    }
    for name, module in packages.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).parents[1] / "unirl" / "distributed" / "weight_sync" / "transfer" / "vllm_native_engine.py"
    spec = importlib.util.spec_from_file_location("_test_vllm_native_engine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tensor:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.shape = (1,)
        self.dtype = _DType()
        self.device = types.SimpleNamespace(type="cuda")
        self._fail = fail

    def detach(self):
        return self

    def contiguous(self):
        if self._fail:
            raise RuntimeError(f"copy failed for {self.name}")
        return self

    def view(self, *_args):
        return self

    def numel(self) -> int:
        return 1


class _Buffer:
    def __getitem__(self, _index):
        return self

    def copy_(self, _tensor, *, non_blocking: bool):
        assert non_blocking


class _Source:
    def __init__(self, tensors: list[_Tensor]) -> None:
        self._tensors = tensors
        self._metadata = [_ParamMeta(tensor.name, tensor.dtype, tensor.shape) for tensor in tensors]

    def metadata(self):
        return list(self._metadata)

    def __iter__(self):
        return iter((tensor.name, tensor) for tensor in self._tensors)


def _make_engine(module, monkeypatch: pytest.MonkeyPatch, tensors: list[_Tensor], boundaries: set[int]):
    phases: list[tuple[str, bool]] = []
    sends: list[list[str]] = []
    synchronizations: list[None] = []

    engine = object.__new__(module.VLLMNativeIPCTrainerEngine)
    engine.source = _Source(tensors)
    engine.packed_buffer_size_bytes = 16
    engine.device_index = 0
    engine.gpu_uuid = "GPU-0"
    engine._materialization_boundaries = frozenset(boundaries)

    def consensus(error, phase):
        phases.append((phase, error is not None))
        if error is not None:
            raise RuntimeError(phase) from error

    engine._consensus = consensus
    engine._all_gather_and_merge_handles = lambda handles: handles
    engine._do_send = lambda **payload: sends.append(payload["names"])

    monkeypatch.setattr(module.torch, "empty", lambda *_args, **_kwargs: _Buffer())
    monkeypatch.setattr(module, "reduce_tensor", lambda _buffer: (None, ("ipc",)))
    monkeypatch.setattr(
        module.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: synchronizations.append(None)),
    )
    return engine, phases, sends, synchronizations


def test_materialization_boundaries_gate_before_next_source(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    engine, phases, sends, synchronizations = _make_engine(
        module,
        monkeypatch,
        [_Tensor("a"), _Tensor("b"), _Tensor("c")],
        {0, 2},
    )

    engine._send_planned_packed()

    assert phases == [
        ("vllm-native-buffer-allocation", False),
        ("vllm-native-materialize-0", False),
        ("vllm-native-materialize-2", False),
        ("vllm-native-transfer-0", False),
    ]
    assert sends == [["a", "b", "c"]]
    assert len(synchronizations) == 2


def test_materialization_failure_uses_next_safe_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    engine, phases, sends, _synchronizations = _make_engine(
        module,
        monkeypatch,
        [_Tensor("a"), _Tensor("b", fail=True), _Tensor("c")],
        {0, 2},
    )

    with pytest.raises(RuntimeError, match="vllm-native-materialize-2"):
        engine._send_planned_packed()

    assert phases[-1] == ("vllm-native-materialize-2", True)
    assert sends == []


@pytest.mark.parametrize("boundaries", [set(), {0}, {-1, 2}, {2, 3}])
def test_materialization_boundaries_require_exact_final_index(
    monkeypatch: pytest.MonkeyPatch,
    boundaries: set[int],
) -> None:
    module = _load_module(monkeypatch)

    with pytest.raises(ValueError):
        module.VLLMNativeIPCTrainerEngine._validate_materialization_boundaries(
            frozenset(boundaries),
            tensor_count=3,
        )
