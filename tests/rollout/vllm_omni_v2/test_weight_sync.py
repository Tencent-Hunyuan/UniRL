"""``WeightSync`` component over a recording fake of the seam protocol.

Asserts state transitions and forwarded payloads — the v1-parity LoRA
lifecycle (cache-on-push, active re-push after wake, fail-fast) in
particular.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
import torch

from unirl.rollout.engine.vllm_omni_v2.weight_sync import WeightSync


class RecordingBackend:
    """Records every verb call; optionally fails the LoRA verbs."""

    def __init__(self, *, fail_lora: bool = False) -> None:
        self.calls: List[tuple] = []
        self.fail_lora = fail_lora

    def _record(self, name: str, **kw: Any) -> None:
        self.calls.append((name, kw))

    def update_from_ipc(self, **kw: Any) -> None:
        self._record("update_from_ipc", **kw)

    def init_weights_group(self, **kw: Any) -> None:
        self._record("init_weights_group", **kw)

    def update_from_distributed(self, **kw: Any) -> None:
        self._record("update_from_distributed", **kw)

    def destroy_weights_group(self, **kw: Any) -> None:
        self._record("destroy_weights_group", **kw)

    def update_from_tensor(self, **kw: Any) -> None:
        self._record("update_from_tensor", **kw)

    def set_lora_handle(self, **kw: Any) -> None:
        if self.fail_lora:
            raise RuntimeError("boom")
        self._record("set_lora_handle", **kw)

    def set_lora_copy(self, **kw: Any) -> None:
        if self.fail_lora:
            raise RuntimeError("boom")
        self._record("set_lora_copy", **kw)

    def param_checksums(self, *, names: List[str]) -> dict:
        self._record("param_checksums", names=names)
        return {0: [{}]}

    def lora_checksums(self, *, adapter_id: int, names: Optional[List[str]]) -> dict:
        self._record("lora_checksums", adapter_id=adapter_id, names=names)
        return {0: [{}]}

    def names(self) -> List[str]:
        return [c[0] for c in self.calls]


def make_ws(*, copy: bool = False, fail_lora: bool = False):
    backend = RecordingBackend(fail_lora=fail_lora)
    return WeightSync(backend, uses_lora=True, lora_copy_transport=copy), backend


LORA = {"transformer.x.lora_A.weight": torch.ones(2, 2)}


def test_plain_forwards_reach_the_seam():
    ws, backend = make_ws()
    ws.init_weights_update_group(
        master_address="h", master_port=1, rank_offset=0, world_size=2, group_name="g"
    )
    ws.update_weights_from_distributed(
        names=["a"], dtypes=["bfloat16"], shapes=[[2, 2]], group_name="g"
    )
    ws.destroy_weights_update_group(group_name="g")
    ws.update_weights_from_tensor(serialized_named_tensors=["blob"])
    ws.loaded_param_checksums(names=["a"])
    ws.loaded_lora_checksums(adapter_id=1)
    assert backend.names() == [
        "init_weights_group",
        "update_from_distributed",
        "destroy_weights_group",
        "update_from_tensor",
        "param_checksums",
        "lora_checksums",
    ]
    assert backend.calls[0][1]["backend"] == "nccl"  # default fan-in


def test_ipc_flips_lora_loaded_only_on_phase2():
    ws, backend = make_ws()
    ws.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=False)
    assert not ws.lora_loaded  # base-weights phase
    ws.update_weights_from_ipc(peft_config=None, base_sync_done=True)
    assert not ws.lora_loaded  # full-weight sync, no adapter
    ws.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=True)
    assert ws.lora_loaded  # phase-2 LoRA registered → activate on generate
    assert backend.names() == ["update_from_ipc"] * 3


def test_set_lora_caches_clones_for_wake_repush():
    ws, backend = make_ws()
    ws.set_lora_from_tensors("ad", LORA, peft_config={"r": 8})
    assert ws.lora_loaded and not ws.lora_dirty
    cached = ws._last_lora_tensors["transformer.x.lora_A.weight"]
    assert torch.equal(cached, LORA["transformer.x.lora_A.weight"])
    assert cached is not LORA["transformer.x.lora_A.weight"]  # cloned, not aliased
    assert backend.names() == ["set_lora_handle"]


def test_copy_variant_routes_to_copy_verb():
    ws, backend = make_ws()
    ws.set_lora_from_tensors_copy("ad", LORA)
    assert backend.names() == ["set_lora_copy"]


def test_released_then_restore_handle_and_copy():
    for copy in (False, True):
        ws, backend = make_ws(copy=copy)
        ws.set_lora_from_tensors("ad", LORA, peft_config={"r": 8})
        ws.mark_weights_released()
        assert ws.lora_dirty and ws.lora_loaded  # intent survives the release
        ws.restore_lora_after_wake()
        assert not ws.lora_dirty
        # The re-push uses the modality's transport (v1:674-685).
        expected = "set_lora_copy" if copy else "set_lora_handle"
        assert backend.names()[-1] == expected


def test_restore_noop_without_cached_adapter():
    ws, backend = make_ws()
    ws.mark_weights_released()
    ws.restore_lora_after_wake()
    assert backend.calls == []  # nothing cached → nothing pushed
    assert not ws._weights_released
    # Still dirty: LoRA is in use but no adapter was ever pushed — the
    # trainer's first sync covers it.
    assert ws.lora_dirty


def test_restore_failure_clears_flag_and_raises():
    ws, backend = make_ws()
    ws.set_lora_from_tensors("ad", LORA)
    backend.fail_lora = True
    ws.mark_weights_released()
    with pytest.raises(RuntimeError, match="LoRA-WAKE"):
        ws.restore_lora_after_wake()
    assert not ws.lora_loaded and ws.lora_dirty
