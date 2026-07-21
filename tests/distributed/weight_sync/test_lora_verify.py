from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint


class LoraVerifyTest(unittest.TestCase):
    def test_readback_descends_into_worker_manager_and_hashes_packed_shards(self) -> None:
        torch = types.ModuleType("torch")
        torch.Tensor = _FakeTensor

        transfer = types.ModuleType("unirl.distributed.weight_sync.transfer")
        transfer.__path__ = []

        bucketed_transfer = types.ModuleType("unirl.distributed.weight_sync.transfer.bucketed_transfer")
        bucketed_transfer.BucketedWeightReceiver = object

        ipc_dispatch = types.ModuleType("unirl.distributed.weight_sync.transfer.ipc_dispatch")
        ipc_dispatch.DIFFRL_LORA_INT_ID = 1
        ipc_dispatch.DIFFRL_LORA_NAME = "test"
        ipc_dispatch.DIFFRL_LORA_PATH = "/tmp/test"
        ipc_dispatch.replica_rank_from_env = lambda: 0
        ipc_dispatch.zmq_handle = lambda **kwargs: kwargs

        checksum = types.ModuleType("unirl.distributed.weight_sync.transfer.checksum")
        checksum.fingerprint_tensor = lambda tensor: tensor.fingerprint

        patches = types.ModuleType("unirl.rollout.engine.vllm_omni.patches")
        patches.__path__ = []

        runtime = types.ModuleType("unirl.rollout.engine.vllm_omni.patches.runtime")
        runtime.OmniTensorLoRARequest = object

        class _VLLMOmniHijack:
            @staticmethod
            def hijack() -> None:
                pass

        runtime.VLLMOmniHijack = _VLLMOmniHijack

        stubs = {
            "torch": torch,
            "unirl.distributed.weight_sync.transfer": transfer,
            "unirl.distributed.weight_sync.transfer.bucketed_transfer": bucketed_transfer,
            "unirl.distributed.weight_sync.transfer.ipc_dispatch": ipc_dispatch,
            "unirl.distributed.weight_sync.transfer.checksum": checksum,
            "unirl.rollout.engine.vllm_omni.patches": patches,
            "unirl.rollout.engine.vllm_omni.patches.runtime": runtime,
        }
        with patch.dict(sys.modules, stubs):
            module = _load_module(
                "_test_ipc_receive_mixin",
                "unirl/rollout/engine/vllm_omni/worker/ipc_receive_mixin.py",
            )

            flat = types.SimpleNamespace(
                lora_a=_FakeTensor("flat-a"),
                lora_b=_FakeTensor("flat-b"),
                bias=None,
                embeddings_tensor=None,
            )
            packed = types.SimpleNamespace(
                lora_a=[_FakeTensor("q-a"), None, _FakeTensor("v-a")],
                lora_b=(_FakeTensor("q-b"), None, _FakeTensor("v-b")),
                bias=None,
                embeddings_tensor=None,
            )
            lora_model = types.SimpleNamespace(loras={"flat": flat, "qkv_proj": packed})
            inner_manager = types.SimpleNamespace(_registered_adapters={7: lora_model})
            outer_manager = types.SimpleNamespace(_adapter_manager=inner_manager)

            worker = object.__new__(module.BucketedIPCReceiveMixin)
            worker.lora_manager = outer_manager
            worker.model_runner = None

            loaded = worker._diffrl_loaded_lora_checksums(adapter_id=7)

        self.assertEqual(
            loaded,
            {
                "flat": {"lora_a": "flat-a", "lora_b": "flat-b"},
                "qkv_proj": {
                    "lora_a.0": "q-a",
                    "lora_a.2": "v-a",
                    "lora_b.0": "q-b",
                    "lora_b.2": "v-b",
                },
            },
        )

    def test_assert_loaded_accepts_mixed_flat_and_packed_checksums(self) -> None:
        remote = types.ModuleType("unirl.distributed.group.remote")

        class _Remote:
            pass

        remote.Remote = _Remote
        with patch.dict(sys.modules, {"unirl.distributed.group.remote": remote}):
            module = _load_module(
                "_test_lora_sync_base",
                "unirl/distributed/weight_sync/lora/base.py",
            )

        sync = object.__new__(module.LoraWeightSyncBase)
        sync._param_prefix = ""
        loaded = {
            0: [
                {
                    "flat": {"lora_a": "flat-a", "lora_b": "flat-b"},
                    "qkv_proj": {
                        "lora_a.0": "q-a",
                        "lora_a.2": "v-a",
                        "lora_b.0": "q-b",
                        "lora_b.2": "v-b",
                    },
                }
            ]
        }

        sync._assert_loaded(
            ["flat-a", "q-a", "v-a"],
            ["flat-b", "q-b", "v-b"],
            loaded,
            label="mixed flat/packed",
        )

    def test_assert_loaded_still_rejects_a_missing_packed_shard(self) -> None:
        remote = types.ModuleType("unirl.distributed.group.remote")

        class _Remote:
            pass

        remote.Remote = _Remote
        with patch.dict(sys.modules, {"unirl.distributed.group.remote": remote}):
            module = _load_module(
                "_test_lora_sync_base_missing",
                "unirl/distributed/weight_sync/lora/base.py",
            )

        sync = object.__new__(module.LoraWeightSyncBase)
        sync._param_prefix = ""
        loaded = {
            0: [
                {
                    "qkv_proj": {
                        "lora_a.0": "q-a",
                        "lora_b.0": "q-b",
                    }
                }
            ]
        }

        with self.assertRaisesRegex(RuntimeError, r"engine loaded 1 / 1"):
            sync._assert_loaded(
                ["k-a", "q-a"],
                ["k-b", "q-b"],
                loaded,
                label="missing packed shard",
            )


if __name__ == "__main__":
    unittest.main()
