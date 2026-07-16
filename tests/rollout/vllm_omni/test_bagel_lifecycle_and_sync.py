from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR, Dispatch
from unirl.distributed.weight_sync.full.cpu_staged import (
    BAGEL_VLLM_OMNI_020_LOAD_PLAN,
    CPUStagedFullWeightSync,
)
from unirl.rollout.engine.vllm_omni.backends.native import VLLMOmniBackend
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine
from unirl.trainer.base import build_sampling_dict
from unirl.trainer.unified_model import UnifiedModelTrainer, _reduce_rollout_boundary_metrics
from unirl.utils.scheduler_utils import AllSDEScheduler


class _Task:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _LifecycleRPC:
    def __init__(self) -> None:
        self.num_stages = 2
        self.calls: list[tuple[str, int]] = []
        self.responses: dict[tuple[str, int], list[str]] = {}

    def queue(self, method: str, stage_id: int, *statuses: str) -> None:
        self.responses[(method, stage_id)] = list(statuses)

    def collective_rpc(self, *, method, stage_ids, **_kwargs):
        stage_id = int(stage_ids[0])
        self.calls.append((method, stage_id))
        status = self.responses.get((method, stage_id), ["SUCCESS"]).pop(0)
        return [[SimpleNamespace(status=status, error_msg=None if status == "SUCCESS" else "injected")]]


def _native_backend(rpc: _LifecycleRPC) -> VLLMOmniBackend:
    return VLLMOmniBackend(
        SimpleNamespace(engine=rpc),
        {"OmniSleepTask": _Task, "OmniWakeTask": _Task},
        tokenizer=None,
        tp_per_stage={0: 1, 1: 1},
    )


def _strict_staged_sync(rollout=None) -> CPUStagedFullWeightSync:
    backend = SimpleNamespace(rollout_adapter_name="default", model=torch.nn.Linear(1, 1))
    return CPUStagedFullWeightSync(
        backend=backend,
        rollout=rollout or SimpleNamespace(),
        verify_names=["language_model.model.layers.0.input_layernorm_moe_gen.weight"],
        load_plan=BAGEL_VLLM_OMNI_020_LOAD_PLAN,
    )


def test_partial_sleep_ack_is_retried_without_hiding_awake_stage() -> None:
    rpc = _LifecycleRPC()
    rpc.queue("handle_sleep_task", 0, "SUCCESS")
    rpc.queue("handle_sleep_task", 1, "ERROR", "SUCCESS")
    backend = _native_backend(rpc)

    with pytest.raises(RuntimeError, match="stage 1"):
        backend.sleep_task()
    assert not backend.all_stages_sleeping()

    backend.sleep_task()
    assert backend.all_stages_sleeping()
    assert rpc.calls == [
        ("handle_sleep_task", 0),
        ("handle_sleep_task", 1),
        ("handle_sleep_task", 1),
    ]


def test_partial_wake_is_conservatively_targeted_by_cleanup_sleep() -> None:
    rpc = _LifecycleRPC()
    backend = _native_backend(rpc)
    backend.sleep_task()
    assert backend.all_stages_sleeping()

    rpc.queue("handle_wake_task", 0, "SUCCESS")
    rpc.queue("handle_wake_task", 1, "ERROR")
    with pytest.raises(RuntimeError, match="stage 1"):
        backend.wake_task()
    assert not backend.all_stages_sleeping()

    backend.sleep_task()
    assert backend.all_stages_sleeping()


@pytest.mark.parametrize("action", ["sleep", "wake"])
def test_engine_fails_closed_after_partial_lifecycle_failure(action: str) -> None:
    class _Backend:
        def __init__(self) -> None:
            self.sleep_calls = 0

        def all_stages_sleeping(self) -> bool:
            return False

        def sleep_task(self) -> None:
            self.sleep_calls += 1
            if self.sleep_calls == 1:
                raise RuntimeError("partial sleep")

        def wake_task(self) -> None:
            raise RuntimeError("partial wake")

    engine = VLLMOmniRolloutEngine.__new__(VLLMOmniRolloutEngine)
    engine._backend = _Backend()
    engine._is_offloaded = action == "wake"

    with pytest.raises(RuntimeError, match=f"partial {action}"):
        getattr(engine, action if action == "sleep" else "wake_up")()

    assert engine.is_offloaded

    if action == "sleep":
        # Unavailable does not suppress a cleanup retry while stages may remain
        # awake; only all_stages_sleeping() makes sleep() an idempotent no-op.
        with pytest.raises(RuntimeError, match="remain awake"):
            engine.sleep()
        assert engine._backend.sleep_calls == 2


def test_failed_rollout_sleep_never_onloads_trainer_gpu_state() -> None:
    class _Rollout:
        shutdown_calls = 0

        def wake_up(self) -> None:
            return None

        def sleep(self) -> None:
            raise RuntimeError("injected partial sleep")

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare_for_compute(self) -> None:
            self.calls.append("compute")

        def prepare_for_rollout(self) -> None:
            self.calls.append("rollout")

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = True
    trainer.weight_sync = None
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with pytest.raises(RuntimeError, match="cleanup failed in rollout.sleep"):
        with trainer._external_single_engine_session(sync_weights=False, onload_trainer_after=True):
            pass
    assert trainer.backend.calls == []
    assert trainer.rollout.shutdown_calls == 1


def test_transient_sleep_failure_retries_before_trainer_onload() -> None:
    class _Rollout:
        def __init__(self) -> None:
            self.sleep_calls = 0

        def wake_up(self) -> None:
            return None

        def sleep(self) -> None:
            self.sleep_calls += 1
            if self.sleep_calls == 1:
                raise RuntimeError("transient")

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def prepare_for_compute(self) -> None:
            self.calls.append("compute")

        def prepare_for_rollout(self) -> None:
            self.calls.append("rollout")

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = True
    trainer.weight_sync = None
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with trainer._external_single_engine_session(sync_weights=False, onload_trainer_after=True):
        pass
    assert trainer.rollout.sleep_calls == 2
    assert trainer.backend.calls == ["compute"]


def test_optimizer_parking_orders_extract_park_wake_push_sleep_discard_restore() -> None:
    events: list[str] = []

    class _Rollout:
        def wake_up(self) -> None:
            events.append("wake")

        def sleep(self) -> None:
            events.append("sleep")

    class _Sync:
        def extract(self) -> None:
            events.append("extract")

        def push(self) -> None:
            events.append("push")

        def discard(self) -> None:
            events.append("discard")

    class _Backend:
        def park_optimizer_state_for_rollout(self):
            events.append("park")
            return [
                {
                    "grad_bytes_cleared": 10.0,
                    "optimizer_state_bytes_parked": 20.0,
                    "optimizer_park_host_time_s": 1.0,
                },
                {
                    "grad_bytes_cleared": 11.0,
                    "optimizer_state_bytes_parked": 21.0,
                    "optimizer_park_host_time_s": 2.0,
                },
            ]

        def restore_optimizer_state_after_rollout(self):
            events.append("restore")
            return [
                {"optimizer_state_bytes_restored": 20.0, "optimizer_restore_host_time_s": 3.0},
                {"optimizer_state_bytes_restored": 21.0, "optimizer_restore_host_time_s": 2.5},
            ]

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = True
    trainer._enable_fsdp_offload = False
    trainer._park_optimizer_state_during_rollout = True
    trainer.weight_sync = _Sync()
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with trainer._external_single_engine_session(sync_weights=True, onload_trainer_after=True) as metrics:
        events.append("generate")

    assert events == ["extract", "park", "wake", "push", "generate", "sleep", "discard", "restore"]
    assert metrics == {
        "grad_bytes_cleared": 21.0,
        "optimizer_state_bytes_parked": 41.0,
        "optimizer_park_host_time_s": 2.0,
        "optimizer_state_bytes_restored": 41.0,
        "optimizer_restore_host_time_s": 3.0,
    }


@pytest.mark.parametrize("failure_at", ["wake", "generate"])
def test_optimizer_parking_restores_after_rollout_failure(failure_at: str) -> None:
    events: list[str] = []

    class _Rollout:
        def wake_up(self) -> None:
            events.append("wake")
            if failure_at == "wake":
                raise RuntimeError("injected wake failure")

        def sleep(self) -> None:
            events.append("sleep")

    class _Backend:
        def park_optimizer_state_for_rollout(self):
            events.append("park")
            return {}

        def restore_optimizer_state_after_rollout(self):
            events.append("restore")
            return {}

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = False
    trainer._park_optimizer_state_during_rollout = True
    trainer.weight_sync = None
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with pytest.raises(RuntimeError, match=f"injected {failure_at} failure"):
        with trainer._external_single_engine_session(sync_weights=False, onload_trainer_after=True):
            events.append("generate")
            if failure_at == "generate":
                raise RuntimeError("injected generate failure")

    expected = ["park", "wake", "sleep", "restore"]
    if failure_at == "generate":
        expected.insert(2, "generate")
    assert events == expected


def test_optimizer_parking_default_off_makes_no_backend_lifecycle_calls() -> None:
    events: list[str] = []

    class _Rollout:
        def wake_up(self) -> None:
            events.append("wake")

        def sleep(self) -> None:
            events.append("sleep")

    class _Backend:
        def park_optimizer_state_for_rollout(self):
            raise AssertionError("default-off path must not park")

        def restore_optimizer_state_after_rollout(self):
            raise AssertionError("default-off path must not restore")

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = False
    trainer._park_optimizer_state_during_rollout = False
    trainer.weight_sync = None
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with trainer._external_single_engine_session(sync_weights=False, onload_trainer_after=True) as metrics:
        events.append("generate")
    assert events == ["wake", "generate", "sleep"]
    assert metrics == {}


def test_failed_rollout_sleep_never_restores_parked_optimizer_onto_gpu() -> None:
    events: list[str] = []

    class _Rollout:
        def wake_up(self) -> None:
            events.append("wake")

        def sleep(self) -> None:
            events.append("sleep")
            raise RuntimeError("injected partial sleep")

        def shutdown(self) -> None:
            events.append("shutdown")

    class _Backend:
        def park_optimizer_state_for_rollout(self):
            events.append("park")
            return {}

        def restore_optimizer_state_after_rollout(self):
            raise AssertionError("optimizer state must stay on CPU while Omni may be awake")

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = False
    trainer._park_optimizer_state_during_rollout = True
    trainer.weight_sync = None
    trainer.rollout = _Rollout()
    trainer.backend = _Backend()

    with pytest.raises(RuntimeError, match="cleanup failed in rollout.sleep"):
        with trainer._external_single_engine_session(sync_weights=False, onload_trainer_after=True):
            pass

    assert events == ["park", "wake", "sleep", "sleep", "shutdown"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"single_engine": False},
        {"rollout_is_trainside": True},
        {"enable_fsdp_offload": True},
        {"backend_persistent_cpu_offload": True},
    ],
)
def test_optimizer_parking_rejects_non_external_or_fsdp_offload_modes(kwargs) -> None:
    contract = {
        "enabled": True,
        "single_engine": True,
        "rollout_is_trainside": False,
        "enable_fsdp_offload": False,
        "backend_persistent_cpu_offload": False,
        **kwargs,
    }
    with pytest.raises(ValueError):
        UnifiedModelTrainer._validate_optimizer_state_parking_contract(**contract)


def test_optimizer_parking_is_default_off_and_metric_reduction_is_empty_safe() -> None:
    parameter = inspect.signature(UnifiedModelTrainer.__init__).parameters["park_optimizer_state_during_rollout"]
    assert parameter.default is False
    assert _reduce_rollout_boundary_metrics(None) == {}


def test_vllm_omni_shutdown_is_driver_dispatchable() -> None:
    config = getattr(VLLMOmniRolloutEngine.shutdown, DISTRIBUTED_CONFIG_ATTR)
    assert config["dispatch_mode"] is Dispatch.BROADCAST


def test_staged_sync_requires_checksum_sentinel() -> None:
    backend = SimpleNamespace(rollout_adapter_name="default", model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="verify_names"):
        CPUStagedFullWeightSync(
            backend=backend,
            rollout=SimpleNamespace(),
            verify_names=[],
        )


@pytest.mark.parametrize(
    ("source", "expected_lane", "expected_target"),
    [
        ("self_attn.q_proj_moe_gen.weight", 0, "self_attn.qkv_proj_moe_gen.weight"),
        ("self_attn.k_proj_moe_gen.bias", 1, "self_attn.qkv_proj_moe_gen.bias"),
        ("self_attn.v_proj_moe_gen.weight", 2, "self_attn.qkv_proj_moe_gen.weight"),
        ("self_attn.q_proj.weight", 0, "self_attn.qkv_proj.weight"),
        ("self_attn.k_proj.bias", 1, "self_attn.qkv_proj.bias"),
        ("self_attn.v_proj.weight", 2, "self_attn.qkv_proj.weight"),
        ("mlp.gate_proj.weight", 0, "mlp.gate_up_proj.weight"),
        ("mlp.up_proj.weight", 1, "mlp.gate_up_proj.weight"),
        ("mlp_moe_gen.gate_proj.weight", 0, "mlp_moe_gen.gate_up_proj.weight"),
        ("mlp_moe_gen.up_proj.weight", 1, "mlp_moe_gen.gate_up_proj.weight"),
        ("input_layernorm_moe_gen.weight", 0, "input_layernorm_moe_gen.weight"),
    ],
)
def test_bagel_020_load_plan_maps_packed_sources(
    source: str,
    expected_lane: int,
    expected_target: str,
) -> None:
    prefix = "language_model.model.layers.0."
    lane, target = CPUStagedFullWeightSync._bagel_vllm_omni_020_target(prefix + source)
    assert lane == expected_lane
    assert target == prefix + expected_target


def test_bagel_020_load_plan_partitions_source_injective_lanes() -> None:
    sync = _strict_staged_sync()
    prefix = "language_model.model.layers.0."
    names = [
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.v_proj.weight",
        prefix + "mlp_moe_gen.gate_proj.weight",
        prefix + "mlp_moe_gen.up_proj.weight",
        prefix + "input_layernorm_moe_gen.weight",
    ]
    sync._snapshot = [(name, torch.ones(1, dtype=torch.bfloat16)) for name in names]

    buckets = list(sync._iter_snapshot_buckets())

    assert [[name for name, _ in bucket] for bucket, _ in buckets] == [
        [names[0], names[3], names[5]],
        [names[1], names[4]],
        [names[2]],
    ]
    assert [is_last for _, is_last in buckets] == [False, False, True]


def test_staged_sync_strict_loader_ack_is_fail_closed() -> None:
    expected = [
        "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "language_model.model.layers.0.mlp_moe_gen.gate_up_proj.weight",
        "language_model.model.layers.0.input_layernorm_moe_gen.weight",
    ]
    good = {0: [[{"received_count": 3, "loaded_count": 3, "loaded_names": list(reversed(expected))}]]}
    CPUStagedFullWeightSync._validate_bucket_ack(
        good,
        stage_id=0,
        expected_received=3,
        expected_loaded_names=expected,
    )

    with pytest.raises(RuntimeError, match="rejected a bucket"):
        CPUStagedFullWeightSync._validate_bucket_ack(
            {0: [[{"received_count": 3, "loaded_count": 2, "loaded_names": expected[:2]}]]},
            stage_id=0,
            expected_received=3,
            expected_loaded_names=expected,
        )
    with pytest.raises(RuntimeError, match="unexpected"):
        CPUStagedFullWeightSync._validate_bucket_ack(
            {
                0: [
                    [
                        {
                            "received_count": 3,
                            "loaded_count": 3,
                            "loaded_names": [*expected[:2], "language_model.wrong.weight"],
                        }
                    ]
                ]
            },
            stage_id=0,
            expected_received=3,
            expected_loaded_names=expected,
        )
    with pytest.raises(RuntimeError, match="loaded_names"):
        CPUStagedFullWeightSync._validate_bucket_ack(
            {0: [[{"received_count": 3, "loaded_count": 3}]]},
            stage_id=0,
            expected_received=3,
            expected_loaded_names=expected,
        )
    with pytest.raises(RuntimeError, match="exactly one"):
        CPUStagedFullWeightSync._validate_bucket_ack(None, stage_id=0, expected_received=3)
    with pytest.raises(RuntimeError, match="wrong stage"):
        CPUStagedFullWeightSync._validate_bucket_ack(
            {1: [[{"received_count": 3, "loaded_count": 3, "loaded_names": expected}]]},
            stage_id=0,
            expected_received=3,
            expected_loaded_names=expected,
        )
    with pytest.raises(RuntimeError, match="exactly one stage result"):
        CPUStagedFullWeightSync._validate_bucket_ack(
            {
                0: [[{"received_count": 3, "loaded_count": 3, "loaded_names": expected}]],
                1: [[{"received_count": 3, "loaded_count": 3, "loaded_names": expected}]],
            },
            stage_id=0,
            expected_received=3,
            expected_loaded_names=expected,
        )


def test_tensor_worker_ack_includes_loaded_destination_names(monkeypatch) -> None:
    pytest.importorskip("zmq")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_omni")

    from unirl.distributed.weight_sync.transfer import sgl_compat
    from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
        BucketedIPCReceiveMixin,
    )

    class _Serializer:
        @staticmethod
        def deserialize(payload):
            return payload

    class _Bucket:
        def __init__(self, *, flattened_tensor, metadata) -> None:
            del flattened_tensor
            self._metadata = metadata

        def reconstruct_tensors(self):
            return self._metadata

    monkeypatch.setattr(sgl_compat, "MultiprocessingSerializer", _Serializer)
    monkeypatch.setattr(sgl_compat, "FlattenedTensorBucket", _Bucket)
    worker = BucketedIPCReceiveMixin()
    worker.local_rank = 0
    worker._diffrl_load_weights = lambda _weights: {"target.z", "target.a"}

    ack = worker.update_weights_from_tensor(
        serialized_named_tensors=[{"flattened_tensor": None, "metadata": [("source.weight", torch.ones(1))]}]
    )

    assert ack == {
        "received_count": 1,
        "loaded_count": 2,
        "loaded_names": ["target.a", "target.z"],
    }


def test_bagel_020_load_plan_flushes_only_final_send(monkeypatch) -> None:
    import sys
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "zmq", ModuleType("zmq"))
    from unirl.distributed.weight_sync.transfer import sgl_compat

    class _FlatBucket:
        def __init__(self, *, named_tensors) -> None:
            self._names = [name for name, _ in named_tensors]

        def get_flattened_tensor(self):
            return None

        def get_metadata(self):
            return self._names

    class _Serializer:
        @staticmethod
        def serialize(payload, *, output_str):
            assert output_str
            return payload

    class _Rollout:
        def __init__(self) -> None:
            self.calls = []

        def update_weights_from_tensor(
            self,
            *,
            serialized_named_tensors,
            flush_cache,
            stage_ids,
            **_kwargs,
        ):
            names = serialized_named_tensors[0]["metadata"]
            loaded_names = [CPUStagedFullWeightSync._bagel_vllm_omni_020_target(name)[1] for name in names]
            self.calls.append((flush_cache, names))
            return {
                stage_ids[0]: [
                    [
                        {
                            "received_count": len(names),
                            "loaded_count": len(loaded_names),
                            "loaded_names": loaded_names,
                        }
                    ]
                ]
            }

    monkeypatch.setattr(sgl_compat, "FlattenedTensorBucket", _FlatBucket)
    monkeypatch.setattr(sgl_compat, "MultiprocessingSerializer", _Serializer)
    rollout = _Rollout()
    sync = _strict_staged_sync(rollout)
    prefix = "language_model.model.layers.0.self_attn."
    names = [
        prefix + "q_proj.weight",
        prefix + "k_proj.weight",
        prefix + "v_proj.weight",
        prefix + "v_proj.bias",
    ]
    sync._snapshot = [(name, torch.ones(1, dtype=torch.bfloat16)) for name in names]
    sync._bucket_bytes = 4

    sync._push_stage(0)

    assert [call_names for _, call_names in rollout.calls] == [[name] for name in names]
    assert [flush for flush, _ in rollout.calls] == [False, False, False, True]


def test_bagel_stage_yaml_uses_vllm_omni_020_flat_extension_keys() -> None:
    cfg = OmegaConf.load("unirl/rollout/engine/vllm_omni/stage_configs/bagel_t2ti_rl.yaml")
    assert all("engine_extras" not in stage for stage in cfg.stages)
    assert cfg.stages[0].max_model_len == 8192
    assert cfg.stages[0].max_num_batched_tokens == 8192
    assert cfg.stages[0].max_num_seqs == 1
    assert cfg.stages[0].kv_cache_memory_bytes == 1024**3
    assert cfg.stages[0].enforce_eager is True
    assert dict(cfg.stages[0].limit_mm_per_prompt) == {"image": 0, "img2img": 0}
    assert cfg.stages[0].worker_extension_cls.endswith("BagelARWeightSyncExtension")
    assert cfg.stages[1].custom_pipeline_args.pipeline_class.endswith("RLBagelPipeline")


def test_bagel_recipe_uses_inference_replay_and_trainable_checksum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPORT_TO_WANDB", "false")
    cfg = OmegaConf.load("examples/unified_model/bagel_vllmomni_t2ti.yaml")
    assert cfg.num_devices == 32
    assert cfg.devices_per_node == 8
    assert cfg.batch_size == 32
    assert cfg.logging.report_to_wandb is False
    assert cfg.pipeline.replay_mode == "inference"
    assert cfg.sync.load_plan == BAGEL_VLLM_OMNI_020_LOAD_PLAN
    assert cfg.sync.verify_names == ["language_model.model.layers.0.input_layernorm_moe_gen.weight"]
    UnifiedModelTrainer._validate_bagel_t2ti_contract(build_sampling_dict(cfg.sampling), cfg.sync)
    scheduler = AllSDEScheduler(
        num_timesteps=cfg.sampling.diffusion.scheduler.num_timesteps,
        timestep_fraction=cfg.sampling.diffusion.scheduler.timestep_fraction,
        num_sde_steps=cfg.sampling.diffusion.scheduler.num_sde_steps,
    )
    assert scheduler.get_sde_indices(step=0) == {2, 3, 4}


def test_bagel_load_plan_requires_both_native_stages() -> None:
    backend = SimpleNamespace(rollout_adapter_name="default", model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="exactly Stage 0 and Stage 1"):
        CPUStagedFullWeightSync(
            backend=backend,
            rollout=SimpleNamespace(),
            stage_ids=[0],
            verify_names=["language_model.model.layers.0.input_layernorm_moe_gen.weight"],
            load_plan=BAGEL_VLLM_OMNI_020_LOAD_PLAN,
        )


def test_bagel_trainer_contract_rejects_missing_or_partial_sync() -> None:
    sampling = {
        "ar": SimpleNamespace(samples_per_prompt=4),
        "diffusion": SimpleNamespace(samples_per_prompt=1),
    }
    with pytest.raises(ValueError, match="strict full-weight sync"):
        UnifiedModelTrainer._validate_bagel_t2ti_contract(sampling, None)

    sync_cfg = OmegaConf.create(
        {
            "_target_": "unirl.distributed.weight_sync.full.CPUStagedFullWeightSync",
            "load_plan": BAGEL_VLLM_OMNI_020_LOAD_PLAN,
            "stage_ids": [0],
        }
    )
    with pytest.raises(ValueError, match=r"exactly \{0, 1\}"):
        UnifiedModelTrainer._validate_bagel_t2ti_contract(sampling, sync_cfg)


def test_bagel_trainer_rejects_skipped_weight_sync() -> None:
    trainer = object.__new__(UnifiedModelTrainer)
    trainer._strict_bagel_t2ti = True
    with pytest.raises(ValueError, match="weight_sync_interval=1"):
        trainer.train(num_rollouts=1, weight_sync_interval=2)
