from __future__ import annotations

import ast
import copy
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

import unirl.utils.profiling as profiling


class _RangeRecorder:
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.paths: list[str] = []
        self.events: list[tuple[str, str]] = []

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        self.stack.append(name)
        path = "/".join(self.stack)
        self.paths.append(path)
        self.events.append(("enter", path))
        try:
            yield
        finally:
            self.events.append(("exit", path))
            self.stack.pop()


class _Tokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = None

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(char) for char in text]


def _build_sglang_inputs(
    seed: int | None,
    *,
    sample_ids: list[str] | None = None,
    prompts: list[str] | None = None,
    samples_pre_expanded: bool = True,
    deterministic: bool | None = None,
):
    from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
    from unirl.rollout.engine.sglang.utils.sampling import resolve_sampling
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    sample_ids = sample_ids or [
        "prompt:group:sample:0",
        "prompt:group:sample:1",
        "prompt:group:sample:2",
    ]
    prompts = prompts or ["a", "b", "c"]
    if deterministic is None:
        deterministic = seed is not None
    config = SimpleNamespace(
        pretrained_model_ckpt_path="unused",
        image_token=None,
        chat_template_kwargs={},
        samples_pre_expanded=samples_pre_expanded,
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.9,
        system_instruction=None,
        engine_kwargs={"enable_deterministic_inference": deterministic},
    )
    req = RolloutReq(
        sample_ids=sample_ids,
        group_ids=["group"] * len(sample_ids),
        primitives={"text": Texts(texts=prompts)},
        sampling_params={
            "ar": ARSamplingParams(
                samples_per_prompt=3,
                temperature=0.6,
                max_new_tokens=16,
                top_p=0.8,
                top_k=0,
                seed=seed,
            )
        },
    )
    adapter = TextLMAdapter(config, tokenizer=_Tokenizer())
    sampling = resolve_sampling(config, req)
    return sampling, adapter.build_inputs(req, sampling=sampling)


def test_sglang_expanded_request_derives_stable_sampling_seeds() -> None:
    from unirl.rollout.engine.sglang.backends.native import payload_to_generate_kwargs
    from unirl.rollout.engine.sglang.utils.sampling import derive_sampling_seed

    sample_ids = [
        "prompt:group:sample:0",
        "prompt:group:sample:1",
        "prompt:group:sample:2",
    ]
    expected = [derive_sampling_seed(1234, sample_id) for sample_id in sample_ids]
    assert len(set(expected)) == len(expected)
    with pytest.raises(ValueError, match="base_seed"):
        derive_sampling_seed(-1, sample_ids[0])
    with pytest.raises(ValueError, match="integer"):
        derive_sampling_seed(True, sample_ids[0])
    with pytest.raises(ValueError, match="integer"):
        derive_sampling_seed(1.5, sample_ids[0])
    with pytest.raises(ValueError, match="sample_id"):
        derive_sampling_seed(1234, "")
    assert 0 <= derive_sampling_seed((1 << 63) - 1, sample_ids[0]) < (1 << 63)

    sampling, prepared = _build_sglang_inputs(1234, sample_ids=sample_ids)
    assert sampling.n == 1
    assert "sampling_seed" not in sampling.block
    assert [payload["sampling_params"]["sampling_seed"] for payload in prepared.wire] == expected

    # Seed assignment is keyed by identity, so request reorder and DP-style
    # slicing cannot change a logical sample's seed.
    reversed_ids = list(reversed(sample_ids))
    _, reordered = _build_sglang_inputs(
        1234,
        sample_ids=reversed_ids,
        prompts=["c", "b", "a"],
    )
    reordered_by_id = {
        sample_id: payload["sampling_params"]["sampling_seed"]
        for sample_id, payload in zip(reversed_ids, reordered.wire)
    }
    assert reordered_by_id == dict(zip(sample_ids, expected))
    _, shard = _build_sglang_inputs(
        1234,
        sample_ids=[sample_ids[1]],
        prompts=["b"],
    )
    assert shard.wire[0]["sampling_params"]["sampling_seed"] == expected[1]

    # Native Engine.async_generate receives the same nested 0.5.12 API key and
    # the mapper does not mutate the HTTP-shaped payload.
    payload = prepared.wire[1]
    snapshot = copy.deepcopy(payload)
    kwargs = payload_to_generate_kwargs(payload)
    assert kwargs["sampling_params"]["sampling_seed"] == expected[1]
    assert payload == snapshot


def test_seeded_sglang_requires_deterministic_preexpanded_requests() -> None:
    from unirl.trainer.ar import ARTrainer
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    with pytest.raises(ValueError, match="enable_deterministic_inference"):
        _build_sglang_inputs(1234, deterministic=False)
    with pytest.raises(ValueError, match="n=1"):
        _build_sglang_inputs(1234, samples_pre_expanded=False)
    with pytest.raises(ValueError, match="unique sample_ids"):
        _build_sglang_inputs(
            1234,
            sample_ids=["duplicate", "duplicate"],
            prompts=["a", "b"],
        )
    full_req = RolloutReq(
        sample_ids=["duplicate", "duplicate"],
        group_ids=["group", "group"],
        primitives={"text": Texts(texts=["a", "a"])},
        sampling_params={"ar": ARSamplingParams(seed=1234)},
    )
    with pytest.raises(ValueError, match="globally unique sample_ids"):
        ARTrainer._validate_seeded_request(full_req)


def test_none_seed_keeps_legacy_sglang_payload_exactly() -> None:
    from unirl.types.sampling import ARSamplingParams

    assert ARSamplingParams().seed is None
    sampling, prepared = _build_sglang_inputs(None)
    assert sampling.base_seed is None
    assert sampling.block == {
        "temperature": 0.6,
        "max_new_tokens": 16,
        "top_p": 0.8,
        "top_k": -1,
        "n": 1,
    }
    assert prepared.wire == [
        {
            "sampling_params": dict(sampling.block),
            "return_logprob": True,
            "logprob_start_len": 0,
            "text": prompt,
        }
        for prompt in ("a", "b", "c")
    ]
    assert all("sampling_seed" not in payload["sampling_params"] for payload in prepared.wire)


def test_vlm_adapter_uses_shared_stable_sampling_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import unirl.rollout.engine.sglang.adapters.vlm as vlm_module
    from unirl.rollout.engine.sglang.adapters.base import MMEncoding
    from unirl.rollout.engine.sglang.adapters.vlm import VLMAdapter
    from unirl.rollout.engine.sglang.utils.sampling import (
        derive_sampling_seed,
        resolve_sampling,
    )
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import ARSamplingParams

    sample_ids = ["prompt:vlm:sample:0", "prompt:vlm:sample:1"]
    config = SimpleNamespace(
        pretrained_model_ckpt_path="unused",
        image_token="<image>",
        chat_template_kwargs={},
        samples_pre_expanded=True,
        temperature=0.7,
        max_new_tokens=16,
        top_p=0.9,
        system_instruction=None,
        engine_kwargs={"enable_deterministic_inference": True},
    )
    req = RolloutReq(
        sample_ids=sample_ids,
        group_ids=["vlm", "vlm"],
        primitives={"text": Texts(texts=["a", "b"])},
        sampling_params={"ar": ARSamplingParams(seed=4321, top_k=0)},
    )
    adapter = VLMAdapter(
        config,
        tokenizer=_Tokenizer(),
        processor=object(),
    )
    images = [object(), object()]
    monkeypatch.setattr(
        adapter,
        "extract_images",
        lambda _req, *, n_prompts: images[:n_prompts],
    )
    monkeypatch.setattr(
        adapter,
        "encode_mm",
        lambda prompt, image, _system_instruction: MMEncoding(
            image=image,
            text=prompt,
            input_ids=[ord(prompt)],
        ),
    )
    monkeypatch.setattr(vlm_module, "pil_to_base64", lambda _image: "encoded")

    sampling = resolve_sampling(config, req)
    prepared = adapter.build_inputs(req, sampling=sampling)
    assert [payload["sampling_params"]["sampling_seed"] for payload in prepared.wire] == [
        derive_sampling_seed(4321, sample_id) for sample_id in sample_ids
    ]


def test_qwen3_dppo_recipe_keeps_determinism_profiling_only() -> None:
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    recipe = root / "examples/ar/qwen3_dppo_4b_base_dapo_sglang.yaml"
    config = OmegaConf.load(recipe)
    assert config.rollout.config.engine_kwargs.enable_deterministic_inference is False
    assert config.sampling.get("seed") is None

    launcher = (root / "scripts/profiling/run_qwen3_ar_full_rl.sh").read_text(encoding="utf-8")
    assert '"+sampling.seed=${SEED}"' in launcher
    assert '"data_source.args.run.seed=${SEED}"' in launcher
    assert '"rollout.config.engine_kwargs.enable_deterministic_inference=true"' in launcher


class _FakeTransformer:
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **_kwargs):
        return {"input_ids": input_ids}

    def __call__(self, *, input_ids: torch.Tensor, return_dict: bool):
        assert return_dict
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 4), -10.0)
        logits[..., 2] = 10.0
        return SimpleNamespace(logits=logits)

    @staticmethod
    def _update_model_kwargs_for_generation(_out, model_kwargs):
        updated = dict(model_kwargs)
        updated["past_key_values"] = object()
        updated["cache_position"] = updated["cache_position"][-1:] + 1
        updated["attention_mask"] = torch.cat(
            [
                updated["attention_mask"],
                torch.ones(
                    (updated["attention_mask"].shape[0], 1),
                    dtype=updated["attention_mask"].dtype,
                ),
            ],
            dim=1,
        )
        return updated


def test_qwen3_local_autoregress_records_real_nested_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    import unirl.models.qwen3.ar as qwen3_ar
    from unirl.models.qwen3.conditions import Qwen3ARConditions
    from unirl.types.conditions import TextTokenCondition
    from unirl.types.sampling import ARSamplingParams

    recorder = _RangeRecorder()
    monkeypatch.setattr(qwen3_ar, "nvtx_range", recorder.range)

    stage = object.__new__(qwen3_ar.Qwen3ARStage)
    stage.model = SimpleNamespace(
        transformer=_FakeTransformer(),
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=None),
    )
    conditions = Qwen3ARConditions(
        prompt=TextTokenCondition(
            input_ids=torch.tensor([[1, 3]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )
    )

    segment = stage.autoregress(
        conditions,
        sampling_params=ARSamplingParams(
            temperature=0.0,
            max_new_tokens=2,
            top_p=1.0,
            top_k=0,
        ),
    )

    assert segment.lengths.tolist() == [2]
    assert recorder.paths == [
        "unirl.ar.rollout",
        "unirl.ar.rollout/unirl.ar.prefill",
        "unirl.ar.rollout/unirl.ar.kv_cache",
        "unirl.ar.rollout/unirl.ar.decode",
        "unirl.ar.rollout/unirl.ar.kv_cache",
    ]


def test_dppo_loss_and_backward_ranges_do_not_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    import unirl.algorithms.dppo as dppo_module
    from unirl.types.segments.text import TextSegment

    class _Stage:
        def __init__(self) -> None:
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def replay(self, _conditions, *, segment, temperature):
            assert temperature == 1.0
            return self.value.expand(segment.tokens.numel())

    recorder = _RangeRecorder()
    monkeypatch.setattr(dppo_module, "nvtx_range", recorder.range)
    stage = _Stage()
    algorithm = dppo_module.DPPO(stage=stage, sampling_temperature=1.0)
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2], dtype=torch.long)],
        log_probs=[torch.zeros(2, dtype=torch.float32)],
    )

    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.ones(1),
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert result.has_backward
    assert stage.value.grad is not None
    assert recorder.events == [
        ("enter", "unirl.ar.loss"),
        ("exit", "unirl.ar.loss"),
        ("enter", "unirl.ar.backward"),
        ("exit", "unirl.ar.backward"),
    ]


def test_all_required_range_names_are_spelled_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "unirl/rollout/engine/sglang/engine.py",
        "unirl/models/qwen3/ar.py",
        "unirl/algorithms/dppo.py",
        "unirl/trainer/ar.py",
        "unirl/train/stack/base.py",
        "unirl/train/backend/base_backend.py",
        "unirl/distributed/weight_sync/full/tensor.py",
        "unirl/reward/service.py",
    ]
    required = {
        "unirl.ar.rollout",
        "unirl.ar.prefill",
        "unirl.ar.decode",
        "unirl.ar.kv_cache",
        "unirl.ar.logprob_replay",
        "unirl.ar.loss",
        "unirl.ar.backward",
        "unirl.ar.optimizer",
        "unirl.ar.weight_sync",
        "unirl.ar.reward",
        "unirl.ar.eval",
        "unirl.rl.train_track",
    }
    found: set[str] = set()
    for relative_path in relative_paths:
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        found.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in required
        )
        assert not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize"
            for node in ast.walk(tree)
        )
    assert found == required

    # UniRL can only observe the SGLang request boundary. Scheduler-internal
    # prefill/decode/KV ranges belong exclusively to the local Qwen3 loop.
    sglang_tree = ast.parse((root / "unirl/rollout/engine/sglang/engine.py").read_text(encoding="utf-8"))
    sglang_range_names = {
        call.args[0].value
        for call in ast.walk(sglang_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "nvtx_range"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }
    assert sglang_range_names == {"unirl.ar.rollout"}


def test_nvtx_range_is_disabled_by_default_without_synchronizing(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("UNIRL_") and ("NVTX" in key or key == "UNIRL_PROFILE"):
            monkeypatch.delenv(key, raising=False)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("disabled nvtx_range must neither emit CUDA ranges nor synchronize")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_call)
    for name in ("range", "range_push", "range_pop"):
        if hasattr(torch.cuda.nvtx, name):
            monkeypatch.setattr(torch.cuda.nvtx, name, unexpected_call)

    with profiling.nvtx_range("disabled-by-default"):
        pass


def test_nvtx_range_balances_enabled_push_pop_without_synchronizing(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str | None]] = []

    def unexpected_sync(*_args, **_kwargs):
        raise AssertionError("nvtx_range must never synchronize CUDA")

    monkeypatch.setattr(profiling, "nvtx_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_sync)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append(("pop", None)))

    with profiling.nvtx_range("balanced"):
        events.append(("body", None))

    assert events == [
        ("push", "balanced"),
        ("body", None),
        ("pop", None),
    ]
