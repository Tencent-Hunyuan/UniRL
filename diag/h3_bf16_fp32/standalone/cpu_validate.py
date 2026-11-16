import argparse
import ast
import copy
import importlib.util
import sys
import time
import json
from pathlib import Path

import psutil
import torch
from accelerate import init_empty_weights

from standalone_micro import (CASES, GIB, Experiment, attach_lora, attention_context, check_host_memory,
                              configure_precision, cpu, distance, load_model, load_vendor, prepare_output,
                              sliced_kwargs, validate_dump, write_json, load_loss_api, loss_from_outputs,
                              own_cotangent, replay_kwargs, run_groups, LORA_TARGETS)
from capture_hook import context_slice, install_capture


def full_autograd(model, dump, micro, loss_api=None):
    # 小尺寸真实 H3 的常规完整图，作为逐层重算的独立导数校验。
    parameters = {f"{i}.{n}": p for i, b in enumerate(model.transformer_blocks) for n, p in b.named_parameters()}
    for p in parameters.values():
        p.requires_grad_(True)
    gradients = {n: torch.zeros_like(p, device="cpu", dtype=torch.float64) for n, p in parameters.items()}
    deltas = [[] for _ in model.transformer_blocks]
    handles, inputs, cotangent_checks = [], {}, []
    for i, block in enumerate(model.transformer_blocks):
        def hook(module, args, i=i):
            x = args[0]
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            inputs[i] = x
            return (x, *args[1:])
        handles.append(block.register_forward_pre_hook(hook))
    with attention_context(torch.device("cpu")):
        for start in range(0, 8, micro):
            kwargs = (replay_kwargs if loss_api is not None else sliced_kwargs)(dump["kwargs"], start, start + micro)
            if loss_api is not None:
                with loss_api.adapters_disabled(model), torch.no_grad():
                    refs = model(**kwargs)
            outputs = model(**kwargs)
            if loss_api is None:
                values = torch.autograd.grad(outputs, [*inputs.values(), *parameters.values()],
                                            grad_outputs=[g[start:start + micro].to(o) for g, o in zip(dump["out_grads"], outputs)])
            else:
                loss, *_ = loss_from_outputs(loss_api, outputs, kwargs, dump["loss_context"], start, start + micro, refs)
                gs = torch.autograd.grad(loss, outputs, retain_graph=True)
                detached_gs, *_ = own_cotangent(loss_api, outputs, kwargs, dump["loss_context"], start, start + micro, refs)
                cotangent_checks.extend(distance(a, b) for a, b in zip(gs, detached_gs))
                values = torch.autograd.grad(loss, [*inputs.values(), *parameters.values()])
            for i, value in enumerate(values[:len(inputs)]):
                deltas[i].append(cpu(value))
            for (name, _), value in zip(parameters.items(), values[len(inputs):]):
                gradients[name].add_(value.detach().double())
    for handle in handles:
        handle.remove()
    return {"deltas": [torch.cat(xs) for xs in deltas], "gradients": gradients,
            "cotangent_checks": cotangent_checks}


def validate_loss_options(vendor_path, vendor, model, masters, kwargs, directory):
    api = load_loss_api(vendor_path)
    model = copy.deepcopy(model)
    model.float().requires_grad_(True)
    with torch.no_grad():
        for name, value in masters.items():
            model.get_parameter(name).copy_(value)
    # 由真实小 H3 和真实 Flash SDE 生成单元测试 transition；不是生产轨迹或精度结论。
    ts = torch.linspace(1, 0, 11, dtype=torch.float64)
    schedules = [(s * ts / (1 + (s - 1) * ts)).float() for s in (12, 3)]
    contexts, inputs = [], []
    for group_index, index in enumerate((0, 1, 4)):
        kw = cpu(kwargs)
        text_length = 3 + group_index
        kw["encoder_hidden_states"] = torch.randn(1, text_length, 10).expand(8, -1, -1).clone()
        kw["token_tags"] = torch.tensor([1] * text_length + [0] * 4 + [2] * 2 + [-1] * 2)
        kw["text_indices"] = torch.arange(text_length)
        kw["video_indices"] = torch.arange(text_length, text_length + 4)
        kw["audio_indices"] = torch.arange(text_length + 4, text_length + 6)
        kw["position_ids"] = torch.arange((text_length + 8) * 3).reshape(-1, 3)
        row_times = torch.full((8, kw["token_tags"].numel()), 1 - schedules[0][index].item())
        row_times[:, kw["audio_indices"]] = 1 - schedules[1][index]
        kw["timestep"], kw["timestep_indices"] = torch.unique(row_times, sorted=True, return_inverse=True)
        inputs.append(kw)
        with torch.no_grad(), attention_context(torch.device("cpu")):
            ys = model(**kw)
        prev = []
        for y, input_key, schedule in zip(ys, ("hidden_states", "audio_hidden_states"), schedules):
            strategy = api.strategy()
            strategy.init_schedule(schedule)
            value, _, _ = strategy.denoise(noise_pred=-y, sample=kw[input_key], sigma=schedule[index],
                                           sigma_next=schedule[index + 1], sigma_max=schedule[1], eta=1.)
            prev.append(value.detach())
        contexts.append({"schema": 1, "video_sigmas": schedules[0], "audio_sigmas": schedules[1],
            "eta": 1., "sde_index_per_sample": torch.full((8,), index, dtype=torch.long),
            "prev_video": prev[0], "prev_audio": prev[1], "advantages": torch.linspace(-6, 6, 8),
            "policy_loss_weights": torch.full((8, 1), 0.8 + index / 10), "adv_clip_max": 5.,
            "clip_range": 1e-4, "beta": 0.004, "local_rows": 24, "split_factor": 1,
            "audio_in_policy": True, "logprob_dtype": "float32"})
    # 直接执行原 row_timestep_plan 的 scalar/batched 两条分支，核对索引维数和表的逐位值。
    spec = importlib.util.spec_from_file_location("micro_real_packing", vendor_path.parent / "packing.py")
    packing = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = packing
    spec.loader.exec_module(packing)
    source = ast.parse((vendor_path.parent.parent / "packing.py").read_text())
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "row_timestep_plan"]
    namespace = {"torch": torch, "build_row_timesteps": packing.build_row_timesteps}
    exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + nodes,
                           type_ignores=[]), "row_timestep_plan", "exec"), namespace)
    layout = packing.MiniMaxH3PackedSequence(sequence_length=kwargs["token_tags"].numel(),
        **{k: kwargs[k] for k in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices")},
        num_condition_video_rows=0, num_condition_audio_rows=0)
    input_checks = []
    for index in (0, 1, 4):
        table8, ids8 = namespace["row_timestep_plan"](layout, video_sigma=schedules[0][index].expand(8),
                                                      audio_sigma=schedules[1][index].expand(8))
        table1, ids1 = namespace["row_timestep_plan"](layout, video_sigma=schedules[0][index].reshape(1),
                                                      audio_sigma=schedules[1][index].reshape(1))
        assert torch.equal(table8, table1) and torch.equal(ids8, ids1.expand(8, -1))
        actual = replay_kwargs(dict(kwargs, timestep=table8, timestep_indices=ids8), 0, 1)
        assert torch.equal(actual["timestep"], table1) and torch.equal(actual["timestep_indices"], ids1)
        input_checks.append({"index": index, "unique_timesteps": table8.numel(), "table_exact": True,
                             "b8_indices_shape": list(ids8.shape), "b1_indices_shape": list(ids1.shape)})
    row_keys = {"prev_video", "prev_audio", "advantages", "policy_loss_weights", "sde_index_per_sample"}
    context = {k: torch.cat([c[k] for c in contexts]) if k in row_keys else v for k, v in contexts[0].items()}
    # 身份基准必须来自将要复现的混合精度 BF16 B8，不能用 FP32 输出代替。
    for name, child in model.named_children():
        child.to(torch.float32 if name in model._keep_in_fp32_modules else torch.bfloat16)
    capture = directory / "capture-three"
    handles = install_capture(model, capture, {"cpu_unit_test": True}, loss_context=context, groups=3)
    with attention_context(torch.device("cpu")):
        with torch.no_grad():
            model(**inputs[0])
        for i, kw in enumerate(inputs):
            y = model(**kw)
            with api.adapters_disabled(model), torch.no_grad():
                ref = model(**kw)
            loss, *_ = loss_from_outputs(api, y, kw, contexts[i], 0, 8, ref)
            loss.backward()
            assert (capture / "CAPTURE_DONE.json").exists() == (i == 2)
            model.zero_grad(set_to_none=True)
    for handle in handles:
        handle.remove()
    bundle = torch.load(capture / "fwd_rank0.pt", weights_only=True)
    validate_dump(bundle)
    assert json.loads((capture / "CAPTURE_DONE.json").read_text())["rows"] == 24
    analytic_checks = []
    for i, item in enumerate(bundle["groups"]):
        assert torch.equal(item["loss_context"]["prev_audio"], contexts[i]["prev_audio"])
        assert item["capture_metadata"]["sde_indices"] == [int(contexts[i]["sde_index_per_sample"][0])] * 8
        gs, *_ = own_cotangent(api, item["outputs"], item["kwargs"], item["loss_context"], 0, 8,
                               item["reference_outputs"])
        assert all(torch.equal(a, b) for a, b in zip(gs, item["out_grads"]))
        # 独立的闭式导数，核对 H3 负号、双流元素数、TGR、KL 和 1/24 缩放。
        context_i = item["loss_context"]
        for j, label in enumerate(("video", "audio")):
            schedule = context_i[label + "_sigmas"]
            idx = context_i["sde_index_per_sample"]
            sigma, nxt = schedule[idx].reshape(-1, 1, 1), schedule[idx + 1].reshape(-1, 1, 1)
            diffusion = schedule[1] * sigma
            c = -(1 + diffusion.square() * (1 - sigma) / (2 * sigma)) * (nxt - sigma)
            tau = diffusion * (sigma - nxt).sqrt()
            sample = item["kwargs"][("hidden_states", "audio_hidden_states")[j]]
            mean = sample * (1 + diffusion.square() / (2 * sigma) * (nxt - sigma)) + c * item["outputs"][j]
            ref_mean = sample * (1 + diffusion.square() / (2 * sigma) * (nxt - sigma)) + c * item["reference_outputs"][j]
            a = context_i["advantages"].clamp(-5, 5).reshape(-1, 1, 1)
            w = context_i["policy_loss_weights"].reshape(-1, 1, 1)
            n = sum(o[0].numel() for o in item["outputs"])
            expected = c / (24 * n * tau.square()) * (
                -a * w * (context_i["prev_" + label] - mean) + 0.004 * (mean - ref_mean))
            torch.testing.assert_close(gs[j], expected, rtol=2e-5, atol=2e-8)
            analytic_checks.append(distance(gs[j], expected))
        b1 = replay_kwargs(item["kwargs"], 2, 3)
        assert b1["timestep_indices"].ndim == 1
        assert torch.equal(b1["timestep"][b1["timestep_indices"]],
                           item["kwargs"]["timestep"][item["kwargs"]["timestep_indices"][2]])
    # 重新恢复混合精度存储，走与生产相同的三组顺序执行入口。
    for name, child in model.named_children():
        child.to(torch.float32 if name in model._keep_in_fp32_modules else torch.bfloat16)
    runs = directory / "three-groups"
    runs.mkdir()
    run_groups(model, bundle, runs, [torch.device("cpu")], vendor, masters, loss_api=api,
               all_groups=True, replay_inputs=True, minimum_host_gib=0.5, profile=False)
    assert (runs / "DONE").exists() and len(json.loads((runs / "groups.json").read_text())["groups"]) == 3
    checks = {}
    for group_index, item in enumerate(bundle["groups"]):
        original = copy.deepcopy(model)
        exp = Experiment(original, item, directory / f"loss-oracle-{group_index}", [torch.device("cpu")], masters,
                         profile=False, minimum_host_gib=0.5, loss_api=api, replay_inputs=True)
        for case, (fp32, micro) in CASES.items():
            exp.forward_case(case)
            exp.cast("", original, fp32)
            oracle = full_autograd(original, item, micro, api)
            distances = []
            for index in reversed(range(len(original.transformer_blocks))):
                actual = exp.reverse_case(case, index)
                distances.append(distance(actual["delta"], oracle["deltas"][index]))
                distances.extend(distance(g, oracle["gradients"][f"{index}.{n}"])
                                 for n, g in actual["gradients"].items())
            assert all(d["max_abs"] == 0 for d in distances)
            assert all(d["max_abs"] == 0 for d in oracle["cotangent_checks"])
            checks[f"group{group_index}/{case}"] = {
                "full_loss_vs_blockwise_max_abs": max(d["max_abs"] for d in distances),
                "derivative_comparisons": len(distances),
                "cotangent_vs_full_loss_max_abs": max(d["max_abs"] for d in oracle["cotangent_checks"]),
                "cotangent_comparisons": len(oracle["cotangent_checks"])}
            exp.boundaries.clear()
    control = Experiment(copy.deepcopy(model), bundle["groups"][0], directory / "captured-reference-control",
                         [torch.device("cpu")], masters, loss_api=api, reference_output="captured",
                         profile=False, minimum_host_gib=0.5)
    control_checks = []
    for case in ("fp32_b8", "bf16_b1"):
        control.forward_case(case)
        micro = CASES[case][1]
        item = bundle["groups"][0]
        for start in range(0, 8, micro):
            expected, *_ = own_cotangent(api, [o[start:start + micro] for o in control.outputs[case]],
                sliced_kwargs(item["kwargs"], start, start + micro), item["loss_context"], start, start + micro,
                [o[start:start + micro] for o in item["reference_outputs"]])
            control_checks.extend(distance(g[start:start + micro], e) for g, e in zip(control.cotangents[case], expected))
        control.boundaries.clear()
    assert all(c["max_abs"] == 0 for c in control_checks)
    # 旧 capture 缺字段必须拒绝，不能默默继续固定 cotangent。
    try:
        Experiment(model, {k: v for k, v in bundle["groups"][0].items() if k != "loss_context"},
                   directory / "must-reject", [torch.device("cpu")], masters, loss_api=api, minimum_host_gib=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("缺少 loss_context 未拒绝")
    result = {"test_type": "small real H3 + real Flash SDE; generated unit inputs only",
              "groups": 3, "sde_indices": [0, 1, 4], "sequence_lengths": [11, 12, 13],
              "capture_roundtrip_exact": True,
              "capture_marker_only_after_third_backward": True, "replay_b1_indices_1d_exact_values": True,
              "missing_context_rejected": True, "real_row_timestep_plan": input_checks,
              "closed_form_cotangent_checks": analytic_checks,
              "captured_reference_control_max_abs": max(c["max_abs"] for c in control_checks), "checks": checks}
    result["identity_and_budget_guards"] = validate_guards(vendor, model, masters, bundle, api, directory)
    write_json(directory / "CPU-LOSS-VALIDATION.json", result)
    return result


def validate_guards(vendor, model, masters, bundle, api, directory):
    results = {}

    def expect_stop(name, candidate, data, candidate_masters, *, loss_api=api, all_groups=True, phase=None,
                    devices=(torch.device("cpu"),)):
        folder = directory / name
        folder.mkdir()
        forwards = []
        handle = candidate.register_forward_pre_hook(lambda module, args: forwards.append(True))
        try:
            run_groups(candidate, data, folder, list(devices), vendor, candidate_masters,
                       loss_api=loss_api, all_groups=all_groups, minimum_host_gib=0.5, profile=False)
        except (ValueError, MemoryError):
            stops = [json.loads(p.read_text()) for p in folder.rglob("STOPPED.json")]
            assert stops, name
            if phase is not None:
                assert any(s.get("phase") == phase for s in stops), name
            else:
                assert stops[0]["reason"] == "resource_budget_exceeded", name
            assert not list(folder.rglob("DONE")) and not list(folder.rglob("block-*.json"))
            assert not list(folder.rglob("probe-input.pt"))
            results[name] = {"h3_forwards": len(forwards), "stops": stops}
        else:
            raise AssertionError(f"{name}: 没有拒绝不一致的输入")
        finally:
            handle.remove()
        return folder

    # 第三组 loss 上下文错组；前两组虽正确，也不能开始任何 H3 forward。
    wrong_group = cpu(bundle)
    wrong_group["groups"][2]["loss_context"] = cpu(bundle["groups"][1]["loss_context"])
    folder = expect_stop("wrong-group", copy.deepcopy(model), wrong_group, masters, phase="capture-loss-check")
    assert results["wrong-group"]["h3_forwards"] == 0
    assert json.loads((folder / "group-00/capture-loss-check.json").read_text())["passed"]
    assert json.loads((folder / "group-01/capture-loss-check.json").read_text())["passed"]
    assert not list(folder.rglob("*-settings.json"))

    # 只针对反向前检查的直接测试：policy 输出正确，但传入别组的 loss 上下文。
    post = directory / "post-forward-wrong-cotangent"
    exp = Experiment(copy.deepcopy(model), wrong_group["groups"][2], post, [torch.device("cpu")], masters,
                     loss_api=api, minimum_host_gib=0.5, profile=False)
    try:
        exp.run(vendor)
    except ValueError:
        stop = json.loads((post / "STOPPED.json").read_text())
        assert stop["phase"] == "bf16-b8-cotangent-check"
        assert json.loads((post / "bf16-b8-output-check.json").read_text())["passed"]
        assert (post / "captured_outputs.json").exists() and (post / "cotangents.json").exists()
        assert not (post / "DONE").exists() and not list(post.glob("block-*.json"))
        assert all(t["reverse_seconds"] == 0 for t in exp.timings.values())
        results["post-forward-wrong-cotangent"] = {"stop": stop, "reverse_calls_completed": 0}
    else:
        raise AssertionError("未拒绝 BF16 B8 cotangent 失配")
    del exp

    if masters:
        # 保存并经正式加载器读取另一个真实 PEFT checkpoint，不替换模型输出。
        wrong_weights = {n: -v if ".lora_B." in n else v for n, v in masters.items()}
        checkpoint = directory / "wrong-adapter.pt"
        torch.save({"save_mode": "adapter", "policy_state_dict": wrong_weights}, checkpoint)
        wrong_model = copy.deepcopy(model)
        wrong_model.delete_adapters("default")
        wrong_masters = attach_lora(wrong_model, checkpoint, 64)
        folder = expect_stop("wrong-checkpoint", wrong_model, bundle["groups"][0], wrong_masters,
                             all_groups=False, phase="bf16-b8-output-check")
        assert json.loads((folder / "capture-loss-check.json").read_text())["passed"]
        assert (folder / "captured_outputs.json").exists() and (folder / "cotangents.json").exists()
        assert results["wrong-checkpoint"]["h3_forwards"] > 0

    # 预算仅使用真实 H3 类的 meta 参数和长度；不分配完整权重，不运行这些预算输入。
    from peft import LoraConfig
    with init_empty_weights(include_buffers=False):
        full = vendor.MiniMaxH3Transformer3DModel()
        targets = [f"transformer_blocks.{i}.{n}" for i in range(len(full.transformer_blocks)) for n in LORA_TARGETS]
        full.add_adapter(LoraConfig(r=32, lora_alpha=64, target_modules=targets))
    for name, lengths, budget_model, failures in (
            ("late-gpu-budget", (4011, 4011, 4058), full, {"gpu"}),
            ("late-host-budget", (4011, 4011, 8000), full, {"gpu", "host"}),
            ("cumulative-disk-budget", (80000, 80000, 80000), model, {"cumulative_disk"})):
        data = cpu(bundle)
        for item, length in zip(data["groups"], lengths):
            item["kwargs"]["token_tags"] = torch.empty(length, device="meta", dtype=torch.long)
        folder = expect_stop(name, budget_model, data, masters,
                             devices=[torch.device("cpu")] * (8 if budget_model is full else 1))
        report = json.loads((folder / "budget-preflight.json").read_text())
        assert len(report["groups"]) == 3 and not report["passed"]
        assert report["groups"][0]["violations"] == report["groups"][1]["violations"] == []
        assert set(report["groups"][2]["violations"]) == failures
        assert results[name]["h3_forwards"] == 0 and not list(folder.rglob("*-settings.json"))
        assert len(list(folder.rglob("resource-plan.json"))) == 3
        results[name]["budget"] = report
    # 正常三组已完成；BF16/FP32 的真实差异不应触发身份停止条件。
    positive = directory / "three-groups"
    assert json.loads((positive / "budget-preflight.json").read_text())["passed"]
    comparisons = json.loads((positive / "group-00/outputs.json").read_text())
    assert any(r["max_abs"] > 0 for stream in comparisons["bf16_b8_vs_fp32_b8"] for r in stream)
    assert (positive / "DONE").exists()
    results["valid_three_groups"] = {"done": True, "experimental_precision_difference_allowed": True}
    write_json(directory / "CPU-GUARDS.json", results)
    return results


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-lora", action="store_true")
    parser.add_argument("--loss-options", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    configure_precision()
    vendor = load_vendor(args.vendor)
    layers = 4
    model = vendor.MiniMaxH3Transformer3DModel(
        hidden_size=16, num_layers=layers, num_refiner_layers=1, num_attention_heads=2,
        attention_head_dim=8, ffn_dim=24, in_channels=2, audio_in_channels=3,
        patch_size=(1, 1, 1), text_dim=10, freq_dim=8, time_embed_hidden_dim=16,
        time_embed_dim=8, rope_freq_dim=1).bfloat16()
    for name, module in model.named_children():
        if name in model._keep_in_fp32_modules:
            module.float()
    model.eval().requires_grad_(False)
    source = model
    pretrained = args.out.parent / (args.out.name + "-pretrained")
    source.save_pretrained(pretrained, max_shard_size="8KB", safe_serialization=True)
    # 本机可用内存约 3 GiB；只对小模型验证采用 1/0.5 GiB 门槛，生产入口固定 50/12。
    prepare_output(args.out, minimum_host_gib=1)
    model = load_model(vendor, pretrained, [torch.device("cpu")], args.out, minimum_host_gib=0.5)
    assert all(torch.equal(value, model.state_dict()[name]) and value.dtype == model.state_dict()[name].dtype
               for name, value in source.state_dict().items())
    assert all(p.device.type == "cpu" and not p.is_meta for p in model.parameters())
    loader = json.loads((args.out / "loader.json").read_text())
    assert loader["shard_count"] > 1 and loader["initial_parameters_all_meta"]
    single_pretrained = args.out.parent / (args.out.name + "-single-pretrained")
    source.save_pretrained(single_pretrained / "transformer", safe_serialization=True)
    single_report = args.out / "single-loader"
    single_report.mkdir()
    single_model = load_model(vendor, single_pretrained, [torch.device("cpu")], single_report,
                              minimum_host_gib=0.5)
    assert all(torch.equal(value, single_model.state_dict()[name]) for name, value in source.state_dict().items())
    assert json.loads((single_report / "loader.json").read_text())["shard_count"] == 1
    del single_model
    # 这里只验证程序，不代表真实 checkpoint/真实轨迹或 GPU kernel 数值证据。
    kwargs = {"hidden_states": torch.randn(8, 4, 2), "audio_hidden_states": torch.randn(8, 2, 3),
              "encoder_hidden_states": torch.randn(1, 3, 10).expand(8, -1, -1).clone(),
              "timestep": torch.linspace(0, 1, 8), "timestep_indices": torch.arange(8)[:, None].expand(8, 11).clone(),
              "token_tags": torch.tensor([1, 1, 1, 0, 0, 0, 0, 2, 2, -1, -1]),
              "position_ids": torch.arange(33).view(11, 3), "text_indices": torch.arange(3),
              "video_indices": torch.arange(3, 7), "audio_indices": torch.arange(7, 9), "return_dict": False}
    with torch.no_grad(), attention_context(torch.device("cpu")):
        outputs = model(**kwargs)
        source_outputs = source(**kwargs)
    assert all(torch.equal(a, b) for a, b in zip(outputs, source_outputs))
    del source, source_outputs
    dump = {"kwargs": kwargs, "outputs": cpu(outputs),
            "out_grads": tuple(torch.randn_like(x) * 0.01 for x in outputs)}
    validate_dump(dump)
    single = sliced_kwargs(kwargs, 3, 4)
    assert single["timestep"].shape == (8,) and single["token_tags"].shape == (11,)
    assert single["timestep_indices"].shape == (1, 11)
    # 共享一维 timestep_indices 也不能误切。
    shared = dict(kwargs, timestep_indices=kwargs["timestep_indices"][0])
    assert sliced_kwargs(shared, 0, 1)["timestep_indices"].shape == (11,)
    capture_directory = args.out.parent / (args.out.name + "-capture")
    capture_model = copy.deepcopy(model).requires_grad_(True)
    handles = install_capture(capture_model, capture_directory, {"cpu_unit_test": True})
    with attention_context(torch.device("cpu")):
        with torch.no_grad():
            capture_model(**kwargs)
        assert not (capture_directory / "CAPTURE_DONE.json").exists()
        ys = capture_model(**kwargs)
        torch.autograd.backward(ys, dump["out_grads"])
    captured = torch.load(capture_directory / "fwd_rank0.pt", weights_only=True)
    validate_dump(captured)
    assert all(torch.equal(x, y) for x, y in zip(captured["out_grads"], dump["out_grads"]))
    assert all(torch.equal(x, y) for x, y in zip(captured["outputs"], dump["outputs"]))
    for handle in handles:
        handle.remove()
    del capture_model, ys
    masters = {}
    if args.with_lora:
        from peft import LoraConfig
        # 用真实 PEFT 注入和序列化，验证 live adapter 加载及 master 精度恢复；不是训练结果。
        adapter_source = copy.deepcopy(model)
        targets = [f"transformer_blocks.{i}.{name}" for i in range(layers) for name in
                   ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")]
        adapter_source.add_adapter(LoraConfig(r=32, lora_alpha=64, target_modules=targets))
        weights = {n: torch.randn_like(p, dtype=torch.float32) * 0.01
                   for n, p in adapter_source.named_parameters() if ".lora_" in n}
        checkpoint = args.out.parent / (args.out.name + "-adapter.pt")
        with checkpoint.open("xb") as stream:
            torch.save({"save_mode": "adapter", "policy_state_dict": weights}, stream)
        masters = attach_lora(model, checkpoint, 64)
        assert all(torch.equal(masters[n], weights[n]) for n in weights)
        del adapter_source
        with torch.no_grad(), attention_context(torch.device("cpu")):
            dump["outputs"] = cpu(model(**kwargs))
    oracle_model = copy.deepcopy(model)
    experiment = Experiment(model, dump, args.out, [torch.device("cpu")], masters, profile=True,
                            minimum_host_gib=0.5)
    rows = experiment.run(vendor)
    assert not experiment.boundaries
    assert sorted(p.name for p in args.out.rglob("*.pt")) == ["probe-input.pt"]
    oracle_checks = {}
    # 常规图与本脚本逐层重算必须一致。重新产生边界仅用于本项小模型验证。
    oracle_dir = args.out / "oracle-recompute"
    second = Experiment(oracle_model, dump, oracle_dir, [torch.device("cpu")], masters, profile=False,
                        minimum_host_gib=0.5)
    for case, (fp32, micro) in CASES.items():
        second.forward_case(case)
        assert all(value.device == second.block_devices[min(i, layers - 1)] and not value.requires_grad
                   for (_, i, _), value in second.boundaries.items())
        assert not list(oracle_dir.rglob("*.pt"))
        second.cast("", oracle_model, fp32)
        oracle = full_autograd(oracle_model, dump, micro)
        checks = []
        for index in reversed(range(layers)):
            result = second.reverse_case(case, index)
            checks.append(distance(result["delta"], oracle["deltas"][index]))
            checks.extend(distance(g, oracle["gradients"][f"{index}.{name}"])
                          for name, g in result["gradients"].items())
        assert all(c["max_abs"] == 0 for c in checks), case
        oracle_checks[case] = {"comparison_count": len(checks), "max_abs": max(c["max_abs"] for c in checks)}
        second.boundaries.clear()
    pair = "fp32_b8_vs_fp32_b1"
    delta_cos = [s["cos"] for row in rows for s in row["comparisons"][pair]["delta"] if s["cos"] is not None]
    grad_cos = [s["cos"] for row in rows for s in row["comparisons"][pair]["gradients"].values() if s["cos"] is not None]
    assert min(delta_cos) > 0.99999 and min(grad_cos) > 0.99999
    assert len(rows) == layers and all(len(r["comparisons"]) == 4 for r in rows)
    assert len(experiment.grad_names[3]) == 2
    probes = {tag: json.loads((args.out / f"probe-{tag}.json").read_text()) for tag in ("fp32", "bf16")}
    assert all(len(p["samples"]) == len(p["operation_order"]) * 8 for p in probes.values())
    # 使用真实内存读数和一个高于物理总内存的门槛，检查失败路径，不伪造 MemAvailable。
    failure_directory = args.out.parent / (args.out.name + "-memory-stop")
    failure_directory.mkdir()
    write_json(failure_directory / "block-00.json", next(r for r in rows if r["block"] == 0))
    before = (failure_directory / "block-00.json").read_bytes()
    try:
        check_host_memory(failure_directory, "cpu_guard_validation", psutil.virtual_memory().total / GIB + 1)
    except MemoryError:
        assert (failure_directory / "STOPPED.json").exists()
        assert (failure_directory / "block-00.json").read_bytes() == before
    else:
        raise AssertionError("内存不足时没有停止")
    startup_directory = args.out.parent / (args.out.name + "-startup-stop")
    try:
        prepare_output(startup_directory, minimum_host_gib=psutil.virtual_memory().total / GIB + 1)
    except MemoryError:
        assert (startup_directory / "STOPPED.json").exists()
    else:
        raise AssertionError("启动内存检查未拒绝")
    result = {"test_type": "randomly initialized small REAL vendor H3; CPU plumbing only",
              "meta_sharded_loader_exact": True, "meta_single_file_loader_exact": True,
              "loaded_safetensors_shards": loader["shard_count"],
              "boundary_tensor_files": 0, "only_tensor_artifact": "probe-input.pt",
              "host_memory_stop_preserved_statistics": True, "host_startup_guard_tested": True,
              "cpu_test_available_thresholds_gib": {"startup": 1, "runtime": 0.5},
              "measured_host": check_host_memory(args.out, "cpu_validation_complete", 0.5),
              "torch": str(torch.__version__), "layers": layers, "hidden": 16, "batch": 8,
              "lora_loader_tested": args.with_lora, "capture_hook_exact_roundtrip": True,
              "sequence": 11, "includes_padding": True, "unique_timesteps": 8,
              "fp32_batch_delta_min_cos": min(delta_cos), "fp32_batch_weight_min_cos": min(grad_cos),
              "fp32_batch_delta_comparisons": len(delta_cos), "fp32_batch_weight_comparisons": len(grad_cos),
              "full_autograd_vs_blockwise": oracle_checks,
              "probe_operations": {tag: len(p["operation_order"]) for tag, p in probes.items()},
              "all_recomputations_exact": True}
    if args.loss_options:
        result["loss_options"] = validate_loss_options(args.vendor, vendor, model, masters, kwargs, args.out)
    result["measured_host"] = check_host_memory(args.out, "all_cpu_validation_complete", 0.5)
    result["cpu_validation_seconds"] = time.perf_counter() - started
    write_json(args.out / "CPU-VALIDATION.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
