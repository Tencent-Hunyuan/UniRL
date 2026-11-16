import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft.tuners.lora import LoraLayer

import standalone_micro as sm


VARIANTS = (
    "baseline", "after_g0", "after_g0_reference", "reference_first", "no_lora",
    "timestep_bf16", "sinusoid_bf16", "time_mlp_bf16", "block_temb_bf16",
    "time_mlp_rows", "adaln_rows", "proj_in_bf16", "audio_proj_in_bf16",
    "proj_out_bf16", "audio_proj_out_bf16", "input_strides", "no_grad", "standalone_cast_cycle",
)


def comparison(actual, expected):
    if len(actual) != 2 or len(expected) != 2:
        raise ValueError("需要两个模态输出")
    rows = []
    for label, a, b in zip(("video", "audio"), actual, expected):
        if a.shape != b.shape:
            raise ValueError("输出形状不同")
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        exact = a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        rows.append({"modality": label, "bitwise_equal": exact,
                     "candidate_dtype": str(a.dtype), "capture_dtype": str(b.dtype),
                     **sm.distance(a, b), "samples": sm.per_sample(a, b)})
    return {"bitwise_equal": all(r["bitwise_equal"] for r in rows), "modalities": rows}


def forward_plan(model, sequence_length, devices, extra_lora_rank=0):
    old = sm.resource_plan(model, sequence_length, devices, extra_lora_rank)
    h = 8 * sequence_length * model.config.hidden_size * 4
    # 沿用原 FP32 单层工作区作保守预算；不留整网图、四种情况或逐层边界。
    # 另计两个跨卡 hidden、单层 cast_cycle 临时升精度、2 GiB 杂项。
    rows = [{"device": r["device"], "native_weights_gib": r["native_weights_gib"],
             "upper_estimate_gib": r["native_weights_gib"] + old["fp32_block_workspace_estimate_gib"]
             + 2 * h / sm.GIB + r["active_fp32_promotion_gib"] + 2}
            for r in old["devices"]]
    return {"is_gpu_measurement": False, "sequence_length": sequence_length, "devices": rows,
            "allocator_limit_gib": 58, "host_start_minimum_gib": 50, "host_runtime_minimum_gib": 12}


def lora_state(model):
    return {n: {"disabled": m.disable_adapters, "merged": bool(m.merged),
                "active": list(m.active_adapters), "scaling": dict(m.scaling)}
            for n, m in model.named_modules() if isinstance(m, LoraLayer)}


@contextmanager
def variant_scope(model, variant, masters):
    handles, originals = [], []
    state = lora_state(model)
    try:
        targets = {"time_mlp_bf16": ("time_embedder",), "proj_in_bf16": ("proj_in",),
                   "audio_proj_in_bf16": ("audio_proj_in",),
                   "proj_out_bf16": ("proj_out",),
                   "audio_proj_out_bf16": ("audio_proj_out",)}.get(variant, ())
        for target in targets:
            for p in model.get_submodule(target).parameters():
                if p.dtype != torch.float32:
                    raise ValueError(f"{target} 不是原 FP32 模块")
                # 保留原 storage，不能把已舍入的 BF16 升回 FP32 冒充恢复。
                originals.append((p, p.data))
                p.data = p.data.to(torch.bfloat16)
        if variant == "sinusoid_bf16":
            handles.append(model.time_proj.register_forward_hook(
                lambda module, args, output: output.bfloat16().to(output.dtype)))
        if variant == "time_mlp_rows":
            original = model.time_embedder.forward

            def separate_rows(value, *args, **kwargs):
                return torch.cat([original(row, *args, **kwargs) for row in value.split(1)])

            model.time_embedder.forward = separate_rows
        if variant == "adaln_rows":
            # 只改变 AdaLN 线性投影的 U 维 GEMM 形状，保留原 SiLU 和 token 索引。
            for module in [*(b.adaln_proj.linear for b in model.transformer_blocks), model.norm_out.linear]:
                original_forward = module.forward

                def rows(value, original_forward=original_forward):
                    return torch.cat([original_forward(row) for row in value.split(1)])

                originals.append((module, original_forward))
                module.forward = rows
        if variant == "standalone_cast_cycle":
            caster = SimpleNamespace(masters=masters, shipped={n: t.dtype for n, t in
                list(model.named_parameters()) + list(model.named_buffers())})
            # 使用原 cast 函数；每次只升一个 block，不执行 FP32 前向或任何反向。
            for name, child in model.named_children():
                modules = [(f"transformer_blocks.{i}", b) for i, b in enumerate(child)] if (
                    name == "transformer_blocks") else [(name, child)]
                for prefix, module in modules:
                    sm.Experiment.cast(caster, prefix, module, True)
                    sm.Experiment.cast(caster, prefix, module, False)
        yield
    finally:
        for handle in handles:
            handle.remove()
        if variant == "time_mlp_rows":
            model.time_embedder.forward = original
        for target, original_value in reversed(originals):
            if isinstance(target, torch.nn.Parameter):
                target.data = original_value
            else:
                target.forward = original_value
        if lora_state(model) != state:
            raise RuntimeError("变体退出后 LoRA 路由未恢复")


def forward(model, item, devices, directory, variant, api, *, reference=False, minimum_host_gib=12):
    root = devices[0]
    handles, trace = [], {}
    grad_enabled = variant != "no_grad" and not reference
    kwargs = sm.move(sm.sliced_kwargs(item["kwargs"], 0, 8), root)
    if variant == "timestep_bf16":
        kwargs["timestep"] = kwargs["timestep"].bfloat16().to(kwargs["timestep"].dtype)
    if variant == "input_strides":
        for key, layout in item["capture_metadata"]["original_input_layout"].items():
            value = kwargs[key]
            shape, stride = tuple(layout["shape"]), tuple(layout["stride"])
            if shape != value.shape or layout["dtype"] != str(value.dtype) or any(s < 0 for s in stride):
                raise ValueError("原始布局与保存 tensor 不符")
            storage = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
            if storage > 4 * value.numel():
                raise ValueError("原 stride 需要超过四倍存储，停止而不自动改布局")
            restored = torch.empty_strided(shape, stride, dtype=value.dtype, device=root)
            restored.copy_(value)
            kwargs[key] = restored

    trace["effective_timestep"] = kwargs["timestep"].tolist()

    def guard(phase):
        return sm.check_host_memory(directory, phase, minimum_host_gib)

    def record(name, module, args, output):
        trace[name] = {"inputs": [str(a.dtype) for a in args if torch.is_tensor(a)],
                       "weights": dict(Counter(str(p.dtype) for p in module.parameters())),
                       "output": str(output.dtype)}
        if name == "time_proj":
            trace[name]["bf16_roundtrip_max_abs"] = (output - output.bfloat16().to(output.dtype)).abs().max().item()

    for name in ("time_proj", "time_embedder", "transformer_blocks.0.adaln_proj.linear", "norm_out.linear"):
        handles.append(model.get_submodule(name).register_forward_hook(
            lambda module, args, output, name=name: record(name, module, args, output)))
    for i, (block, device) in enumerate(zip(model.transformer_blocks, sm.block_placement(len(model.transformer_blocks), devices))):
        def before(module, args, i=i, device=device):
            guard(f"forward:block:{i}")
            if len(args) != 5:
                raise ValueError("H3 block 接口改变")
            x, temb, ids, rotary, mask = sm.move(args, device)
            if variant == "block_temb_bf16":
                temb = temb.bfloat16()
            if i == 0:
                trace["block0_temb_dtype"] = str(temb.dtype)
            # 与 standalone 一样保留逐层 grad-enabled 分派，但不保留跨层图、不做反向。
            x = x.detach().requires_grad_(grad_enabled)
            return x, temb, ids, rotary, mask

        handles.extend([block.register_forward_pre_hook(before),
                        block.register_forward_hook(lambda module, args, output: output.detach())])

    def tail_before(module, args):
        x, *rest = sm.move(args, root)
        return (x.detach().requires_grad_(grad_enabled), *rest)

    handles.append(model.norm_out.register_forward_pre_hook(tail_before))
    # vendor 对两个输出头共用 video 头的 dtype；独立降精度时显式按各头转换。
    # 基准中两个头都是 FP32，这两个 hook 是 no-op，不改变输入值。
    for head in (model.proj_out, model.audio_proj_out):
        handles.append(head.register_forward_pre_hook(
            lambda module, args: (args[0].to(module.weight.dtype),)))
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    guard("forward:begin")
    try:
        with sm.attention_context(root), torch.set_grad_enabled(grad_enabled), (
                api.adapters_disabled(model) if reference or variant == "no_lora" else nullcontext()):
            outputs = sm.cpu(model(**kwargs))
    finally:
        for handle in handles:
            handle.remove()
    peaks = {}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peaks[str(device)] = {"allocated_gib": torch.cuda.max_memory_allocated(device) / sm.GIB,
                                  "reserved_gib": torch.cuda.max_memory_reserved(device) / sm.GIB}
            if peaks[str(device)]["allocated_gib"] > 58:
                raise MemoryError("显存 allocated 超过 58 GiB")
    return outputs, {"seconds": time.perf_counter() - start, "gpu_peaks": peaks,
                     "host": guard("forward:end"), "grad_enabled": grad_enabled, "trace": trace}


def run_suite(model, bundle, devices, directory, vendor, masters, variants=VARIANTS, group_order=(1, 0),
              minimum_host_gib=12):
    directory = Path(directory)
    sm.validate_dump(bundle)
    if len(bundle.get("groups", [])) != 3:
        raise ValueError("需要新的 rank0 三组 capture")
    groups = bundle["groups"]
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError("重复或未知变体")
    if sorted(group_order) != [0, 1] or not variants or variants[0] != "baseline":
        raise ValueError("先跑 baseline，组顺序必须是 1,0 或 0,1")
    model.eval().requires_grad_(False)
    if model.gradient_checkpointing or model.token_refiner.gradient_checkpointing:
        raise ValueError("诊断不使用 activation checkpointing")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError("要求 dropout=0")
        if getattr(module, "fused_projections", False):
            raise ValueError("要求未融合 QKV")
        if hasattr(module, "processor"):
            if module.processor._parallel_config is not None:
                raise ValueError("不使用 context parallel")
            module.processor._attention_backend = "native"
    state = lora_state(model)
    if not state or any(s["disabled"] or s["merged"] or s["active"] != ["default"] for s in state.values()):
        raise ValueError("需要已启用、未 merge 的 live default LoRA")
    for n, p in model.named_parameters():
        if p.dtype != sm.shipped_dtype(model, n):
            raise ValueError(f"加载 dtype 不符合 standalone: {n}")
    api = sm.load_loss_api(vendor.__file__)
    plans = [forward_plan(model, groups[g]["kwargs"]["token_tags"].numel(), devices) for g in group_order]
    result = {"schema": 1, "complete": False, "test_type": "CPU small real H3" if devices[0].type == "cpu"
              else "GPU real checkpoint/capture", "model_loads": 1, "backward_calls": 0,
              "fresh_first_group": group_order[0], "group_order": list(group_order), "plans": plans,
              "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
              "records": []}
    with sm.attention_context(devices[0]):
        result["settings"] = sm.settings(model, devices[0])
    sm.write_json(directory / "results.json", result)
    if max(r["upper_estimate_gib"] for p in plans for r in p["devices"]) > 58:
        raise MemoryError("前向解析预算超过 58 GiB，不改变 B8 或序列长度")
    baselines = {}

    def measured(variant, group, reference=False, warmup=False):
        outputs, info = forward(model, groups[group], devices, directory, variant, api,
                                reference=reference, minimum_host_gib=minimum_host_gib)
        row = {"variant": variant, "group": group, "reference": reference, "warmup": warmup,
               "timesteps": groups[group]["kwargs"]["timestep"].tolist(),
               "against_capture_policy": comparison(outputs, groups[group]["outputs"]),
               "against_capture_reference": comparison(outputs, groups[group]["reference_outputs"]), **info}
        if group in baselines and not warmup:
            row["against_fresh_baseline"] = comparison(outputs, baselines[group])
        result["records"].append(row)
        sm.write_json(directory / "results.json", result)
        print(f"{variant} group={group} reference={reference} warmup={warmup} "
              f"exact={row['against_capture_policy']['bitwise_equal']}", flush=True)
        return outputs

    for variant in variants:
        sm.check_host_memory(directory, f"variant:{variant}", minimum_host_gib)
        with variant_scope(model, variant, masters):
            for group in group_order:
                if variant in ("after_g0", "after_g0_reference"):
                    measured(variant, 0, warmup=True)
                    if variant == "after_g0_reference":
                        measured(variant, 0, reference=True, warmup=True)
                if variant == "reference_first":
                    measured(variant, group, reference=True, warmup=True)
                outputs = measured(variant, group)
                if variant == "baseline":
                    baselines[group] = outputs
        if lora_state(model) != state or any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("出现 LoRA 路由残留或参数梯度")
    # 检查所有临时变体退出后基准仍原样恢复，不能污染后续诊断。
    for group in group_order:
        outputs = measured("baseline_restored", group)
        if not comparison(outputs, baselines[group])["bitwise_equal"]:
            raise RuntimeError("诊断变体改变了后续基准输出")
    result["complete"] = True
    result["forward_count"] = len(result["records"])
    sm.write_json(directory / "results.json", result)
    (directory / "DONE").write_text("forward comparisons complete; capture identity is reported, not assumed\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="仅 BF16 B8 前向；一次加载，先直接跑组 1")
    for flag in ("vendor", "pretrained", "dump", "live-checkpoint", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--lora-alpha", type=float, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--group-order", choices=("1,0", "0,1"), default="1,0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("需要恰好八张已授权、空闲的 CUDA 卡；CPU 验证使用 cpu_validate.py")
    sm.prepare_output(args.out)
    sm.configure_precision()
    devices = [torch.device(f"cuda:{i}") for i in range(8)]
    for device in devices:
        total = torch.cuda.get_device_properties(device).total_memory
        if total < sm.GPU_LIMIT:
            raise MemoryError("目标卡容量不足 58 GiB")
        torch.cuda.set_per_process_memory_fraction(sm.GPU_LIMIT / total, device)
    try:
        vendor = sm.load_vendor(args.vendor)
        bundle = torch.load(args.dump, weights_only=True, mmap=True, map_location="cpu")
        sm.validate_dump(bundle)
        model = sm.load_model(vendor, args.pretrained, devices, args.out)
        masters = sm.attach_lora(model, args.live_checkpoint, args.lora_alpha)
        run_suite(model, bundle, devices, args.out, vendor, masters, args.variants,
                  tuple(map(int, args.group_order.split(","))))
    except Exception as error:
        # 保留部分 JSON 后原样抛出；不重试、不改变诊断条件、不记录机器路径。
        if not (args.out / "STOPPED.json").exists():
            sm.write_json(args.out / "STOPPED.json", {"reason": type(error).__name__, "complete": False})
        raise


if __name__ == "__main__":
    main()
