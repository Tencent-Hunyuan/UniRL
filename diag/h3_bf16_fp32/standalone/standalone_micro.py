import argparse
import ast
from collections import Counter
from contextlib import contextmanager, nullcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from types import SimpleNamespace

import psutil
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from diffusers.models.attention_dispatch import attention_backend
from torch.nn.attention import SDPBackend, sdpa_kernel


CASES = {"fp32_b8": (True, 8), "fp32_b1": (True, 1),
         "bf16_b8": (False, 8), "bf16_b1": (False, 1)}
PAIRS = [("fp32_b8", "fp32_b1"), ("bf16_b8", "bf16_b1"),
         ("bf16_b8", "fp32_b8"), ("bf16_b1", "fp32_b1")]
# --rounding-control：fp32 计算 + LoRA master 先舍入到 bf16 再升回 fp32，拆开 adapter 舍入与计算精度
ROUNDING_CASES = {"fp32_b8": (True, 8), "fp32r_b8": (True, 8), "bf16_b8": (False, 8)}
ROUNDING_PAIRS = [("fp32r_b8", "fp32_b8"), ("bf16_b8", "fp32r_b8"), ("bf16_b8", "fp32_b8")]
ROUNDED_LORA = {"fp32r_b8"}
BATCH_KEYS = {"hidden_states", "audio_hidden_states", "encoder_hidden_states"}
LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
GIB = 2**30
DISK_LIMIT = 20 * GIB
GPU_LIMIT = 58 * GIB
COTANGENT_RTOL = 1e-5
COTANGENT_ATOL_RMS = 1e-6


def check_host_memory(directory, phase, minimum_gib=12):
    available = psutil.virtual_memory().available
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    row = {"phase": phase, "time_unix": time.time(), "available_gib": available / GIB,
           "rss_gib": psutil.Process().memory_info().rss / GIB, "peak_rss_gib": peak / GIB,
           "minimum_available_gib": minimum_gib}
    with (Path(directory) / "resources.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    if available < minimum_gib * GIB or peak >= 40 * GIB:
        write_json(Path(directory) / "STOPPED.json", {
            "reason": "host_memory_limit", **row,
            "completed_blocks": sorted(p.name for p in Path(directory).glob("block-*.json"))})
        raise MemoryError(f"{phase}: 主内存低于安全余量或进程 RSS 达到 40 GiB")
    return row


def prepare_output(directory, minimum_host_gib=50):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if shutil.disk_usage(directory.parent).free < DISK_LIMIT:
        raise RuntimeError("工作盘必须至少剩余 20 GiB；不使用其他分区")
    directory.mkdir()
    check_host_memory(directory, "startup", minimum_host_gib)


def block_placement(count, devices):
    return [devices[min(i * len(devices) // count, len(devices) - 1)] for i in range(count)]


def weight_device(name, block_devices, root):
    return block_devices[int(name.split(".")[1])] if name.startswith("transformer_blocks.") else root


def shipped_dtype(model, name):
    return torch.float32 if any(n in name.split(".") for n in model._keep_in_fp32_modules) else torch.bfloat16


def load_model(vendor, pretrained, devices, directory, minimum_host_gib=12):
    pretrained = Path(pretrained)
    folder = pretrained / "transformer" if (pretrained / "transformer").is_dir() else pretrained
    config = vendor.MiniMaxH3Transformer3DModel.load_config(folder, local_files_only=True)
    # include_buffers=False 保留按公式生成的微小 RoPE buffer，不把它留成未初始化的 meta。
    with init_empty_weights(include_buffers=False):
        model = vendor.MiniMaxH3Transformer3DModel.from_config(config)
    if not all(p.is_meta for p in model.parameters()):
        raise RuntimeError("meta 初始化没有覆盖所有参数")
    model.requires_grad_(False)
    placement = block_placement(len(model.transformer_blocks), devices)
    index = folder / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(folder / filename, framework="pt", device="cpu") as reader:
            mapping = {name: filename for name in reader.keys()}
    expected = model.state_dict()
    if mapping.keys() != expected.keys():
        raise ValueError("safetensors 与 vendor config 的参数/buffer 名称不完全一致")
    largest = 0
    for name, filename in mapping.items():
        check_host_memory(directory, f"load:{name}", minimum_host_gib)
        device = weight_device(name, placement, devices[0])
        # 每个 tensor 单独打开/关闭映射；即使是单个超大文件，也不累计整文件的驻留 CPU 页。
        with safe_open(folder / filename, framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
            largest = max(largest, value.numel() * value.element_size())
            finite(name, value)
            set_module_tensor_to_device(model, name, device, value=value,
                                        dtype=shipped_dtype(model, name), clear_cache=False)
            del value
        # Accelerate 在写入前已经逐项校验 shape。
        check_host_memory(directory, f"loaded:{name}", minimum_host_gib)
    for name, value in list(model.named_buffers()):
        if name not in expected:
            if value.is_meta:
                raise ValueError(f"未初始化的非持久 buffer: {name}")
            set_module_tensor_to_device(model, name, weight_device(name, placement, devices[0]),
                                        value=value, dtype=value.dtype, clear_cache=False)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("加载后仍有 meta tensor")
    write_json(Path(directory) / "loader.json", {"initial_parameters_all_meta": True,
               "tensor_count": len(mapping), "shard_count": len(set(mapping.values())),
               "largest_single_source_tensor_gib": largest / GIB,
               "cpu_staging": "one tensor; mmap closed after each assignment",
               "block_devices": [str(d) for d in placement]})
    return model.eval()


def load_vendor(path):
    # 直接载入提供的真实 vendor 文件，不导入 UniRL 的分布式训练依赖。
    spec = importlib.util.spec_from_file_location("micro_h3_vendor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {k: tree(v, function) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree(v, function) for v in value)
    return value


def cpu(value):
    return tree(value, lambda t: t.detach().cpu().clone())


def move(value, device):
    return tree(value, lambda t: t.to(device))


def save(path, value):
    path = Path(path)
    sizes = []
    tree(value, lambda t: sizes.append(t.numel() * t.element_size()) or t)
    used = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    if used + sum(sizes) + 16 * 2**20 > DISK_LIMIT:
        raise RuntimeError("输出将超过 20 GiB；不写入大 tensor")
    with path.open("xb") as stream:
        torch.save(value, stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def finite(name, tensor):
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name}: NaN/Inf")


def distance(a, b):
    # CPU float64 分块归约；第二个参数是相对误差的参考。
    if a.shape != b.shape:
        raise ValueError((a.shape, b.shape))
    aa = bb = ab = dd = 0.0
    maximum = 0.0
    different = count = 0
    for x, y in zip(a.reshape(-1).split(1 << 20), b.reshape(-1).split(1 << 20)):
        x = x.detach().to(device="cpu", dtype=torch.float64)
        y = y.detach().to(device="cpu", dtype=torch.float64)
        finite("candidate", x)
        finite("reference", y)
        d = x - y
        aa += x.square().sum().item()
        bb += y.square().sum().item()
        ab += (x * y).sum().item()
        dd += d.square().sum().item()
        maximum = max(maximum, d.abs().max().item() if d.numel() else 0.0)
        different += (x != y).sum().item()
        count += x.numel()
    return {"cos": ab / math.sqrt(aa * bb) if aa and bb else None,
            "rel_l2": math.sqrt(dd / bb) if bb else None,
            "norm_ratio": math.sqrt(aa / bb) if bb else None,
            "max_abs": maximum, "fraction_different": different / count if count else None,
            "candidate_norm": math.sqrt(aa), "reference_norm": math.sqrt(bb)}


def combine_distances(metrics):
    # 拼接梯度向量的统计，不把几十亿参数再拼成一个大 tensor。
    aa = bb = ab = dd = 0.0
    for item in metrics:
        a, b = item["candidate_norm"], item["reference_norm"]
        aa += a * a
        bb += b * b
        ab += (item["cos"] or 0.0) * a * b
        dd += (item["rel_l2"] * b) ** 2 if b else a * a
    return {"cos": ab / math.sqrt(aa * bb) if aa and bb else None,
            "rel_l2": math.sqrt(dd / bb) if bb else None,
            "candidate_norm": math.sqrt(aa), "reference_norm": math.sqrt(bb)}


def per_sample(a, b):
    return [dict(sample=i, **distance(x, y)) for i, (x, y) in enumerate(zip(a, b))]


def sliced_kwargs(kwargs, start, stop):
    # 只切真实 batch 轴。timestep 是共享的 unique 表，长度恰为 8 也不能切。
    result = dict(kwargs)
    for key in BATCH_KEYS:
        result[key] = kwargs[key][start:stop]
    if kwargs["timestep_indices"].ndim == 2:
        result["timestep_indices"] = kwargs["timestep_indices"][start:stop]
    result["return_dict"] = False
    return result


def load_loss_api(vendor_path):
    # 直接复用该快照的 SDE 类和纯函数；AST 只选函数，不加载 Ray/训练入口，也不改函数体。
    unirl = Path(vendor_path).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("micro_sde_kernels", unirl / "sde/kernels.py")
    kernels = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kernels
    spec.loader.exec_module(kernels)
    namespace = {"torch": torch, "contextmanager": contextmanager}
    for filename, names in (("algorithms/base.py", {"select_active_tokens", "grpo_clip_loss", "gaussian_kl_div"}),
                            ("models/types/diffusion.py", {"combine_modality_logp"}),
                            ("train/lora.py", {"adapters_disabled"})):
        parsed = ast.parse((unirl / filename).read_text())
        nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError("UniRL loss 函数接口已改变")
        future = ast.parse("from __future__ import annotations").body
        exec(compile(ast.Module(body=future + nodes, type_ignores=[]), filename, "exec"), namespace)
    return SimpleNamespace(strategy=kernels.FlashSDEStrategy, **{n: namespace[n] for n in
                           ("grpo_clip_loss", "gaussian_kl_div", "combine_modality_logp", "adapters_disabled")})


def replay_kwargs(kwargs, start, stop):
    result = sliced_kwargs(kwargs, start, stop)
    # 同 prompt 同时间步的 B1 在真实 row_timestep_plan 中走 scalar 分支，产生一维映射。
    if stop - start == 1:
        rows = result["timestep"][result["timestep_indices"]].reshape(-1)
        result["timestep"], result["timestep_indices"] = torch.unique(rows, sorted=True, return_inverse=True)
    return result


def loss_from_outputs(api, outputs, kwargs, context, start, stop, reference_outputs):
    device = outputs[0].device
    if context["schema"] != 1 or not context["audio_in_policy"] or context["logprob_dtype"] != "float32":
        raise ValueError("只支持本次 Flash 双流、FP32 logp 的 replay")
    if context["beta"] != 0.004 or context["split_factor"] != 1 or context["local_rows"] != 24:
        raise ValueError("需要本次 beta=0.004、full split、local_rows=24")
    indices = context["sde_index_per_sample"][start:stop].to(device=device, dtype=torch.long)
    means, residuals, stds, logps, kl = [], [], [], [], []
    if len(outputs) != 2 or len(reference_outputs) != 2:
        raise ValueError("需要 video/audio 两个 policy 和 reference 输出")
    for output, ref, input_key, label in zip(outputs, reference_outputs,
            ("hidden_states", "audio_hidden_states"), ("video", "audio")):
        schedule = context[label + "_sigmas"].to(device=device, dtype=torch.float32)
        sigma, sigma_next = schedule[indices], schedule[indices + 1]
        strategy = api.strategy()
        strategy.init_schedule(schedule)
        sample = kwargs[input_key].to(device)
        prev = context["prev_" + label][start:stop].to(device)
        if prev.shape != output.shape or ref.shape != output.shape:
            raise ValueError("loss 上下文与模型输出形状不同")
        arguments = dict(sample=sample, prev_sample=prev, sigma=sigma, sigma_next=sigma_next,
                         eta=context["eta"], sigma_max=schedule[1], step_index=0)
        rounded_prev, logp, mean = strategy.denoise(noise_pred=-output, **arguments)
        with torch.no_grad():
            _, _, ref_mean = strategy.denoise(noise_pred=-ref.detach().to(output), **arguments)
        std = strategy.transition_std(sigma=sigma, sigma_next=sigma_next, eta=context["eta"],
                                      sigma_max=schedule[1]).reshape(-1, 1)
        if logp is None or not bool((std > 0).all()):
            raise ValueError("需要非零 SDE transition std")
        kl.append(api.gaussian_kl_div(mean.unsqueeze(1), ref_mean.unsqueeze(1), std).mean(dim=(2, 3)))
        means.append(mean)
        residuals.append(rounded_prev.detach() - mean.detach())
        stds.append(std)
        logps.append(logp)
    nv, na = outputs[0][0].numel(), outputs[1][0].numel()
    new_logp = api.combine_modality_logp(*logps, n_video=nv, n_audio=na).reshape(-1, 1).float()
    adv = context["advantages"][start:stop].detach().to(new_logp)
    adv = adv.clamp(-context["adv_clip_max"], context["adv_clip_max"]).reshape(-1, 1)
    elements, _ = api.grpo_clip_loss(new_logp=new_logp, old_logp=new_logp.detach(), advantages=adv,
                                    clip_range=context["clip_range"])
    weights = context["policy_loss_weights"][start:stop].to(new_logp)
    if weights.shape != new_logp.shape:
        raise ValueError("policy_loss_weights 必须是 [B,1]")
    scale = (stop - start) / context["local_rows"]
    pg = (elements * weights).mean() * scale
    # 保留 FlashGRPO._reference_kl 的两次加权运算顺序；KL 不乘 TGR。
    kl_loss = (kl[0] * (nv / (nv + na)) + kl[1] * (na / (nv + na))).mean() * context["beta"] * scale
    # 原实现先相加再乘 scale；不要用 pg + kl_loss 替代总 loss 的浮点运算顺序。
    loss = ((elements * weights).mean() + context["beta"] *
            (kl[0] * (nv / (nv + na)) + kl[1] * (na / (nv + na))).mean()) * scale
    return loss, pg, kl_loss, means, residuals, stds


def own_cotangent(api, outputs, kwargs, context, start, stop, reference_outputs):
    leaves = [o.detach().requires_grad_(True) for o in outputs]
    loss, pg, kl, means, residuals, stds = loss_from_outputs(
        api, leaves, kwargs, context, start, stop, reference_outputs)
    pg_grads = torch.autograd.grad(pg, leaves, retain_graph=True)
    kl_grads = torch.autograd.grad(kl, leaves, retain_graph=True)
    grads = torch.autograd.grad(loss, leaves)
    stats = {"loss": loss.item(), "pg_scaled": pg.item(), "kl_scaled": kl.item(),
             "samples": [{"sample": start + i,
                "modalities": [{"residual_rms": r[i].float().square().mean().sqrt().item(),
                                "transition_std": s[i].item(),
                                "pg_norm": p[i].float().norm().item(),
                                "kl_norm": k[i].float().norm().item()}
                               for r, s, p, k in zip(residuals, stds, pg_grads, kl_grads)]}
                         for i in range(stop - start)]}
    for g in grads:
        finite("recomputed cotangent", g)
    return grads, stats, cpu(means), cpu(residuals)


def check_capture_identity(directory, phase, actual, expected, device, *, exact=False):
    # 容差按每条、每模态的参考 RMS 缩放，不能用固定 atol 掩盖真实大模型的微小 cotangent。
    rows = []
    if len(actual) != 2 or len(expected) != 2:
        raise ValueError("身份检查需要两个模态")
    for modality, (a, b) in enumerate(zip(actual, expected)):
        a, b = a.detach().to(device), b.detach().to(device)
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append({"modality": modality, "passed": False, "reason": "shape_or_dtype_mismatch"})
            continue
        for sample, (x, y) in enumerate(zip(a, b)):
            stats = distance(x, y)
            rtol = 0.0 if exact else COTANGENT_RTOL
            atol = 0.0 if exact else COTANGENT_ATOL_RMS * stats["reference_norm"] / math.sqrt(y.numel())
            close = torch.isclose(x, y, rtol=rtol, atol=atol)
            rows.append({"modality": modality, "sample": sample, **stats, "rtol": rtol, "atol": atol,
                         "passed": bool(close.all()), "outside_tolerance": int((~close).sum().item())})
    report = {"phase": phase, "comparison_device": str(device), "passed": all(r["passed"] for r in rows),
              "exact": exact, "samples": rows}
    write_json(Path(directory) / f"{phase}.json", report)
    if not report["passed"]:
        write_json(Path(directory) / "STOPPED.json", {
            "reason": "capture_identity_mismatch", "phase": phase,
            "detail": "保存的 loss 字段/输出梯度不对应" if phase == "capture-loss-check" else
                      "BF16 B8 与 capture 不对应；核对 checkpoint、输入、reference 与执行设置",
            "failed_samples": [{k: r[k] for k in ("modality", "sample") if k in r}
                               for r in rows if not r["passed"]]})
        raise ValueError(f"capture_identity_mismatch: {phase}；已保存比较，停止本次实验")
    return report


def configure_precision():
    if not torch.__version__.startswith("2.13."):
        raise RuntimeError("使用 PyTorch 2.13；不混用新旧精度 API")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("TF32 环境覆盖与 V2 条件冲突")
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)


def settings(model, device):
    import diffusers
    m = torch.backends.cuda.matmul
    result = {"torch": str(torch.__version__), "diffusers": diffusers.__version__,
              "cuda_build": torch.version.cuda, "device_type": device.type,
              "global_fp32": torch.backends.fp32_precision, "matmul_fp32": m.fp32_precision,
              "cudnn_fp32": torch.backends.cudnn.fp32_precision,
              "bf16_reduction": m.allow_bf16_reduced_precision_reduction,
              "bf16_split_k": m.allow_bf16_reduced_precision_reduction_split_k,
              "fp16_reduction": m.allow_fp16_reduced_precision_reduction,
              "fp16_split_k": m.allow_fp16_reduced_precision_reduction_split_k,
              "fp16_accumulation": m.allow_fp16_accumulation,
              "math_low_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "autocast": torch.is_autocast_enabled(device.type),
              "attention_dispatch": "native", "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
              "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
              "sdpa_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
              "sdpa_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
              "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
              "processor_backends": sorted({str(m.processor._attention_backend) for m in model.modules()
                                             if hasattr(m, "processor")}),
              "environment": {k: os.environ.get(k) for k in (
                  "TORCH_BLAS_PREFER_CUBLASLT", "DISABLE_ADDMM_CUDA_LT", "CUBLAS_WORKSPACE_CONFIG",
                  "CUBLASLT_WORKSPACE_SIZE", "NVIDIA_TF32_OVERRIDE", "DIFFUSERS_ATTN_BACKEND")}}
    if device.type == "cuda":
        result.update(gpu_type=torch.cuda.get_device_name(device),
                      preferred_blas=str(torch.backends.cuda.preferred_blas_library()),
                      cublas_workspace_bytes=torch.backends.cuda.cublas_workspace_size(),
                      cublaslt_workspace_bytes=torch.backends.cuda.cublaslt_workspace_size())
    return result


@contextmanager
def attention_context(device):
    # CUDA 禁止 S² math 分配；没有可用 fused 后端就报错，绝不拆 B=8。
    context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                           SDPBackend.CUDNN_ATTENTION]) if device.type == "cuda" else nullcontext()
    with attention_backend("native"), context, torch.autocast(device.type, enabled=False):
        yield


def attach_lora(model, checkpoint, alpha):
    from peft import LoraConfig
    raw = torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")
    if raw["save_mode"] != "adapter":
        raise ValueError("需要 checkup 使用的 live adapter checkpoint.pt")
    values = {k.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): v
              for k, v in raw["policy_state_dict"].items()
              if k.endswith(".default.weight") and (".lora_A." in k or ".lora_B." in k)}
    expected = 12 * len(model.transformer_blocks)
    if len(values) != expected:
        raise ValueError(f"预期 {expected} 个 live LoRA tensor，实际 {len(values)}")
    targets = [n.removesuffix(".lora_A.default.weight") for n in values if ".lora_A." in n]
    ranks = {values[n + ".lora_A.default.weight"].shape[0] for n in targets}
    expected_targets = {f"transformer_blocks.{i}.{n}" for i in range(len(model.transformer_blocks))
                        for n in LORA_TARGETS}
    if ranks != {32} or alpha != 64 or set(targets) != expected_targets:
        raise ValueError("本任务需要 rank=32、alpha=64、每 block 指定的六个目标投影")
    model.add_adapter(LoraConfig(r=ranks.pop(), lora_alpha=alpha, target_modules=targets,
                                lora_dropout=0.0, bias="none"), adapter_name="default")
    actual = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    if actual.keys() != values.keys():
        raise ValueError("live LoRA 参数名称不匹配")
    masters = {}
    with torch.no_grad():
        for name, p in actual.items():
            value = values[name]
            finite(name, value)
            if value.dtype != torch.float32 or value.shape != p.shape:
                raise ValueError(name)
            masters[name] = value.clone()
            p.data = value.to(device=p.device, dtype=torch.bfloat16)
    return masters


def resource_plan(model, sequence_length, devices, extra_lora_rank=0):
    # 参数项是精确计数；单层工作区是保守公式估算，不冒充 CUDA 测量。
    count = len(model.transformer_blocks)
    slots = [min(i * len(devices) // count, len(devices) - 1) for i in range(count)]
    native = [0] * len(devices)
    promotion, gradient_numel = [0] * count, [0] * count
    outside_promotion = master_bytes = total_fp32 = largest_gradient = 0
    for name, value in model.named_parameters():
        size = value.numel()
        low_bytes = size * (4 if shipped_dtype(model, name) == torch.float32 else 2)
        total_fp32 += size * 4
        if name.startswith("transformer_blocks."):
            i = int(name.split(".")[1])
            native[slots[i]] += low_bytes
            promotion[i] += size * 4 - low_bytes
            local = name.split(".", 2)[2]
            if i < 3 or local in {"attn.to_q.weight", "attn.to_q.base_layer.weight",
                                  "ff.net.2.weight", "ff.net.2.base_layer.weight"}:
                gradient_numel[i] += size
                largest_gradient = max(largest_gradient, size * 8)
        else:
            native[0] += low_bytes
            outside_promotion += size * 4 - low_bytes
        if ".lora_" in name:
            master_bytes += size * 4
    if extra_lora_rank:
        if master_bytes:
            raise ValueError("已有 LoRA 时不能重复加入解析预算")
        for i, block in enumerate(model.transformer_blocks):
            size = sum(extra_lora_rank * (block.get_submodule(n).in_features + block.get_submodule(n).out_features)
                       for n in LORA_TARGETS)
            native[slots[i]] += size * 2
            promotion[i] += size * 2
            total_fp32 += size * 4
            master_bytes += size * 4
            if i < 3:
                gradient_numel[i] += size
    config = model.config
    h = 8 * sequence_length * config.hidden_size * 4
    q = 8 * sequence_length * config.num_attention_heads * config.attention_head_dim * 4
    f = 8 * sequence_length * config.ffn_dim * 4
    boundary_slots = slots + [slots[-1]]
    boundaries = [boundary_slots.count(i) * 3 * h for i in range(len(devices))]
    workspace = 10 * h + 10 * q + 8 * f + 2 * GIB
    rows = []
    for slot, device in enumerate(devices):
        indices = [i for i in range(count) if slots[i] == slot]
        active_promotion = max((promotion[i] for i in indices), default=0)
        forward = native[slot] + boundaries[slot] + active_promotion + workspace
        forward += outside_promotion if slot == 0 else 0
        reverse = max((native[slot] + promotion[i] + workspace + gradient_numel[i] * 4
                       + sum(3 * h for j, s in enumerate(boundary_slots) if j <= i + 1 and s == slot)
                       for i in indices), default=0)
        rows.append({"device": str(device), "blocks": indices,
                     "native_weights_gib": native[slot] / GIB,
                     "active_fp32_promotion_gib": active_promotion / GIB,
                     "all_four_case_boundaries_gib": boundaries[slot] / GIB,
                     "fp32_forward_upper_estimate_gib": forward / GIB,
                     "fp32_b8_vjp_upper_estimate_gib": reverse / GIB})
    probe_cpu = 10 * h + 8 * q + 5 * f
    host = max(probe_cpu, 4 * max(gradient_numel) * 8)
    host += 2 * max(2 * f, largest_gradient) + master_bytes + 3 * h + 6 * GIB
    disk = h + sequence_length**2 + 1 * GIB
    return {"is_gpu_measurement": False, "sequence_length": sequence_length,
            "full_fp32_weights_gib": total_fp32 / GIB, "devices": rows,
            "fp32_block_workspace_estimate_gib": workspace / GIB,
            "host_upper_estimate_gib": host / GIB, "probe_cpu_references_gib": probe_cpu / GIB,
            "output_upper_estimate_gib": disk / GIB, "disk_limit_gib": 20,
            "gpu_allocator_limit_gib": 58, "host_start_minimum_gib": 50, "host_runtime_minimum_gib": 12}


class Experiment:
    def __init__(self, model, dump, directory, devices, masters=None, profile=True, minimum_host_gib=12,
                 loss_api=None, reference_output="recompute", replay_inputs=False):
        self.model, self.dump, self.out = model, dump, Path(directory)
        self.minimum_host_gib = minimum_host_gib
        self.loss_api, self.reference_output, self.replay_inputs = loss_api, reference_output, replay_inputs
        if loss_api is not None and ("loss_context" not in dump or "reference_outputs" not in dump):
            raise ValueError("旧 capture 没有 loss 字段；需按 RUN.md 重新截取，不能推测缺失值")
        self.cotangents, self.loss_stats, self.loss_means, self.residuals = {}, {}, {}, {}
        self.devices, self.root, self.profile = devices, devices[0], profile
        self.blocks = list(model.transformer_blocks)
        self.masters = masters or {}
        self.shipped = {n: t.dtype for n, t in list(model.named_parameters()) + list(model.named_buffers())}
        for n in self.masters:
            self.shipped[n] = torch.bfloat16
        model.eval().requires_grad_(False)
        if model.gradient_checkpointing or model.token_refiner.gradient_checkpointing:
            raise ValueError("关闭 activation checkpointing")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p != 0:
                raise ValueError("要求 dropout=0")
            if getattr(module, "fused_projections", False):
                raise ValueError("要求未融合 QKV")
            if hasattr(module, "processor"):
                if module.processor._parallel_config is not None:
                    raise ValueError("不使用 context parallel")
                module.processor._attention_backend = "native"
        self.block_devices = block_placement(len(self.blocks), devices)
        for name, child in model.named_children():
            if name != "transformer_blocks":
                child.to(self.root)
        for block, device in zip(self.blocks, self.block_devices):
            block.to(device)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "DONE").exists():
            raise FileExistsError("不能覆盖完成的实验")
        self.boundaries = {}
        self.outputs, self.deltas, self.contexts, self.timings = {}, {}, {}, {}
        self.memory = {case: {} for case in CASES}
        self.grad_names = []
        for i, block in enumerate(self.blocks):
            names = []
            for name, parameter in block.named_parameters():
                selected = i < 3 or name in {"attn.to_q.weight", "attn.to_q.base_layer.weight",
                                             "ff.net.2.weight", "ff.net.2.base_layer.weight"}
                parameter.requires_grad_(selected)
                if selected:
                    names.append(name)
            if not names:
                raise ValueError(f"block {i}: 缺少待测权重")
            self.grad_names.append(names)

    def cast(self, prefix, module, fp32, rounded=False):
        for name, p in module.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            dtype = torch.float32 if fp32 else self.shipped[key]
            if key in self.masters:
                master = self.masters[key].to(torch.bfloat16) if rounded else self.masters[key]
                p.data = master.to(device=p.device, dtype=dtype)
            elif p.dtype != dtype:
                p.data = p.data.to(dtype)
        for name, submodule in module.named_modules():
            for key, value in submodule._buffers.items():
                if value is not None and value.is_floating_point():
                    full = ".".join(x for x in (prefix, name, key) if x)
                    dtype = torch.float32 if fp32 else self.shipped[full]
                    submodule._buffers[key] = value.to(dtype)

    def check_memory(self, phase):
        return check_host_memory(self.out, phase, self.minimum_host_gib)

    def keep_boundary(self, case, block, start, value):
        key = (case, block, start)
        device = self.block_devices[min(block, len(self.blocks) - 1)]
        if key in self.boundaries or value.device != device:
            raise ValueError("重复边界或边界不在对应 block 的设备")
        self.boundaries[key] = value.detach()

    def hidden(self, case, block):
        micro = CASES[case][1]
        pieces = [self.boundaries[case, block, start] for start in range(0, 8, micro)]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def reset_memory(self):
        for device in self.devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

    def record_memory(self, case, phase):
        for device in self.devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                key = f"{phase}:{device}"
                previous = self.memory[case].get(key, {"allocated_gib": 0.0, "reserved_gib": 0.0})
                self.memory[case][key] = {
                    "allocated_gib": max(previous["allocated_gib"], torch.cuda.max_memory_allocated(device) / 2**30),
                    "reserved_gib": max(previous["reserved_gib"], torch.cuda.max_memory_reserved(device) / 2**30)}
                if torch.cuda.max_memory_allocated(device) > GPU_LIMIT:
                    raise MemoryError("PyTorch allocated 超过 58 GiB")

    def forward_case(self, case):
        self.reset_memory()
        started = time.perf_counter()
        fp32, micro = CASES[case]
        rounded = case in ROUNDED_LORA
        self.check_memory(f"{case}:forward_begin")
        # 非当前 block 保持原存储精度；在进入 block 时才升精度，计算后无损恢复。
        for i, block in enumerate(self.blocks):
            self.cast(f"transformer_blocks.{i}", block, False)
        for name, child in self.model.named_children():
            if name != "transformer_blocks":
                self.cast(name, child, fp32)
        outputs, delta, contexts = [], [], []
        cotangents, loss_stats, loss_means, residuals = [], [], [], []
        current = {"start": 0, "active": None, "reference": False}
        handles = []

        def release_active():
            previous = current["active"]
            if previous is not None:
                self.cast(f"transformer_blocks.{previous}", self.blocks[previous], False)
                current["active"] = None
        for i, (block, device) in enumerate(zip(self.blocks, self.block_devices)):
            def before(module, args, kwargs, i=i, device=device):
                if kwargs or len(args) != 5:
                    raise ValueError("vendor block 接口改变")
                self.check_memory(f"{case}:forward:{i}:before")
                release_active()
                self.cast(f"transformer_blocks.{i}", module, fp32, rounded)
                current["active"] = i
                if i == 0:
                    current["block0_dtypes"] = dict(Counter(str(p.dtype) for p in module.parameters()))
                args = move(args, device)
                x = args[0].detach().requires_grad_(True)
                finite("block input", x)
                if not current["reference"]:
                    self.keep_boundary(case, i, current["start"], x)
                    if i == 0:
                        contexts.append(cpu(args[1:]))
                return (x, *args[1:]), kwargs

            def after(module, args, output, i=i):
                finite("block output", output)
                self.check_memory(f"{case}:forward:{i}:after")
                # 当前图先释放，下一个 pre-hook 再恢复权重，避免保留两个 block 的 FP32 图。
                return output.detach()

            handles += [block.register_forward_pre_hook(before, with_kwargs=True),
                        block.register_forward_hook(after)]

        def tail_before(module, args):
            release_active()
            # 最末边界留在最后一张卡，只有本次输出头计算把输入搬到 root。
            if not current["reference"]:
                self.keep_boundary(case, len(self.blocks), current["start"], args[0])
            args = move(args, self.root)
            x = args[0].detach().requires_grad_(True)
            if not current["reference"]:
                current["tail"] = x
            return (x, *args[1:])

        handles.append(self.model.norm_out.register_forward_pre_hook(tail_before))
        try:
            with attention_context(self.root):
                metadata = settings(self.model, self.root)
                for start in range(0, 8, micro):
                    current["start"] = start
                    slicer = replay_kwargs if self.replay_inputs else sliced_kwargs
                    kwargs = slicer(self.dump["kwargs"], start, start + micro)
                    if self.loss_api is not None and self.reference_output == "recompute":
                        # 无 FSDP、dropout=0；先算 reference 避免与 policy 输出头的图同时占显存。
                        current["reference"] = True
                        try:
                            with self.loss_api.adapters_disabled(self.model), torch.no_grad():
                                refs = self.model(**move(kwargs, self.root))
                        finally:
                            current["reference"] = False
                    y = self.model(**move(kwargs, self.root))
                    if self.loss_api is None:
                        gs = [g[start:start + micro].to(o) for g, o in zip(self.dump["out_grads"], y)]
                    else:
                        if self.reference_output == "captured":
                            refs = [v[start:start + micro].to(o) for v, o in zip(self.dump["reference_outputs"], y)]
                        gs, stats, means, rs = own_cotangent(self.loss_api, y, kwargs, self.dump["loss_context"],
                                                           start, start + micro, refs)
                        cotangents.append(cpu(gs))
                        loss_stats.append(stats)
                        loss_means.append(means)
                        residuals.append(rs)
                        del refs

                    if len(y) != len(self.dump["out_grads"]):
                        raise ValueError("输出数量与 out_grads 不同")
                    for v in y:
                        finite("output", v)
                    dx, = torch.autograd.grad(y, current.pop("tail"), grad_outputs=gs)
                    finite("tail delta", dx)
                    outputs.append(cpu(y))
                    delta.append(cpu(dx))
                    del y, dx, gs
        finally:
            for handle in handles:
                handle.remove()
        release_active()
        for name, child in self.model.named_children():
            if name != "transformer_blocks":
                self.cast(name, child, False)
        metadata["block0_compute_parameter_dtypes"] = current["block0_dtypes"]
        metadata["weight_residency"] = "only active block promoted; outside modules promoted during forward"
        self.outputs[case] = [torch.cat([part[i] for part in outputs]) for i in range(len(outputs[0]))]
        self.deltas[case] = torch.cat(delta)
        self.contexts[case] = contexts
        if self.loss_api is not None:
            for destination, values in ((self.cotangents, cotangents), (self.loss_means, loss_means),
                                         (self.residuals, residuals)):
                destination[case] = [torch.cat([part[i] for part in values]) for i in range(2)]
            self.loss_stats[case] = loss_stats
        self.record_memory(case, "forward")
        self.timings[case] = {"forward_seconds": time.perf_counter() - started, "reverse_seconds": 0.0}
        metadata["inputs"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
                              for k, v in self.dump["kwargs"].items() if torch.is_tensor(v)}
        write_json(self.out / f"{case}-settings.json", metadata)
        print(json.dumps({"event": "forward_done", "case": case, **self.timings[case]}), flush=True)

    def locate(self):
        # 记录首个逐位差异及首个相对误差 > 1e-5 的边界；入口差异单列。
        boundaries = []
        for i in range(len(self.blocks) + 1):
            self.check_memory(f"locate:{i}")
            a, b = self.hidden("bf16_b8", i), self.hidden("bf16_b1", i)
            boundaries.append({"boundary": i, "samples": per_sample(a, b)})
            del a, b
        first = next((r["boundary"] for r in boundaries
                      if any(s["max_abs"] > 0 for s in r["samples"])), None)
        threshold = next((r["boundary"] for r in boundaries
                          if any((s["rel_l2"] or 0) > 1e-5 for s in r["samples"])), None)
        block = 0 if first in (None, 0) else first - 1
        result = {"first_bitwise_boundary": first, "first_rel_l2_over_1e_minus_5_boundary": threshold,
                  "difference_before_block0": first == 0, "fixed_input_probe_block": block,
                  "boundaries": boundaries}
        write_json(self.out / "location.json", result)
        return block

    def reverse_case(self, case, index):
        self.reset_memory()
        started = time.perf_counter()
        fp32, micro = CASES[case]
        block, device = self.blocks[index], self.block_devices[index]
        self.check_memory(f"{case}:reverse:{index}:before")
        self.cast(f"transformer_blocks.{index}", block, fp32, case in ROUNDED_LORA)
        named = dict(block.named_parameters())
        params = [named[n] for n in self.grad_names[index]]
        summed, dxs = {}, []
        recompute = []
        for j, start in enumerate(range(0, 8, micro)):
            x = self.boundaries[case, index, start].detach().requires_grad_(True)
            context = move(self.contexts[case][j], device)
            dy = self.deltas[case][start:start + micro].to(device)
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] +
                ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                record_shapes=False) if self.profile and index == 0 and start == 0 else None
            with attention_context(device), prof if prof is not None else nullcontext():
                y = block(x, *context)
                expected = self.boundaries[case, index + 1, start].to(device)
                # 重算必须逐位等于原前向；不把改变了的函数继续当作原函数求导。
                equal = torch.equal(y.detach(), expected)
                if not equal:
                    raise RuntimeError(f"{case} block={index} sample={start}: 重算不逐位相同")
                gradients = torch.autograd.grad(y, [x, *params], grad_outputs=dy.to(y))
            if prof is not None:
                events = sorted({(event.name, str(event.input_shapes)) for event in prof.events()
                                 if event.device_type == torch.autograd.DeviceType.CUDA or
                                 any(s in event.name.lower() for s in ("attention", "mm", "linear"))})
                write_json(self.out / f"{case}-block0-profile.json", {"events": events, "record_shapes": False,
                           "cuda_events_present": any(e.device_type == torch.autograd.DeviceType.CUDA
                                                      for e in prof.events())})
            recompute.append(equal)
            finite("delta", gradients[0])
            dxs.append(cpu(gradients[0]))
            for name, value in zip(self.grad_names[index], gradients[1:]):
                # 逐条产生的梯度先转 CPU FP64 再求和，排除低精度叶子累加的影响。
                g = value.detach().cpu().double()
                finite(name, g)
                if name in summed:
                    summed[name].add_(g)
                else:
                    summed[name] = g
                self.check_memory(f"{case}:reverse:{index}:gradient:{name}")
            del x, y, dy, context, gradients, expected, g, value
        self.cast(f"transformer_blocks.{index}", block, False)
        self.check_memory(f"{case}:reverse:{index}:after")
        self.record_memory(case, "reverse")
        self.timings[case]["reverse_seconds"] += time.perf_counter() - started
        self.deltas[case] = torch.cat(dxs)
        return {"hidden": self.hidden(case, index), "delta": self.deltas[case],
                "gradients": summed, "recompute_exact": all(recompute)}

    def run(self, vendor):
        started = time.perf_counter()
        for case in CASES:
            self.forward_case(case)
        write_json(self.out / "outputs.json", {
            f"{a}_vs_{b}": [per_sample(x, y) for x, y in zip(self.outputs[a], self.outputs[b])]
            for a, b in PAIRS})
        write_json(self.out / "captured_outputs.json", {
            case: [per_sample(x, y) for x, y in zip(self.outputs[case], self.dump["outputs"])]
            for case in CASES})
        if self.loss_api is not None:
            write_json(self.out / "cotangents.json", {
                "reference_output": self.reference_output, "loss_stats": self.loss_stats,
                "comparisons": {f"{a}_vs_{b}": [per_sample(x, y) for x, y in
                    zip(self.cotangents[a], self.cotangents[b])] for a, b in PAIRS},
                "against_captured": {case: [per_sample(x, y) for x, y in
                    zip(self.cotangents[case], self.dump["out_grads"])] for case in CASES},
                "mean_error_over_fp32_residual": {case: [[
                    (x[i] - y[i]).double().norm().item() / r[i].double().norm().item()
                    if r[i].count_nonzero() else None for i in range(8)]
                    for x, y, r in zip(self.loss_means[case], self.loss_means["fp32_b8"],
                                       self.residuals["fp32_b8"])] for case in CASES}})
        # 只检查复现身份；不把 BF16/FP32 或 B1/B8 的实验差异当作失败。
        check_capture_identity(self.out, "bf16-b8-output-check", self.outputs["bf16_b8"],
                               self.dump["outputs"], self.root, exact=True)
        gs = self.cotangents["bf16_b8"] if self.loss_api is not None else [
            g.to(o) for g, o in zip(self.dump["out_grads"], self.outputs["bf16_b8"])]
        check_capture_identity(self.out, "bf16-b8-cotangent-check", gs, self.dump["out_grads"], self.root)
        # 舍入对照没有 B1 case，跳过 B8/B1 差异定位与单层探针
        probe_index = None
        if "bf16_b1" in CASES:
            probe_index = self.locate()
            probe_hidden = self.hidden("fp32_b8", probe_index)
            probe_context = self.contexts["fp32_b8"][0]
            save(self.out / "probe-input.pt", {"block": probe_index, "hidden": cpu(probe_hidden),
                                               "context": probe_context})
            probe_block(self, vendor, probe_index, probe_hidden, probe_context)
            del probe_hidden
        summaries = []
        for index in reversed(range(len(self.blocks))):
            results = {case: self.reverse_case(case, index) for case in CASES}
            comparisons = {}
            for a, b in PAIRS:
                x, y = results[a], results[b]
                comparisons[f"{a}_vs_{b}"] = {
                    "hidden": per_sample(x["hidden"], y["hidden"]),
                    "delta": per_sample(x["delta"], y["delta"]),
                    "gradients": {n: distance(x["gradients"][n], y["gradients"][n])
                                  for n in self.grad_names[index]}}
                comparison = comparisons[f"{a}_vs_{b}"]
                comparison["gradient_vector"] = combine_distances(comparison["gradients"].values())
            row = {"block": index, "recompute_exact": {k: v["recompute_exact"] for k, v in results.items()},
                   "comparisons": comparisons}
            write_json(self.out / f"block-{index:02d}.json", row)
            summaries.append(row)
            del results, x, y
            # h[index] 下一轮仍用于校验上一层的输出；释放已经用完的设备边界。
            for case, (_, micro) in CASES.items():
                for start in range(0, 8, micro):
                    del self.boundaries[case, index + 1, start]
            self.check_memory(f"block:{index}:statistics_saved")
            print(json.dumps({"event": "reverse_block_done", "block": index}), flush=True)
        self.boundaries.clear()
        write_json(self.out / "summary.json", {"schema": 2, "cases": CASES, "pairs": PAIRS,
                   "boundary_storage": "block device; no disk", "gpu_allocator_limit_gib": 58,
                   "host_memory": self.check_memory("finished"),
                   "output_bytes": sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()),
                   "timings": self.timings, "memory_peaks_by_case_phase": self.memory,
                   "total_seconds_including_statistics": time.perf_counter() - started,
                   "block_count": len(self.blocks), "fixed_input_probe_block": probe_index,
                   "blocks_0_2_gradient_vector": {
                       f"{a}_vs_{b}": combine_distances(
                           row["comparisons"][f"{a}_vs_{b}"]["gradient_vector"]
                           for row in summaries if row["block"] < 3) for a, b in PAIRS},
                   "weight_identity": "live LoRA + base" if self.masters else "base only; no checkpoint LoRA",
                   "gradient_sum_dtype": "cpu float64",
                   "timestep_table": "repacked scalar B1" if self.replay_inputs else "fixed, never compacted",
                   "all_recomputations_exact": all(all(r["recompute_exact"].values()) for r in summaries)})
        write_json(self.out / "scope.json", {
            "cotangent": "own UniRL PG+KL" if self.loss_api is not None else "captured fixed",
            "reference_output": self.reference_output if self.loss_api is not None else None,
            "replay_inputs": self.replay_inputs,
            "capture_metadata": self.dump.get("capture_metadata", {}),
            "blocks_0_2_lora_only": {f"{a}_vs_{b}": combine_distances(
                value for row in summaries if row["block"] < 3
                for name, value in row["comparisons"][f"{a}_vs_{b}"]["gradients"].items()
                if ".lora_" in name) for a, b in PAIRS}})
        (self.out / "DONE").write_text(f"{len(CASES)} cases and comparisons complete\n")
        return summaries


def probe_block(experiment, vendor, index, hidden, context):
    block, device = experiment.blocks[index], experiment.block_devices[index]
    for fp32 in (True, False):
        tag = "fp32" if fp32 else "bf16"
        experiment.cast(f"transformer_blocks.{index}", block, fp32)
        dtype = block.norm1.weight.dtype
        # 两种 micro 共享同一个 fp32 入口，先整体转同一 dtype；表、RoPE、mask 也完全固定。
        common_hidden = hidden.to(dtype)
        common_context = context
        reports, names, references = [], [], {}
        state = {}
        experiment.reset_memory()

        def record(name, tensor):
            experiment.check_memory(f"probe:{tag}:{name}:before")
            if state["micro"] == 8:
                references[name] = cpu(tensor)
                names.append(name)
            else:
                reference = references[name]
                reports.append({"operation": name, "sample": state["start"],
                                **distance(reference[state["start"]:state["start"] + 1], tensor)})
                if state["start"] == 7:
                    del references[name]
            experiment.check_memory(f"probe:{tag}:{name}:after")

        handles = []
        module_names = ["norm1", "attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k",
                        "attn.to_out.0", "attn.to_out.1", "norm2", "ff.net.0.proj", "ff.net.0", "ff.net.2"]
        for name in module_names:
            def hook(module, args, output, name=name):
                record(name, output)
                if name == "ff.net.0.proj":
                    value, gate = output.chunk(2, dim=-1)
                    record("ff_value_projection", value)
                    record("ff_gate_projection", gate)
            handles.append(block.get_submodule(name).register_forward_hook(hook))

        def adaln(module, args, output):
            state["gates"] = output[2], output[5]

        def attn_input(module, args):
            record("adaln_msa", args[0])

        def attn_output(module, args, output):
            record("gate_x_attn", state["gates"][0][state["indices"]] * output)

        def ff_input(module, args):
            record("adaln_mlp", args[0])

        def ff_output(module, args, output):
            record("gate_x_ff", state["gates"][1][state["indices"]] * output)

        handles += [block.adaln_proj.register_forward_hook(adaln),
                    block.attn.register_forward_pre_hook(attn_input),
                    block.attn.register_forward_hook(attn_output),
                    block.ff.register_forward_pre_hook(ff_input), block.ff.register_forward_hook(ff_output)]
        original_rope, original_dispatch = vendor._apply_rotary_emb, vendor.dispatch_attention_fn

        def rope(*args, **kwargs):
            result = original_rope(*args, **kwargs)
            name = "rope_q" if state["rope_calls"] == 0 else "rope_k"
            state["rope_calls"] += 1
            record(name, result)
            return result

        def dispatch(*args, **kwargs):
            result = original_dispatch(*args, **kwargs)
            record("attention_core", result)
            return result

        vendor._apply_rotary_emb, vendor.dispatch_attention_fn = rope, dispatch
        try:
            # 保持 grad-enabled 的 forward 分派条件，避免测成 inference 专用 kernel。
            with torch.enable_grad(), attention_context(device):
                for micro in (8, 1):
                    for start in range(0, 8, micro):
                        temb, ids, rotary, mask = move(common_context, device)
                        if ids.ndim == 2:
                            ids = ids[start:start + micro]
                        state.update(micro=micro, start=start, rope_calls=0, indices=ids)
                        x = common_hidden[start:start + micro].to(device).detach().requires_grad_(True)
                        output = block(x, temb, ids, rotary, mask)
                        record("block_output", output)
                        del output, x
        finally:
            vendor._apply_rotary_emb, vendor.dispatch_attention_fn = original_rope, original_dispatch
            for handle in handles:
                handle.remove()
        if len(reports) != len(names) * 8 or len(set(names)) != len(names):
            raise RuntimeError("probe 操作计数不一致")
        first = next((n for n in names if any(r["operation"] == n and r["max_abs"] > 0 for r in reports)), None)
        write_json(experiment.out / f"probe-{tag}.json", {"block": index, "shared_fp32_input_cast": str(dtype),
                   "operation_order": names, "first_bitwise_difference": first, "samples": reports,
                   "relative_error_reference": "B1", "forward_only": True, "grad_enabled": True})
        if references:
            raise RuntimeError("局部 probe 未释放全部参考 tensor")
        experiment.cast(f"transformer_blocks.{index}", block, False)
        experiment.record_memory(f"{tag}_b8", "probe")
        experiment.check_memory(f"probe:{tag}:finished")


def validate_dump(dump):
    if "groups" in dump:
        if len(dump["groups"]) != 3:
            raise ValueError("rank0 全量 capture 必须恰有三个 B8")
        for item in dump["groups"]:
            validate_dump(item)
        return
    kwargs = dump["kwargs"]
    if kwargs.get("adaln_cache_entry") is not None or kwargs.get("attention_kwargs") not in (None, {}):
        raise ValueError("要求 V2，无 AdaLN cache / attention kwargs")
    if kwargs["hidden_states"].shape[0] != 8:
        raise ValueError("需要同 prompt 的真实 8 条样本")
    for key in BATCH_KEYS:
        if kwargs[key].shape[0] != 8:
            raise ValueError(key)
    if len(dump["outputs"]) != 2 or len(dump["out_grads"]) != 2:
        raise ValueError("需要 video/audio 两个输出及其 cotangent")
    for o, g in zip(dump["outputs"], dump["out_grads"]):
        if o.shape != g.shape or o.shape[0] != 8:
            raise ValueError("输出与 cotangent 形状不同")
        finite("captured output", o)
        finite("captured cotangent", g)
    if not any(bool(g.count_nonzero()) for g in dump["out_grads"]):
        raise ValueError("全部 cotangent 为零")
    if "loss_context" in dump or "reference_outputs" in dump:
        context = dump["loss_context"]
        if len(dump["reference_outputs"]) != 2:
            raise ValueError("需要两个 reference 输出")
        for output, ref, key in zip(dump["outputs"], dump["reference_outputs"], ("prev_video", "prev_audio")):
            if ref.shape != output.shape or context[key].shape != output.shape:
                raise ValueError("loss context 形状不匹配")
            finite("reference output", ref)
            finite(key, context[key])
        if context["sde_index_per_sample"].shape != (8,) or context["policy_loss_weights"].shape != (8, 1):
            raise ValueError("SDE index/TGR 必须逐样本保存")
        if context["advantages"].numel() != 8:
            raise ValueError("缺少逐样本 advantage")
        for key in ("advantages", "policy_loss_weights", "video_sigmas", "audio_sigmas"):
            finite(key, context[key])


def run_groups(model, dump, directory, devices, vendor, masters, *, loss_api=None,
               reference_output="recompute", all_groups=False, replay_inputs=False, minimum_host_gib=12, profile=True):
    validate_dump(dump)
    groups = dump.get("groups", [dump])
    if all_groups and len(groups) != 3:
        raise ValueError("--groups all 需要 rank0 三组的新 capture")
    selected = groups if all_groups else groups[:1]
    records, folders, budgets = [], [], []
    total_disk = 0
    # 所有组选定后先整体预检；任何一组超预算，都不开始第一组 H3 forward。
    for i, item in enumerate(selected):
        folder = Path(directory) / f"group-{i:02d}" if all_groups else Path(directory)
        folder.mkdir(exist_ok=True)
        plan = resource_plan(model, item["kwargs"]["token_tags"].numel(), devices)
        extra = 0.5 if loss_api is not None else 0
        plan["loss_extra_reserve_gib"] = extra
        total_disk += plan["output_upper_estimate_gib"]
        write_json(folder / "resource-plan.json", plan)
        gpu = max(max(p["fp32_forward_upper_estimate_gib"], p["fp32_b8_vjp_upper_estimate_gib"])
                  for p in plan["devices"]) + extra
        host = plan["host_upper_estimate_gib"] + extra
        budgets.append({"group": i, "gpu_gib": gpu, "host_gib": host, "cumulative_disk_gib": total_disk,
                        "violations": [name for name, exceeded in (("gpu", gpu > 58), ("host", host >= 40),
                                       ("cumulative_disk", total_disk >= 20)) if exceeded]})
        folders.append(folder)
    budget_report = {"passed": not any(b["violations"] for b in budgets), "groups": budgets}
    write_json(Path(directory) / "budget-preflight.json", budget_report)
    if not budget_report["passed"]:
        write_json(Path(directory) / "STOPPED.json", {"reason": "resource_budget_exceeded", **budget_report})
        raise MemoryError("选中组的整体预算未通过；未运行 H3，不改变 micro/序列长度")
    api = loss_api
    if api is None and any("loss_context" in item for item in selected):
        api = load_loss_api(vendor.__file__)
    for item, folder in zip(selected, folders):
        if "loss_context" not in item:
            if loss_api is not None:
                raise ValueError("loss 模式需要新的 capture loss_context")
            write_json(folder / "capture-loss-check.json", {
                "status": "not_available", "reason": "旧 fixed capture 无 loss_context；只检查 BF16 B8 输出身份"})
            continue
        # 只使用保存的 tensor，在同一计算设备重建并比较；不调用 H3。
        with torch.enable_grad(), torch.autocast(devices[0].type, enabled=False):
            gs, *_ = own_cotangent(api, move(item["outputs"], devices[0]), item["kwargs"],
                                   item["loss_context"], 0, 8, move(item["reference_outputs"], devices[0]))
        check_capture_identity(folder, "capture-loss-check", gs, item["out_grads"], devices[0])
        del gs
    # 顺序完成并释放一组，不让 24 条边界同时驻留；不把各组 cosine 相加冒充 24 条总梯度。
    for i, (item, folder) in enumerate(zip(selected, folders)):
        exp = Experiment(model, item, folder, devices, masters, loss_api=loss_api,
                         reference_output=reference_output, replay_inputs=replay_inputs,
                         minimum_host_gib=minimum_host_gib, profile=profile)
        exp.run(vendor)
        records.append({"group": i, "directory": folder.name,
                        "scope": json.loads((folder / "scope.json").read_text())})
        del exp
    if all_groups:
        write_json(Path(directory) / "groups.json", {"rows": 24, "groups": records,
                   "aggregation": "per group; no cross-group gradient vector retained"})
        (Path(directory) / "DONE").write_text("three B8 groups complete\n")


def main():
    parser = argparse.ArgumentParser(description="H3 micro8/1：设备层边界、逐 tensor 加载、单层 VJP")
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live-checkpoint", type=Path)
    group.add_argument("--base-only", action="store_true")
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--cotangent", choices=("fixed", "loss"), default="fixed")
    parser.add_argument("--reference-output", choices=("recompute", "captured"), default="recompute")
    parser.add_argument("--groups", choices=("first", "all"), default="first")
    parser.add_argument("--replay-inputs", action="store_true")
    parser.add_argument("--rounding-control", action="store_true")
    args = parser.parse_args()
    if args.rounding_control:
        CASES.clear()
        CASES.update(ROUNDING_CASES)
        PAIRS[:] = ROUNDING_PAIRS
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("完整模型要求恰好 8 张已授权、空闲的可见 CUDA 卡；CPU 验证用 cpu_validate.py")
    if args.live_checkpoint and args.lora_alpha is None:
        raise ValueError("--lora-alpha 必须来自训练配置，不猜测默认值")
    prepare_output(args.out)
    configure_precision()
    devices = [torch.device(f"cuda:{i}") for i in range(8)]
    # 只限制本进程的 allocator；超过限额直接 OOM，不影响其他进程、不自动重试。
    for device in devices:
        total = torch.cuda.get_device_properties(device).total_memory
        if total < GPU_LIMIT:
            raise RuntimeError("目标卡容量不足 58 GiB")
        torch.cuda.set_per_process_memory_fraction(GPU_LIMIT / total, device)
    try:
        vendor = load_vendor(args.vendor)
        dump = torch.load(args.dump, map_location="cpu", weights_only=True, mmap=True)
        validate_dump(dump)
        model = load_model(vendor, args.pretrained, devices, args.out)
        masters = attach_lora(model, args.live_checkpoint, args.lora_alpha) if args.live_checkpoint else {}
        run_groups(model, dump, args.out, devices, vendor, masters,
                   loss_api=load_loss_api(args.vendor) if args.cotangent == "loss" else None,
                   reference_output=args.reference_output, all_groups=args.groups == "all",
                   replay_inputs=args.replay_inputs)
    except (MemoryError, torch.OutOfMemoryError) as error:
        if not (args.out / "STOPPED.json").exists():
            write_json(args.out / "STOPPED.json", {"reason": type(error).__name__,
                       "gpu_allocator_limit_gib": 58,
                       "completed_blocks": sorted(p.name for p in args.out.glob("block-*.json"))})
        raise


if __name__ == "__main__":
    main()
