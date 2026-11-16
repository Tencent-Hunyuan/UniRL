import json
import os
from pathlib import Path

import torch


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def make_loss_context(algorithm, part, training_progress):
    from unirl.algorithms.base import resolve_clip_range_from_schedule
    stage, segment = algorithm.stage, part.segment
    if stage.strategy.canonical_name != "flash" or not stage.audio_joint_sde:
        raise ValueError("需要 Flash 双流 SDE")
    if not algorithm.batch_normalize_tgr or not algorithm._checkup_on_policy:
        raise ValueError("需要已 prepare 的 TGR 与本次前向 detach anchor")
    if algorithm.beta != 0.004 or part.batch_size != 24 or stage.logprob_dtype != torch.float32:
        raise ValueError("需要本次 rank0 24 条、beta=0.004、FP32 logp")
    return cpu_tree({"schema": 1, "video_sigmas": segment.sigmas,
        "audio_sigmas": stage.audio_schedule(segment.sigmas), "eta": algorithm.params.eta,
        "sde_index_per_sample": segment.sde_index_per_sample,
        "prev_video": segment.latents[:, 1], "prev_audio": segment.aux_latents[:, 1],
        "advantages": part.advantages, "policy_loss_weights": segment.policy_loss_weights,
        "adv_clip_max": algorithm.adv_clip_max,
        "clip_range": resolve_clip_range_from_schedule(algorithm.clip_range, algorithm.clip_schedule, training_progress),
        "beta": algorithm.beta, "local_rows": 24, "split_factor": 1,
        "audio_in_policy": True, "logprob_dtype": "float32"})


def context_slice(context, start, stop):
    row_keys = {"sde_index_per_sample", "prev_video", "prev_audio", "advantages", "policy_loss_weights"}
    return {k: cpu_tree(v[start:stop] if k in row_keys else v) for k, v in context.items()}


def install_capture(model, directory, metadata, *, loss_context=None, groups=1):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "fwd_rank0.pt").exists() or (directory / "CAPTURE_DONE.json").exists():
        raise FileExistsError("拒绝覆盖已有 capture")
    if groups not in (1, 3):
        raise ValueError("只能截第一组或 rank0 的全部三组")
    completed, state = [], {}

    def before(module, args, kwargs):
        if len(completed) == groups:
            return
        if not torch.is_grad_enabled():
            if loss_context is not None and state.get("await_reference"):
                if "reference_outputs" in state:
                    raise ValueError("同一组出现多次 reference forward")
                state["reference_pending"] = True
            return
        if state:
            raise RuntimeError("上一组 backward 尚未完整截取")
        if args or kwargs.get("return_dict", True):
            raise ValueError("需要 H3 kwargs 调用与 tuple 输出")
        if kwargs["hidden_states"].shape[0] != 8:
            raise ValueError("第一个带梯度调用不是 B=8")
        if kwargs.get("adaln_cache_entry") is not None:
            raise ValueError("需要 V2，禁用 AdaLN cache")
        state["input_layout"] = {k: {"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)}
                                 for k, v in kwargs.items() if torch.is_tensor(v)}
        state["kwargs"] = cpu_tree(kwargs)
        state["pending"] = True
        state["precision"] = {
            "torch": str(torch.__version__), "tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "processor_backends": sorted({str(m.processor._attention_backend) for m in module.modules()
                                           if hasattr(m, "processor")})}
        if state["precision"]["tf32"] or state["precision"]["bf16_reduction"]:
            raise ValueError("capture 的实际设置不符合 V2")

    def after(module, args, kwargs, outputs):
        if state.pop("reference_pending", False):
            if len(outputs) != 2 or any(o.requires_grad for o in outputs):
                raise ValueError("KL reference 应为两个无梯度输出")
            if any(torch.is_tensor(v) and not torch.equal(v, state["kwargs"][k]) for k, v in cpu_tree(kwargs).items()):
                raise ValueError("reference 与 policy 输入不同")
            state["reference_outputs"] = cpu_tree(outputs)
            return
        if not state.pop("pending", False):
            return
        state["await_reference"] = loss_context is not None
        if len(outputs) != 2 or not all(o.requires_grad for o in outputs):
            raise ValueError("需要 video/audio 两个可微输出")
        saved_outputs = [o.detach().cpu().clone() for o in outputs]
        grads = [None, None]

        def receive(index, gradient):
            grads[index] = gradient.detach().cpu().clone()
            if not all(g is not None for g in grads):
                return
            for output, grad in zip(saved_outputs, grads):
                if output.shape != grad.shape or not bool(torch.isfinite(output).all() & torch.isfinite(grad).all()):
                    raise ValueError("capture 非有限值或形状不同")
            payload = {"kwargs": state["kwargs"], "outputs": saved_outputs, "out_grads": grads,
                       "capture_metadata": {**metadata, "precision": state["precision"], "group": len(completed),
                                            "original_input_layout": state["input_layout"]}}
            if loss_context is not None:
                if "reference_outputs" not in state:
                    raise RuntimeError("没有截到 KL reference 输出，不能重建 loss")
                start = len(completed) * 8
                payload["loss_context"] = context_slice(loss_context, start, start + 8)
                payload["reference_outputs"] = state["reference_outputs"]
                payload["capture_metadata"]["sde_indices"] = payload["loss_context"]["sde_index_per_sample"].tolist()
                for key, input_key in (("prev_video", "hidden_states"), ("prev_audio", "audio_hidden_states")):
                    if payload["loss_context"][key].shape != state["kwargs"][input_key].shape:
                        raise ValueError("Part 与截取组不匹配")
            completed.append(payload)
            state.clear()
            if len(completed) != groups:
                return
            payload = completed[0] if groups == 1 else {"schema": 2, "groups": completed}
            temporary = directory / "fwd_rank0.pt.partial"
            with temporary.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "fwd_rank0.pt")
            marker = directory / "CAPTURE_DONE.json.partial"
            marker.write_text(json.dumps({"schema": 1, "batch": 8, "groups": groups, "rows": groups * 8,
                                          "output_shapes": [list(o.shape) for o in saved_outputs],
                                          "bytes": (directory / "fwd_rank0.pt").stat().st_size,
                                          "owner": os.environ.get("CHECKUP_OWNER")}) + "\n")
            os.replace(marker, directory / "CAPTURE_DONE.json")
            print("MICRO_CAPTURE_COMPLETE", flush=True)

        for index, output in enumerate(outputs):
            output.register_hook(lambda gradient, index=index: receive(index, gradient))

    return [model.register_forward_pre_hook(before, with_kwargs=True),
            model.register_forward_hook(after, with_kwargs=True)]
