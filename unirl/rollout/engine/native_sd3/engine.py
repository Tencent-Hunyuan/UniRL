"""Native SD3 rollout worker with rapidly refreshable Hopper FP8 scouting."""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
from datetime import timedelta
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.sd3.bundle import SD3Bundle
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine, RolloutCapabilities
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.primitives import Images
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.utils.dtypes import parse_torch_dtype

from .config import NativeSD3EngineConfig
from .quantization import FP8Controller, RoutedTransformer, convert_transformer_for_fp8

logger = logging.getLogger(__name__)

_DTYPES: Dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _dtype_from_name(value: str) -> torch.dtype:
    key = value if value.startswith("torch.") else f"torch.{value}"
    if key not in _DTYPES:
        raise KeyError(f"NativeSD3RolloutEngine does not support wire dtype {value!r}.")
    return _DTYPES[key]


class NativeSD3RolloutEngine(BaseRolloutEngine):
    """A worker-local Diffusers SD3 pipeline plus raw-NCCL BF16 weight refresh."""

    _component_name = "native_sd3"
    capabilities = RolloutCapabilities(
        rollout_precisions=frozenset({"bf16", "fp8"}),
        reward_image_resize=True,
        transactional_weight_publication=True,
    )

    def __init__(
        self,
        config: NativeSD3EngineConfig,
        *,
        model_config: SD3PipelineConfig,
        strategy: StepStrategy,
        device: Optional[torch.device] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int)):
            raise TypeError(f"NativeSD3RolloutEngine.rank must be an integer or None, got {rank!r}.")
        self.rank = 0 if rank is None else rank
        if bool(model_config.meta_init_transformer):
            raise ValueError("NativeSD3RolloutEngine requires meta_init_transformer=false for its eager model copy.")
        if config.fp8_enabled:
            model_dtype = parse_torch_dtype(model_config.model_precision, field_name="model_precision")
            autocast_dtype = parse_torch_dtype(model_config.autocast_precision, field_name="autocast_precision")
            if model_dtype != torch.bfloat16 or autocast_dtype != torch.bfloat16:
                raise ValueError(
                    "NativeSD3 FP8 conversion requires BF16 model and autocast precision; "
                    f"got model={model_dtype}, autocast={autocast_dtype}."
                )
        runtime_model_config = dataclasses.replace(model_config, device=self.device, load_vae=True)
        self.bundle = SD3Bundle.from_config(runtime_model_config)
        self.pipeline = SD3Pipeline(
            bundle=self.bundle,
            strategy=strategy,
            shift=float(runtime_model_config.shift),
            autocast_precision=runtime_model_config.autocast_precision,
            trajectory_precision=runtime_model_config.trajectory_precision,
            logprob_precision=runtime_model_config.logprob_precision,
            batch_replay_steps=runtime_model_config.batch_replay_steps,
        )
        self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
            self.bundle.pretrained_path,
            shift=float(runtime_model_config.shift),
        )

        self._controller = FP8Controller(config)
        transformer = self.bundle.transformer.eval().requires_grad_(False)
        self._parameter_targets, report = convert_transformer_for_fp8(
            transformer,
            config=config,
            controller=self._controller,
        )
        if config.compile_model:
            transformer = torch.compile(transformer, mode=config.compile_mode)
        self.bundle.transformer = RoutedTransformer(transformer, self._controller).eval()

        self._weight_groups: Dict[str, dist.ProcessGroup] = {}
        self._weight_group_timeouts: Dict[str, float] = {}
        self._failed_weight_groups: set[str] = set()
        self._generate_lock = threading.Lock()
        self._shutdown = False
        self._weights_valid = True
        self._publication_names: Optional[set[str]] = None
        self._prepared_bucket = None
        self._version = 0
        logger.info(
            "NativeSD3 rollout ready rank=%d fp8=%s converted=%d skipped=%d",
            self.rank,
            config.fp8_enabled,
            len(report.replaced),
            len(report.skipped),
        )
        if report.replaced:
            logger.info("NativeSD3 FP8 linears: %s", ", ".join(report.replaced))

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown:
                raise RuntimeError("NativeSD3RolloutEngine.generate called after shutdown.")
            if not self._weights_valid:
                raise RuntimeError("NativeSD3RolloutEngine has an incomplete failed weight publication.")
            result = self._generate_core(sample)
            return self._stamp_output_version(result)

    def _generate_core(self, sample: Sample) -> Sample:
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        ensure_sample_sigmas(sample, self.schedule_policy)
        batch_size = int(gen.batch_size)
        fbs = self.config.forward_batch_size
        if fbs is None or batch_size <= fbs:
            return self._generate_batch(sample)

        input_parts = sample.parts[:-1]
        chunks: List[Part] = []
        for start in range(0, batch_size, fbs):
            end = min(start + fbs, batch_size)
            request = Sample(parts=[*input_parts, gen.slice(start, end)])
            chunks.append(self._generate_batch(request).parts[-1])
        return sample.replace_frontier(Part.concat(chunks))

    def _generate_batch(self, sample: Sample) -> Sample:
        params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        with self._controller.rollout(
            mode=params.rollout_precision,
            total_steps=int(params.num_inference_steps),
        ):
            result = self.pipeline.generate(sample)
        return self._resize_reward_images(result, params.reward_image_size)

    @staticmethod
    def _resize_reward_images(sample: Sample, image_size: Optional[int]) -> Sample:
        if image_size is None:
            return sample
        frontier = sample.parts[-1]
        images = frontier.primitives.get("image")
        if not isinstance(images, Images):
            raise TypeError("NativeSD3 reward_image_size requires image output.")
        dense = images.to_dense()
        resized = F.interpolate(
            dense,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0.0, 1.0)
        return sample.replace_frontier(
            dataclasses.replace(frontier, primitives={**frontier.primitives, "image": Images.from_dense(resized)})
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
        timeout_s: Optional[float] = None,
    ) -> None:
        del track_prefix
        from unirl.utils.distributed_utils import eager_connect_process_group, init_process_group

        if not isinstance(master_address, str) or not master_address:
            raise TypeError(f"master_address must be a non-empty string, got {master_address!r}.")
        for name, value, minimum in (
            ("master_port", master_port, 1),
            ("rank_offset", rank_offset, 0),
            ("world_size", world_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(f"timeout_s must be finite and > 0 when set, got {timeout_s!r}.")
        if not isinstance(group_name, str) or not group_name:
            raise TypeError(f"group_name must be a non-empty string, got {group_name!r}.")
        if group_name in self._weight_groups:
            raise RuntimeError(f"Native SD3 weight group {group_name!r} is already initialized.")
        group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank_offset,
            group_name=group_name,
            timeout=timedelta(seconds=timeout_s) if timeout_s is not None else None,
        )
        try:
            eager_connect_process_group(group, torch.device(self.device))
        except Exception:
            dist.destroy_process_group(group)
            raise
        self._weight_groups[group_name] = group
        self._weight_group_timeouts[group_name] = float(timeout_s) if timeout_s is not None else 300.0

    def begin_weights_update(self, *, group_name: str, track_prefix: str = "") -> None:
        del track_prefix
        if group_name not in self._weight_groups:
            raise RuntimeError(f"No native SD3 weight group {group_name!r}; initialize it before sync.")
        if group_name in self._failed_weight_groups:
            raise RuntimeError(f"Native SD3 weight group {group_name!r} is failed and cannot be reused.")
        with self._generate_lock:
            self._weights_valid = False
            self._publication_names = set()
            self._prepared_bucket = None

    def prepare_weights_update(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        """Allocate, validate, and arm async receives before the sender broadcasts."""
        del track_prefix
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError(f"names/dtypes/shapes length mismatch: {len(names)}/{len(dtypes)}/{len(shapes)}")
        group = self._weight_groups.get(group_name)
        if group is None:
            raise RuntimeError(f"No native SD3 weight group {group_name!r}; initialize it before sync.")
        if group_name in self._failed_weight_groups:
            raise RuntimeError(f"Native SD3 weight group {group_name!r} is failed and cannot be reused.")
        with self._generate_lock:
            if self._publication_names is None:
                raise RuntimeError("Native SD3 weight bucket arrived outside begin_weights_update().")
            if self._prepared_bucket is not None:
                raise RuntimeError("Native SD3 already has a prepared weight bucket awaiting broadcast.")
            signature = self._bucket_signature(names, dtypes, shapes)
            received = [
                (
                    name,
                    torch.empty(shape, dtype=_dtype_from_name(dtype_name), device=self.device),
                )
                for name, dtype_name, shape in zip(names, dtypes, signature[2])
            ]
            resolved, loaded_names = self._resolve_weight_targets(
                received,
                previously_received=self._publication_names,
            )
            try:
                works = [dist.broadcast(tensor, src=0, group=group, async_op=True) for _, tensor in resolved]
            except Exception:
                self._fail_weight_group(group_name)
                raise
            self._prepared_bucket = (signature, resolved, loaded_names, works)

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del target_modules, flush_cache, track_prefix
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError(f"names/dtypes/shapes length mismatch: {len(names)}/{len(dtypes)}/{len(shapes)}")
        group = self._weight_groups.get(group_name)
        if group is None:
            raise RuntimeError(f"No native SD3 weight group {group_name!r}; initialize it before sync.")
        if group_name in self._failed_weight_groups:
            raise RuntimeError(f"Native SD3 weight group {group_name!r} is failed and cannot be reused.")

        with self._generate_lock:
            if self._publication_names is None:
                raise RuntimeError("Native SD3 weight bucket arrived outside begin_weights_update().")
            try:
                signature = self._bucket_signature(names, dtypes, shapes)
                prepared = self._prepared_bucket
                if prepared is None or prepared[0] != signature:
                    raise RuntimeError("Native SD3 weight bucket was not prepared with matching metadata.")
                _, resolved, loaded_names, works = prepared
                timeout = timedelta(seconds=self._weight_group_timeouts[group_name])
                for work in works:
                    if not work.wait(timeout=timeout):
                        raise TimeoutError("Native SD3 timed out waiting for an armed weight receive.")
                with torch.no_grad():
                    for target, tensor in resolved:
                        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                self._publication_names.update(loaded_names)
            except Exception:
                self._fail_weight_group(group_name)
                raise
            finally:
                self._prepared_bucket = None

    def finish_weights_update(self, *, group_name: str, track_prefix: str = "") -> None:
        del track_prefix
        if group_name not in self._weight_groups:
            raise RuntimeError(f"No native SD3 weight group {group_name!r}; initialize it before sync.")
        if group_name in self._failed_weight_groups:
            raise RuntimeError(f"Native SD3 weight group {group_name!r} is failed and cannot be reused.")
        with self._generate_lock:
            received = self._publication_names
            received_names = received if received is not None else set()
            expected = set(self._parameter_targets)
            if self._prepared_bucket is not None or received is None or received != expected:
                missing = sorted(expected - received_names)
                unexpected = sorted(received_names - expected)
                self._publication_names = None
                self._fail_weight_group(group_name)
                raise RuntimeError(
                    f"Native SD3 weight publication is incomplete: missing={missing[:8]}, unexpected={unexpected[:8]}."
                )
            self._controller.mark_weights_dirty()
            self._weights_valid = True
            self._publication_names = None

    def _resolve_weight_targets(
        self,
        tensors: List[tuple[str, torch.Tensor]],
        *,
        previously_received: Optional[set[str]] = None,
    ) -> tuple[List[tuple[torch.Tensor, torch.Tensor]], set[str]]:
        resolved: List[tuple[torch.Tensor, torch.Tensor]] = []
        names: set[str] = set()
        for wire_name, tensor in tensors:
            if not wire_name.startswith("transformer."):
                raise ValueError(f"NativeSD3 weight sync requires 'transformer.' prefix, got {wire_name!r}.")
            name = wire_name.removeprefix("transformer.")
            if name in names or (previously_received is not None and name in previously_received):
                raise ValueError(f"NativeSD3 weight sync received duplicate parameter {wire_name!r}.")
            target = self._parameter_targets.get(name)
            if target is None:
                raise KeyError(f"NativeSD3 weight sync has no target for {wire_name!r}.")
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"NativeSD3 weight shape mismatch for {wire_name}: "
                    f"target={tuple(target.shape)} wire={tuple(tensor.shape)}."
                )
            names.add(name)
            resolved.append((target, tensor))
        return resolved, names

    @staticmethod
    def _bucket_signature(
        names: List[str], dtypes: List[str], shapes: List[List[int]]
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, ...], ...]]:
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("Native SD3 weight names must be non-empty strings.")
        if any(not isinstance(dtype, str) or not dtype for dtype in dtypes):
            raise TypeError("Native SD3 weight dtypes must be non-empty strings.")
        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise TypeError(f"Native SD3 weight shapes must contain non-negative integers, got {shape!r}.")
        return tuple(names), tuple(dtypes), tuple(tuple(shape) for shape in shapes)

    def _fail_weight_group(self, group_name: str) -> None:
        self._failed_weight_groups.add(group_name)
        self._weights_valid = False
        self._prepared_bucket = None
        self._controller.mark_weights_dirty()
        group = self._weight_groups.get(group_name)
        abort = getattr(group, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                logger.exception("Failed to abort Native SD3 weight group %r", group_name)

    def destroy_weights_update_group(self, *, group_name: str, track_prefix: str = "") -> None:
        del track_prefix
        group = self._weight_groups.get(group_name)
        if group is not None:
            dist.destroy_process_group(group)
            self._weight_groups.pop(group_name, None)
            self._weight_group_timeouts.pop(group_name, None)
            self._failed_weight_groups.discard(group_name)

    def health_check(self) -> bool:
        return not self._shutdown and self._weights_valid and self.bundle.transformer is not None

    def shutdown(self) -> None:
        with self._generate_lock:
            self._shutdown = True
            first_error: Optional[Exception] = None
            for name, group in list(self._weight_groups.items()):
                try:
                    dist.destroy_process_group(group)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                else:
                    self._weight_groups.pop(name, None)
                    self._weight_group_timeouts.pop(name, None)
                    self._failed_weight_groups.discard(name)
            if first_error is not None:
                raise first_error


__all__ = ["NativeSD3RolloutEngine"]
