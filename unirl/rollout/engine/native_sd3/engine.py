"""Native SD3 rollout worker with rapidly refreshable Hopper FP8 scouting."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.sd3.bundle import SD3Bundle
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.primitives import Images
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams

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

    def __init__(
        self,
        config: NativeSD3EngineConfig,
        *,
        model_config: SD3PipelineConfig,
        strategy: StepStrategy,
        device: Optional[torch.device] = None,
        rank: int = 0,
    ) -> None:
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = int(rank)
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
        if bool(config.compile_model):
            transformer = torch.compile(transformer, mode=str(config.compile_mode))
        self.bundle.transformer = RoutedTransformer(transformer, self._controller).eval()
        self.pipeline.diffusion.model = self.bundle

        self._weight_groups: Dict[str, dist.ProcessGroup] = {}
        self._generate_lock = threading.Lock()
        self._shutdown = False
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
            result = self._generate_core(sample)
            return self._stamp_output_version(result)

    def _generate_core(self, sample: Sample) -> Sample:
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        ensure_sample_sigmas(sample, self.schedule_policy)
        batch_size = int(gen.batch_size)
        fbs = self.config.forward_batch_size
        if fbs is None or batch_size <= int(fbs):
            return self._generate_batch(sample)

        input_parts = sample.parts[:-1]
        chunks: List[Part] = []
        for start in range(0, batch_size, int(fbs)):
            end = min(start + int(fbs), batch_size)
            request = Sample(parts=[*input_parts, gen.slice(start, end)])
            chunks.append(self._generate_batch(request).parts[-1])
        return sample.replace_frontier(Part.concat(chunks))

    def _generate_batch(self, sample: Sample) -> Sample:
        params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        with self._controller.rollout(
            mode=str(params.rollout_precision),
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
            size=(int(image_size), int(image_size)),
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
    ) -> None:
        del track_prefix
        from unirl.utils.distributed_utils import init_process_group

        group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{int(master_port)}",
            world_size=int(world_size),
            rank=int(rank_offset),
            group_name=str(group_name),
        )
        self._weight_groups[str(group_name)] = group

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
        del target_modules, track_prefix
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError(f"names/dtypes/shapes length mismatch: {len(names)}/{len(dtypes)}/{len(shapes)}")
        group = self._weight_groups.get(str(group_name))
        if group is None:
            raise RuntimeError(f"No native SD3 weight group {group_name!r}; initialize it before sync.")

        received: List[tuple[str, torch.Tensor]] = []
        for name, dtype_name, shape in zip(names, dtypes, shapes):
            tensor = torch.empty(
                tuple(int(dim) for dim in shape),
                dtype=_dtype_from_name(dtype_name),
                device=self.device,
            )
            dist.broadcast(tensor, src=0, group=group)
            received.append((str(name), tensor))
        self._load_weights(received)
        if flush_cache:
            self._controller.mark_weights_dirty()

    @torch.no_grad()
    def _load_weights(self, tensors: List[tuple[str, torch.Tensor]]) -> None:
        for wire_name, tensor in tensors:
            name = wire_name.removeprefix("transformer.")
            target = self._parameter_targets.get(name)
            if target is None:
                raise KeyError(f"NativeSD3 weight sync has no target for {wire_name!r}.")
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"NativeSD3 weight shape mismatch for {wire_name}: "
                    f"target={tuple(target.shape)} wire={tuple(tensor.shape)}."
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))

    def destroy_weights_update_group(self, *, group_name: str, track_prefix: str = "") -> None:
        del track_prefix
        self._weight_groups.pop(str(group_name), None)

    def health_check(self) -> bool:
        return not self._shutdown and self.bundle.transformer is not None

    def shutdown(self) -> None:
        with self._generate_lock:
            self._shutdown = True
            self._weight_groups.clear()


__all__ = ["NativeSD3RolloutEngine"]
