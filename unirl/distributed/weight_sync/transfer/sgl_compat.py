# Copyright 2023-2024 SGLang Team; SPDX-License-Identifier: Apache-2.0
"""Vendored sglang weight-sync utilities (CUDA-only) for the vLLM-Omni flow."""

from __future__ import annotations

import base64
import io
import pickle
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Callable, List, Tuple, Union

import torch
from torch.multiprocessing import reductions

_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6


def monkey_patch_torch_reductions():
    """Monkey patching before Torch https://github.com/pytorch/pytorch/pull/149248 is fixed"""
    if hasattr(reductions, "_reduce_tensor_original"):
        return

    reductions._reduce_tensor_original = reductions.reduce_tensor
    reductions._rebuild_cuda_tensor_original = reductions.rebuild_cuda_tensor

    reductions.reduce_tensor = _reduce_tensor_modified
    reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
    reductions.init_reductions()


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid)
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    args = _modify_tuple(args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid)
    return reductions._rebuild_cuda_tensor_original(*args)


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device_maybe_uuid: Union[int, str]) -> int:
    if isinstance(device_maybe_uuid, int):
        return device_maybe_uuid

    if isinstance(device_maybe_uuid, str):
        for device in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(device).uuid) == device_maybe_uuid:
                return device
        raise Exception("Invalid device_uuid=" + device_maybe_uuid)

    raise Exception(f"Unknown type: {device_maybe_uuid=}")


def _modify_tuple(t, index: int, modifier: Callable):
    return *t[:index], modifier(t[index]), *t[index + 1 :]


@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket"""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """A bucket flattening many tensors into one, preserving the metadata needed for reconstruction."""

    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        """Initialize a tensor bucket from a list of named tensors OR from pre-flattened data."""
        if named_tensors is not None:
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            self.flattened_tensor: torch.Tensor = None

            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            current_idx = 0
            flattened_tensors: List[torch.Tensor] = [None] * len(named_tensors)

            for i, (name, tensor) in enumerate(named_tensors):
                flattened = tensor.flatten().view(torch.uint8)
                flattened_tensors[i] = flattened

                numel = flattened.numel()
                metadata_obj = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                self.metadata[i] = metadata_obj
                current_idx += numel

            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError("Must provide either named_tensors or both flattened_tensor and metadata")
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    def get_metadata(self) -> List[FlattenedTensorMetadata]:
        """Get metadata for all tensors in the bucket"""
        return self.metadata

    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """Reconstruct original tensors from flattened tensor with optimized performance."""
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = self.flattened_tensor[meta.start_idx : meta.end_idx].view(meta.dtype).reshape(meta.shape)

            reconstructed[i] = (meta.name, tensor)

        return reconstructed


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        """Serialize a Python object using ForkingPickler."""
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            output = base64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        """Deserialize a previously serialized object."""
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)

        return SafeUnpickler(io.BytesIO(data)).load()


class SafeUnpickler(pickle.Unpickler):
    ALLOWED_MODULE_PREFIXES = {
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch.nn.parameter.",
        "torch.autograd.function.",
        "torch.distributed.",
        "torch.distributed._shard.",
        "torch.distributed._composable.",
        "torch._C._distributed_c10d.",
        "torch._C._distributed_fsdp.",
        "torch.distributed.optim.",
        "multiprocessing.resource_sharer.",
        "multiprocessing.reduction.",
        "pickletools.",
        "peft.",
        "transformers.",
        "huggingface_hub.",
        "unirl.",
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
    }

    def find_class(self, module, name):
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}), to prevent exploitation of CVE-2025-10164"
            )
        if any((module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)

        raise RuntimeError(f"Blocked unsafe class loading ({module}.{name}), to prevent exploitation of CVE-2025-10164")
