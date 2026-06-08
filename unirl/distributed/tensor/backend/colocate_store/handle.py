"""TensorHandle for the colocate_store backend.

Re-exports the canonical TensorHandle from the gpu_store backend so both
backends share a single handle type. See gpu_store/handle.py.
"""

from unirl.distributed.tensor.backend.gpu_store.handle import TensorHandle

__all__ = ["TensorHandle"]
