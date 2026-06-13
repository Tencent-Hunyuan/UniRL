"""ColocateTensorHandle for the colocate_store backend.

A thin subclass of the canonical :class:`GPUTensorHandle` (gpu_store) — identical
lifecycle and store-key semantics, but a distinct type so colocate refs are
``TensorSpan[ColocateTensorHandle]``. See gpu_store/handle.py.
"""

from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle


class ColocateTensorHandle(GPUTensorHandle):
    pass


__all__ = ["ColocateTensorHandle"]
