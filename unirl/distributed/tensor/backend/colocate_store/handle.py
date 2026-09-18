"""ColocateTensorHandle for the colocate_store backend."""

from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle


class ColocateTensorHandle(GPUTensorHandle):
    pass


__all__ = ["ColocateTensorHandle"]
