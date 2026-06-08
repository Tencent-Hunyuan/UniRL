"""Weight-sync handlers.

The active (v2) handlers are resolved by Hydra via their full ``_target_``
dotted paths and so are not re-exported here:

- full-weight sync: ``unirl.distributed.weight_sync.full.{nccl,ipc,tensor}``
  (``NCCLWeightSync`` / ``IPCWeightSync`` / ``TensorWeightSync``)
- LoRA sync: ``unirl.distributed.weight_sync.lora.LocalLoraWeightSync``
  (colocate sibling push) / ``...lora.RemoteLoraWeightSync`` (cross-process Ray
  push: separate-slab + HI3)

The active handler is built from ``cfg.sync`` by the trainer (via ``remote_hydra``) during setup.
"""

__all__: list[str] = []
