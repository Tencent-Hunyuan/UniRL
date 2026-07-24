"""v2 full-weight checkpoint-path sync (COLOCATE, single-node).

Simplest full-weight transport: the train slab serializes the freshly-trained
full base weights to a file on a shared/local path, and each co-located rollout
engine loads it via ``update_weights_from_path``. Full-weight analogue of the
LoRA / tensor / nccl / ipc handlers, used to bring up the FastVideo engine
before the faster zero-copy transports are wired.

Mirrors ``TensorWeightSync`` (colocate, ``rollout`` is a LOCAL sibling) but the
handoff is a torch.save file instead of a serialized tensor bag:

  rank0: ``_iter_full_tensors`` (FSDP all-gather; all ranks in lockstep) →
         ``torch.save`` to ``{sync_dir}/weights_v{N}`` + ``.ready`` marker
  every rank: wait for marker → ``self._rollout.update_weights_from_path(path)``

The weight walk yields the bare trainable-module (transformer) keys, which is
exactly what the FastVideo engine's ``transformer.load_state_dict`` expects — so
no ``name_remap`` is needed for the default FastVideo case. All torch imports
are deferred so the driver can import this module for ``remote(...)``.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


def _shared_run_id(explicit: Optional[str]) -> str:
    """Return a filesystem-safe id shared by every rank in this Ray job."""
    raw = str(explicit or os.environ.get("UNIRL_RUN_ID", "")).strip()
    if not raw:
        try:
            import ray

            job_id = ray.get_runtime_context().get_job_id()
            as_hex = getattr(job_id, "hex", None)
            raw = str(as_hex() if callable(as_hex) else job_id)
        except Exception as exc:
            raise RuntimeError(
                "CheckpointWeightSync needs a run-unique id to avoid stale/concurrent "
                "checkpoint markers. Set run_id=... or UNIRL_RUN_ID when no Ray job "
                "context is available."
            ) from exc
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError(f"CheckpointWeightSync run_id {raw!r} has no filesystem-safe characters.")
    return safe


class CheckpointWeightSync(FullWeightSync):
    """Colocate full-weight sync via a torch.save checkpoint file."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        sync_dir: str = "/tmp/unirl_fastvideo_weight_sync",
        run_id: Optional[str] = None,
        wait_timeout_s: float = 1200.0,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        # NOTE: no ``bucket_size_mb`` — the base's size-bounded bucketing
        # (``_iter_buckets``) is for streaming transports; this handler writes the
        # whole state_dict in one ``torch.save``, so bucketing never runs.
        super().__init__(
            backend=backend,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout  # local engine sibling (colocate)
        # Isolate concurrent/restarted jobs. Ray's job id is identical on every
        # train rank; the rollout class / track prefix isolates multiple checkpoint
        # bridges within one job.
        scope = self._track_prefix or type(rollout).__name__
        scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope).strip("._") or "default"
        self._dir = os.path.join(str(sync_dir), _shared_run_id(run_id), scope)
        self._wait_timeout_s = float(wait_timeout_s)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Publish the full weights to a file and load it into the local engine.

        Runs on every train rank (``BROADCAST``). ``_iter_full_tensors`` all-gathers
        each FSDP shard on every rank in lockstep, so all ranks must iterate; only
        rank-0 keeps the materialized tensors and writes the file. The path is
        deterministic from ``weight_version`` (incremented in lockstep on every
        rank), so all ranks agree on it without a broadcast.
        """
        import torch

        version = int(self.weight_version)
        path = os.path.join(self._dir, f"weights_v{version}.pt")
        marker = path + ".ready"

        if self._my_rank == 0:
            os.makedirs(self._dir, exist_ok=True)
            # A restarted actor can reuse the same version inside the same Ray
            # job. Remove that version's stale publication before the first FSDP
            # all-gather; the collective then keeps every rank from reaching the
            # marker wait until this cleanup has happened.
            for stale in (path, path + ".tmp", marker, marker + ".tmp"):
                try:
                    os.remove(stale)
                except FileNotFoundError:
                    pass
            self._cleanup_old_versions(version)

        state_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in self._iter_full_tensors():
            if self._my_rank == 0:
                state_dict[name] = tensor.detach().to("cpu", copy=True)

        if self._my_rank == 0:
            tmp = path + ".tmp"
            torch.save(state_dict, tmp)
            os.replace(tmp, path)  # atomic publish
            marker_tmp = marker + ".tmp"
            with open(marker_tmp, "w") as fh:
                fh.write(f"version={version}\npath={path}\n")
            os.replace(marker_tmp, marker)  # marker becomes visible atomically
            del state_dict

        self._wait_for_marker(marker, path)
        self._rollout.update_weights_from_path(path, track_prefix=self._track_prefix)

        self.weight_version += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_old_versions(self, publishing_version: int) -> None:
        """Keep only the version being published and the last good fallback.

        Cleanup runs at the *next* sync, after the previous driver-level
        BROADCAST completed and after rollout ``wake_up`` consumed its cached
        checkpoint. Retaining ``publishing_version - 1`` keeps wake fail-closed
        if the new publication/load fails partway through.
        """
        keep = {publishing_version, max(0, publishing_version - 1)}
        pattern = re.compile(r"^weights_v(\d+)\.pt(?:\..*)?$")
        try:
            entries = list(os.scandir(self._dir))
        except FileNotFoundError:
            return
        for entry in entries:
            match = pattern.match(entry.name)
            if match is None or int(match.group(1)) in keep:
                continue
            try:
                os.remove(entry.path)
            except FileNotFoundError:
                pass

    def _wait_for_marker(self, marker: str, checkpoint_path: str) -> None:
        t0 = time.time()
        while not (os.path.exists(marker) and os.path.exists(checkpoint_path)):
            if time.time() - t0 > self._wait_timeout_s:
                raise TimeoutError(
                    f"CheckpointWeightSync: checkpoint publication not ready after "
                    f"{self._wait_timeout_s}s: path={checkpoint_path}, marker={marker}"
                )
            time.sleep(0.2)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def cleanup(self) -> None:
        """Remove this run's checkpoint namespace during trainer teardown."""
        if self._my_rank != 0:
            return
        shutil.rmtree(self._dir, ignore_errors=True)
        # Remove now-empty scope/run/root directories without touching siblings
        # belonging to another bridge or concurrent Ray job.
        parent = os.path.dirname(self._dir)
        root = os.path.dirname(parent)
        for directory in (parent, root):
            try:
                os.rmdir(directory)
            except OSError:
                pass


__all__ = ["CheckpointWeightSync"]
