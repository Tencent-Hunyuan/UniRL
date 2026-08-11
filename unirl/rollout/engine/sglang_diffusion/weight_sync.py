"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component.

``WeightSync`` is a plain object the engine constructs over the seam: it takes the
backend and the LoRA spec explicitly and owns all sync/LoRA state
(``_lora_loaded`` / ``_active_adapter``). Method names mirror
the frozen ``base.py`` surface minus ``track_prefix`` (the engine's forwards absorb
that, along with the per-worker ``Worker.call`` dispatch concern), so a grep for a
trainer-side entry point lands here.

The transports declared are exactly what SGLang supports: tensor-bag, NCCL
(init/transfer/destroy), LoRA-from-tensors, and the checksum query. There is no
IPC method — the engine simply doesn't define ``update_weights_from_ipc``, so it
inherits ``BaseRolloutEngine``'s ``NotImplementedError`` (SGLang has no IPC
receiver).

The "weights released" event: the engine's ``sleep()`` calls
:meth:`mark_weights_released` after releasing memory (the released tags include the
transformer weights), so ``lora_dirty`` flips and the next sync re-pushes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang_diffusion.backends import Backend
from unirl.utils.peft_merge import adapt_lora_for_sglang

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(
        self,
        backend: Backend,
        *,
        pipeline_prefix: str,
        target_modules: List[str],
        uses_lora: bool,
        lora_tensor_routes: Optional[Dict[str, str]] = None,
    ) -> None:
        self._backend = backend
        self._pipeline_prefix = pipeline_prefix
        self._target_modules = list(target_modules)
        self._uses_lora = uses_lora
        self._lora_tensor_routes = dict(lora_tensor_routes or {})
        self._active_adapter: Optional[str] = None
        self._lora_loaded = False

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        if not serialized_named_tensors:
            raise ValueError("serialized_named_tensors must be non-empty")
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=list(target_modules or self._target_modules),
            load_format=load_format,
            flush_cache=flush_cache,
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
    ) -> None:
        self._backend.init_weights_group(
            master_address=master_address,
            master_port=int(master_port),
            rank_offset=int(rank_offset),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
        )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        if not names:
            raise ValueError("names must be non-empty for distributed update")
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            target_modules=list(target_modules or self._target_modules),
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=str(group_name))

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Push a LoRA adapter from in-memory tensors.

        Stable nicknames are reused on every push so refreshed tensors replace
        their registry entries without leaking GPU-resident adapter copies.
        """
        stripped = adapt_lora_for_sglang(
            lora_tensors,
            pipeline_prefix=self._pipeline_prefix,
        )
        adapter_alpha = None
        if peft_config is not None:
            adapter_alpha = peft_config.get("lora_alpha")
        routed = self._route_lora_tensors(stripped)
        alpha = float(adapter_alpha) if adapter_alpha is not None else None
        for target, tensors in routed.items():
            nickname = adapter_name if target == "all" else f"{adapter_name}__{target}"
            status = self._backend.set_lora(
                lora_nickname=nickname,
                lora_tensors=tensors,
                target=target,
                lora_alpha=alpha,
            )
            logger.info(
                "SGLang LoRA target %s active on %s/%s layers (nickname=%s)",
                target,
                status.get("active_layers", "?"),
                status.get("total_layers", "?"),
                nickname,
            )
        self._active_adapter = adapter_name
        self._lora_loaded = True

        layer_names = set()
        for key in stripped:
            if key.endswith(".alpha"):
                continue
            base = key
            for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            layer_names.add(base)
        logger.info(
            "SGLang LoRA loaded from tensors (adapter=%s, targets=%s) — %d source layers",
            adapter_name,
            ",".join(routed),
            len(layer_names),
        )

    def _route_lora_tensors(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        if not self._lora_tensor_routes:
            return {"all": tensors}

        routed: Dict[str, Dict[str, torch.Tensor]] = {target: {} for target in self._lora_tensor_routes.values()}
        unmatched = []
        for key, tensor in tensors.items():
            matches = [
                (prefix, target) for prefix, target in self._lora_tensor_routes.items() if key.startswith(prefix)
            ]
            if len(matches) != 1:
                unmatched.append(key)
                continue
            prefix, target = matches[0]
            local_key = key[len(prefix) :]
            if local_key in routed[target]:
                raise ValueError(f"Duplicate routed LoRA tensor {local_key!r} for SGLang target {target!r}")
            routed[target][local_key] = tensor

        if unmatched:
            examples = ", ".join(unmatched[:4])
            raise ValueError(
                "LoRA tensor routing requires exactly one model-prefix match per tensor; "
                f"{len(unmatched)} tensor(s) did not match uniquely. Examples: {examples}"
            )
        missing = [target for target, values in routed.items() if not values]
        if missing:
            raise ValueError(
                "LoRA tensor routing produced no tensors for required SGLang target(s): " + ", ".join(missing)
            )
        return routed

    def loaded_param_checksums(self, *, names: List[str]) -> Dict[int, List[Dict[str, str]]]:
        output = self._backend.weights_checksum(module_names=list(names))
        return {0: [{str(k): str(v) for k, v in output.items()}]}

    def mark_weights_released(self) -> None:
        """The engine released the runtime weights — the loaded LoRA pool is gone."""
        self._lora_loaded = False

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._uses_lora and not self._lora_loaded


__all__ = ["WeightSync"]
