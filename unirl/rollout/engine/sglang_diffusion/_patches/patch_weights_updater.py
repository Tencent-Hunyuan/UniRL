"""Re-host the ``sglang-drl`` fork's ``WeightsUpdater`` in-memory-tensor path."""

from __future__ import annotations

import gc
import logging
import re
from collections import defaultdict
from collections.abc import Iterable

import torch

_METHODS_SENTINEL = "_unirl_named_tensor_methods"
_LWIM_SENTINEL = "_unirl_lora_name_remap"

_log = logging.getLogger(__name__)


def _resolve_param_names_mapping(module) -> dict:
    """Return the model's ``param_names_mapping`` dict, or ``{}``."""
    mapping = getattr(type(module), "param_names_mapping", None)
    if not isinstance(mapping, dict):
        mapping = getattr(module, "param_names_mapping", None)
    return mapping if isinstance(mapping, dict) else {}


def _write_fused_shard(param: torch.Tensor, tensor: torch.Tensor, shard_id: int, num_shards: int) -> None:
    """Write ``tensor`` into the ``shard_id``-th slice (dim 0) of a fused param."""
    wl = getattr(param, "weight_loader", None)
    if wl is not None:
        try:
            wl(param, tensor.to(param.dtype), shard_id)
            return
        except Exception:  # pragma: no cover - signature varies; manual fallback below
            pass
    data = param.data
    total = int(data.shape[0])
    size = int(tensor.shape[0])
    offset = total - size if shard_id == num_shards - 1 else shard_id * size
    if offset < 0 or offset + size > total:
        raise ValueError(
            f"fused shard {shard_id}/{num_shards}: size={size} at offset={offset} does not fit fused param dim0={total}"
        )
    data[offset : offset + size].copy_(tensor.to(param.dtype))


def _apply_fused_param_mapping(module, named_tensors):
    """Apply the model's ``param_names_mapping`` to the incoming named tensors."""
    mapping = _resolve_param_names_mapping(module)
    if not mapping:
        return list(named_tensors)

    model_params = dict(module.named_parameters())
    leftover: list = []
    fused = renamed = dropped = 0
    for name, tensor in named_tensors:
        if name in model_params:
            leftover.append((name, tensor))
            continue
        handled = False
        for pat, val in mapping.items():
            m = re.match(pat, name)
            if m is None:
                continue
            if isinstance(val, (tuple, list)) and len(val) == 3:
                replacement, shard_id, num_shards = val
                param = model_params.get(m.expand(replacement))
                if param is None:
                    continue
                _write_fused_shard(param, tensor, int(shard_id), int(num_shards))
                fused += 1
                handled = True
                break
            if isinstance(val, str):
                if val == "":
                    dropped += 1
                    handled = True
                    break
                target = re.sub(pat, val, name)
                if target in model_params:
                    leftover.append((target, tensor))
                    renamed += 1
                    handled = True
                    break
        if not handled:
            leftover.append((name, tensor))
    if fused or renamed or dropped:
        _log.info(
            "weight-sync: param_names_mapping applied — %d fused, %d renamed, %d dropped",
            fused,
            renamed,
            dropped,
        )
    unmatched = [n for n, _ in leftover if n not in model_params]
    if unmatched:
        _log.warning(
            "weight-sync: %d tensor(s) matched no model param after param_names_mapping "
            "(e.g. %s) — likely a mapping gap; these will not update any weight",
            len(unmatched),
            unmatched[:5],
        )
    return leftover


def patch_weights_updater() -> None:
    """Install the fork's in-memory named-tensor weight-update path."""
    import sglang.multimodal_gen.runtime.loader.weights_updater as wu
    from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin

    logger = wu.logger
    _load_weights_into_module = wu._load_weights_into_module

    if not getattr(wu.load_weights_into_model, _LWIM_SENTINEL, False):
        from torch.distributed.tensor import DTensor, distribute_tensor

        def _build_lora_name_remap(model_params: dict) -> dict:
            """Build bidirectional remap for LoRA-wrapped param names."""
            remap = {}
            for param_name in model_params:
                if ".base_layer." in param_name:
                    stripped = param_name.replace(".base_layer.", ".")
                    remap[stripped] = param_name
                else:
                    for suffix in (".weight", ".bias"):
                        if param_name.endswith(suffix):
                            prefix = param_name[: -len(suffix)]
                            candidate = prefix + ".base_layer" + suffix
                            if candidate not in model_params:
                                remap[candidate] = param_name
            return remap

        def load_weights_into_model(weights_iter, model_params: dict) -> None:
            """Copy weights from weights_iter into model_params in-place."""
            lora_remap = _build_lora_name_remap(model_params)

            _matched = 0
            _skipped: list[str] = []
            for name, loaded_weight in weights_iter:
                if name not in model_params:
                    name = lora_remap.get(name, name)
                if name not in model_params:
                    _skipped.append(name)
                    continue
                _matched += 1
                param = model_params[name]
                if param.shape != loaded_weight.shape:
                    raise ValueError(f"Shape mismatch for {name}: model={param.shape}, loaded={loaded_weight.shape}")
                if isinstance(param, DTensor):
                    distributed_weight = distribute_tensor(
                        loaded_weight.to(param.dtype),
                        param.device_mesh,
                        param.placements,
                    )
                    param._local_tensor.copy_(distributed_weight._local_tensor)
                else:
                    param.data.copy_(loaded_weight.to(param.dtype))

            # Report unmatched weight names; skipped updates leave rollout weights stale.
            if _skipped:
                logger.warning(
                    "load_weights_into_model: matched=%d SKIPPED=%d unmatched name(s) "
                    "(not in target module params — naming/fusion mismatch; those "
                    "weights were NOT updated). Sample: %s",
                    _matched,
                    len(_skipped),
                    _skipped[:10],
                )

        load_weights_into_model._unirl_lora_name_remap = True  # type: ignore[attr-defined]
        wu._build_lora_name_remap = _build_lora_name_remap
        wu.load_weights_into_model = load_weights_into_model

    if getattr(wu.WeightsUpdater, _METHODS_SENTINEL, False):
        return

    def update_weights_from_named_tensors(
        self,
        named_tensors,
        *,
        target_modules: list[str] | None = None,
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> tuple[bool, str]:
        """Update module weights from in-memory named tensors."""
        if named_tensors is None:
            return False, "named_tensors is required"

        try:
            modules_to_update = self._collect_modules(target_modules)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        if not modules_to_update:
            return False, "No matching modules found for in-memory update."

        try:
            normalized = self._normalize_named_tensors(
                named_tensors=named_tensors,
                load_format=load_format,
            )
            module_payloads = self._split_named_tensors_by_module(
                normalized,
                modules_to_update,
            )
        except Exception as e:
            logger.error("Failed to parse in-memory tensor payload: %s", e, exc_info=True)
            return False, f"Failed to parse in-memory tensor payload: {e}"

        if not module_payloads:
            return False, "No tensors in payload matched requested modules."

        logger.info(
            "Updating %d modules from in-memory tensors.",
            len(module_payloads),
        )
        success, message = self._apply_named_tensor_weights(
            modules_to_update=modules_to_update,
            module_payloads=module_payloads,
        )
        self._post_update_cleanup(success, flush_cache, modules_to_update)

        logger.info(message)
        return success, message

    def _normalize_named_tensors(
        self,
        *,
        named_tensors,
        load_format: str | None,
    ) -> list[tuple[str, torch.Tensor]]:
        if load_format == "flattened_bucket":
            if not isinstance(named_tensors, dict):
                raise ValueError("flattened_bucket format expects a dict payload with flattened_tensor and metadata")
            flattened_tensor = named_tensors.get("flattened_tensor")
            metadata = named_tensors.get("metadata")
            if flattened_tensor is None or metadata is None:
                raise ValueError("flattened_bucket payload must contain flattened_tensor and metadata")
            try:
                from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
            except Exception:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_tensor,
                metadata=metadata,
            )
            return list(bucket.reconstruct_tensors())

        if isinstance(named_tensors, dict):
            iterable: Iterable[tuple[str, torch.Tensor]] = named_tensors.items()
        else:
            iterable = named_tensors

        normalized: list[tuple[str, torch.Tensor]] = []
        for item in iterable:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("named_tensors must be iterable of (name, tensor) pairs")
            name, tensor = item
            if not isinstance(name, str):
                raise ValueError(f"Tensor name must be str, got: {type(name).__name__}")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Tensor payload for {name} must be torch.Tensor, got: {type(tensor).__name__}")
            normalized.append((name, tensor))
        return normalized

    def _split_named_tensors_by_module(
        self,
        normalized_named_tensors: list[tuple[str, torch.Tensor]],
        modules_to_update: list[tuple[str, torch.nn.Module]],
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        module_names = {name for name, _ in modules_to_update}
        module_param_name_sets = {
            module_name: set(dict(module.named_parameters()).keys()) for module_name, module in modules_to_update
        }
        by_module: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)

        for name, tensor in normalized_named_tensors:
            assigned_module = None
            inner_name = name

            if "." in name:
                prefix, suffix = name.split(".", 1)
                if prefix in module_names:
                    assigned_module = prefix
                    inner_name = suffix

            if assigned_module is None:
                if len(modules_to_update) == 1:
                    assigned_module = modules_to_update[0][0]
                    inner_name = name
                else:
                    matched = [
                        module_name
                        for module_name, param_names in module_param_name_sets.items()
                        if name in param_names
                    ]
                    if len(matched) == 1:
                        assigned_module = matched[0]
                        inner_name = name

            if assigned_module is None:
                continue

            by_module[assigned_module].append((inner_name, tensor))

        return dict(by_module)

    def _flush_module_runtime_cache(self, modules_to_update: list[tuple[str, torch.nn.Module]]) -> None:
        for _, module in modules_to_update:
            if isinstance(module, TeaCacheMixin):
                module.reset_teacache_state()

    def _post_update_cleanup(
        self,
        success: bool,
        flush_cache: bool,
        modules_to_update: list[tuple[str, torch.nn.Module]],
    ) -> None:
        """Post weight-update cleanup aligned with LLM methodology."""
        if not success:
            gc.collect()
            return

        updated_names = {name for name, _ in modules_to_update}
        if flush_cache and hasattr(self.pipeline, "handle_weight_sync"):
            self.pipeline.handle_weight_sync(updated_names)

        if flush_cache:
            self._flush_module_runtime_cache(modules_to_update)
            torch.cuda.empty_cache()

    def _apply_named_tensor_weights(
        self,
        modules_to_update: list[tuple[str, torch.nn.Module]],
        module_payloads: dict[str, list[tuple[str, torch.Tensor]]],
    ) -> tuple[bool, str]:
        updated_modules: list[str] = []

        for module_name, module in modules_to_update:
            module_tensors = module_payloads.get(module_name)
            if not module_tensors:
                continue
            try:
                module_tensors = _apply_fused_param_mapping(module, module_tensors)
                _load_weights_into_module(module, module_tensors)
                updated_modules.append(module_name)
            except Exception as e:
                rollback_list = updated_modules + [module_name]
                logger.error(
                    "In-memory weight update failed for module '%s': %s. Rolling back modules: %s",
                    module_name,
                    e,
                    rollback_list,
                    exc_info=True,
                )
                self._rollback(rollback_list)
                return False, (
                    f"Failed to update module '{module_name}': {e}. All modules rolled back to original weights."
                )

        if not updated_modules:
            return False, "No module parameters were updated from in-memory payload."

        names = ", ".join(updated_modules)
        return True, f"Updated {len(updated_modules)} modules ({names}) from in-memory payload."

    wu.WeightsUpdater.update_weights_from_named_tensors = update_weights_from_named_tensors
    wu.WeightsUpdater._normalize_named_tensors = _normalize_named_tensors
    wu.WeightsUpdater._split_named_tensors_by_module = _split_named_tensors_by_module
    wu.WeightsUpdater._flush_module_runtime_cache = _flush_module_runtime_cache
    wu.WeightsUpdater._post_update_cleanup = _post_update_cleanup
    wu.WeightsUpdater._apply_named_tensor_weights = _apply_named_tensor_weights
    setattr(wu.WeightsUpdater, _METHODS_SENTINEL, True)
