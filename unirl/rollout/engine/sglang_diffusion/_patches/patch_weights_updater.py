"""Re-host the ``sglang-drl`` fork's ``WeightsUpdater`` in-memory-tensor path.

Stock upstream ``WeightsUpdater`` (``runtime/loader/weights_updater.py``) only
updates weights from disk (``update_weights_from_disk``). The fork added an
in-memory named-tensor path so the RL trainer can push freshly-optimized weights
straight into the live pipeline (no disk round-trip), used by
``GPUWorker.update_weights_from_tensor`` / ``update_weights_from_distributed``
(installed by ``patch_gpu_worker``).

This patch ports that WITHOUT editing sglang source:

  * ``setattr`` the 6 net-new ``WeightsUpdater`` methods (bodies verbatim from
    the fork diff): ``update_weights_from_named_tensors``,
    ``_normalize_named_tensors``, ``_split_named_tensors_by_module``,
    ``_flush_module_runtime_cache``, ``_post_update_cleanup``,
    ``_apply_named_tensor_weights``. These reuse upstream's ``_collect_modules``
    / ``_rollback`` (unchanged) and upstream's module-global
    ``_load_weights_into_module`` (the fork's offload mixin differs, so we KEEP
    upstream's per the task).
  * REPLACE the module-level ``load_weights_into_model`` with the fork's version
    (adds a bidirectional ``.base_layer.`` LoRA name-remap) and ``setattr`` the
    fork-new ``_build_lora_name_remap`` at module level. Upstream's
    ``_load_weights_into_module`` calls ``load_weights_into_model`` by module
    global, so it picks up the replacement automatically.

Re-homing notes:
  * The verbatim method bodies are nested fns here, so their free globals
    (``gc``, ``torch``, ``defaultdict``, ``Iterable``, ``TeaCacheMixin``,
    ``_load_weights_into_module``) resolve via THIS module's scope (LEGB), not
    sglang's. We bind/import each below so behaviour is identical.
  * ``_post_update_cleanup`` calls ``self.pipeline.handle_weight_sync(...)``
    guarded by ``hasattr`` -- upstream pipeline lacks ``handle_weight_sync``;
    ``patch_lora_tensors`` adds it to ``LoRAPipeline``. The hasattr guard is kept
    regardless (other pipelines have no such method). See RISK in the PR notes.
  * ``FlattenedTensorBucket`` is imported lazily inside
    ``_normalize_named_tensors`` (only on ``load_format='flattened_bucket'``),
    with the fork's try/except fallback; both import sites exist upstream
    (``sglang.srt.weight_sync.tensor_bucket`` and
    ``sglang.srt.model_executor.model_runner``).

Idempotent via sentinel guards. Import-safe (sglang imported inside the fn).
"""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Iterable

import torch

_METHODS_SENTINEL = "_unirl_named_tensor_methods"
_LWIM_SENTINEL = "_unirl_lora_name_remap"


def patch_weights_updater() -> None:
    """Install the fork's in-memory named-tensor weight-update path."""
    import sglang.multimodal_gen.runtime.loader.weights_updater as wu
    from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin

    # Upstream module-globals the verbatim bodies depend on. Bound here so the
    # nested fns resolve them through THIS scope identically to the fork.
    logger = wu.logger
    _load_weights_into_module = wu._load_weights_into_module

    # --- (1) REPLACE module-level load_weights_into_model (+ remap helper) ----
    # Upstream's load_weights_into_model has no LoRA name-remap. The fork added a
    # bidirectional ``.base_layer.`` remap so weight-sync names match whether or
    # not the model wrapped a layer with a ``.base_layer.`` indirection. We
    # REPLACE the module function (so upstream ``_load_weights_into_module``,
    # which calls it by module global, picks it up) and install the helper.
    if not getattr(wu.load_weights_into_model, _LWIM_SENTINEL, False):
        from torch.distributed.tensor import DTensor, distribute_tensor

        def _build_lora_name_remap(model_params: dict) -> dict:
            """Build bidirectional remap for LoRA-wrapped param names.

            After LoRA wrapping, some layers have .base_layer. in their names.
            Callers may send names with or without it. This remap handles both:
              xxx.weight → xxx.base_layer.weight  (caller stripped .base_layer.)
              xxx.base_layer.weight → xxx.weight  (caller kept .base_layer. but model didn't wrap)
            """
            remap = {}
            for param_name in model_params:
                if ".base_layer." in param_name:
                    stripped = param_name.replace(".base_layer.", ".")
                    remap[stripped] = param_name
                else:
                    # Only add reverse remap for plausible base_layer patterns
                    # e.g. attn.to_q.weight → attn.to_q.base_layer.weight
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

            for name, loaded_weight in weights_iter:
                if name not in model_params:
                    name = lora_remap.get(name, name)
                if name not in model_params:
                    continue
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

        load_weights_into_model._unirl_lora_name_remap = True  # type: ignore[attr-defined]
        wu._build_lora_name_remap = _build_lora_name_remap
        wu.load_weights_into_model = load_weights_into_model
        # Note: upstream's ``_load_weights_into_module`` calls
        # ``load_weights_into_model`` via its module global, so it picks up the
        # replacement above; the ``_load_weights_into_module`` object itself is
        # unchanged, so the local alias bound at the top stays valid.

    # --- (2) setattr the 6 net-new WeightsUpdater methods (verbatim) ----------
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
        """Update module weights from in-memory named tensors.

        Args:
            named_tensors: Tensor payload. Supported:
                - list[(name, tensor)] / tuple[(name, tensor)]
                - dict[name, tensor]
                - flattened bucket dict when ``load_format='flattened_bucket'``
            target_modules: Restrict update to these modules.
            load_format: Optional payload format (e.g., ``flattened_bucket``).
            flush_cache: Whether to reset TeaCache state for updated modules.
        """
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
        """Post weight-update cleanup aligned with LLM methodology.

        On failure: gc.collect() to free dangling refs from partial loads.
        On success + flush_cache: reset TeaCache state, then empty CUDA cache.
        On success + no flush_cache: no cleanup.
        """
        if not success:
            gc.collect()
            return

        # Handle LoRA state only after ALL buckets/dtypes are done (flush_cache=True).
        # Calling per-bucket would corrupt state: partial updates + unmerge/merge
        # cause base weights to be reverted or LoRA to be double-applied.
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
