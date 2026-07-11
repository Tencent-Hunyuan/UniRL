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
import logging
import re
from collections import defaultdict
from collections.abc import Iterable

import torch

_METHODS_SENTINEL = "_unirl_named_tensor_methods"
_LWIM_SENTINEL = "_unirl_lora_name_remap"

_log = logging.getLogger(__name__)


def _resolve_param_names_mapping(module) -> dict:
    """Return the model's ``param_names_mapping`` dict, or ``{}``.

    SGLang models that FUSE projections (Z-Image's ``ZImageTransformer2DModel``
    maps ``to_q/to_k/to_v`` -> ``to_qkv`` and ``feed_forward.w1/w3`` -> ``w13``)
    carry a class-level ``param_names_mapping`` of
    ``{regex: (replacement, shard_id, num_shards)}`` (the same dict the checkpoint
    loader applies). The in-memory named-tensor update path does NOT apply it, so
    without this the trainer's separate-projection tensors match no fused param and
    are silently dropped — the bug behind a flat reward curve on fused-attention
    models like Z-Image.
    """
    mapping = getattr(type(module), "param_names_mapping", None)
    if not isinstance(mapping, dict):
        mapping = getattr(module, "param_names_mapping", None)
    return mapping if isinstance(mapping, dict) else {}


def _write_fused_shard(param: torch.Tensor, tensor: torch.Tensor, shard_id: int, num_shards: int) -> None:
    """Write ``tensor`` into the ``shard_id``-th slice (dim 0) of a fused param.

    SGLang fused projections pack ``[shard_0 | shard_1 | … | shard_{n-1}]``
    contiguously along dim 0 in shard-id order. Two layouts occur:

    * **all-equal** — ``q==k==v`` (Z-Image ``to_qkv``), ``w1==w3`` (``w13``): each
      slice is ``shard_id * size``.
    * **trailing-unequal** — HunyuanVideo single-block ``linear1 = [q, k, v, mlp]``
      packs three ``H``-sized attention shards + one ``4H``-sized MLP shard. The
      leading shards are equal (``shard_id * size``); the LAST shard is larger and
      sits at the tail (``dim0 - size``).

    Placing the trailing shard at the tail (rather than the legacy ``dim0 //
    num_shards`` equal split, which slices four ``1.75H`` chunks for ``linear1`` and
    crashes writing an ``H`` tensor into a ``1.75H`` slot) is exact for both layouts
    and needs no sibling shards — so it is robust to the sender's bucketing (a
    block's q/k/v/mlp may arrive in different buckets). It deliberately does NOT
    support an unequal MIDDLE shard (no SGLang fused param has one). The param's own
    ``weight_loader`` is tried first when it accepts a shard id (TP-correct); plain
    ``ReplicatedLinear`` (``tp_size=1``) takes no shard id, so it falls through here.
    """
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
    # Leading shards are equal-sized; a (possibly larger) trailing shard sits at the
    # tail. For all-equal fusions the two formulas coincide (``(n-1)*size == dim0 - size``).
    offset = total - size if shard_id == num_shards - 1 else shard_id * size
    if offset < 0 or offset + size > total:
        raise ValueError(
            f"fused shard {shard_id}/{num_shards}: size={size} at offset={offset} does not fit fused param dim0={total}"
        )
    data[offset : offset + size].copy_(tensor.to(param.dtype))


def _apply_fused_param_mapping(module, named_tensors):
    """Apply the model's ``param_names_mapping`` to the incoming named tensors.

    A model's ``param_names_mapping`` (the same dict its checkpoint loader applies)
    has two entry kinds; a model may use either or both:

    * **fused projections** — ``{regex: (replacement, shard_id, num_shards)}`` —
      write the trainer's separate-projection tensor into a dim-0 slice of the
      model's fused param (Z-Image ``to_q/k/v -> to_qkv``, ``w1/w3 -> w13``).
    * **plain renames** — ``{regex: replacement_str}`` — the model simply renamed
      a param vs the checkpoint. WAN's mapping is entirely of this kind
      (``patch_embedding.* -> patch_embedding.proj.*``,
      ``blocks.N.attn1.to_q.* -> blocks.N.to_q.*``,
      ``ffn.net.0.proj -> ffn.fc_in``, …). An EMPTY replacement means the model
      dropped that param (e.g. WAN ``attn2.norm_added_q``) — the tensor is discarded.

    Returns the leftover ``(name, tensor)`` list (renamed where applicable) for the
    exact-match loader. No-op when the module declares no mapping.

    Before this handled the rename kind, simple-rename models (WAN) matched NOTHING
    in the in-memory update path → 112/113 transformer tensors silently skipped →
    the rollout engine ran stale base weights → flat reward curve (cross-engine
    divergence), exactly the fused-model bug one step removed.
    """
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
                    # model dropped this param — nothing to load.
                    dropped += 1
                    handled = True
                    break
                target = re.sub(pat, val, name)
                if target in model_params:
                    leftover.append((target, tensor))
                    renamed += 1
                    handled = True
                    break
                # rename produced a non-param name; keep trying other patterns.
        if not handled:
            leftover.append((name, tensor))
    if fused or renamed or dropped:
        _log.info(
            "weight-sync: param_names_mapping applied — %d fused, %d renamed, %d dropped",
            fused,
            renamed,
            dropped,
        )
    # A leftover name that is still not a real model param slipped through every
    # mapping branch (unmatched pattern, or a rename whose target does not exist).
    # It will silently no-op in the exact-match loader — exactly the stale-weight
    # failure this mapping is meant to prevent — so surface it loudly instead.
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

            # Silent weight-drop is dangerous: a name matching no model param is
            # skipped above (``continue``), so a sender/receiver naming or fusion
            # mismatch (diffusers separate ``to_q/to_k/to_v`` vs SGLang's fused
            # ``to_qkv``) leaves those weights at their loaded base value with NO
            # error — the rollout silently never sees the trained update. Fused
            # projections are consumed upstream by ``_apply_fused_param_mapping``;
            # anything still unmatched here is a real mismatch, surfaced loudly.
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
                # Fused-projection models (Z-Image: to_q/k/v -> to_qkv,
                # feed_forward.w1/w3 -> w13) need the model's param_names_mapping
                # applied so the trainer's separate-projection tensors land in the
                # right fused shard. Consume those here; the rest fall through to
                # the exact-match loader below.
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
