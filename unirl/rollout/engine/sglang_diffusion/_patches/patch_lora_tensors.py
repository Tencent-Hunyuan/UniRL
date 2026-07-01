"""Re-host the ``sglang-drl`` fork's in-memory LoRA path on stock upstream.

This is the heaviest patch. The fork let the RL trainer push freshly-optimized
LoRA tensors straight into the live ``LoRAPipeline`` (no safetensors round-trip)
and keep them UNMERGED so the forward pass computes ``base + A@B`` on the fly --
required because the adapter weights change every training step and a weight-sync
overwrites the base weights underneath them.

THREE-WAY DIVERGENCE (important). At the fork point ``e9b570654`` the pipeline's
LoRA merge policy was a simple always-merge. The fork added a 2-value
``lora_merge_mode`` (``"merge"`` = bake into base weights, ``"online"`` = compute
on-the-fly) with ``self.auto_merge = (mode == "merge")``. INDEPENDENTLY, upstream
evolved a *different* 3-value policy (``LORA_MERGE_MODES = ("auto","merge",
"dynamic")``) with ``_resolve_lora_merge_mode`` / ``_should_merge_lora_for_layers``
and ``merge_weights`` / ``merge_mode`` params threaded through ``set_lora`` ->
``_apply_lora_to_layers`` -> ``set_lora_weights``. So the fork's ``set_lora`` body
and upstream's are structurally incompatible -- a blanket REPLACE would destroy
upstream's merge-mode system. We therefore re-home the fork's *semantics* onto
upstream's plumbing rather than copying its ``set_lora``:

  (a) Register ``"online"`` as a valid merge mode (extend ``LORA_MERGE_MODES``)
      and make ``_should_merge_lora_for_layers`` treat it as no-merge. Upstream
      then drives ``merge_weights=False`` down to ``set_lora_weights`` for the
      RL path -- exactly what the fork's ``auto_merge=False`` did, via upstream's
      own machinery. This is why we do NOT need to patch ``auto_merge`` onto
      ``BaseLayerWithLoRA.__init__`` / ``wrap_with_lora_layer`` /
      ``convert_module_lora_layers``: upstream's ``merge_weights`` plumbing (which
      did not exist at the fork point) supersedes the fork's ``auto_merge`` flag.

  (b) AROUND-wrap ``__init__`` to set ``self.lora_merge_mode`` / ``self.auto_merge``
      and, in ``"online"`` mode with no ``lora_path``, eagerly wrap layers
      (``convert_to_lora_layers``) so weight-sync targets exist before any LoRA
      arrives (the fork's ``elif lora_merge_mode == "online"`` branch).

  (c) ``setattr`` the fork-new ``_register_lora_state_dict`` (verbatim) and
      ``load_lora_adapter_from_tensors`` (verbatim). We KEEP upstream's diverged
      ``load_lora_adapter`` (it reads ``adapter_config.json`` + tracks
      ``loaded_adapter_alphas`` + takes ``weight_name`` -- the fork lacks all
      that); only the in-memory entry point is new.

  (d) AROUND-wrap ``set_lora`` to accept ``lora_tensors=None``. When tensors are
      supplied: register them via ``load_lora_adapter_from_tensors`` and
      invalidate the cached per-module config (so upstream's
      ``_check_lora_config_matches`` does not short-circuit -- LoRA weights change
      every step and MUST be re-applied), then delegate to upstream ``set_lora``
      with ``lora_path=None``. Upstream finds the adapter already in
      ``self.lora_adapters`` (so its ``path is None`` guard does not raise) and
      applies it under the resolved (online=no-merge) mode.

  (e) ``setattr`` the fork-new ``handle_weight_sync`` (verbatim) -- called by
      ``WeightsUpdater._post_update_cleanup`` (``patch_weights_updater``) after a
      weight-sync to mark layers unmerged + refresh the ``cpu_weight`` snapshot.

  (f) ``setattr`` the fork-new ``BaseLayerWithLoRA.update_base_weight_snapshot``
      (used by ``handle_weight_sync``) and REPLACE ``LinearWithLoRA.forward`` to
      apply the fork's two changes (drop ``@torch.compile`` -- the compiled graph
      goes stale when online LoRA weights change every step; drop the
      ``delta.reshape`` -- ``nn.Linear`` preserves input dims) WHILE keeping
      upstream's dtype-casting (which did not exist at the fork point).

Idempotent via sentinel guards. Import-safe (sglang imported inside the fn).

RISKS / non-cleanly-rehomable pieces -- see PR notes:
  * ``"online"`` is NOT an upstream merge mode; (a) registers it. If upstream
    later adds its own ``"online"`` with different semantics this collides.
  * The fork's ``auto_merge`` ctor thread is intentionally NOT ported (superseded
    by upstream ``merge_weights``); this is a behavioural mapping, not a verbatim
    copy. Pinned only by UniRL's own LoRA weight-sync tests.
  * ``LinearWithLoRA.forward`` is a behavioural REPLACE that re-vendors upstream's
    body; an upstream bump to that forward needs a re-sync.
"""

from __future__ import annotations

_MODES_SENTINEL = "_unirl_online_merge_mode"
_INIT_SENTINEL = "_unirl_lora_online_init"
_METHODS_SENTINEL = "_unirl_lora_from_tensors_methods"
_SETLORA_SENTINEL = "_unirl_lora_tensors_param"
_LAYER_SENTINEL = "_unirl_lora_online_layer"


def patch_lora_tensors() -> None:
    """Install the fork's in-memory / online LoRA path on upstream ``LoRAPipeline``."""
    import sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline as lp
    import sglang.multimodal_gen.runtime.server_args as sa

    LoRAPipeline = lp.LoRAPipeline
    logger = lp.logger

    # --- (a) Register "online" as a valid merge mode --------------------------
    # Upstream LORA_MERGE_MODES = ("auto","merge","dynamic"). UniRL ships
    # lora_merge_mode="online" (rollout/engine/sglang/config.py), which upstream's
    # _resolve_lora_merge_mode would reject. Extend the constant *in both modules*
    # that reference it (server_args defines it; lora_pipeline imported it by
    # value at import time, so rebind that binding too).
    if "online" not in sa.LORA_MERGE_MODES:
        sa.LORA_MERGE_MODES = tuple(sa.LORA_MERGE_MODES) + ("online",)
    if "online" not in lp.LORA_MERGE_MODES:
        lp.LORA_MERGE_MODES = tuple(lp.LORA_MERGE_MODES) + ("online",)

    # Make _should_merge_lora_for_layers treat "online" as no-merge (compute LoRA
    # on the fly in forward). AROUND-wrap so all other modes keep upstream policy.
    if not getattr(LoRAPipeline._should_merge_lora_for_layers, _MODES_SENTINEL, False):
        _orig_should_merge = LoRAPipeline._should_merge_lora_for_layers

        def _should_merge_lora_for_layers(self, module_name, lora_layers, merge_mode):
            if merge_mode == "online":
                return False
            return _orig_should_merge(self, module_name, lora_layers, merge_mode)

        _should_merge_lora_for_layers._unirl_online_merge_mode = True  # type: ignore[attr-defined]
        LoRAPipeline._should_merge_lora_for_layers = _should_merge_lora_for_layers

    # --- (b) AROUND-wrap __init__: set merge-mode attrs + online prewrap -------
    if not getattr(LoRAPipeline.__init__, _INIT_SENTINEL, False):
        _orig_init = LoRAPipeline.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Fork additions: expose the configured merge mode on the instance.
            self.lora_merge_mode = self.server_args.lora_merge_mode
            self.auto_merge = self.lora_merge_mode == "merge"
            # RL workflow: when online and no startup lora_path, upstream __init__
            # did NOT wrap any layers. Wrap them now so weight-sync targets exist
            # before the first adapter arrives. (When lora_path is set, upstream
            # __init__ already called convert_to_lora_layers + set_lora.)
            if self.lora_path is None and self.lora_merge_mode == "online":
                self.convert_to_lora_layers()

        __init__._unirl_lora_online_init = True  # type: ignore[attr-defined]
        LoRAPipeline.__init__ = __init__

    # --- (c)+(e) setattr fork-new pipeline methods (verbatim) -----------------
    if not getattr(LoRAPipeline, _METHODS_SENTINEL, False):
        # Bind upstream module-globals the verbatim bodies reference, so the
        # nested fns resolve them through THIS scope (LEGB) identically.
        import hashlib
        from collections import defaultdict
        from collections.abc import Hashable
        from typing import Any

        import torch
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
        from sglang.multimodal_gen.runtime.pipelines_core.lora_format_adapter import (
            normalize_lora_state_dict,
        )

        def _register_lora_state_dict(
            self,
            lora_state_dict: dict,
            lora_nickname: str,
            lora_path,
            rank: int,
        ) -> None:
            """Shared logic: normalize names, merge fused params, store in lora_adapters."""
            if lora_nickname in self.lora_adapters:
                self.lora_adapters[lora_nickname].clear()

            config = self.server_args.pipeline_config.dit_config.arch_config

            param_names_mapping_fn = get_param_names_mapping(
                config.param_names_mapping or self.modules["transformer"].param_names_mapping
            )
            lora_param_names_mapping_fn = get_param_names_mapping(
                config.lora_param_names_mapping or self.modules["transformer"].lora_param_names_mapping
            )

            to_merge_params: defaultdict[Hashable, dict[Any, Any]] = defaultdict(dict)
            for name, weight in lora_state_dict.items():
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                # misc-format -> HF-format
                name, _, _ = lora_param_names_mapping_fn(name)
                # HF-format (LoRA) -> SGLang-dit-format
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(name)
                # for fuse B(out_dim, r) @ A(r, in_dim) -> (N, out_dim, r) @ (N, r, in_dim)
                # see param mapping in HunyuanVideoArchConfig
                if merge_index is not None:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        sorted_tensors = [to_merge_params[target_name][i] for i in range(num_params_to_merge)]
                        # Use stack instead of cat because it needs to be compatible with TP.
                        weight = torch.stack(sorted_tensors, dim=0)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in self.lora_adapters[lora_nickname]:
                    raise ValueError(
                        f"Dit target weight name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
            if lora_path is not None:
                self.loaded_adapter_paths[lora_nickname] = lora_path
            logger.info("Rank %d: registered LoRA adapter %s", rank, lora_path or lora_nickname)

        def load_lora_adapter_from_tensors(
            self,
            lora_tensors: dict,
            lora_nickname: str,
            rank: int,
        ) -> None:
            """Load LoRA adapter from in-memory tensors instead of a file path."""
            lora_state_dict = normalize_lora_state_dict(lora_tensors, logger=logger)
            self._register_lora_state_dict(lora_state_dict, lora_nickname, None, rank)

        def handle_weight_sync(self, updated_module_names: set) -> None:
            """Handle LoRA state after ALL weight sync buckets have been applied.

            Called once after all buckets/dtypes are done (gated by flush_cache in
            the weight updater). At this point base_layer.weight, lora_A, and
            lora_B have all been overwritten by the sync with new raw values.

            We must NOT unmerge (that would restore OLD base from cpu_weight,
            overwriting the new sync values). Instead:
            1. Mark layers as unmerged (weight sync replaced merged weights with raw)
            2. Refresh cpu_weight snapshot from the new raw base weights
            3. Leave LoRA unmerged — forward computes LoRA on-the-fly
            """
            if not self.lora_initialized:
                return

            # Map module names to their LoRA layer dicts
            module_to_lora_layers = {
                "transformer": self.lora_layers,
                "transformer_2": self.lora_layers_transformer_2,
                "critic": self.lora_layers_critic,
            }

            for module_name in updated_module_names:
                lora_layers_dict = module_to_lora_layers.get(module_name)
                if not lora_layers_dict:
                    continue
                if not self.cur_adapter_name.get(module_name):
                    continue  # No LoRA active for this module

                # Weight sync already replaced base_layer.weight with new raw base.
                # Just update the snapshot and mark as unmerged so forward uses
                # online LoRA computation.
                lora_a_hash = None
                for name, layer in lora_layers_dict.items():
                    layer.merged = False
                    layer.update_base_weight_snapshot()
                    if lora_a_hash is None and layer.lora_A is not None:
                        lora_a_hash = hashlib.sha256(
                            layer.lora_A.data.contiguous().cpu().float().numpy().tobytes()
                        ).hexdigest()[:16]

                self.is_lora_merged[module_name] = False

                logger.info(
                    "LoRA state refreshed after weight sync for %s (mode=%s, lora_A_hash=%s)",
                    module_name,
                    self.lora_merge_mode,
                    lora_a_hash,
                )

        LoRAPipeline._register_lora_state_dict = _register_lora_state_dict
        LoRAPipeline.load_lora_adapter_from_tensors = load_lora_adapter_from_tensors
        LoRAPipeline.handle_weight_sync = handle_weight_sync
        setattr(LoRAPipeline, _METHODS_SENTINEL, True)

    # --- (d) AROUND-wrap set_lora to accept lora_tensors= ---------------------
    if not getattr(LoRAPipeline.set_lora, _SETLORA_SENTINEL, False):
        _orig_set_lora = LoRAPipeline.set_lora

        def set_lora(
            self,
            lora_nickname,
            lora_path=None,
            target="all",
            strength=1.0,
            merge_weights=None,
            merge_mode=None,
            lora_tensors=None,
        ):
            """Upstream ``set_lora`` + a fork ``lora_tensors=`` in-memory branch.

            When ``lora_tensors`` is given (RL weight-sync path), register the
            tensors as ``lora_nickname`` and invalidate the cached per-module
            config so upstream re-applies them (LoRA weights change every step),
            then delegate to upstream ``set_lora`` with ``lora_path=None``.
            Otherwise behaviour is byte-for-byte upstream.
            """
            if lora_tensors is not None:
                # set_lora always pre-wraps layers when uninitialized, but the
                # registration path below needs self.modules["transformer"] param
                # mappings only; conversion is handled by upstream set_lora. Mirror
                # upstream's normalization to learn the rank for this call.
                rank = self._distributed_rank()
                # Single-nickname contract for the from-tensors path (the fork's
                # SetLoraFromTensorsReq carries a single nickname).
                nickname = lora_nickname[0] if isinstance(lora_nickname, list) else lora_nickname
                if not self.lora_initialized:
                    with self._temporarily_disable_offload(target="all", use_module_names_only=True):
                        self.convert_to_lora_layers()
                # Always reload from tensors — LoRA weights change after each
                # training step and must be refreshed on every sync.
                self.load_lora_adapter_from_tensors(lora_tensors, nickname, rank)
                # Invalidate cached config so _check_lora_config_matches does not
                # short-circuit re-application of the (changed) adapter.
                tgt_list = target if isinstance(target, list) else [target]
                for tgt in tgt_list:
                    target_modules, _ = self._get_target_lora_layers(tgt)
                    for module_name, _layers in target_modules:
                        self.cur_adapter_config.pop(module_name, None)
                # Delegate to upstream with lora_path=None (adapter is now present
                # in self.lora_adapters, so upstream's path-None guard won't raise).
                return _orig_set_lora(
                    self,
                    lora_nickname,
                    lora_path=None,
                    target=target,
                    strength=strength,
                    merge_weights=merge_weights,
                    merge_mode=merge_mode,
                )

            return _orig_set_lora(
                self,
                lora_nickname,
                lora_path=lora_path,
                target=target,
                strength=strength,
                merge_weights=merge_weights,
                merge_mode=merge_mode,
            )

        set_lora._unirl_lora_tensors_param = True  # type: ignore[attr-defined]
        LoRAPipeline.set_lora = set_lora

        # Small helper so the wrapper does not duplicate the dist-rank logic
        # upstream uses inline (`dist.get_rank()`); kept defensive for the
        # not-yet-initialized-process-group case in unit tests.
        if not hasattr(LoRAPipeline, "_distributed_rank"):
            import torch.distributed as dist

            def _distributed_rank(self) -> int:
                return dist.get_rank() if dist.is_initialized() else 0

            LoRAPipeline._distributed_rank = _distributed_rank

    # --- (f) layers/lora/linear.py: snapshot refresh + forward REPLACE --------
    _patch_lora_linear()


def _patch_lora_linear() -> None:
    """Port the fork's ``layers/lora/linear.py`` changes onto upstream.

    Two pieces:
      * ``BaseLayerWithLoRA.update_base_weight_snapshot`` (fork-new; used by
        ``handle_weight_sync``) -- additive setattr.
      * ``LinearWithLoRA.forward`` -- behavioural REPLACE re-vendoring upstream's
        body minus ``@torch.compile`` and minus the ``delta.reshape``.
    """
    import sglang.multimodal_gen.runtime.layers.lora.linear as ll
    import torch
    from torch.distributed.tensor import DTensor

    BaseLayerWithLoRA = ll.BaseLayerWithLoRA
    LinearWithLoRA = ll.LinearWithLoRA

    if not getattr(BaseLayerWithLoRA, _LAYER_SENTINEL, False):

        def update_base_weight_snapshot(self) -> None:
            """Refresh the CPU weight snapshot from the current base layer weights."""
            self.cpu_weight = self.base_layer.weight.detach().to("cpu").clone()

        BaseLayerWithLoRA.update_base_weight_snapshot = update_base_weight_snapshot
        setattr(BaseLayerWithLoRA, _LAYER_SENTINEL, True)

    # REPLACE LinearWithLoRA.forward. Re-vendors UPSTREAM's body (keeping its
    # lora_dtype / dtype-cast, which did not exist at the fork point) with the
    # fork's two changes applied: no @torch.compile, no delta.reshape.
    if not getattr(LinearWithLoRA.forward, _LAYER_SENTINEL, False):

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            lora_A = self.lora_A
            lora_B = self.lora_B
            if isinstance(self.lora_B, DTensor):
                lora_B = self.lora_B.to_local()
                lora_A = self.lora_A.to_local()

            if not self.merged and not self.disable_lora:
                lora_dtype = lora_A.dtype
                x_lora = x.to(dtype=lora_dtype)
                lora_A_sliced = self.slice_lora_a_weights(lora_A.to(device=x.device, non_blocking=True))
                lora_B_sliced = self.slice_lora_b_weights(lora_B.to(device=x.device, non_blocking=True))
                delta = x_lora @ lora_A_sliced.T @ lora_B_sliced.T
                if self.lora_alpha != self.lora_rank:
                    delta = delta * (
                        self.lora_alpha / self.lora_rank  # type: ignore
                    )  # type: ignore
                delta = delta * self.strength
                # nn.Linear preserves input dimensions — no reshape needed.
                out = self.base_layer(x)
                return out + delta.to(dtype=out.dtype)
            else:
                return self.base_layer(x)

        forward._unirl_lora_online_layer = True  # type: ignore[attr-defined]
        LinearWithLoRA.forward = forward
