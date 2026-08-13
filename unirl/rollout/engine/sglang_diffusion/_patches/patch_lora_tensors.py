"""Re-host the ``sglang-drl`` fork's in-memory LoRA path on stock upstream."""

from __future__ import annotations

_MODES_SENTINEL = "_unirl_online_merge_mode"
_INIT_SENTINEL = "_unirl_lora_online_init"
_METHODS_SENTINEL = "_unirl_lora_from_tensors_methods"
_SETLORA_SENTINEL = "_unirl_lora_tensors_param"
_LAYER_SENTINEL = "_unirl_lora_online_layer"
_APPLY_SENTINEL = "_unirl_adapter_alpha_expand"

_ADAPTER_ALPHA_KEY = "__adapter_alpha__"


def patch_lora_tensors() -> None:
    """Install the fork's in-memory / online LoRA path on upstream ``LoRAPipeline``."""
    import sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline as lp
    import sglang.multimodal_gen.runtime.server_args as sa

    LoRAPipeline = lp.LoRAPipeline
    logger = lp.logger

    if "online" not in sa.LORA_MERGE_MODES:
        sa.LORA_MERGE_MODES = tuple(sa.LORA_MERGE_MODES) + ("online",)
    if "online" not in lp.LORA_MERGE_MODES:
        lp.LORA_MERGE_MODES = tuple(lp.LORA_MERGE_MODES) + ("online",)

    if not getattr(LoRAPipeline._should_merge_lora_for_layers, _MODES_SENTINEL, False):
        _orig_should_merge = LoRAPipeline._should_merge_lora_for_layers

        def _should_merge_lora_for_layers(self, module_name, lora_layers, merge_mode):
            if merge_mode == "online":
                return False
            return _orig_should_merge(self, module_name, lora_layers, merge_mode)

        _should_merge_lora_for_layers._unirl_online_merge_mode = True  # type: ignore[attr-defined]
        LoRAPipeline._should_merge_lora_for_layers = _should_merge_lora_for_layers

    if not getattr(LoRAPipeline.__init__, _INIT_SENTINEL, False):
        _orig_init = LoRAPipeline.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.lora_merge_mode = self.server_args.lora_merge_mode
            self.auto_merge = self.lora_merge_mode == "merge"
            if self.lora_path is None and self.lora_merge_mode == "online":
                self.convert_to_lora_layers()

        __init__._unirl_lora_online_init = True  # type: ignore[attr-defined]
        LoRAPipeline.__init__ = __init__

    if not getattr(LoRAPipeline, _METHODS_SENTINEL, False):
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
            adapter_alpha=None,
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
                if name.endswith(".alpha"):
                    continue
                name = name.replace(".weight", "")
                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(name)
                if merge_index is not None:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        sorted_tensors = [to_merge_params[target_name][i] for i in range(num_params_to_merge)]
                        weight = torch.stack(sorted_tensors, dim=0)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in self.lora_adapters[lora_nickname]:
                    raise ValueError(
                        f"Dit target weight name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
            if adapter_alpha is not None:
                self.lora_adapters[lora_nickname][_ADAPTER_ALPHA_KEY] = torch.tensor(float(adapter_alpha))
            if lora_path is not None:
                self.loaded_adapter_paths[lora_nickname] = lora_path
            logger.info("Rank %d: registered LoRA adapter %s", rank, lora_path or lora_nickname)

        def load_lora_adapter_from_tensors(
            self,
            lora_tensors: dict,
            lora_nickname: str,
            rank: int,
            adapter_alpha=None,
        ) -> None:
            """Load LoRA adapter from in-memory tensors instead of a file path."""
            lora_state_dict = normalize_lora_state_dict(lora_tensors, logger=logger)
            self._register_lora_state_dict(lora_state_dict, lora_nickname, None, rank, adapter_alpha=adapter_alpha)

        def handle_weight_sync(self, updated_module_names: set) -> None:
            """Handle LoRA state after ALL weight sync buckets have been applied."""
            if not self.lora_initialized:
                return

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
                    continue

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
            lora_alpha=None,
        ):
            """Upstream ``set_lora`` + a fork ``lora_tensors=`` in-memory branch."""
            if lora_tensors is not None:
                rank = self._distributed_rank()
                nickname = lora_nickname[0] if isinstance(lora_nickname, list) else lora_nickname
                if not self.lora_initialized:
                    with self._temporarily_disable_offload(target="all", use_module_names_only=True):
                        self.convert_to_lora_layers()
                self.load_lora_adapter_from_tensors(lora_tensors, nickname, rank, adapter_alpha=lora_alpha)
                tgt_list = target if isinstance(target, list) else [target]
                for tgt in tgt_list:
                    target_modules, _ = self._get_target_lora_layers(tgt)
                    for module_name, _layers in target_modules:
                        self.cur_adapter_config.pop(module_name, None)
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

        if not hasattr(LoRAPipeline, "_distributed_rank"):
            import torch.distributed as dist

            def _distributed_rank(self) -> int:
                return dist.get_rank() if dist.is_initialized() else 0

            LoRAPipeline._distributed_rank = _distributed_rank

    if not getattr(LoRAPipeline._apply_lora_to_layers, _APPLY_SENTINEL, False):
        _orig_apply = LoRAPipeline._apply_lora_to_layers

        def _apply_lora_to_layers(self, lora_layers, lora_nicknames, *args, **kwargs):
            for nickname in lora_nicknames:
                adapter = self.lora_adapters.get(nickname)
                if not adapter or _ADAPTER_ALPHA_KEY not in adapter:
                    continue
                adapter_alpha = adapter[_ADAPTER_ALPHA_KEY]
                for name in lora_layers:
                    if name + ".lora_A" in adapter and name + ".alpha" not in adapter:
                        adapter[name + ".alpha"] = adapter_alpha
            return _orig_apply(self, lora_layers, lora_nicknames, *args, **kwargs)

        _apply_lora_to_layers._unirl_adapter_alpha_expand = True  # type: ignore[attr-defined]
        LoRAPipeline._apply_lora_to_layers = _apply_lora_to_layers

    _patch_lora_linear()


def _patch_lora_linear() -> None:
    """Port the fork's ``layers/lora/linear.py`` changes onto upstream."""
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
                out = self.base_layer(x)
                return out + delta.to(dtype=out.dtype)
            else:
                return self.base_layer(x)

        forward._unirl_lora_online_layer = True  # type: ignore[attr-defined]
        LinearWithLoRA.forward = forward
