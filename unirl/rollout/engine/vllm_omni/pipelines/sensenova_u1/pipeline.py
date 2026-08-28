"""RL-aware SenseNova-U1.5 vLLM-Omni pipeline."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    _patchify,
    _to_pil,
    _unpatchify,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    clear_flash_kv_cache,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    resolve_request_noise,
)
from unirl.rollout.engine.vllm_omni.pipelines.sensenova_u1.weight_names import (
    missing_weight_sync_names,
)
from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.sampling import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype


def _cache_row_to_cpu(cache: Any, row: int, batch_size: int) -> Any:
    """Copy one expanded DynamicCache row without retaining worker CUDA tensors."""
    if cache is None:
        return None
    result = copy.copy(cache)
    result.layers = []
    for source_layer in cache.layers:
        target_layer = copy.copy(source_layer)
        for name, value in vars(source_layer).items():
            if name.startswith("flash_"):
                if hasattr(target_layer, name):
                    delattr(target_layer, name)
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                value = value[row : row + 1]
            setattr(target_layer, name, value.detach().to("cpu").clone())
        result.layers.append(target_layer)
    return result


def _capture_conditions(caches: Dict[str, Any], p: SimpleNamespace) -> Dict[str, List[Any]]:
    """Materialize the worker's prefix caches in the trainer replay format."""
    batch_size = int(p.batch_size)
    use_cfg = float(p.cfg_scale) > 1.0 and "uncond" in caches
    if float(p.cfg_scale) > 1.0 and "img_cond" not in caches and "uncond" not in caches:
        raise RuntimeError("SenseNova T2I CFG requested an unconditional branch, but the worker returned no cache.")
    # One engine request contains multiple outputs of the same prompt. Prefix
    # caches and indexes are therefore identical expanded views; copy them once
    # and preserve aliases so pickle sends one payload per prompt group.
    condition_cache = _cache_row_to_cpu(caches["cond"], 0, batch_size)
    condition_index = caches["idx_cond"].detach().to("cpu").clone()
    condition_caches = [condition_cache] * batch_size
    if use_cfg:
        uncondition_cache = _cache_row_to_cpu(caches["uncond"], 0, batch_size)
        uncondition_index = caches["idx_uncond"].detach().to("cpu").clone()
        uncondition_caches = [uncondition_cache] * batch_size
        uncondition_indexes: List[Optional[torch.Tensor]] = [uncondition_index] * batch_size
    else:
        uncondition_caches = [None] * batch_size
        uncondition_indexes = [None] * batch_size

    return {
        "prompts": [str(p.prompt)] * batch_size,
        "condition_caches": condition_caches,
        "uncondition_caches": uncondition_caches,
        "condition_image_indexes": [condition_index] * batch_size,
        "uncondition_image_indexes": uncondition_indexes,
        "image_shapes": [(int(p.image_size[1]), int(p.image_size[0]))] * batch_size,
    }


class RLSenseNovaU1Pipeline(SenseNovaU1Pipeline):
    """SenseNova pixel-flow inference with driver x_T and FlowGRPO capture."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._pending_initial_noise: Optional[torch.Tensor] = None

    def _parse_request(self, req: OmniDiffusionRequest) -> SimpleNamespace:
        p = super()._parse_request(req)
        sampling = req.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        full_sigmas = extra.get("unirl_sigmas")
        p.sigmas = (
            torch.as_tensor(full_sigmas, dtype=torch.float32, device=self.device) if full_sigmas is not None else None
        )
        p.eta = float(getattr(sampling, "eta", 0.0) or 0.0)
        p.sde_indices = tuple(sorted({int(index) for index in extra.get("sde_indices", [])}))
        p.trajectory_dtype = parse_torch_dtype(
            extra.get("trajectory_precision", "bf16"),
            field_name="trajectory_precision",
        )
        p.sde_seed = int(extra.get("sde_seed", p.seed))
        self._pending_initial_noise = resolve_request_noise(
            req,
            caller="RLSenseNovaU1Pipeline._parse_request",
        )
        return p

    def _init_noise_and_schedule(self, p: SimpleNamespace) -> SimpleNamespace:
        ns = super()._init_noise_and_schedule(p)
        if p.sigmas is not None:
            if p.sigmas.ndim != 1 or int(p.sigmas.numel()) != int(p.num_steps) + 1:
                raise ValueError(
                    "RLSenseNovaU1Pipeline: driver sigma schedule must have "
                    f"{int(p.num_steps) + 1} entries, got shape={tuple(p.sigmas.shape)}."
                )
            if not bool(torch.all(p.sigmas[:-1] >= p.sigmas[1:])):
                raise ValueError("RLSenseNovaU1Pipeline: driver sigma schedule must be non-increasing.")
            ns.timesteps = 1.0 - p.sigmas
        else:
            p.sigmas = 1.0 - ns.timesteps

        initial_noise = self._pending_initial_noise
        self._pending_initial_noise = None
        if initial_noise is not None:
            expected = (int(p.batch_size), 3, int(p.image_size[1]), int(p.image_size[0]))
            if tuple(initial_noise.shape) != expected:
                raise RuntimeError(
                    f"RLSenseNovaU1Pipeline: driver x_T shape {tuple(initial_noise.shape)} != expected {expected}."
                )
            ns.image_prediction = initial_noise.to(
                device=self.device,
                dtype=p.trajectory_dtype,
            ) * float(ns.noise_scale)
        else:
            ns.image_prediction = ns.image_prediction.to(dtype=p.trajectory_dtype)
        return ns

    def validate_weight_sync_names(self, weights: List[tuple[str, torch.Tensor]]) -> None:
        """Reject full-weight buckets containing names this pipeline would skip."""
        parameter_names = set(dict(self.named_parameters()))
        missing = missing_weight_sync_names((name for name, _ in weights), parameter_names)
        if missing:
            sample = ", ".join(missing[:5])
            raise RuntimeError(
                f"SenseNova full-weight sync would silently skip {len(missing)} parameter(s); first names: [{sample}]"
            )

    def _run_denoising_loop(
        self,
        ns: SimpleNamespace,
        caches: Dict[str, Any],
        p: SimpleNamespace,
        think_text: str = "",
    ) -> DiffusionOutput:
        """Run upstream model predictions with UniRL's stochastic transition."""
        merge_size = self.merge_size
        image_prediction = ns.image_prediction
        sigmas = p.sigmas.to(device=self.device, dtype=torch.float32)
        strategy = FlowSDEStrategy()
        strategy.init_schedule(sigmas)
        sigma_max = sigmas[1] if int(sigmas.numel()) > 1 else sigmas[0]
        sde_indices = frozenset(int(index) for index in p.sde_indices)
        if sde_indices and p.eta <= 0.0:
            raise ValueError("RLSenseNovaU1Pipeline: non-empty sde_indices require eta > 0.")

        trajectory_positions = set(compute_trajectory_positions(set(sde_indices), int(p.num_steps)))
        trajectory_positions.add(int(p.num_steps))
        stored_positions: List[int] = []
        trajectory: List[torch.Tensor] = []
        if 0 in trajectory_positions:
            stored_positions.append(0)
            trajectory.append(_patchify(image_prediction, self.patch_size * merge_size).detach().clone())
        log_probs: List[torch.Tensor] = []
        generator = torch.Generator(self.device).manual_seed(int(p.sde_seed))

        for step_i in range(p.num_steps):
            t = ns.timesteps[step_i]
            t_next = ns.timesteps[step_i + 1]
            z = _patchify(image_prediction, self.patch_size * merge_size)
            image_input = _patchify(image_prediction, self.patch_size, channel_first=True)
            image_embeds = self._extract_feature(
                image_input.view(p.batch_size * ns.grid_h * ns.grid_w, -1),
                gen_model=True,
                grid_hw=ns.grid_hw,
            ).view(p.batch_size, ns.token_h * ns.token_w, -1)

            t_expanded = t.expand(p.batch_size * ns.token_h * ns.token_w)
            timestep_embeddings = self.fm_modules["timestep_embedder"](t_expanded).view(
                p.batch_size,
                ns.token_h * ns.token_w,
                -1,
            )
            if self.top_cfg.add_noise_scale_embedding:
                noise_scale_tensor = torch.full_like(
                    t_expanded,
                    ns.noise_scale / self.top_cfg.noise_scale_max_value,
                )
                timestep_embeddings = timestep_embeddings + self.fm_modules["noise_scale_embedder"](
                    noise_scale_tensor
                ).view(
                    p.batch_size,
                    ns.token_h * ns.token_w,
                    -1,
                )
            image_embeds = image_embeds + timestep_embeddings

            velocity = self._denoise_step(image_prediction, ns, t, z, image_embeds, caches, p, step_i)
            if step_i in sde_indices:
                unit_next, log_prob, _ = strategy.denoise(
                    noise_pred=-velocity / float(ns.noise_scale),
                    sample=z / float(ns.noise_scale),
                    sigma=sigmas[step_i],
                    sigma_next=sigmas[step_i + 1],
                    eta=float(p.eta),
                    generator=generator,
                    sigma_max=float(sigma_max),
                    step_index=step_i,
                )
                z = unit_next * float(ns.noise_scale)
                if log_prob is None:
                    raise RuntimeError(f"RLSenseNovaU1Pipeline: SDE step {step_i} produced no log probability.")
                log_probs.append(log_prob.to(torch.float32))
            else:
                z = z + (t_next - t) * velocity

            z = z.to(dtype=p.trajectory_dtype)
            if step_i + 1 in trajectory_positions:
                stored_positions.append(step_i + 1)
                trajectory.append(z.detach().clone())
            image_prediction = _unpatchify(
                z,
                self.patch_size * merge_size,
                p.image_size[1],
                p.image_size[0],
            )

        for key in ("cond", "uncond", "img_cond"):
            if key in caches and not isinstance(caches[key], dict):
                clear_flash_kv_cache(caches[key])

        images = _to_pil(image_prediction)
        custom_output: Dict[str, Any] = {
            "sde_step_indices": list(sorted(sde_indices)),
            "trajectory_indices": stored_positions,
            "sensenova_u1_capture": _capture_conditions(caches, p),
        }
        if think_text:
            custom_output["think_text"] = think_text
        return DiffusionOutput(
            output=images if len(images) != 1 else images[0],
            trajectory_latents=torch.stack(trajectory, dim=1),
            trajectory_timesteps=sigmas,
            trajectory_log_probs=torch.stack(log_probs, dim=1) if log_probs else None,
            custom_output=custom_output,
            to_cpu=True,
        )


__all__ = ["RLSenseNovaU1Pipeline"]
