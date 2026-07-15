"""RL-aware BAGEL-7B-MoT pipeline subclass.

``forward`` follows the RL interception protocol (install → arm → run → harvest):
install the trajectory-capturing SDE scheduler + noise tap + fp32 RoPE/RMSNorm
patches, arm per-request x_T/SDE, delegate to upstream, then harvest the trajectory.
Conditioning is NOT tapped — the driver ships prompts and the trainer rebuilds the
(frozen) KV contexts at replay. Loaded in vLLM-Omni's worker via custom_pipeline_args.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    drain_trajectory_into,
    resolve_request_noise,
    stamp_custom_output,
)
from unirl.rollout.engine.vllm_omni.pipelines.bagel.bagel_flow_match_sde_scheduler import (
    BagelFlowSDEScheduler,
)
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

_BAGEL_KV_REPLAY_TRACE_KEY = "unirl_bagel_t2ti_trace"
_BAGEL_T2TI_REPLAY_OUTPUT_KEY = "bagel_t2ti_replay"


class RLBagelPipeline(BagelPipeline):
    """BAGEL pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Trajectory-capturing SDE scheduler; generate_image reads it via self.scheduler.
        self._sde_scheduler = BagelFlowSDEScheduler()
        self._sde_scheduler_installed = False
        self._noise_tap_installed = False
        self._rope_fp32_patched = False
        self._rmsnorm_fp32_patched = False
        # Per-request x_T hand-off: armed every request, consumed once by the
        # prepare_vae_latent tap. None = upstream RNG draw fires.
        self._pending_initial_noise: Optional[torch.Tensor] = None
        # Stored trajectory dtype (matches trainside trajectory_precision).
        self._trajectory_dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Point ``self.scheduler`` at the trajectory-capturing SDE scheduler — always
        installed (even eta=0) since replay needs the captured trajectory; kwargs empty."""
        if self._sde_scheduler_installed:
            return
        self.scheduler = self._sde_scheduler
        self.scheduler_kwargs = {}
        self._sde_scheduler_installed = True

    def _install_rope_fp32(self) -> None:
        """Force the rotary cos/sin into fp32 to bit-match trainside — the worker rotary
        runs under autocast(bf16) with no guard, so its cos/sin diverge every step."""
        if self._rope_fp32_patched:
            return
        try:
            rotary = self.bagel.language_model.model.rotary_emb
        except AttributeError:
            # Topology changed (e.g. und-only build); skip rather than crash.
            self._rope_fp32_patched = True
            return

        if getattr(rotary, "_unirl_fp32_forward", False):
            self._rope_fp32_patched = True
            return

        orig_forward = rotary.forward

        @torch.no_grad()
        def fp32_forward(x: torch.Tensor, position_ids: torch.Tensor):
            inv_freq_expanded = rotary.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            # Force fp32 for the matmul + trig (autocast off), like vendored Qwen2RotaryEmbedding.
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            cos = cos * rotary.attention_scaling
            sin = sin * rotary.attention_scaling
            # Return fp32 (not bf16): keeps q/k fp32 through rotary_op so the rotated
            # q/k match trainside (the attention forward downcasts to bf16 itself).
            return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)

        rotary.forward = fp32_forward  # type: ignore[assignment]
        rotary._unirl_fp32_forward = True  # type: ignore[attr-defined]
        # Keep a handle for debugging / potential revert; never restored in-run.
        rotary._unirl_orig_forward = orig_forward  # type: ignore[attr-defined]
        logger.warning("[PATCH-INSTALLED] rope_fp32 modules=1 (rotary_emb)")
        self._rope_fp32_patched = True

    def _install_rmsnorm_fp32(self) -> None:
        """Make every worker RMSNorm bit-match the trainside ``Qwen2RMSNorm`` — vLLM
        rounds the fp32 q/k-norm to bf16 before the multiply, a LoRA-growing velocity gap."""
        if self._rmsnorm_fp32_patched:
            return
        try:
            from vllm.model_executor.layers.layernorm import RMSNorm as _VllmRMSNorm
        except Exception:
            self._rmsnorm_fp32_patched = True
            return

        def _make_fp32_forward(module: Any):
            eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            orig = module.forward

            def fp32_forward(x: torch.Tensor, residual: Optional[torch.Tensor] = None):
                # Fused add-then-norm isn't on the gen velocity path; defer to the
                # original kernel to keep its contract.
                if residual is not None:
                    return orig(x, residual)
                input_dtype = x.dtype
                h = x.to(torch.float32)
                variance = h.pow(2).mean(-1, keepdim=True)
                h = h * torch.rsqrt(variance + eps)
                # Literal Qwen2RMSNorm: weight in native dtype, multiply promotes
                # when h.to(input_dtype) is fp32 (the gen q/k path).
                return module.weight * h.to(input_dtype)

            return fp32_forward

        patched = 0
        for module in self.bagel.modules():
            if isinstance(module, _VllmRMSNorm) and not getattr(module, "_unirl_fp32_rmsnorm", False):
                module._unirl_orig_forward = module.forward  # type: ignore[attr-defined]
                module.forward = _make_fp32_forward(module)  # type: ignore[assignment]
                module._unirl_fp32_rmsnorm = True  # type: ignore[attr-defined]
                patched += 1
        logger.warning("[PATCH-INSTALLED] rmsnorm_fp32 modules=%d", patched)
        self._rmsnorm_fp32_patched = True

    def _install_noise_tap(self) -> None:
        """Wrap ``bagel.prepare_vae_latent`` to swap the driver-authored x_T (consume-once)
        in for upstream's RNG-drawn ``packed_init_noises``, leaving other inputs untouched."""
        if self._noise_tap_installed:
            return

        orig = self.bagel.prepare_vae_latent
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            out = orig(*args, **kw)
            noise = pipeline_self._pending_initial_noise
            if noise is not None:
                pipeline_self._pending_initial_noise = None
                ref = out.get("packed_init_noises")
                if ref is None:
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: prepare_vae_latent returned no 'packed_init_noises' to override."
                    )
                # Driver x_T is [1, seq, C]; packed_init_noises is unbatched [seq, C].
                # Squeeze the leading 1 and validate the packed geometry matches.
                if noise.dim() == ref.dim() + 1 and int(noise.shape[0]) == 1:
                    noise = noise.squeeze(0)
                if tuple(noise.shape) != tuple(ref.shape):
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: driver x_T shape "
                        f"{tuple(noise.shape)} != worker packed_init_noises shape "
                        f"{tuple(ref.shape)} — check the recipe's "
                        "init_noise_latent_shape (bagel_latent_shape) vs the "
                        "request's height/width."
                    )
                # Match the worker draw's dtype/device (upstream moves to device after).
                out["packed_init_noises"] = noise.to(dtype=ref.dtype, device=ref.device)
            return out

        self.bagel.prepare_vae_latent = tapped  # type: ignore[assignment]
        self._noise_tap_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate + σ_max + storage dtype."""
        sp = req.sampling_params
        eta = float(getattr(sp, "eta", 0.0) or 0.0)
        extra = getattr(sp, "extra_args", None) or {}
        traj_dtype_name = extra.get("trajectory_precision")
        traj_dtype = (
            parse_torch_dtype(traj_dtype_name, field_name="trajectory_precision")
            if traj_dtype_name
            else self._trajectory_dtype
        )
        # σ_max (trainside schedule[1]): load-bearing for the first SDE step's
        # std_dev_t clamp — must match trainside or the ratio drifts off 1.
        sigma_max = extra.get("sigma_max")
        self._sde_scheduler.set_for_request(
            eta=eta,
            sde_indices=extra.get("sde_indices"),
            sigma_max=float(sigma_max) if sigma_max is not None else None,
            trajectory_dtype=traj_dtype,
            noise_seed=int(extra["sde_seed"]) if extra.get("sde_seed") is not None else None,
        )

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLBagelPipeline._arm_initial_noise")

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        """Overwrite upstream's trajectory capture with the SDE scheduler's — sets
        latents/timesteps/log_probs + sparse sde_step_indices (the build_image_segment wire)."""
        drain_trajectory_into(out, self._sde_scheduler)

    def _native_t2ti_replay_payload(self, req: OmniDiffusionRequest) -> Optional[dict[str, Any]]:
        """Validate Stage-0's transfer trace and echo Stage-1's observations.

        Presence of the private transfer key identifies a native BAGEL T2TI
        request. Plain single-stage ``bagel_t2i`` has no injected cache and
        returns ``None`` here, preserving its existing prompt-rebuild path.
        """
        sp = req.sampling_params
        metadata = getattr(sp, "kv_metadata", None) or {}
        trace = metadata.get(_BAGEL_KV_REPLAY_TRACE_KEY)
        if trace is None:
            return None
        if not isinstance(trace, dict):
            raise RuntimeError(
                f"RLBagelPipeline: native T2TI KV replay trace must be a mapping; got {type(trace).__name__}."
            )

        required = {"cache_input_ids", "chunk_offsets", "kv_length", "ropes"}
        missing = sorted(required - set(trace))
        if missing:
            raise RuntimeError(f"RLBagelPipeline: native T2TI KV replay trace is missing fields {missing}.")

        cache_input_ids = [int(token) for token in trace["cache_input_ids"]]
        chunk_offsets = [int(offset) for offset in trace["chunk_offsets"]]
        kv_length = int(trace["kv_length"])
        stage0_ropes = [int(rope) for rope in trace["ropes"]]
        if not cache_input_ids or len(cache_input_ids) != kv_length:
            raise RuntimeError(
                "RLBagelPipeline: Stage-0 cache-input trace length does not match its KV length: "
                f"tokens={len(cache_input_ids)}, kv_length={kv_length}."
            )
        if (
            len(chunk_offsets) < 2
            or chunk_offsets[0] != 0
            or chunk_offsets[-1] != kv_length
            or any(b <= a for a, b in zip(chunk_offsets, chunk_offsets[1:]))
        ):
            raise RuntimeError(
                "RLBagelPipeline: invalid Stage-0 cache-input chunk offsets "
                f"{chunk_offsets!r} for kv_length={kv_length}."
            )
        if not stage0_ropes:
            raise RuntimeError("RLBagelPipeline: Stage-0 T2TI ropes must be non-empty.")

        injected_kv = getattr(sp, "past_key_values", None)
        try:
            received_kv_length = int(injected_kv.key_cache[0].shape[0])
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            raise RuntimeError(
                "RLBagelPipeline: native T2TI replay metadata arrived without a readable injected KV cache."
            ) from exc
        received_ropes = [int(rope) for rope in (metadata.get("ropes") or [received_kv_length])]
        if received_kv_length != kv_length:
            raise RuntimeError(
                "RLBagelPipeline: Stage-1 received KV length differs from Stage 0: "
                f"received={received_kv_length}, stage0={kv_length}."
            )
        if received_ropes != stage0_ropes:
            raise RuntimeError(
                "RLBagelPipeline: Stage-1 received ropes differ from Stage 0: "
                f"received={received_ropes!r}, stage0={stage0_ropes!r}."
            )

        if metadata.get("image_shape") is not None:
            image_shape = [int(value) for value in metadata["image_shape"]]
        else:
            max_hw = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
            image_shape = [
                int(sp.height) if sp.height is not None else max_hw,
                int(sp.width) if sp.width is not None else max_hw,
            ]
        if len(image_shape) != 2 or any(value <= 0 for value in image_shape):
            raise RuntimeError(f"RLBagelPipeline: invalid native T2TI image shape {image_shape!r}.")

        return {
            "cache_input_ids": cache_input_ids,
            "chunk_offsets": chunk_offsets,
            "kv_length": kv_length,
            "ropes": stage0_ropes,
            "received_kv_length": received_kv_length,
            "received_ropes": received_ropes,
            "image_shape": image_shape,
        }

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_noise_tap()
        # fp32 RoPE + RMSNorm: bit-match the trainside forward so the rollout↔replay
        # log-prob ratio stays ≈ 1 (see the install methods).
        self._install_rope_fp32()
        self._install_rmsnorm_fp32()

        self._arm_sde(req)
        self._arm_initial_noise(req)

        # Validate the native transfer before Stage 1 consumes it. This payload
        # contains only reconstruction metadata; cache tensors stay in vLLM-Omni.
        replay_payload = self._native_t2ti_replay_payload(req)

        # Delegate the full pipeline to upstream; the noise tap fires inside and the
        # scheduler captures the trajectory as the loop runs.
        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        if replay_payload is not None:
            stamp_custom_output(out, _BAGEL_T2TI_REPLAY_OUTPUT_KEY, replay_payload)
        return out


__all__ = ["RLBagelPipeline"]
