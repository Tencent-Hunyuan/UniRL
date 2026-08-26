from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from unirl.algorithms.base import _require_replay_anchor_for_batched_replay, _transition_sigma
from unirl.models.hunyuan_video10.conditions import HunyuanVideo10Conditions
from unirl.models.hunyuan_video10.diffusion import HunyuanVideo10DiffusionStage, HunyuanVideo10DiffusionStep
from unirl.models.hunyuan_video15.conditions import HunyuanVideo15Conditions
from unirl.models.hunyuan_video15.diffusion import HunyuanVideo15DiffusionStage, HunyuanVideo15DiffusionStep
from unirl.models.ltx2.conditions import LTX2Conditions
from unirl.models.ltx2.diffusion import LTX2DiffusionStage
from unirl.models.types.batched_replay import BatchedStepReplayMixin
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.diffusion import WAN21DiffusionStage, WAN21DiffusionStep
from unirl.models.wan22.diffusion import WAN22DiffusionStage, WAN22DiffusionStep
from unirl.models.wan22_v2v.pipeline import WAN22V2VPipeline
from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition, TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

BATCH = 2
STEPS = 4
CHANNELS = 4
SIGMAS = torch.tensor([0.9, 0.7, 0.4, 0.1], dtype=torch.float32)


def _text(batch_size: int = BATCH) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=torch.arange(batch_size * 6 * 8, dtype=torch.float32).reshape(batch_size, 6, 8) / 100,
        attn_mask=torch.ones(batch_size, 6, dtype=torch.long),
    )


def _segment(
    shape: tuple[int, ...],
    *,
    sde_indices: tuple[int, ...] = (2, 0),
    aux_shape: tuple[int, ...] | None = None,
) -> LatentSegment:
    generator = torch.Generator().manual_seed(123)
    latents = torch.randn(BATCH, STEPS, *shape, generator=generator).requires_grad_(True)
    aux = (
        torch.randn(BATCH, STEPS, *aux_shape, generator=generator).requires_grad_(True)
        if aux_shape is not None
        else None
    )
    return LatentSegment(
        latents=latents,
        aux_latents=aux,
        sigmas=SIGMAS,
        indices=torch.arange(STEPS),
        sde_indices=torch.tensor(sde_indices),
    )


def _clone_segment(segment: LatentSegment) -> LatentSegment:
    return LatentSegment(
        latents=segment.latents.detach().clone().requires_grad_(True),
        aux_latents=(
            segment.aux_latents.detach().clone().requires_grad_(True) if segment.aux_latents is not None else None
        ),
        sigmas=segment.sigmas.clone(),
        indices=segment.indices.clone(),
        sde_indices=segment.sde_indices.clone(),
    )


class _VideoTransformer(torch.nn.Module):
    def __init__(self, *, output_channels: int = CHANNELS) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.output_channels = int(output_channels)
        self.forward_batches: list[int] = []

    def forward(self, **kwargs):
        hidden = kwargs["hidden_states"][:, : self.output_channels]
        timestep = kwargs["timestep"].to(hidden).reshape(-1, 1, 1, 1, 1) / 1000
        self.forward_batches.append(int(hidden.shape[0]))
        return (hidden * self.scale + timestep,)


class _LTXTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.forward_batches: list[int] = []

    def forward(self, **kwargs):
        video = kwargs["hidden_states"]
        audio = kwargs["audio_hidden_states"]
        timestep = kwargs["timestep"].to(video).reshape(-1, 1, 1) / 1000
        self.forward_batches.append(int(video.shape[0]))
        video_context = audio.mean(dim=(1, 2), keepdim=True)
        audio_context = video.mean(dim=(1, 2), keepdim=True)
        return (
            video * self.scale + timestep + 0.01 * video_context,
            audio * self.scale + timestep + 0.01 * audio_context,
        )


class _DualTransformer(_VideoTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.routes: list[tuple[bool, int]] = []

    def forward(self, *, use_high_noise: bool, **kwargs):
        self.routes.append((bool(use_high_noise), int(kwargs["hidden_states"].shape[0])))
        return super().forward(**kwargs)


class _GenericStep:
    def step_with_logp(
        self,
        model,
        conditions,
        *,
        strategy,
        sample,
        sigma,
        sigma_next,
        guidance_scale,
        prev_sample,
        eta,
        sigma_max,
        step_index,
    ):
        del conditions, guidance_scale
        noise_pred = model.transformer(hidden_states=sample, timestep=sigma * 1000)[0]
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )


class _GenericBatchedStage(BatchedStepReplayMixin):
    def __init__(self) -> None:
        self.model = SimpleNamespace(transformer=_VideoTransformer())
        self.step = _GenericStep()
        self.strategy = FlowSDEStrategy()
        self.logprob_dtype = torch.float32
        self.trajectory_dtype = torch.float32

    @staticmethod
    def _tile_conditions(conditions: WAN21Conditions, repeats: int) -> WAN21Conditions:
        return WAN21Conditions.concat([conditions] * repeats)


def _params(*, guidance_scale: float = 1.0, num_frames: int = 5) -> DiffusionSamplingParams:
    return DiffusionSamplingParams(
        num_inference_steps=3,
        guidance_scale=guidance_scale,
        eta=0.7,
        height=64,
        width=64,
        num_frames=num_frames,
    )


def _assert_grouped_matches_serial(
    stage_factory: Callable[[bool], object],
    conditions,
    *,
    segment: LatentSegment,
    params: DiffusionSamplingParams,
    target: list[int],
) -> None:
    serial_segment = segment
    grouped_segment = _clone_segment(segment)
    serial = stage_factory(False)
    grouped = stage_factory(True)

    serial_result = serial.replay(conditions, segment=serial_segment, params=params, step_indices=target)
    grouped_result = grouped.replay(conditions, segment=grouped_segment, params=params, step_indices=target)
    serial_result.log_probs.sum().backward()
    grouped_result.log_probs.sum().backward()

    torch.testing.assert_close(grouped_result.log_probs, serial_result.log_probs)
    torch.testing.assert_close(grouped_result.prev_sample_means, serial_result.prev_sample_means)
    torch.testing.assert_close(grouped_segment.latents.grad, serial_segment.latents.grad)
    torch.testing.assert_close(
        grouped.trainable_module().scale.grad,
        serial.trainable_module().scale.grad,
        rtol=5e-4,
        atol=2e-3,
    )
    if serial_segment.aux_latents is not None:
        torch.testing.assert_close(grouped_segment.aux_latents.grad, serial_segment.aux_latents.grad)


@pytest.mark.parametrize("family", ["hunyuan_video10", "hunyuan_video15", "wan21"])
def test_dense_video_grouped_replay_matches_serial_forward_and_backward(family: str) -> None:
    text = _text()
    if family == "hunyuan_video10":
        conditions = HunyuanVideo10Conditions(
            text_llama=text,
            pooled_clip=TextEmbedCondition(embeds=torch.ones(BATCH, 8)),
        )

        def stage_factory(grouped: bool):
            bundle = SimpleNamespace(
                transformer=_VideoTransformer(),
                vae=SimpleNamespace(config=SimpleNamespace(latent_channels=CHANNELS)),
                device=torch.device("cpu"),
            )
            return HunyuanVideo10DiffusionStage(
                model=bundle,
                step=HunyuanVideo10DiffusionStep(),
                strategy=FlowSDEStrategy(),
                batch_replay_steps=grouped,
            )

    elif family == "hunyuan_video15":
        conditions = HunyuanVideo15Conditions(text_mllm=text, text_glyph=text)

        def stage_factory(grouped: bool):
            bundle = SimpleNamespace(
                transformer=_VideoTransformer(),
                vae=SimpleNamespace(config=SimpleNamespace(latent_channels=CHANNELS)),
                device=torch.device("cpu"),
            )
            return HunyuanVideo15DiffusionStage(
                model=bundle,
                step=HunyuanVideo15DiffusionStep(),
                strategy=FlowSDEStrategy(),
                batch_replay_steps=grouped,
                vision_num_semantic_tokens=4,
                vision_states_dim=8,
            )

    else:
        conditions = WAN21Conditions(
            text=text,
            image_latent=ImageLatentCondition(latents=torch.ones(BATCH, 2, 2, 2, 2)),
            image_embed=ImageEmbedCondition(embeds=torch.ones(BATCH, 3, 8)),
        )

        def stage_factory(grouped: bool):
            bundle = SimpleNamespace(
                transformer=_VideoTransformer(),
                vae=SimpleNamespace(config=SimpleNamespace(z_dim=CHANNELS)),
                device=torch.device("cpu"),
            )
            return WAN21DiffusionStage(
                model=bundle,
                step=WAN21DiffusionStep(),
                strategy=FlowSDEStrategy(),
                batch_replay_steps=grouped,
            )

    _assert_grouped_matches_serial(
        stage_factory,
        conditions,
        segment=_segment((CHANNELS, 2, 2, 2)),
        params=_params(),
        target=[2, 0],
    )


def test_existing_mixin_batches_all_targets_in_one_forward() -> None:
    stage = _GenericBatchedStage()
    segment = _segment((CHANNELS, 2, 2, 2), sde_indices=(0, 1, 2))
    result = stage._replay_batched_steps(
        WAN21Conditions(text=_text()),
        segment=segment,
        params=_params(),
        target=[0, 1, 2],
        sigmas=SIGMAS,
        sigma_max=SIGMAS[1],
        device=torch.device("cpu"),
    )
    assert stage.model.transformer.forward_batches == [BATCH * 3]
    assert tuple(result.log_probs.shape) == (BATCH, 3)


def test_wan21_grouped_replay_preserves_cfg_batch_order() -> None:
    text = _text()
    conditions = WAN21Conditions(text=text, negative_text=TextEmbedCondition(embeds=-text.embeds))

    def stage_factory(grouped: bool):
        bundle = SimpleNamespace(
            transformer=_VideoTransformer(),
            vae=SimpleNamespace(config=SimpleNamespace(z_dim=CHANNELS)),
            device=torch.device("cpu"),
        )
        return WAN21DiffusionStage(
            model=bundle,
            step=WAN21DiffusionStep(),
            strategy=FlowSDEStrategy(),
            batch_replay_steps=grouped,
        )

    _assert_grouped_matches_serial(
        stage_factory,
        conditions,
        segment=_segment((CHANNELS, 2, 2, 2)),
        params=_params(guidance_scale=2.0),
        target=[2, 0],
    )


def test_hunyuan_video15_grouped_replay_preserves_cfg_batch_order() -> None:
    text = _text()
    conditions = HunyuanVideo15Conditions(
        text_mllm=text,
        text_glyph=text,
        negative_text_mllm=TextEmbedCondition(embeds=-text.embeds, attn_mask=text.attn_mask),
        negative_text_glyph=TextEmbedCondition(embeds=-text.embeds, attn_mask=text.attn_mask),
    )

    def stage_factory(grouped: bool):
        bundle = SimpleNamespace(
            transformer=_VideoTransformer(),
            vae=SimpleNamespace(config=SimpleNamespace(latent_channels=CHANNELS)),
            device=torch.device("cpu"),
        )
        return HunyuanVideo15DiffusionStage(
            model=bundle,
            step=HunyuanVideo15DiffusionStep(),
            strategy=FlowSDEStrategy(),
            batch_replay_steps=grouped,
            vision_num_semantic_tokens=4,
            vision_states_dim=8,
        )

    _assert_grouped_matches_serial(
        stage_factory,
        conditions,
        segment=_segment((CHANNELS, 2, 2, 2)),
        params=_params(guidance_scale=2.0),
        target=[2, 0],
    )


def test_wan22_splits_grouped_replay_at_expert_boundary() -> None:
    conditions = WAN21Conditions(text=_text())
    route_sigmas = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.1], dtype=torch.float32)

    def stage_factory(grouped: bool):
        bundle = SimpleNamespace(
            transformer=_DualTransformer(),
            vae=SimpleNamespace(config=SimpleNamespace(z_dim=CHANNELS)),
            device=torch.device("cpu"),
            boundary_ratio=0.5,
            guidance_scale_2=None,
        )
        return WAN22DiffusionStage(
            model=bundle,
            step=WAN22DiffusionStep(),
            strategy=FlowSDEStrategy(),
            batch_replay_steps=grouped,
        )

    segment = LatentSegment(
        latents=torch.randn(BATCH, 5, CHANNELS, 2, 2, 2, generator=torch.Generator().manual_seed(321)).requires_grad_(
            True
        ),
        sigmas=route_sigmas,
        indices=torch.arange(5),
        sde_indices=torch.tensor([0, 1, 2, 3]),
    )
    serial_segment = segment
    grouped_segment = _clone_segment(segment)
    serial = stage_factory(False)
    grouped = stage_factory(True)
    target = [0, 1, 2, 3]
    params = DiffusionSamplingParams(num_inference_steps=4, guidance_scale=1.0, eta=0.7)
    serial_result = serial.replay(conditions, segment=serial_segment, params=params, step_indices=target)
    grouped_result = grouped.replay(conditions, segment=grouped_segment, params=params, step_indices=target)

    torch.testing.assert_close(grouped_result.log_probs, serial_result.log_probs)
    torch.testing.assert_close(grouped_result.prev_sample_means, serial_result.prev_sample_means)
    assert grouped.model.transformer.routes == [(True, BATCH * 2), (False, BATCH * 2)]


def test_wan22_v2v_trimmed_schedule_keeps_groupable_step_indices() -> None:
    remapped = WAN22V2VPipeline._sde_indices_in_trimmed_frame(
        [0, 2, 5, 9],
        t_full=10,
        t_eff=4,
    )
    assert remapped == [0, 1, 2, 3]


@pytest.mark.parametrize(("audio_joint_sde", "guidance_scale"), [(True, 1.0), (False, 1.0), (True, 2.0)])
def test_ltx2_grouped_replay_matches_serial_video_audio_backward(
    audio_joint_sde: bool,
    guidance_scale: float,
) -> None:
    text = _text()
    conditions = LTX2Conditions(
        text=text,
        negative_text=(
            TextEmbedCondition(embeds=-text.embeds, attn_mask=text.attn_mask) if guidance_scale > 1.0 else None
        ),
    )

    def stage_factory(grouped: bool):
        bundle = SimpleNamespace(transformer=_LTXTransformer(), has_audio=True)
        return LTX2DiffusionStage(
            bundle,
            strategy=FlowSDEStrategy(),
            audio_joint_sde=audio_joint_sde,
            batch_replay_steps=grouped,
        )

    _assert_grouped_matches_serial(
        stage_factory,
        conditions,
        segment=_segment((12, CHANNELS), aux_shape=(5, CHANNELS)),
        params=_params(guidance_scale=guidance_scale, num_frames=9),
        target=[2, 0],
    )


@pytest.mark.parametrize(
    ("sample_ndim", "expected_shape"),
    [
        (6, (1, 2, 1, 1, 1, 1)),
        (4, (1, 2, 1, 1)),
    ],
)
def test_transition_sigma_broadcasts_to_video_and_packed_av_means(
    sample_ndim: int,
    expected_shape: tuple[int, ...],
) -> None:
    stage = SimpleNamespace(strategy=FlowSDEStrategy())
    segment = _segment((CHANNELS, 2, 2, 2))
    sigma = _transition_sigma(
        stage,
        segment=segment,
        target_steps=[2, 0],
        eta=0.7,
        device=torch.device("cpu"),
        sample_ndim=sample_ndim,
    )
    assert tuple(sigma.shape) == expected_shape


def test_grouped_video_replay_requires_replay_anchor() -> None:
    stage = SimpleNamespace(batch_replay_steps=True)
    with pytest.raises(ValueError, match="old_logp_source='replay'"):
        _require_replay_anchor_for_batched_replay(stage, "rollout", algo="FlowGRPO")
    _require_replay_anchor_for_batched_replay(stage, "replay", algo="FlowGRPO")
    experimental = SimpleNamespace(batch_replay_steps=True, allow_batched_rollout_anchor=True)
    _require_replay_anchor_for_batched_replay(experimental, "rollout", algo="FlowGRPO")
