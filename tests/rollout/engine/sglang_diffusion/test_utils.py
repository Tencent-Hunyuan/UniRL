"""Pure tests for the ``sglang_diffusion`` utils — canned data, no SGLang, no GPU.

These cover the model-agnostic mechanics the adapter conversion methods call:
prompt de-expansion, encoder-output fusion, media decode, trajectory→segment
assembly (incl. selective trim + native log-probs), decoded stacking, and text
condition fusion.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.types.primitives import Images
from unirl.types.segments.latent import make_image_segment, make_video_segment
from unirl.types.trajectory_store import compute_trajectory_positions


def _result(**kw):
    """A duck-typed stand-in for SGLang's GenerationResult."""
    base = dict(
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        samples=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        encoder_attention_mask=None,
        negative_prompt_embeds=None,
        neg_pooled_prompt_embeds=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def test_deexpand_collapses_uniform_groups():
    prompts = ["a", "a", "b", "b"]
    gids = ["g0", "g0", "g1", "g1"]
    assert utils.deexpand_prompts_from_groups(prompts, gids) == (["a", "b"], 2)


@pytest.mark.parametrize(
    "prompts,gids",
    [
        (["a", "a", "b"], ["g0", "g0", "g1"]),       # heterogeneous K
        (["a", "x", "b", "b"], ["g0", "g0", "g1", "g1"]),  # mismatched within group
        (["a", "b"], ["g0", "g1"]),                  # K == 1
    ],
)
def test_deexpand_falls_through(prompts, gids):
    out, k = utils.deexpand_prompts_from_groups(prompts, gids)
    assert (out, k) == (prompts, 1)


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #


def test_fuse_encoder_outputs_token_level_cats_on_seq():
    a = torch.zeros(2, 3, 8)
    b = torch.zeros(2, 5, 8)
    fused = utils.fuse_encoder_outputs([a, b])
    assert fused.shape == (2, 8, 8)  # seq 3 + 5


def test_fuse_encoder_outputs_pooled_cats_on_hidden():
    a = torch.zeros(2, 8)
    b = torch.zeros(2, 4)
    fused = utils.fuse_encoder_outputs([a, b])
    assert fused.shape == (2, 12)


def test_fuse_encoder_outputs_none_and_single():
    assert utils.fuse_encoder_outputs(None) is None
    t = torch.zeros(2, 3, 8)
    assert utils.fuse_encoder_outputs(t) is t


def test_decode_sample_hwc_to_chw_and_clamp():
    # H, W deliberately not in {1,3,4}: otherwise the channels-first heuristic
    # (inherited from the old engine) treats shape[0] as the channel dim.
    sample = torch.full((5, 6, 3), 2.0)  # [H, W, C], out of [0,1]
    canonical = utils.decode_sample(sample)
    assert canonical.shape == (3, 5, 6)
    assert float(canonical.max()) <= 1.0


def test_decode_sample_unwraps_video_audio_tuple():
    video = torch.zeros(2, 5, 6, 3)  # [T, H, W, C], H/W not channel-ambiguous
    canonical = utils.decode_sample((video, torch.zeros(1)))
    assert canonical.shape == (3, 2, 5, 6)  # [C, T, H, W]


# --------------------------------------------------------------------------- #
# tracks: segment assembly
# --------------------------------------------------------------------------- #


def _traj(b, tp1, c=2, h=4, w=4):
    return torch.randn(b, tp1, c, h, w)


def test_build_latent_segment_basic_image():
    sigmas = torch.tensor([1.0, 0.5, 0.0])  # T=2, T+1=3
    traj = _traj(2, 3)
    res = [_result(trajectory_latents=traj, trajectory_timesteps=sigmas)]
    seg = utils.build_latent_segment(
        traj,
        results=res,
        expected_sigmas=sigmas,
        num_steps=2,
        sde_indices=None,
        use_native_logprob=False,
    )
    assert seg.latents.shape == (2, 3, 2, 4, 4)
    assert torch.equal(seg.sigmas, sigmas)
    assert seg.sde_logp is None
    assert seg.modality == make_image_segment().modality  # IMAGE by default


def test_build_latent_segment_selective_trim():
    num_steps = 4
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)  # T+1 = 5
    traj = _traj(1, num_steps + 1)
    res = [_result(trajectory_latents=traj, trajectory_timesteps=sigmas)]
    sde_indices = [1]
    seg = utils.build_latent_segment(
        traj,
        results=res,
        expected_sigmas=sigmas,
        num_steps=num_steps,
        sde_indices=sde_indices,
        use_native_logprob=False,
    )
    expected_keep = sorted(set(compute_trajectory_positions({1}, num_steps)) | {num_steps})
    expected_keep = [p for p in expected_keep if 0 <= p < num_steps + 1]
    assert seg.latents.shape[1] == len(expected_keep)
    assert seg.indices.tolist() == expected_keep
    # terminal clean latent always preserved
    assert num_steps in seg.indices.tolist()


def test_build_latent_segment_native_logprob():
    sigmas = torch.linspace(1.0, 0.0, 5)
    traj = _traj(2, 5)
    logp = torch.randn(2, 1)  # [B, len(sde_indices)]
    res = [_result(trajectory_latents=traj, trajectory_timesteps=sigmas, trajectory_log_probs=logp)]
    seg = utils.build_latent_segment(
        traj,
        results=res,
        expected_sigmas=sigmas,
        num_steps=4,
        sde_indices=[1],
        use_native_logprob=True,
    )
    assert seg.sde_logp is not None and seg.sde_logp.shape == (2, 1)


def test_build_latent_segment_native_logprob_slices_full_schedule():
    num_steps = 4
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    traj = _traj(1, num_steps + 1)
    logp_full = torch.arange(num_steps, dtype=torch.float32).reshape(1, num_steps)
    res = [_result(trajectory_latents=traj, trajectory_timesteps=sigmas, trajectory_log_probs=logp_full)]
    seg = utils.build_latent_segment(
        traj,
        results=res,
        expected_sigmas=sigmas,
        num_steps=num_steps,
        sde_indices=[2],
        use_native_logprob=True,
    )
    assert seg.sde_logp.shape == (1, 1)
    assert float(seg.sde_logp[0, 0]) == 2.0  # column 2 selected


def test_derive_timestep_alignment_rejects_length_mismatch():
    sigmas = torch.tensor([1.0, 0.5, 0.0])  # len 3
    traj = _traj(1, 2)  # T+1 = 2, mismatched
    res = [_result(trajectory_latents=traj, trajectory_timesteps=sigmas[:2])]
    with pytest.raises(Exception, match="trajectory length"):
        utils.derive_timestep_alignment(
            trajectories_tensor=traj, expected_sigmas=sigmas, results=res
        )


# --------------------------------------------------------------------------- #
# tracks: decoded + conditions
# --------------------------------------------------------------------------- #


def test_stack_decoded_images_packs_chw():
    res = [
        _result(samples=torch.rand(3, 4, 4)),
        _result(samples=torch.rand(3, 4, 4)),
    ]
    out = utils.stack_decoded_images(res)
    assert isinstance(out, Images)
    assert out.pixels.shape == (2, 3, 4, 4)


def test_stack_decoded_images_drops_video(caplog):
    res = [_result(samples=torch.rand(3, 2, 4, 4))]  # [C, T, H, W] video
    out = utils.stack_decoded_images(res)
    assert out is None  # dropped, nothing image-shaped


def test_fuse_text_conditions_text_and_negative():
    res = [
        _result(
            prompt_embeds=torch.zeros(1, 3, 8),
            pooled_prompt_embeds=torch.zeros(1, 8),
            negative_prompt_embeds=torch.zeros(1, 3, 8),
        )
    ]
    text, neg = utils.fuse_text_conditions(res)
    assert text is not None and text.embeds.shape == (1, 3, 8)
    assert neg is not None and neg.embeds.shape == (1, 3, 8)


def test_fuse_text_conditions_requires_prompt_embeds():
    with pytest.raises(Exception, match="missing prompt_embeds"):
        utils.fuse_text_conditions([_result()])
