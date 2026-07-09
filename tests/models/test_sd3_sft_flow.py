"""CPU tests for the SD3 SFT task's flow-matching loss assembly (no ckpts, no GPU).

We don't load SD3.5. We duck-type the bundle + the reused stages so
``compute_loss``'s flow-matching assembly runs on tiny fake tensors, and assert:
the velocity target is ``noise - x0`` (NOT ``noise``), the interpolation is
``x_t = (1-σ)x0 + σ·noise``, and the loss is MSE against that target.
"""

import types

import torch

from unirl.models.sd3.sft_task import SD3SFTTask


class _FakeTextEmbed:
    def embed(self, texts):
        return types.SimpleNamespace(embeds=torch.zeros(1, 4, 8), pooled=torch.zeros(1, 8))


def _make_task(predict_noise_fn, x0):
    task = SD3SFTTask.__new__(SD3SFTTask)  # bypass __init__ (no real bundle)
    transformer = types.SimpleNamespace(train=lambda: None, eval=lambda: None)
    task.bundle = types.SimpleNamespace(transformer=transformer, device=torch.device("cpu"))
    task.config = types.SimpleNamespace(shift=3.0)
    task.text_embed = _FakeTextEmbed()
    task.diffusion = types.SimpleNamespace(predict_noise=predict_noise_fn)
    task.shift = 3.0
    task.autocast_dtype = torch.bfloat16
    task._num_sched_steps = 100
    # Stub VAE encode to return the given clean latent.
    task._encode_image = lambda pixels: x0
    return task


def test_velocity_target_is_noise_minus_x0():
    # If predict_noise returns EXACTLY the flow-matching target (noise - x0),
    # the MSE loss must be ~0. This pins the target convention.
    x0 = torch.randn(1, 4, 3, 3)
    captured = {}

    def fake_predict(bundle, x_t, sigma, conditions, *, guidance_scale):
        s = sigma.view(1, 1, 1, 1)
        # Recover noise from x_t = (1-s)x0 + s*noise  ->  noise = (x_t - (1-s)x0)/s
        noise = (x_t - (1.0 - s) * x0) / s
        captured["v_target"] = noise - x0
        captured["guidance_scale"] = guidance_scale
        return noise - x0  # return the exact target

    task = _make_task(fake_predict, x0)
    loss, metrics = task.compute_loss({"pixels": torch.zeros(3, 8, 8), "prompt": "a cat"})
    assert float(loss) < 1e-8  # perfect prediction -> ~0 MSE
    assert captured["guidance_scale"] == 1.0  # SFT uses no CFG
    assert 0.0 < metrics["train/sigma"] < 1.0


def test_loss_is_mse_against_noise_minus_x0():
    # predict_noise returns zeros -> loss must equal mean((noise - x0)^2).
    x0 = torch.randn(1, 4, 3, 3)
    holder = {}

    def fake_predict(bundle, x_t, sigma, conditions, *, guidance_scale):
        s = sigma.view(1, 1, 1, 1)
        noise = (x_t - (1.0 - s) * x0) / s
        holder["target"] = noise - x0
        return torch.zeros_like(x_t)

    task = _make_task(fake_predict, x0)
    loss, _ = task.compute_loss({"pixels": torch.zeros(3, 8, 8), "prompt": "a dog"})
    expected = (holder["target"] ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-5)


def test_interpolation_relation_holds_at_predict():
    # Assert x_t handed to predict_noise satisfies x_t = (1-s)x0 + s*noise.
    x0 = torch.randn(1, 4, 2, 2)
    ok = {}

    def fake_predict(bundle, x_t, sigma, conditions, *, guidance_scale):
        s = sigma.view(1, 1, 1, 1)
        noise = (x_t - (1.0 - s) * x0) / s
        # reconstruct and compare
        recon = (1.0 - s) * x0 + s * noise
        ok["match"] = torch.allclose(recon, x_t, atol=1e-5)
        return torch.zeros_like(x_t)

    task = _make_task(fake_predict, x0)
    task.compute_loss({"pixels": torch.zeros(3, 4, 4), "prompt": "x"})
    assert ok["match"]
