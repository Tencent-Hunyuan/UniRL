"""Parity harness: UniRL SD3 SFT flow-matching loss  vs  diffusers-standard formula.

No checkpoints, no external images, CPU — a deterministic numerical check that
SD3SFTTask's flow-matching math (x_t = (1-σ)x0 + σ·noise; target v = noise - x0;
loss = MSE(pred, v)) matches the reference convention used across flow-matching
diffusion trainers (and encoded in FlowSDEStrategy.step's velocity drift).

We stub the transformer (predict_noise) with a known output and drive
SD3SFTTask.compute_loss over random latents, then recompute the loss the
"textbook" way and assert equality. This proves the target convention
(v = noise - x0, NOT noise) and the interpolation are correct against the
reference — the diffusion analogue of the qwen3 CE parity check.
"""

import sys
import types

import torch

from unirl.models.sd3.sft_task import SD3SFTTask


def build_task(x0, predict_fn):
    task = SD3SFTTask.__new__(SD3SFTTask)
    task.bundle = types.SimpleNamespace(
        transformer=types.SimpleNamespace(train=lambda: None, eval=lambda: None),
        device=torch.device("cpu"),
    )
    task.config = types.SimpleNamespace(shift=3.0)
    task.text_embed = types.SimpleNamespace(
        embed=lambda texts: types.SimpleNamespace(embeds=torch.zeros(1, 4, 8), pooled=torch.zeros(1, 8))
    )
    task.diffusion = types.SimpleNamespace(predict_noise=predict_fn)
    task.shift = 3.0
    task.autocast_dtype = torch.bfloat16
    task._encode_image = lambda pixels: x0
    return task


def main():
    torch.manual_seed(0)
    diffs = []

    # Case 1: predict = exact target (noise - x0) -> loss must be ~0.
    for trial in range(5):
        x0 = torch.randn(1, 16, 8, 8)
        captured = {}

        def predict(bundle, x_t, sigma, conditions, *, guidance_scale):
            s = sigma.view(1, 1, 1, 1)
            noise = (x_t - (1.0 - s) * x0) / s  # invert x_t = (1-s)x0 + s*noise
            captured["v"] = noise - x0
            captured["gs"] = guidance_scale
            # diffusers-standard flow-matching target IS (noise - x0); return it.
            return noise - x0

        task = build_task(x0, predict)
        loss, m = task.compute_loss({"pixels": torch.zeros(3, 8, 8), "prompt": "p"})
        assert captured["gs"] == 1.0, "SFT must use guidance_scale=1.0"
        diffs.append(float(loss))  # should be ~0
    zero_max = max(diffs)
    print(f"exact-target loss max = {zero_max:.2e}  (expect ~0)")

    # Case 2: predict = zeros -> our loss must equal textbook MSE(0, noise-x0).
    ref_diffs = []
    for trial in range(5):
        x0 = torch.randn(1, 16, 8, 8)
        holder = {}

        def predict(bundle, x_t, sigma, conditions, *, guidance_scale):
            s = sigma.view(1, 1, 1, 1)
            noise = (x_t - (1.0 - s) * x0) / s
            holder["target"] = noise - x0
            return torch.zeros_like(x_t)

        task = build_task(x0, predict)
        loss, _ = task.compute_loss({"pixels": torch.zeros(3, 8, 8), "prompt": "p"})
        ref = (holder["target"] ** 2).mean()  # textbook MSE against v = noise - x0
        ref_diffs.append(abs(float(loss) - float(ref)))
    ref_max = max(ref_diffs)
    print(f"vs textbook MSE(v=noise-x0) max |Δ| = {ref_max:.2e}")

    ok = zero_max < 1e-6 and ref_max < 1e-6
    print("SD3 FLOW PARITY:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
