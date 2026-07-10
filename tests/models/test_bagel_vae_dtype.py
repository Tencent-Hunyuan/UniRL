from types import SimpleNamespace

import torch
from torch import nn

from unirl.models.bagel.pipeline import _VaeEncodeDtypeAdapter
from unirl.models.bagel.vae import BagelVAEDecodeStage
from unirl.types.segments.latent import LatentSegment


class _EncodeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.bfloat16), requires_grad=False)
        self.input_dtype = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.input_dtype = x.dtype
        return x * self.scale


class _DecodeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.repeat(1, 3, 1, 1) * self.scale


def test_encode_adapter_tracks_live_vae_and_consumer_dtypes() -> None:
    vae = _EncodeVae()
    consumer = nn.Linear(1, 1, bias=False).to(dtype=torch.float32)
    adapter = _VaeEncodeDtypeAdapter(vae, consumer)
    x = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    out = adapter.encode(x)
    assert vae.input_dtype == torch.bfloat16
    assert out.dtype == torch.float32

    vae.to(dtype=torch.float32)
    consumer.to(dtype=torch.bfloat16)
    out = adapter.encode(x)
    assert vae.input_dtype == torch.float32
    assert out.dtype == torch.bfloat16


def test_checkpoint_decode_keeps_vae_fp32_after_backward() -> None:
    vae = _DecodeVae()
    bundle = SimpleNamespace(vae=vae, latent_patch_size=1, latent_channels=1, latent_downsample=1)
    stage = BagelVAEDecodeStage(bundle, decode_batch_size=1)
    latents = torch.randn(1, 1, 4, 1, dtype=torch.float32, requires_grad=True)
    segment = LatentSegment(latents=latents)

    decoded = stage.decode(segment, image_shape=(2, 2), grad=True, activation_checkpoint=True)
    decoded.pixels.mean().backward()

    assert next(vae.parameters()).dtype == torch.float32
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
