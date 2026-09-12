"""Differentiable scoring through a managed child (segmented autograd)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import torch

from unirl.reward.tensor_ipc import decode_tensor, encode_tensor

if TYPE_CHECKING:
    from unirl.reward.managed_process import ManagedScorerProcessBackend


class _ManagedScore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, images: torch.Tensor, backend, prompts: list[str]) -> torch.Tensor:
        if getattr(backend, "_transport", "http") != "cuda_ipc":
            raise RuntimeError(
                "differentiable scoring requires the cuda_ipc data plane; "
                "set process.transport='cuda_ipc' on the managed spec"
            )
        if not images.is_cuda:
            raise RuntimeError("differentiable scoring requires CUDA image tensors")
        call_id = uuid.uuid4().hex
        scorer_name = backend.spec.scorer.name
        # detach+contiguous gives the wire a stable storage to share; the child
        # copies out of it (torch.stack) while this request is in flight.
        shipped = images.detach().contiguous()
        payload = {
            "protocol_version": "1",
            "grad_mode": True,
            "call_id": call_id,
            "requests": [
                {
                    "history": [{"text": prompt, "image_ipc": encode_tensor(shipped[i])}],
                    "required_rewards": [scorer_name],
                }
                for i, prompt in enumerate(prompts)
            ],
        }
        response = backend._session.post(f"{backend.base_url}/score", json=payload, timeout=600.0)
        response.raise_for_status()
        body = response.json()
        for index, errs in enumerate(body.get("errors", [])):
            if errs:
                raise RuntimeError(f"differentiable score failed for item {index}: {errs}")
        results = body["results"]
        metric = next(iter(results[0][scorer_name]))
        scores = torch.tensor(
            [results[i][scorer_name][metric] for i in range(len(prompts))],
            device=images.device,
            dtype=torch.float32,
        )
        ctx.unirl_backend = backend
        ctx.unirl_call_id = call_id
        ctx.unirl_device = images.device
        ctx.unirl_shape = tuple(images.shape)
        return scores

    @staticmethod
    def backward(ctx, grad_scores: torch.Tensor):
        backend = ctx.unirl_backend
        response = backend._session.post(
            f"{backend.base_url}/backward",
            json={
                "call_id": ctx.unirl_call_id,
                "grad_scores": [float(v) for v in grad_scores.detach().cpu()],
            },
            timeout=600.0,
        )
        response.raise_for_status()
        blobs = response.json()["grad_ipc"]
        # Clone immediately: the child only pins the shared gradient storage
        # until its next backward (bounded keepalive), so materialize now.
        grads = torch.stack([decode_tensor(blob).clone() for blob in blobs]).to(ctx.unirl_device)
        if tuple(grads.shape) != ctx.unirl_shape:
            raise RuntimeError(
                f"child returned image grads of shape {tuple(grads.shape)}, expected {ctx.unirl_shape}"
            )
        return grads, None, None


def score_differentiable(
    backend: "ManagedScorerProcessBackend",
    images: torch.Tensor,
    prompts: list[str],
) -> torch.Tensor:
    """Score ``images`` [B,C,H,W] through the managed child with grad attached."""
    return _ManagedScore.apply(images, backend, prompts)
