"""Auto-backward through the transport: enable_grad tracking + grad flow.

Uses a real ``Handle`` (``create_remote``) so ``@distributed`` methods dispatch
and the ``RPCBackwardNode`` chain + ``_auto_backward`` (DP_SCATTER) run
end-to-end. Parametrized over gpu_store + colocate via ``multigpu_pool``.
Marker: multigpu.
"""

from __future__ import annotations

import pytest
import torch

from unirl.distributed.tensor.grad_context import current_grad_context, enable_grad

pytestmark = pytest.mark.multigpu


def _handle(pool):
    from tests.transport.harness import TProbe

    return pool.create_remote(TProbe, device_ids=list(range(pool.num_devices)))


def test_enable_grad_nesting_raises(multigpu_pool):
    with enable_grad():
        assert current_grad_context() is not None
        with pytest.raises(RuntimeError):
            with enable_grad():
                pass
    assert current_grad_context() is None


def test_forward_records_backward_node(multigpu_pool):
    h = _handle(multigpu_pool)
    inp = h.make(4, 8, 0.0)  # (8, 8) spanning both devices
    with enable_grad() as ctx:
        out = h.scale(inp)
        assert len(ctx.nodes) == 1
        node = ctx.nodes[0]
        assert inp in node.input_metas and out in node.output_metas
    # out.grad never seeded → backward is skipped cleanly; nothing accumulated.
    assert inp.grad is None


def test_backward_flows_grad(multigpu_pool):
    h = _handle(multigpu_pool)
    inp = h.make(4, 8, 0.0)
    # retain_grad() is required: GradContext.__exit__ clears .grad on every tracked
    # meta unless retained (mirrors PyTorch non-leaf grad freeing), so without this
    # inp.grad would be wiped after the auto-backward populates it.
    inp.retain_grad()
    ones = h.make_ones(4, 8)  # made OUTSIDE enable_grad → seeding is not re-tracked
    with enable_grad():
        out = h.scale(inp)  # out = inp * 3
        out.grad = ones  # dL/dout = 1
    # dL/dinp = 3 * dL/dout = 3
    assert inp.grad is not None
    g = inp.grad.local()
    assert torch.allclose(g, torch.full((8, 8), 3.0)), g
