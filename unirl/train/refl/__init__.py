"""ReFL (direct differentiable-reward backprop) training-side roles.

``ReFLPolicy`` is the worker-side ``Remote`` that fuses FSDP-wrapped SD3 sampling
+ VAE decode (grad-enabled, DRaFT-K) with the optimizer, so a frozen
differentiable reward on a sibling role can backprop end-to-end onto the policy
weights via the distributed ``enable_grad()`` context. The driver orchestrator
lives in ``unirl/trainer/refl.py``.
"""

from unirl.train.refl.policy import ReFLPolicy

__all__ = ["ReFLPolicy"]
