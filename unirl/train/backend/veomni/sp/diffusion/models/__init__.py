"""Per-model SP boundary adapters.

Importing this subpackage registers each model's ``_wrap_*`` into ``FORWARD_WRAPPERS``
via the ``@register`` decorator in :mod:`unirl.train.backend.veomni.sp.diffusion.ulysses`.

Add a model:
  1. Create ``models/<name>.py`` with a ``@register("<DiffusersClassName>")``-decorated
     ``_wrap_<name>(model, sp_group)``. Use ``_install_boundary_hooks`` for a block-level
     boundary (slice at ``blocks[0]``, gather at ``norm_out`` -- see qwen_image/sd3/wan),
     or register model-level hooks inline for a non-block layout (see flux2).
  2. Import it below so registration fires on package import.
  3. Add an ``sp_<name>_parity.py`` test (sp=1 vs sp=2 output parity).
"""

from unirl.train.backend.veomni.sp.diffusion.models import (  # noqa: F401 -- import-for-registration
    flux2,
    qwen_image,
    sd3,
    wan,
)
