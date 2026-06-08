"""HPSv3 scorer backed by the official `hpsv3` pip package.

HPSv3RewardInferencer.reward returns one (mu, sigma) tensor per image;
we expose mu as the `hpsv3` sub-metric. The underlying
`qwen_vl_utils.fetch_image` accepts PIL.Image objects directly (see
that module's ``fetch_image`` L101) so we pass images in-memory — no
JPEG round-trip to a tempfile.

Activation memory of the Qwen2-VL-7B backbone grows with batch size,
so large batches easily overrun a 95 GB H20. ``score`` chunks incoming
items into ``max_batch_size`` slices, calls the inferencer once per
slice, and reclaims activation memory between slices with
``torch.cuda.empty_cache()``. The per-call batch is the only knob that
bounds peak activation — raise only as memory allows.

hpsv3 must be installed from the upstream ``upgrade_transformers_version``
branch — see install.sh for the pinned commit. The PyPI 1.0.0 wheel is
incompatible with transformers ≥4.50.

Local checkpoint layout (official HF release):
  <weights_dir>/HPSv3.safetensors   (+ optional YAML under <weights_dir>/config/)
"""

from __future__ import annotations

from pathlib import Path

import torch

from reward_service.scorers._common import split_last_turn
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


class HPSv3Scorer(BaseScorer):
    name = "hpsv3"
    sub_metric_names = ("hpsv3",)

    def __init__(
        self,
        weights_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda",
        max_batch_size: int = 4,
    ) -> None:
        """Load the HPSv3 reward model.

        Args:
            weights_path: Directory containing ``HPSv3.safetensors`` (or the
                file itself). ``None`` falls back to the hpsv3 package's
                default HF download path.
            config_path: Optional override for the HPSv3 config YAML.
                ``None`` uses the one bundled with the hpsv3 package.
            device: Target device (``"cuda"`` / ``"cpu"``).
            max_batch_size: Maximum samples per inferencer forward. Incoming
                items are chunked into slices of this size so peak
                activation memory stays bounded (Qwen2-VL-7B on a 95 GB H20
                OOMs somewhere around batch=16 at image_token=1024). Must
                be positive.
        """
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.max_batch_size = max_batch_size
        # Cached so the per-chunk hot path doesn't re-query the CUDA runtime
        # on every score() call.
        self._cuda_available = torch.cuda.is_available()

        from hpsv3 import HPSv3RewardInferencer

        kwargs: dict = {"device": device}
        if weights_path:
            p = Path(weights_path)
            if p.is_dir():
                ckpt = p / "HPSv3.safetensors"
                if not ckpt.exists():
                    raise FileNotFoundError(f"HPSv3.safetensors not found under {p}")
                kwargs["checkpoint_path"] = str(ckpt)
            else:
                kwargs["checkpoint_path"] = str(p)
        if config_path:
            kwargs["config_path"] = config_path
        self._inferencer = HPSv3RewardInferencer(**kwargs)

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []
        # The inferencer's "image_paths" kwarg is a misnomer: its upstream
        # fetch_image accepts PIL.Image directly, so we skip the JPEG
        # tempfile dance.
        prompts, images = split_last_turn(items)

        out: list[dict[str, float]] = []
        for start in range(0, len(items), self.max_batch_size):
            stop = start + self.max_batch_size
            rewards = self._inferencer.reward(
                image_paths=images[start:stop],
                prompts=prompts[start:stop],
            )
            for r in rewards:
                mu = r[0].item() if hasattr(r[0], "item") else float(r[0])
                out.append({"hpsv3": float(mu)})
            # Release the activation buffers between slices so the next
            # forward starts from the weights-only footprint.
            if self._cuda_available:
                torch.cuda.empty_cache()
        return out


register("hpsv3", HPSv3Scorer)
