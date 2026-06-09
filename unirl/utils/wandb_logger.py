"""
WandB Logger for unirl Training.

Provides comprehensive logging for training metrics, rollout statistics,
and image samples. Designed to match the logging behavior of DanceGRPO,
FlowGRPO, DiffusionNFT, and MixGRPO for comparison and reproducibility.

Usage:
    from unirl.utils.wandb_logger import init_logger, get_logger

    # Initialize (typically in train.py)
    logger = init_logger(project="unirl", run_name="exp1", config=args)

    # Log training metrics
    logger.log_step(step=100, metrics={"loss": 0.5, "policy_loss": 0.3})

    # Log rollout metrics
    logger.log_rollout(rollout_id=10, metrics={"reward_mean": 0.8})
"""

import os
from typing import Any, Dict, List, Optional

import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class UniRLWandBLogger:
    """WandB logger for unirl training.

    Logs metrics compatible with DanceGRPO, FlowGRPO, DiffusionNFT, and MixGRPO
    for cross-validation and comparison.

    Attributes:
        enabled: Whether logging is enabled
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        image_log_interval: How often to log images (in rollouts)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        log_dir: Optional[str] = None,
        rank: int = 0,
        image_log_interval: int = 10,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        entity: Optional[str] = None,
        require_success: bool = False,
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            run_name: WandB run name
            config: Training configuration (dict or object with __dict__)
            log_dir: WandB run directory (if provided)
            rank: Process rank (only rank 0 logs)
            image_log_interval: How often to log images (in rollouts)
            enabled: Whether to enable logging
            tags: List of tags for the WandB run. Defaults to ['unirl'] if not provided.
            entity: WandB entity (team or username). If None, uses the default entity.
            require_success: Raise immediately if WandB is unavailable or init fails.
        """
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.log_dir = str(log_dir) if log_dir else None
        self.image_log_interval = image_log_interval
        self.rank = rank
        self.tags = tags if tags is not None else ["unirl"]
        self.require_success = bool(require_success)
        self._initialized = False

        # Only enable on rank 0
        self.enabled = enabled and rank == 0

        if self.enabled and project:
            if not WANDB_AVAILABLE:
                self._handle_init_failure("wandb package is not installed but WandB reporting was requested")
                return
            self._init_wandb(config)

    @property
    def initialized(self) -> bool:
        """Whether wandb.init completed successfully."""
        return bool(self._initialized)

    def _handle_init_failure(
        self,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Disable the logger or raise immediately when strict mode is enabled."""
        self.enabled = False
        full_message = f"{message}: {exc}" if exc is not None else message
        if self.require_success:
            raise RuntimeError(full_message) from exc
        print(f"Warning: {full_message}")

    def _init_wandb(self, config: Optional[Any] = None):
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            self._handle_init_failure("wandb package is not installed but WandB reporting was requested")
            return

        try:
            # Convert config to dict if needed
            config_dict = None
            if config is not None:
                if isinstance(config, dict):
                    config_dict = config
                elif hasattr(config, "__dict__"):
                    config_dict = vars(config)

            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)

            init_kwargs = dict(
                project=self.project,
                name=self.run_name,
                config=config_dict,
                dir=self.log_dir,
                tags=self.tags,
            )
            if self.entity:
                init_kwargs["entity"] = self.entity
            wandb.init(**init_kwargs)
            self._init_metric_axes()
            self._initialized = True
        except Exception as e:
            self._handle_init_failure("Failed to initialize wandb", exc=e)

    def _init_metric_axes(self) -> None:
        """Define metric namespaces and their step axes."""
        if not WANDB_AVAILABLE:
            return
        try:
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            # rollout/step tracks the outer rollout-train loop step.
            # It behaves like a framework-level global step, but is not the same
            # thing as optimizer update count when one rollout yields multiple updates.
            wandb.define_metric("rollout/step")
            wandb.define_metric("rollout/*", step_metric="rollout/step")
            wandb.define_metric("perf/*", step_metric="rollout/step")
            wandb.define_metric("sync/*", step_metric="rollout/step")
            wandb.define_metric("buffer/*", step_metric="rollout/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
        except Exception as e:
            print(f"Warning: Failed to define wandb metrics: {e}")

    @staticmethod
    def _coerce_metric_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.numel() == 0:
                return None
            if tensor.numel() == 1:
                return float(tensor.item())
            return float(tensor.to(dtype=torch.float32).mean().item())
        return None

    @staticmethod
    def _apply_prefix(key: str, prefix: str) -> str:
        if not prefix:
            return key
        return key if key.startswith(prefix) else f"{prefix}{key}"

    def log_with_step(
        self,
        *,
        step_key: str,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Log metrics with an explicit namespace step key."""
        if not self.enabled or not self._initialized:
            return

        try:
            log_dict: Dict[str, Any] = {step_key: int(step)}
            for key, value in metrics.items():
                metric_key = self._apply_prefix(str(key), prefix)
                if metric_key == step_key:
                    continue
                scalar = self._coerce_metric_value(value)
                if scalar is None:
                    continue
                log_dict[metric_key] = scalar
            wandb.log(log_dict)
        except Exception as e:
            print(f"Warning: Failed to log metrics ({step_key}): {e}")

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train/",
    ):
        """Log per-step training metrics.

        Metrics typically include:
        - loss: Total loss
        - policy_loss: Policy gradient loss
        - kl_loss: KL divergence loss
        - approx_kl: Approximate KL divergence
        - clip_fraction: Fraction of ratios clipped
        - ratio_mean/std: Importance sampling ratio stats
        - grad_norm: Gradient norm
        - lr: Learning rate

        Args:
            step: Global step number
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names (default: "train/")
        """
        self.log_with_step(
            step_key="train/step",
            step=step,
            metrics=metrics,
            prefix=prefix,
        )

    def log_rollout(
        self,
        rollout_id: int,
        metrics: Dict[str, Any],
    ):
        """Log per-rollout metrics.

        Metrics typically include:
        - reward_mean: Mean reward across samples
        - reward_std: Reward standard deviation
        - advantage_mean: Mean advantage
        - advantage_std: Advantage standard deviation
        - num_samples: Number of samples in rollout
        - zero_std_ratio: Ratio of prompts with zero reward std

        Args:
            rollout_id: Outer rollout-train loop step. Similar to a global step for
                this framework, but not guaranteed to equal optimizer update count.
            metrics: Dictionary of metrics to log
        """
        self.log_with_step(
            step_key="rollout/step",
            step=rollout_id,
            metrics=metrics,
            prefix="rollout/",
        )

    def log_perf(
        self,
        rollout_id: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Log performance metrics keyed by rollout step."""
        self.log_with_step(
            step_key="rollout/step",
            step=rollout_id,
            metrics=metrics,
            prefix="perf/",
        )

    def log_generated_media(
        self,
        rollout_id: int,
        media_preview: Any,
        *,
        key: str = "rollout/generated_media",
        video_key: Optional[str] = None,
        video_fps: int = 8,
    ) -> None:
        """Log rollout media preview payload produced by the rollout pipeline.

        Accepts either a ``unirl.types.sample.MediaPreview`` dataclass
        (the canonical internal form) or a legacy ``{"images", "prompts",
        "rewards"}`` dict. Image-only, video-only, and image+video previews
        are all supported. Non-matching payloads are silently ignored.

        Images and videos go to *separate* wandb keys so wandb renders each
        as its native panel type. Both are logged in a single ``wandb.log``
        call sharing ``"rollout/step"`` so the panels line up on the same
        step axis. Captions ("``{prompt:.100} | reward: {r:.2f}``") are
        built once and applied to both ``wandb.Image`` and ``wandb.Video``
        so a sample's image and video panels show identical caption text.

        Args:
            rollout_id: outer loop step (shared step axis).
            media_preview: a ``MediaPreview`` dataclass or legacy dict.
            key: wandb key for the images panel.
            video_key: wandb key for the videos panel. Defaults to
                ``"rollout/generated_videos"`` when ``key`` is its default
                (``"rollout/generated_media"``); otherwise derives by
                replacing a trailing ``"_images"`` / ``"_media"`` with
                ``"_videos"``, or falls back to ``f"{key}/videos"``.
            video_fps: framerate for ``wandb.Video`` mp4 encoding.
        """
        if media_preview is None:
            return

        if isinstance(media_preview, dict):
            images = media_preview.get("images")
            videos = media_preview.get("videos")
            prompts = media_preview.get("prompts")
            rewards = media_preview.get("rewards")
        else:
            images = getattr(media_preview, "images", None)
            videos = getattr(media_preview, "videos", None)
            prompts = getattr(media_preview, "prompts", None)
            rewards = getattr(media_preview, "rewards", None)

        has_images = isinstance(images, list) and bool(images)
        has_videos = isinstance(videos, list) and bool(videos)
        if not has_images and not has_videos:
            return

        if not isinstance(prompts, list):
            prompts = []
        if not self.enabled or not self._initialized:
            return

        # Normalize rewards to a flat list[float] once, shared across panels.
        reward_values: Optional[List[float]] = None
        if rewards is not None:
            if isinstance(rewards, dict):
                rewards_extracted = rewards.get("avg", rewards.get("rewards"))
                reward_values = list(rewards_extracted) if rewards_extracted is not None else None
            elif isinstance(rewards, torch.Tensor):
                reward_values = rewards.detach().cpu().reshape(-1).tolist()
            else:
                try:
                    reward_values = [float(r) for r in rewards]
                except Exception:
                    reward_values = None

        # Resolve video_key. Common default: paired sibling under
        # "rollout/generated_videos" when key is the standard image one.
        if video_key is None:
            if key == "rollout/generated_media":
                video_key = "rollout/generated_videos"
            elif key.endswith("_images"):
                video_key = key[: -len("_images")] + "_videos"
            elif key.endswith("_media"):
                video_key = key[: -len("_media")] + "_videos"
            else:
                video_key = f"{key}/videos"

        try:
            n = max(len(images) if has_images else 0, len(videos) if has_videos else 0)

            def _caption_for(idx: int) -> str:
                prompt = str(prompts[idx]) if idx < len(prompts) else ""
                if reward_values is not None and idx < len(reward_values):
                    return f"{prompt[:100]} | reward: {reward_values[idx]:.2f}"
                return f"{prompt[:100]}"

            payload: Dict[str, Any] = {"rollout/step": int(rollout_id)}

            if has_images:
                wandb_images = [
                    wandb.Image(images[idx], caption=_caption_for(idx)) for idx in range(min(len(images), n))
                ]
                payload[key] = wandb_images

            if has_videos:
                wandb_videos: List[Any] = []
                for idx in range(min(len(videos), n)):
                    vid = videos[idx]
                    if not torch.is_tensor(vid):
                        continue
                    if vid.dim() != 4:
                        raise ValueError(
                            f"log_generated_media: video at idx {idx} must be 4D "
                            f"[C, T, H, W], got shape {tuple(vid.shape)}"
                        )
                    # ``wandb.Video`` accepts a (T, C, H, W) uint8 ndarray in
                    # [0, 255]. Our preview tensors are float [0, 1] in
                    # (C, T, H, W); permute, clamp, scale, cast.
                    arr = (
                        vid.detach()
                        .cpu()
                        .to(dtype=torch.float32)
                        .clamp(0.0, 1.0)
                        .mul(255.0)
                        .to(dtype=torch.uint8)
                        .permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
                        .numpy()
                    )
                    wandb_videos.append(wandb.Video(arr, caption=_caption_for(idx), fps=int(video_fps)))
                if wandb_videos:
                    payload[video_key] = wandb_videos

            wandb.log(payload)
        except Exception as e:
            print(f"Warning: Failed to log generated media: {e}")

    def log_eval(
        self,
        step: int,
        eval_metrics: Dict[str, Any],
    ):
        """Log evaluation metrics.

        Args:
            step: Global step number
            eval_metrics: Dictionary of evaluation metrics
        """
        self.log_with_step(
            step_key="eval/step",
            step=step,
            metrics=eval_metrics,
            prefix="eval/",
        )

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self._initialized:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")


# Global logger instance
_global_logger: Optional[UniRLWandBLogger] = None


def get_logger() -> Optional[UniRLWandBLogger]:
    """Get the global wandb logger instance."""
    return _global_logger


def set_logger(logger: UniRLWandBLogger):
    """Set the global wandb logger instance."""
    global _global_logger
    _global_logger = logger


def init_logger(
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Any] = None,
    log_dir: Optional[str] = None,
    rank: int = 0,
    tags: Optional[List[str]] = None,
    entity: Optional[str] = None,
    require_success: bool = False,
    **kwargs,
) -> UniRLWandBLogger:
    """Initialize and set the global wandb logger.

    Args:
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        rank: Process rank
        tags: List of tags for the WandB run. Defaults to ['unirl'] if not provided.
        entity: WandB entity (team or username). If None, uses the default entity.
        require_success: Raise immediately if WandB init fails.
        **kwargs: Additional arguments for UniRLWandBLogger

    Returns:
        The initialized logger
    """
    global _global_logger
    _global_logger = UniRLWandBLogger(
        project=project,
        run_name=run_name,
        config=config,
        log_dir=log_dir,
        rank=rank,
        tags=tags,
        entity=entity,
        require_success=require_success,
        **kwargs,
    )
    return _global_logger


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate metrics from multiple training actors.

    Args:
        metrics_list: List of metric dicts from each actor

    Returns:
        Aggregated metrics (mean of each key)
    """
    if not metrics_list:
        return {}

    aggregated = {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    for key in all_keys:
        values = []
        for m in metrics_list:
            if key in m:
                val = m[key]
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else val.mean().item()
                if isinstance(val, bool):
                    values.append(float(val))
                elif isinstance(val, (int, float)):
                    values.append(float(val))
        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated


def aggregate_stage_results(results: List[Any]) -> Dict[str, float]:
    """Average :class:`TrackMiniBatchResult` metrics across the per-actor list.

    Driver-side aggregator for ONE track's per-actor results
    (``per_track_results[track_name]`` shape from ``train_group.train``).
    Each :class:`TrackMiniBatchResult.metrics` is already aggregated
    across micro-batches inside the actor via ``aggregate_numeric_metrics``
    (see ``training/stack.py``). This helper:

    1. Stamps the scalar fields ``loss / grad_norm / lr / has_backward``
       onto each per-actor dict.
    2. Forwards every algorithm-emitted metric key
       (e.g. ``ratio_mean``, ``clip_fraction``, ``approx_kl``).
    3. Averages numerically via ``aggregate_numeric_metrics``.

    The caller adds the ``<track>/`` namespace prefix when merging across
    tracks (see ``train.py``'s per-track aggregation loop).
    """
    if not results:
        return {}
    # Lazy import — keeps wandb_logger.py importable without pulling in
    # the training stack on cold paths (e.g. tests).
    from unirl.utils.misc import aggregate_numeric_metrics

    per_actor_dicts: List[Dict[str, Any]] = []
    for r in results:
        d: Dict[str, Any] = {
            "loss": float(r.loss),
            "grad_norm": float(r.grad_norm),
            "lr": float(r.lr),
            "has_backward": float(bool(r.has_backward)),
        }
        metrics = getattr(r, "metrics", None)
        if metrics:
            d.update({str(k): v for k, v in dict(metrics).items()})
        per_actor_dicts.append(d)
    return aggregate_numeric_metrics(per_actor_dicts)
