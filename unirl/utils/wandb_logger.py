"""
WandB Logger for unirl Training.

Provides comprehensive logging for training metrics, rollout statistics,
and image samples. Designed to match the logging behavior of DanceGRPO,
FlowGRPO, DiffusionNFT, and MixGRPO for comparison and reproducibility.

Usage:
    from unirl.utils.wandb_logger import init_logger

    # Initialize (typically via BaseTrainer._init_wandb)
    logger = init_logger(project="unirl", run_name="exp1", config=args)

    # Log training metrics
    logger.log_step(step=100, metrics={"loss": 0.5, "policy_loss": 0.3})

    # Log rollout metrics
    logger.log_rollout(rollout_id=10, metrics={"reward_mean": 0.8})
"""

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from unirl.train.stack import TrainStepResult

module_logger = logging.getLogger(__name__)


class PhaseTimer:
    """Per-phase wall-clock timer for one train step.

    Construction starts the step total; each ``phase(name)`` block accumulates
    into :attr:`phases` (re-entering a name adds to it, so a phase split across
    code paths still reports one number). Feed the results straight to
    :meth:`UniRLWandBLogger.log_rollout_step`::

        timer = PhaseTimer()
        with timer.phase("generate"):
            resp = self.rollout.generate(req)
        ...
        logger.log_rollout_step(
            rollout_id, result, resp,
            step_time_s=timer.total(), phase_times=timer.phases,
        )

    Phases sum to ~``total()``; the residual is whatever ran outside any
    ``phase`` block (cheap glue like logging).
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.phases: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the enclosed block and accumulate it under ``name``."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - t)

    def total(self) -> float:
        """Wall-clock seconds since construction (the whole step)."""
        return time.perf_counter() - self._t0


#: (handle attr, method, phase name) — the standard per-step collaborator
#: handles every v2 trainer drives; missing ones (e.g. trainside has no
#: ``weight_sync``) are skipped by :func:`install_phase_timing`.
_STEP_PHASE_SPECS = (
    ("rollout", "wake_up", "wake_up"),
    ("rollout", "generate", "generate"),
    ("rollout", "sleep", "sleep"),
    ("weight_sync", "sync", "weight_sync"),
    ("reward", "score_and_attach", "reward"),
    ("stack", "train_track", "train"),
)


def install_phase_timing(trainer: Any) -> None:
    """Attribute every train step into ``perf/<phase>_time_s`` — no trainer edits.

    Wraps ``trainer.train_step`` to arm a fresh :class:`PhaseTimer` per step and,
    lazily on the first step (the collaborators are created by the subclass
    ``__init__`` after this installs, and ``_init_wandb`` replaces the logger
    before stepping), wraps the standard collaborators from
    ``_STEP_PHASE_SPECS`` to accumulate their wall-clocks, plus the live
    logger's :meth:`UniRLWandBLogger.log_rollout_step` to inject the collected
    ``phase_times`` unless the caller already passed them. The trainers' own
    ``log_rollout_step(step_time_s=...)`` call sites stay the boundary —
    untouched.

    Handle methods are instance attributes (``handle.py`` binds them via
    ``setattr``), so instance-level re-``setattr`` wrapping is the framework's
    own extension mechanism. ``evaluate`` between steps also hits the wrapped
    collaborators, but it accumulates into the stale timer of the
    already-logged step and is discarded at the next re-arm.

    Timing semantics: handle dispatch is a blocking barrier (``handle_fn``
    does ``ray.get`` on all workers before returning), so each phase is the
    step's true critical-path wall-clock and phases sum to ~the step total.
    If a collaborator ever becomes async-submit (returns before the work
    finishes), its phase collapses to submission time and the wait leaks into
    the residual — a sudden near-zero phase plus a large
    ``rollout_time_s - sum(phases)`` residual is the tell.
    """
    inner = getattr(trainer, "train_step", None)
    if not callable(inner):
        return

    @functools.wraps(inner)
    def _steady_step(*args, **kwargs):
        trainer._step_timer = PhaseTimer()  # re-arm: fresh phases for this step
        return inner(*args, **kwargs)

    @functools.wraps(inner)
    def _first_step(*args, **kwargs):
        # First step is the earliest point the collaborators and the live logger
        # are all constructed; wrap them once, then rebind to the lean steady
        # wrapper so later steps just re-arm (no per-step branch, no latch flag).
        trainer._step_timer = PhaseTimer()
        _wrap_step_collaborators(trainer)
        trainer.train_step = _steady_step
        return inner(*args, **kwargs)

    trainer._step_timer = PhaseTimer()  # target for any pre-step evaluate()
    trainer.train_step = _first_step


def _timed_call(trainer: Any, fn, phase: str):
    """Return ``fn`` wrapped to accumulate its wall-clock under ``phase``."""

    @functools.wraps(fn)
    def _timed(*args, **kwargs):
        with trainer._step_timer.phase(phase):
            return fn(*args, **kwargs)

    return _timed


def _wrap_step_collaborators(trainer: Any) -> None:
    """Time each present collaborator method, and teach the logger to emit phases."""
    for handle_attr, method, phase in _STEP_PHASE_SPECS:
        handle = getattr(trainer, handle_attr, None)
        fn = getattr(handle, method, None)
        if not callable(fn):
            continue
        setattr(handle, method, _timed_call(trainer, fn, phase))

    # Inject the phases we collected into the logger boundary, unless the
    # trainer already passed its own.
    log_inner = trainer.wandb_logger.log_rollout_step

    @functools.wraps(log_inner)
    def _log_with_phases(*args, **kwargs):
        if kwargs.get("phase_times") is None and trainer._step_timer.phases:
            kwargs["phase_times"] = dict(trainer._step_timer.phases)
        return log_inner(*args, **kwargs)

    trainer.wandb_logger.log_rollout_step = _log_with_phases


class UniRLWandBLogger:
    """WandB logger for unirl training.

    Logs metrics compatible with DanceGRPO, FlowGRPO, DiffusionNFT, and MixGRPO
    for cross-validation and comparison.

    Attributes:
        enabled: Whether logging is enabled
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        media_log_interval: How often to log generated media (in rollouts)
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        log_dir: Optional[str] = None,
        rank: int = 0,
        media_log_interval: int = 1,
        media_max_items: int = 8,
        log_media: bool = False,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        optimizer_step: int = 0,
    ):
        """Initialize WandB logger.

        Enabling reporting inherently requires a successful init: when
        ``enabled`` and a ``project`` are given, a failed/unavailable wandb
        init raises (you asked for wandb and it could not start) rather than
        silently training without logging.

        Args:
            project: WandB project name
            run_name: WandB run name
            config: Training configuration (dict or object with __dict__)
            log_dir: WandB run directory (if provided)
            rank: Process rank (only rank 0 logs)
            media_log_interval: How often to log generated media (in rollouts)
            media_max_items: Max per-track media samples to log per logged rollout
            log_media: Master switch for generated-media logging
            enabled: Whether to enable logging (disabled => no-op null-object)
            tags: List of tags for the WandB run. Defaults to ['unirl'] if not provided.
            entity: WandB entity (team or username). If None, uses the default entity.
            run_id: Resume this wandb run id (from a checkpoint's
                trainer_state.json) instead of starting a fresh run.
            optimizer_step: Seed for the ``train/`` step axis on resume.
        """
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self.log_dir = str(log_dir) if log_dir else None
        self.media_log_interval = max(1, int(media_log_interval))
        self.media_max_items = max(1, int(media_max_items))
        self.log_media = bool(log_media)
        self.rank = rank
        self.tags = tags if tags is not None else ["unirl"]
        self.run_id = run_id
        self._initialized = False
        # Optimizer-step counter for the ``train/`` panel (moved here from
        # BaseTrainer so all step-axis bookkeeping lives in the logger).
        self._optimizer_step = int(optimizer_step)

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

    @property
    def optimizer_step(self) -> int:
        """Current ``train/`` step-axis value — checkpointed for resume."""
        return self._optimizer_step

    def _handle_init_failure(
        self,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Raise when an *enabled* wandb run fails to initialize.

        Reached only from the ctor's ``enabled and project`` branch, so a failure
        here means reporting was explicitly requested and could not start —
        surface it loudly instead of silently degrading to no logging.
        """
        full_message = f"{message}: {exc}" if exc is not None else message
        raise RuntimeError(full_message) from exc

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
            if self.run_id:
                # Resume the checkpoint's run ("allow": append if the id
                # exists, else create it) so curves continue in one run.
                init_kwargs["id"] = self.run_id
                init_kwargs["resume"] = "allow"
            wandb.init(**init_kwargs)
            self.run_id = wandb.run.id
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

    def should_log_media(self, rollout_id: int) -> bool:
        """Whether generated media should be captured/logged for this rollout.

        Gated so trainers don't build (CPU/PIL-heavy) previews when media
        logging is off, disabled, or this rollout isn't on the cadence.
        """
        return self.enabled and self.log_media and (int(rollout_id) % self.media_log_interval == 0)

    def log_rollout_step(
        self,
        rollout_id: int,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
        resp: Any,
        *,
        step_time_s: Optional[float] = None,
        phase_times: Optional[Dict[str, float]] = None,
        trunc_len: Optional[int] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one rollout's metrics to wandb. No-op when disabled.

        The single per-step entry point shared by every trainer. It consumes
        only framework-universal objects: a :class:`RolloutResp` (``resp``)
        and a :class:`TrainStepResult` (single-track) or a ``{track:
        TrainStepResult}`` dict (multi-track). All wandb/metric/step logic
        lives here so trainers stay logging-free.

        - ``rollout/*``: reward/advantage (and AR response-length) distribution
          stats from ``resp.tracks`` via ``compute_rollout_resp_metrics``, plus
          any ``extra_metrics`` (e.g. ``sync_weights``) merged in.
        - ``train/*``: optimizer scalars + algorithm metrics, per-update aware
          (see :meth:`_log_train`).
        - ``perf/rollout_time_s``: optional wall-clock for the step.
        - ``perf/<phase>_time_s``: optional per-phase wall-clocks from
          ``phase_times`` (e.g. ``generate``/``weight_sync``/``reward``/
          ``train``), so the step total can be attributed without log
          archaeology.

        Generated media is NOT logged here: this runs after ``train_track``,
        so a preview still attached to the track would have ridden into the
        DP_SCATTER training dispatch. ``BaseTrainer._drop_decoded`` uploads
        previews via :meth:`log_generated_media` at this same step value and
        frees them before dispatch.
        """
        if not self.enabled or not self._initialized:
            return
        # Lazy import keeps wandb_logger importable without the training stack.
        from unirl.utils.wandb_metrics import compute_rollout_resp_metrics

        step = rollout_id + 1
        rollout_metrics = compute_rollout_resp_metrics(resp=resp, trunc_len=trunc_len)
        if extra_metrics:
            rollout_metrics.update(extra_metrics)
        self.log_rollout(step, rollout_metrics)

        self._log_train(results)

        perf: Dict[str, float] = {}
        if step_time_s is not None:
            perf["rollout_time_s"] = float(step_time_s)
        if phase_times:
            perf.update({f"{name}_time_s": float(v) for name, v in phase_times.items()})
        if perf:
            self.log_perf(step, perf)

    def _log_train(
        self,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
    ) -> None:
        """Emit ``train/*`` points, per optimizer step, single- and multi-track.

        Step-axis matrix (``train/step`` == ``self._optimizer_step``):

        - single result, ``per_update`` empty → one aggregate point per backward.
        - single result, ``per_update`` len N>1 → N points (one per optimizer
          update), metrics unprefixed (the on-policy update0 then off-policy drift).
        - dict results → metrics namespaced ``<track>/<key>``. Cross-track
          per-update merge ONLY when every track's ``per_update`` shares the same
          length L>1 (one optimizer driving all tracks, e.g. unified_model);
          otherwise one aggregate point per rollout. This never interleaves the
          independent optimizers of a per-track recipe (e.g. PE), and is
          byte-identical to the legacy path for every single-update recipe.
        """
        if not self.enabled or not self._initialized:
            return

        if isinstance(results, dict):
            per_update_lens = [len(getattr(r, "per_update", ()) or ()) for r in results.values()]
            mergeable = bool(per_update_lens) and len(set(per_update_lens)) == 1 and per_update_lens[0] > 1
            if mergeable:
                length = per_update_lens[0]
                for i in range(length):
                    merged: Dict[str, Any] = {
                        f"{name}/{key}": value
                        for name, result in results.items()
                        for key, value in dict(result.per_update[i]).items()
                    }
                    self._optimizer_step += 1
                    self.log_step(self._optimizer_step, merged)
                return
            train_metrics: Dict[str, Any] = {
                f"{name}/{key}": value
                for name, result in results.items()
                for key, value in aggregate_stage_results([result]).items()
            }
            if any(bool(getattr(r, "has_backward", False)) for r in results.values()):
                self._optimizer_step += 1
                self.log_step(self._optimizer_step, train_metrics)
            return

        # Single-track result.
        per_update = getattr(results, "per_update", ()) or ()
        if len(per_update) > 1:
            for metrics in per_update:
                self._optimizer_step += 1
                self.log_step(self._optimizer_step, dict(metrics))
        elif getattr(results, "has_backward", False):
            self._optimizer_step += 1
            self.log_step(self._optimizer_step, dict(aggregate_stage_results([results])))

    def log_progress(
        self,
        rollout_id: int,
        num_rollouts: int,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
        mean_reward: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Emit the one-line stdout progress summary for a rollout.

        NOT gated by ``enabled`` — console progress prints even when wandb
        reporting is off. Generic over single- and multi-track ``results``:
        a single result renders ``loss/grad_norm/lr`` (+ ``ratio``/``clip``
        when the algorithm reported them); a dict renders one ``name[...]``
        group per track, preserving the richer per-track line trainers used
        to hand-format.
        """
        log = logger if logger is not None else module_logger

        def _metric(metrics: Any, key: str) -> Optional[float]:
            value = (metrics or {}).get(key) if metrics is not None else None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _fmt(result: Any) -> str:
            parts = f"loss={result.loss:.4f} gn={result.grad_norm:.4f} lr={result.lr:.2e}"
            metrics = getattr(result, "metrics", None)
            ratio_mean = _metric(metrics, "ratio_mean")
            ratio_std = _metric(metrics, "ratio_std")
            clip_fraction = _metric(metrics, "clip_fraction")
            if ratio_mean is not None:
                parts += f" ratio={ratio_mean:.4f}"
                if ratio_std is not None:
                    parts += f"±{ratio_std:.4f}"
            if clip_fraction is not None:
                parts += f" clip={clip_fraction:.2f}"
            return parts

        if isinstance(results, dict):
            body = "  ".join(f"{name}[{_fmt(result)}]" for name, result in results.items())
        else:
            body = _fmt(results)
        suffix = ("  " + " ".join(f"{k}={v}" for k, v in extra.items())) if extra else ""
        log.info(
            "rollout %d/%d  reward=%.4f  %s%s",
            rollout_id + 1,
            num_rollouts,
            mean_reward,
            body,
            suffix,
        )

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self._initialized:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")


def init_logger(
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Any] = None,
    log_dir: Optional[str] = None,
    rank: int = 0,
    tags: Optional[List[str]] = None,
    entity: Optional[str] = None,
    log_media: bool = False,
    media_max_items: int = 8,
    media_log_interval: int = 1,
    enabled: bool = True,
    **kwargs,
) -> UniRLWandBLogger:
    """Construct a :class:`UniRLWandBLogger`.

    Always returns a logger instance. Pass ``enabled=False`` (the BaseTrainer
    factory does this when reporting is off) for a no-op null-object whose wandb
    methods short-circuit while ``log_progress`` still prints. An *enabled* run
    that fails to init raises (success is inherent to enabling — no opt-out flag).

    Args:
        project: WandB project name
        run_name: WandB run name
        config: Training configuration
        rank: Process rank
        tags: List of tags for the WandB run. Defaults to ['unirl'] if not provided.
        entity: WandB entity (team or username). If None, uses the default entity.
        log_media: Master switch for generated-media logging.
        media_max_items: Max per-track media samples per logged rollout.
        media_log_interval: How often (in rollouts) to log media.
        enabled: Whether logging is enabled at all.
        **kwargs: Additional arguments for UniRLWandBLogger

    Returns:
        The constructed logger
    """
    return UniRLWandBLogger(
        project=project,
        run_name=run_name,
        config=config,
        log_dir=log_dir,
        rank=rank,
        tags=tags,
        entity=entity,
        log_media=log_media,
        media_max_items=media_max_items,
        media_log_interval=media_log_interval,
        enabled=enabled,
        **kwargs,
    )


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
