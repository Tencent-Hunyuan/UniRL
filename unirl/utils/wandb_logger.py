"""WandB Logger for unirl Training."""

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


def _write_video_with_audio(
    frames: Any,
    fps: int,
    audio: torch.Tensor,
    audio_sample_rate: int,
) -> str:
    """Mux video frames + audio waveform into a single mp4 file using PyAV."""
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        from fractions import Fraction

        import av

        container = av.open(path, mode="w")
        video_stream = container.add_stream("libx264", rate=int(fps))
        video_stream.width = int(frames.shape[2])
        video_stream.height = int(frames.shape[1])
        video_stream.pix_fmt = "yuv420p"

        audio_stream = container.add_stream("aac", rate=audio_sample_rate)
        audio_stream.codec_context.sample_rate = audio_sample_rate
        audio_stream.codec_context.layout = "stereo"
        audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)

        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        samples = audio.float().cpu()
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)
        if samples.shape[0] == 1:
            samples = samples.expand(2, -1)
        samples = samples.T
        samples = torch.clamp(samples, -1.0, 1.0)
        int16_samples = (samples * 32767.0).to(torch.int16)

        audio_frame = av.AudioFrame.from_ndarray(
            int16_samples.contiguous().reshape(1, -1).numpy(),
            format="s16",
            layout="stereo",
        )
        audio_frame.sample_rate = audio_sample_rate

        target_format = audio_stream.codec_context.format or "fltp"
        target_layout = audio_stream.codec_context.layout or "stereo"
        resampler = av.audio.resampler.AudioResampler(
            format=target_format,
            layout=target_layout,
            rate=audio_sample_rate,
        )
        audio_next_pts = 0
        for rframe in resampler.resample(audio_frame):
            if rframe.pts is None:
                rframe.pts = audio_next_pts
            audio_next_pts += rframe.samples
            rframe.sample_rate = audio_sample_rate
            container.mux(audio_stream.encode(rframe))
        for packet in audio_stream.encode():
            container.mux(packet)

        container.close()
    except ImportError:
        module_logger.warning("PyAV (av) not installed; writing video without audio. Install with: pip install av")
        os.unlink(path)
        fd2, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd2)
        import imageio

        imageio.mimwrite(path, frames, fps=fps, format="FFMPEG", codec="libx264", pixelformat="yuv420p")
    return path


class PhaseTimer:
    """Per-phase wall-clock timer for one train step."""

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


_STEP_PHASE_SPECS = (
    ("rollout", "wake_up", "wake_up"),
    ("rollout", "generate", "generate"),
    ("rollout", "sleep", "sleep"),
    ("weight_sync", "sync", "weight_sync"),
    ("reward", "score_and_attach", "reward"),
    ("stack", "train_track", "train"),
)


def install_phase_timing(trainer: Any) -> None:
    """Attribute every train step into ``perf/<phase>_time_s`` — no trainer edits."""
    inner = getattr(trainer, "train_step", None)
    if not callable(inner):
        return

    @functools.wraps(inner)
    def _steady_step(*args, **kwargs):
        trainer._step_timer = PhaseTimer()
        return inner(*args, **kwargs)

    @functools.wraps(inner)
    def _first_step(*args, **kwargs):
        trainer._step_timer = PhaseTimer()
        _wrap_step_collaborators(trainer)
        trainer.train_step = _steady_step
        return inner(*args, **kwargs)

    trainer._step_timer = PhaseTimer()
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

    log_inner = trainer.wandb_logger.log_rollout_step

    @functools.wraps(log_inner)
    def _log_with_phases(*args, **kwargs):
        if kwargs.get("phase_times") is None and trainer._step_timer.phases:
            kwargs["phase_times"] = dict(trainer._step_timer.phases)
        return log_inner(*args, **kwargs)

    trainer.wandb_logger.log_rollout_step = _log_with_phases


class UniRLWandBLogger:
    """WandB logger for unirl training."""

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
        """Initialize WandB logger."""
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
        self._optimizer_step = int(optimizer_step)
        self.memory_monitor = None

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
        """Raise when an *enabled* wandb run fails to initialize."""
        full_message = f"{message}: {exc}" if exc is not None else message
        raise RuntimeError(full_message) from exc

    def _init_wandb(self, config: Optional[Any] = None):
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            self._handle_init_failure("wandb package is not installed but WandB reporting was requested")
            return

        try:
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
            # Bind nested train metrics explicitly so W&B uses train/step.
            wandb.define_metric("train/ar/*", step_metric="train/step")
            wandb.define_metric("train/image/*", step_metric="train/step")
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
        """Log per-step training metrics."""
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
        """Log per-rollout metrics."""
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
        step_key: str = "rollout/step",
    ) -> None:
        """Log rollout media preview payload produced by the rollout pipeline."""
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

        if video_key is None:
            if key == "rollout/generated_media":
                video_key = "rollout/generated_videos"
            elif key.endswith("_images"):
                video_key = key[: -len("_images")] + "_videos"
            elif key.endswith("_media"):
                video_key = key[: -len("_media")] + "_videos"
            else:
                video_key = f"{key}/videos"

        _muxed_paths: List[str] = []
        try:
            n = max(len(images) if has_images else 0, len(videos) if has_videos else 0)

            def _caption_for(idx: int) -> str:
                prompt = str(prompts[idx]) if idx < len(prompts) else ""
                if reward_values is not None and idx < len(reward_values):
                    return f"{prompt[:100]} | reward: {reward_values[idx]:.2f}"
                return f"{prompt[:100]}"

            payload: Dict[str, Any] = {step_key: int(rollout_id)}

            if has_images:
                wandb_images = [
                    wandb.Image(images[idx], caption=_caption_for(idx)) for idx in range(min(len(images), n))
                ]
                payload[key] = wandb_images

            if has_videos:
                wandb_videos: List[Any] = []
                audios = getattr(media_preview, "audios", None) or []
                audio_sr = getattr(media_preview, "audio_sample_rate", None)
                for idx in range(min(len(videos), n)):
                    vid = videos[idx]
                    if not torch.is_tensor(vid):
                        continue
                    if vid.dim() != 4:
                        raise ValueError(
                            f"log_generated_media: video at idx {idx} must be 4D "
                            f"[C, T, H, W], got shape {tuple(vid.shape)}"
                        )
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
                    audio_wf = audios[idx] if idx < len(audios) else None
                    if audio_wf is not None and audio_sr is not None and torch.is_tensor(audio_wf):
                        arr_hwc = arr.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
                        path = _write_video_with_audio(arr_hwc, int(video_fps), audio_wf, int(audio_sr))
                        _muxed_paths.append(path)
                        wandb_videos.append(wandb.Video(path, caption=_caption_for(idx), format="mp4"))
                    else:
                        wandb_videos.append(wandb.Video(arr, caption=_caption_for(idx), fps=int(video_fps)))
                if wandb_videos:
                    payload[video_key] = wandb_videos

            wandb.log(payload)
        except Exception as e:
            print(f"Warning: Failed to log generated media: {e}")
        finally:
            for _p in _muxed_paths:
                try:
                    os.unlink(_p)
                except OSError:
                    pass

    def log_eval(
        self,
        step: int,
        eval_metrics: Dict[str, Any],
    ):
        """Log evaluation metrics."""
        self.log_with_step(
            step_key="eval/step",
            step=step,
            metrics=eval_metrics,
            prefix="eval/",
        )

    def should_log_media(self, rollout_id: int) -> bool:
        """Whether generated media should be captured/logged for this rollout."""
        return self.enabled and self.log_media and (int(rollout_id) % self.media_log_interval == 0)

    def should_log_eval_media(self) -> bool:
        """Whether eval generations should be captured/logged — ``eval_interval`` is the only cadence."""
        return self.enabled and self.log_media

    def log_rollout_step(
        self,
        rollout_id: int,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
        sample: Any,
        *,
        step_time_s: Optional[float] = None,
        phase_times: Optional[Dict[str, float]] = None,
        trunc_len: Optional[int] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log one rollout's metrics to wandb. No-op when disabled."""
        mem_summary = self.memory_monitor.step_summary(step=rollout_id + 1) if self.memory_monitor is not None else None
        if not self.enabled or not self._initialized:
            return
        from unirl.utils.wandb_metrics import compute_rollout_sample_metrics

        step = rollout_id + 1
        rollout_metrics = compute_rollout_sample_metrics(sample=sample, trunc_len=trunc_len)
        if extra_metrics:
            rollout_metrics.update(extra_metrics)
        self.log_rollout(step, rollout_metrics)

        self._log_train(results)

        perf: Dict[str, float] = {}
        if step_time_s is not None:
            perf["step_time_s"] = float(step_time_s)
        if phase_times:
            perf.update({f"{name}_time_s": float(v) for name, v in phase_times.items()})
        if mem_summary:
            perf.update(mem_summary)
        if perf:
            self.log_perf(step, perf)

    def _log_train(
        self,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
    ) -> None:
        """Emit ``train/*`` points, one per optimizer update, single- and multi-track."""
        if not self.enabled or not self._initialized:
            return

        if isinstance(results, dict):
            per_track_updates: Dict[str, List[Dict[str, Any]]] = {}
            for name, result in results.items():
                per_update = getattr(result, "per_update", ()) or ()
                if len(per_update) > 1:
                    per_track_updates[name] = [dict(m) for m in per_update]
                else:
                    per_track_updates[name] = [dict(aggregate_stage_results([result]))]
            length = max((len(v) for v in per_track_updates.values()), default=0)
            if length <= 1:
                if any(bool(getattr(r, "has_backward", False)) for r in results.values()):
                    merged = {
                        f"{name}/{key}": value
                        for name, updates in per_track_updates.items()
                        for key, value in updates[0].items()
                    }
                    self._optimizer_step += 1
                    self.log_step(self._optimizer_step, merged)
                return
            for i in range(length):
                merged = {
                    f"{name}/{key}": value
                    for name, updates in per_track_updates.items()
                    if i < len(updates)
                    for key, value in updates[i].items()
                }
                if not merged:
                    continue
                self._optimizer_step += 1
                self.log_step(self._optimizer_step, merged)
            return

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
        """Emit the one-line stdout progress summary for a rollout."""
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
            k3_mean = _metric(metrics, "k3_mean")
            absdiff_mean = _metric(metrics, "rollout_replay_logp_absdiff_mean")
            if k3_mean is not None:
                parts += f" k3={k3_mean:.2e}"
            if absdiff_mean is not None:
                parts += f" |Δlogp|={absdiff_mean:.2e}"
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
    """Construct a :class:`UniRLWandBLogger`."""
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
    """Aggregate metrics from multiple training actors."""
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
    """Average :class:`TrackMiniBatchResult` metrics across the per-actor list."""
    if not results:
        return {}
    from unirl.utils.metrics import aggregate_numeric_metrics

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
