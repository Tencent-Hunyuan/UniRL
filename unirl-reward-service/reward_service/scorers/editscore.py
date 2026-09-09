"""EditScore VLM judge for instruction-guided image editing.

The scorer owns the boundaries missing from the upstream package: safe LoRA
cache publication, vLLM lifecycle kwargs, batched generation, and strict
validation of untrusted judge output. Each item carries the source image in
``history[0]`` and the edited image in ``history[1]``.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import json
import math
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any, Iterator

from PIL import Image

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)

_BACKBONES: dict[str, tuple[str, str] | None] = {
    "qwen3vl": None,
    "qwen25vl": None,
    "qwen3vl_vllm": ("editscore.mllm_tools.qwen3vl_vllm", "Qwen3VLForConditionalGeneration"),
    "qwen25vl_vllm": ("editscore.mllm_tools.qwen25vl_vllm", "Qwen2_5_VLForConditionalGeneration"),
}
_SLEEP_MANAGED_LLM_KWARGS = {"enable_prefix_caching", "mm_processor_cache_gb"}
_MAX_PARSE_ATTEMPTS = 3
_CACHE_MANIFEST = ".unirl-editscore-cache.json"
_PreparedRow = tuple[int, str, Any, Any]


class _InvalidJudgeOutput(ValueError):
    """One row could not be parsed into valid EditScore values."""


def _validate_cache(path: Path, expected: dict[str, object]) -> None:
    manifest = path / _CACHE_MANIFEST
    try:
        actual = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"EditScore cache {path} has no valid {_CACHE_MANIFEST}; use an empty path so UniRL can build it atomically"
        ) from exc
    if actual != expected:
        raise ValueError(f"EditScore cache {path} was built for {actual!r}, not {expected!r}")


@contextmanager
def _cache_writer_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _merge_checkpoint(path: Path, *, model: str, lora: str, backbone: str) -> None:
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor

    model_cls = getattr(transformers, _BACKBONES[backbone][1])
    merged = model_cls.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(merged, lora).merge_and_unload()
    merged.save_pretrained(path)
    AutoProcessor.from_pretrained(model).save_pretrained(path)


def _prepare_merged_checkpoint(
    *,
    model: str,
    lora: str,
    backbone: str,
    cache_dir: str | None,
) -> str:
    expected = {
        "format_version": 1,
        "backbone": backbone,
        "model_name_or_path": model,
        "lora_path": lora,
    }
    if cache_dir is None:
        import torch

        digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12]
        target = Path(torch.hub.get_dir()) / "EditScore" / f"{Path(model).name}_{Path(lora).name}_{digest}"
    else:
        target = Path(cache_dir).expanduser()

    if target.exists():
        _validate_cache(target, expected)
        return str(target)

    lock = target.parent / f".{target.name}.lock"
    with _cache_writer_lock(lock):
        if target.exists():
            _validate_cache(target, expected)
            return str(target)

        cutoff = time.time() - 24 * 60 * 60
        for stale in target.parent.glob(f".{target.name}.tmp-*"):
            try:
                if stale.stat().st_mtime < cutoff:
                    shutil.rmtree(stale, ignore_errors=True)
            except OSError:
                pass

        temporary = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            temporary.mkdir()
            _merge_checkpoint(temporary, model=model, lora=lora, backbone=backbone)
            (temporary / _CACHE_MANIFEST).write_text(
                json.dumps(expected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, target)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not target.exists():
                    raise
                _validate_cache(target, expected)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return str(target)


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _engine_kwargs(
    *,
    extra_llm_kwargs: dict[str, Any] | None,
    gpu_memory_utilization: float | None,
    enable_sleep_mode: bool,
) -> dict[str, Any]:
    injected = dict(extra_llm_kwargs or {})
    overlap = {"enable_sleep_mode"} & set(injected)
    if gpu_memory_utilization is not None and "gpu_memory_utilization" in injected:
        overlap.add("gpu_memory_utilization")
    if enable_sleep_mode:
        overlap |= _SLEEP_MANAGED_LLM_KWARGS & set(injected)
    if overlap:
        raise ValueError(f"extra_llm_kwargs must not set dedicated or lifecycle-managed options {sorted(overlap)}")
    if gpu_memory_utilization is not None:
        injected["gpu_memory_utilization"] = gpu_memory_utilization
    elif "gpu_memory_utilization" in injected:
        utilization = _finite_number("extra_llm_kwargs['gpu_memory_utilization']", injected["gpu_memory_utilization"])
        if not 0.0 < utilization <= 1.0:
            raise ValueError("extra_llm_kwargs['gpu_memory_utilization'] must be in (0, 1]")
    if enable_sleep_mode:
        injected.update(
            enable_sleep_mode=True,
            mm_processor_cache_gb=0,
            enable_prefix_caching=False,
        )
    return injected


@contextmanager
def _inject_llm_kwargs(backbone: str, injected: dict[str, Any]) -> Iterator[None]:
    """Temporarily add kwargs to EditScore's hardcoded vLLM constructor."""
    if not injected:
        yield
        return

    module = importlib.import_module(_BACKBONES[backbone][0])
    original_llm = module.LLM

    def llm_with_injected_kwargs(**kwargs):
        return original_llm(**{**kwargs, **injected})

    module.LLM = llm_with_injected_kwargs
    try:
        yield
    finally:
        module.LLM = original_llm


class EditScoreScorer(BaseScorer):
    name = "editscore"
    version = "1"
    input_kind = "image"
    # The direct server checks this class attribute before construction. The
    # constructor separately rejects a boot-offloaded instance without sleep
    # mode before loading any model.
    supports_offload = True
    # The consumer's default reduction is "first", so expose the calibrated
    # headline score first while retaining the diagnostic axes.
    sub_metric_names = ("overall", "prompt_following", "consistency", "perceptual_quality")

    def __init__(
        self,
        model_name_or_path: str,
        backbone: str = "qwen3vl_vllm",
        lora_path: str | None = None,
        cache_dir: str | None = None,
        score_range: int = 25,
        num_pass: int = 1,
        temperature: float = 0.7,
        seed: int = 42,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float | None = None,
        enable_sleep_mode: bool = False,
        extra_llm_kwargs: dict[str, Any] | None = None,
        max_image_side: int | None = 1536,
        batched: bool = True,
    ) -> None:
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string")
        if not isinstance(backbone, str) or backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(_BACKBONES)}, got {backbone!r}")
        is_vllm = _BACKBONES[backbone] is not None
        for name, value in (("lora_path", lora_path), ("cache_dir", cache_dir)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        for name, value in (
            ("score_range", score_range),
            ("num_pass", num_pass),
            ("tensor_parallel_size", tensor_parallel_size),
            ("max_model_len", max_model_len),
            ("max_num_batched_tokens", max_num_batched_tokens),
            ("max_num_seqs", max_num_seqs),
        ):
            _require_positive_int(name, value)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if _finite_number("temperature", temperature) < 0.0:
            raise ValueError("temperature must be non-negative")
        if (
            gpu_memory_utilization is not None
            and not 0.0 < _finite_number("gpu_memory_utilization", gpu_memory_utilization) <= 1.0
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        for name, value in (("enable_sleep_mode", enable_sleep_mode), ("batched", batched)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if extra_llm_kwargs is not None and not isinstance(extra_llm_kwargs, dict):
            raise TypeError("extra_llm_kwargs must be a dict or None")
        if max_image_side is not None:
            _require_positive_int("max_image_side", max_image_side)
        if cache_dir is not None and (lora_path is None or not is_vllm):
            raise ValueError("cache_dir is only valid for a LoRA used with a vLLM backbone")
        if enable_sleep_mode and not is_vllm:
            raise ValueError("enable_sleep_mode requires a vLLM backbone")
        if (gpu_memory_utilization is not None or extra_llm_kwargs) and not is_vllm:
            raise ValueError("vLLM engine kwargs require a vLLM backbone")
        injected = _engine_kwargs(
            extra_llm_kwargs=extra_llm_kwargs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=enable_sleep_mode,
        )

        if os.environ.get("UNIRL_SCORER_BOOT_OFFLOADED") == "1" and not enable_sleep_mode:
            raise ValueError("boot-offloaded EditScore requires enable_sleep_mode=True")

        from editscore import EditScore

        resolved_model = model_name_or_path
        resolved_lora = lora_path
        if lora_path is not None and is_vllm:
            resolved_model = _prepare_merged_checkpoint(
                model=model_name_or_path,
                lora=lora_path,
                backbone=backbone,
                cache_dir=cache_dir,
            )
            # The cache contains merged weights; prevent EditScore from merging
            # the adapter again or trusting its non-atomic cache implementation.
            resolved_lora = None

        with _inject_llm_kwargs(backbone, injected):
            self.es = EditScore(
                backbone=backbone,
                model_name_or_path=resolved_model,
                lora_path=resolved_lora,
                score_range=score_range,
                num_pass=num_pass,
                temperature=temperature,
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
            )

        self.supports_offload = enable_sleep_mode
        self._max_image_side = max_image_side
        self._use_batch_inference = batched and is_vllm

    def _resize_image_to_max_side(self, image: Image.Image) -> Image.Image:
        """Bound vision tokens before upstream processing.

        Two 1536px square images use roughly 4.6k vision tokens; increasing
        this limit may also require increasing ``max_model_len``.
        """
        side = self._max_image_side
        if side is not None and max(image.size) > side:
            image = image.copy()
            image.thumbnail((side, side), Image.Resampling.LANCZOS)
        return image

    def _unpack_item(self, item: ScoreItem) -> tuple[str, Image.Image, Image.Image]:
        if len(item.history) < 2:
            raise ValueError(f"EditScore requires 2 history turns, got {len(item.history)}")
        prompt, source_image = item.history[0]
        _, edited_image = item.history[1]
        if not isinstance(prompt, str):
            raise TypeError(f"EditScore prompt must be a string, got {type(prompt).__name__}")
        if not isinstance(source_image, Image.Image) or not isinstance(edited_image, Image.Image):
            raise TypeError("EditScore requires PIL source and edited images")
        return (
            prompt,
            self._resize_image_to_max_side(source_image),
            self._resize_image_to_max_side(edited_image),
        )

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        import torch

        rows: list[_PreparedRow] = []
        results = [{name: float("nan") for name in self.sub_metric_names} for _ in items]
        for i, item in enumerate(items):
            try:
                prompt, source_image, edited_image = self._unpack_item(item)
                sc_message = self.es.model.prepare_input(
                    [source_image, edited_image],
                    self.es.SC_prompt.replace("<instruction>", prompt),
                )
                pq_message = self.es.model.prepare_input(edited_image, self.es.PQ_prompt)
                row = (i, prompt, sc_message, pq_message)
            except torch.OutOfMemoryError:
                raise
            except Exception:
                logger.exception("EditScore failed to score item %d", i)
                continue
            if self._use_batch_inference:
                rows.append(row)
            else:
                [output] = self._score_rows([row])
                if output is not None:
                    results[i] = output

        if rows:
            scored = self._score_rows(rows)
            for row, output in zip(rows, scored):
                if output is not None:
                    results[row[0]] = output

        return results

    def _run_inference(self, messages: list[Any], *, seed: int) -> list[str]:
        if self._use_batch_inference:
            outputs = self.es.model.batch_inference(messages, seed=seed)
        else:
            outputs = [self.es.model.inference(message, seed=seed) for message in messages]
        if not isinstance(outputs, list) or len(outputs) != len(messages):
            raise RuntimeError(
                f"EditScore inference returned {type(outputs).__name__} with "
                f"{len(outputs) if isinstance(outputs, list) else 'unknown'} rows for {len(messages)} inputs"
            )
        if not all(isinstance(output, str) for output in outputs):
            raise RuntimeError("EditScore inference returned a non-string output")
        return outputs

    @staticmethod
    def _parse_scores(
        text: str,
        *,
        prompt: str,
        score_range: int,
        expected_count: int,
    ) -> tuple[float, ...]:
        from editscore.utils import mllm_output_to_dict

        try:
            parsed = mllm_output_to_dict(
                text,
                give_up_parsing=False,
                text_prompt=prompt,
                score_range=score_range,
            )
        except Exception as exc:
            raise _InvalidJudgeOutput("EditScore parser raised") from exc
        if not isinstance(parsed, dict):
            raise _InvalidJudgeOutput(f"EditScore parser returned {parsed!r}")

        scores = parsed.get("score")
        if not isinstance(scores, list) or len(scores) != expected_count:
            raise _InvalidJudgeOutput(f"expected {expected_count} scores, got {scores!r}")
        normalized = []
        for value in scores:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise _InvalidJudgeOutput(f"scores must be real numbers, got {scores!r}")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= score_range:
                raise _InvalidJudgeOutput(f"scores must be finite and in [0, {score_range}], got {scores!r}")
            normalized.append(number)
        return tuple(normalized)

    @classmethod
    def _parse_metrics(
        cls,
        sc_text: str,
        pq_text: str,
        *,
        prompt: str,
        score_range: int,
    ) -> dict[str, float]:
        sc_scores = cls._parse_scores(
            sc_text,
            prompt=prompt,
            score_range=score_range,
            expected_count=2,
        )
        pq_scores = cls._parse_scores(
            pq_text,
            prompt=prompt,
            score_range=score_range,
            expected_count=2,
        )
        scale = score_range / 10
        prompt_following, consistency = (value / scale for value in sc_scores)
        perceptual_quality = min(pq_scores) / scale
        overall = math.sqrt(min(prompt_following, consistency) * perceptual_quality)
        return {
            "overall": overall,
            "prompt_following": prompt_following,
            "consistency": consistency,
            "perceptual_quality": perceptual_quality,
        }

    def _score_rows(
        self,
        rows: list[_PreparedRow],
    ) -> list[dict[str, float] | None]:
        """Score valid rows with one parser and failure policy for every backend."""
        es = self.es

        per_pass: list[list[dict[str, float] | None]] = []
        for pass_index in range(es.num_pass):
            pass_outs: list[dict[str, float] | None] = [None] * len(rows)
            pending = list(range(len(rows)))
            for attempt in range(_MAX_PARSE_ATTEMPTS):
                if not pending:
                    break
                retry_seed = es.seed + pass_index + attempt * es.num_pass
                sc_texts = self._run_inference(
                    [rows[idx][2] for idx in pending],
                    seed=retry_seed,
                )
                pq_texts = self._run_inference(
                    [rows[idx][3] for idx in pending],
                    seed=retry_seed,
                )

                retry: list[int] = []
                for row_index, sc_text, pq_text in zip(pending, sc_texts, pq_texts):
                    prompt = rows[row_index][1]
                    try:
                        pass_outs[row_index] = self._parse_metrics(
                            sc_text,
                            pq_text,
                            prompt=prompt,
                            score_range=es.score_range,
                        )
                    except _InvalidJudgeOutput as exc:
                        if attempt < _MAX_PARSE_ATTEMPTS - 1:
                            retry.append(row_index)
                        else:
                            logger.warning(
                                "EditScore output remained invalid after %d attempts for item %d: %s",
                                _MAX_PARSE_ATTEMPTS,
                                rows[row_index][0],
                                exc,
                            )
                pending = retry
            per_pass.append(pass_outs)

        merged: list[dict[str, float] | None] = []
        for idx in range(len(rows)):
            complete = [output for pass_outputs in per_pass if (output := pass_outputs[idx]) is not None]
            if len(complete) != es.num_pass:
                merged.append(None)
                continue
            merged.append({name: fmean(output[name] for output in complete) for name in self.sub_metric_names})
        return merged

    def _engine(self):
        return self.es.model.model  # EditScore -> Qwen3VL wrapper -> vllm.LLM

    def _require_sleep_mode(self, action: str) -> None:
        if not self.supports_offload:
            raise RuntimeError(
                f"EditScore {action} requires enable_sleep_mode: true in the scorer params — "
                "without it the vLLM engine cannot leave the GPU, and silently no-oping would "
                "report an offloaded state the memory does not reflect (per_call needs sleep mode)"
            )

    def onload(self) -> None:
        self._require_sleep_mode("onload")
        self._engine().wake_up()

    def offload(self) -> None:
        self._require_sleep_mode("offload")
        self._engine().sleep(level=1)

    def close(self) -> None:
        if hasattr(self, "es"):
            del self.es


register("editscore", EditScoreScorer)
