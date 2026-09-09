"""Fail-closed runtime contract for the public TP4 parity recipe."""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from omegaconf import DictConfig, OmegaConf

PUBLIC_REFERENCE = "public_reference"
EXPECTED_TP = 4
EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "transformers": "5.6.0",
    "vllm": "0.22.0",
}
EXPECTED_MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
EXPECTED_SYNC_TARGET = "unirl.distributed.weight_sync.full.fsdp_vllm.FSDPVLLMFullWeightSync"
PLUGIN_DISTRIBUTION = "unirl-train-inference-parity-vllm"
PLUGIN_ENTRYPOINT = "unirl_train_inference_parity"

_REQUIRED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":16:8",
    "FLASH_ATTENTION_DETERMINISTIC": "1",
    "HF_HUB_OFFLINE": "1",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "UNIRL_PARITY_ENABLE": "1",
    "UNIRL_PARITY_MODEL": "qwen3_moe_30b_a3b",
    "UNIRL_PARITY_PATCHES": "common,qwen3_moe_30b_a3b",
    "UNIRL_PARITY_PROFILE": PUBLIC_REFERENCE,
    "UNIRL_PARITY_STRICT": "1",
    "VLLM_BATCH_INVARIANT": "0",
    "VLLM_PLUGINS": PLUGIN_ENTRYPOINT,
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


def _select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, path, default=default)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_gsm8k_jsonl(path: Path, *, min_rows: int) -> list[str]:
    """Return GSM8K schema/uniqueness errors for one JSONL file."""
    errors: list[str] = []
    prompt_ids: set[str] = set()
    rows = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                prompt_id = row.get("prompt_id")
                prompt = row.get("prompt")
                metadata = row.get("metadata")
                if not isinstance(prompt_id, str) or not prompt_id.startswith("gsm8k-"):
                    errors.append(f"{path}:{line_number} has non-GSM8K prompt_id={prompt_id!r}")
                    break
                if prompt_id in prompt_ids:
                    errors.append(f"{path}:{line_number} duplicates prompt_id={prompt_id!r}")
                    break
                prompt_ids.add(prompt_id)
                if not isinstance(prompt, str) or not prompt.strip():
                    errors.append(f"{path}:{line_number} has an empty prompt")
                    break
                if not isinstance(metadata, dict) or metadata.get("answer") is None:
                    errors.append(f"{path}:{line_number} has no metadata.answer")
                    break
    except (OSError, json.JSONDecodeError, TypeError) as error:
        errors.append(f"could not validate GSM8K JSONL {path}: {type(error).__name__}: {error}")
        return errors
    if rows < min_rows:
        errors.append(f"GSM8K JSONL {path} has {rows} rows; requires at least {min_rows}")
    return errors


@dataclass(frozen=True)
class ParityRuntimeContract:
    """The one pre-launch source of truth for this experiment."""

    cfg: DictConfig
    train_tp: int
    num_devices: int
    rollout_tp: int
    max_prompt_length: int
    max_new_tokens: int
    model_path: str
    model_revision: str
    data_path: str
    staging_dir: str
    artifact_path: str

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ParityRuntimeContract":
        return cls(
            cfg=cfg,
            train_tp=_int(_select(cfg, "parity.train_tp", 0)),
            num_devices=_int(_select(cfg, "num_devices", 0)),
            rollout_tp=_int(_select(cfg, "rollout.config.tp_size", 0)),
            max_prompt_length=_int(_select(cfg, "parity.max_prompt_length", 0)),
            max_new_tokens=_int(_select(cfg, "parity.max_new_tokens", 0)),
            model_path=str(_select(cfg, "bundle.config.pretrained_model_ckpt_path", "")),
            model_revision=str(_select(cfg, "bundle.config.model_revision", "")),
            data_path=str(_select(cfg, "data_source.args.run.data_path", "")),
            staging_dir=str(_select(cfg, "parity.staging_dir", "")),
            artifact_path=str(_select(cfg, "parity.artifact_path", "")),
        )

    @property
    def environment(self) -> Dict[str, str]:
        values = dict(_REQUIRED_ENVIRONMENT)
        values["UNIRL_PARITY_TRAIN_TP"] = str(self.train_tp)
        return values

    def flags(self) -> Dict[str, Any]:
        """Return the resolved numerical switches written into verification JSON."""
        return {
            "profile": _select(self.cfg, "parity.profile"),
            "train_tp": self.train_tp,
            "rollout_tp": self.rollout_tp,
            "num_devices": self.num_devices,
            "model_precision": _select(self.cfg, "bundle.config.model_precision"),
            "model_revision": self.model_revision,
            "autocast_precision": _select(self.cfg, "pipeline.autocast_precision"),
            "logprob_precision": _select(self.cfg, "pipeline.logprob_precision"),
            "exact_actor_logprobs": _select(self.cfg, "pipeline.exact_actor_logprobs"),
            "temperature": _select(self.cfg, "sampling.temperature"),
            "top_p": _select(self.cfg, "sampling.top_p"),
            "top_k": _select(self.cfg, "sampling.top_k"),
            "logprobs_mode": _select(self.cfg, "rollout.config.engine_kwargs.logprobs_mode"),
            "max_prompt_length": self.max_prompt_length,
            "max_new_tokens": self.max_new_tokens,
            "max_model_len": _select(self.cfg, "rollout.config.engine_kwargs.max_model_len"),
            "num_updates_per_batch": _select(self.cfg, "stack.num_updates_per_batch"),
            "data_shuffle": _select(self.cfg, "data_source.args.run.shuffle"),
            "data_seed": _select(self.cfg, "data_source.args.run.seed"),
            "ignore_eos": _select(self.cfg, "rollout.config.ignore_eos"),
            "enable_prefix_caching": _select(self.cfg, "rollout.config.engine_kwargs.enable_prefix_caching"),
            "enable_chunked_prefill": _select(self.cfg, "rollout.config.engine_kwargs.enable_chunked_prefill"),
            "trust_remote_code": {
                "actor": _select(self.cfg, "bundle.config.trust_remote_code"),
                "rollout": _select(self.cfg, "rollout.config.trust_remote_code"),
            },
        }

    def validate(self) -> None:
        """Aggregate every pre-launch mismatch and raise once."""
        errors: list[str] = []

        def expect(path: str, expected: Any) -> None:
            actual = _select(self.cfg, path)
            if not _same(actual, expected):
                errors.append(f"{path}={actual!r}; expected {expected!r}")

        expect("parity.profile", PUBLIC_REFERENCE)
        expect("parity.train_tp", EXPECTED_TP)
        expect("num_devices", EXPECTED_TP)
        expect("devices_per_node", EXPECTED_TP)
        expect("rollout.config.tp_size", EXPECTED_TP)
        expect("bundle.config.model_revision", EXPECTED_MODEL_REVISION)
        expect("rollout.config.model_revision", EXPECTED_MODEL_REVISION)
        expect(
            "rollout._target_",
            "unirl.rollout.engine.vllm.engine.VLLMRolloutEngine",
        )

        expect("bundle.config.model_precision", "bf16")
        expect("bundle.config.autocast_precision", "bf16")
        expect("pipeline.autocast_precision", "bf16")
        expect("bundle.config.logprob_precision", "fp32")
        expect("pipeline.logprob_precision", "fp32")
        expect("bundle.config.exact_actor_logprobs", True)
        expect("pipeline.exact_actor_logprobs", True)
        expect(
            "bundle.config.external_libs",
            ["experimental.train_inference_parity.models.qwen3_moe_30b_a3b.fsdp"],
        )
        expect(
            "algorithm._target_",
            "experimental.train_inference_parity.algorithm.TrainInferenceParityGRPO",
        )

        for path in (
            "rollout.config.temperature",
            "sampling.temperature",
            "algorithm.sampling_temperature",
        ):
            expect(path, 1.0)
        for path in ("rollout.config.top_p", "sampling.top_p"):
            expect(path, 1.0)
        for path in ("rollout.config.top_k", "sampling.top_k"):
            expect(path, 0)
        expect(
            "rollout.config.engine_kwargs.logprobs_mode",
            "processed_logprobs",
        )
        expect("rollout.config.engine_kwargs.dtype", "bfloat16")
        expect("rollout.config.engine_kwargs.enforce_eager", True)
        expect("rollout.config.engine_kwargs.moe_backend", "triton")
        expect("rollout.config.startup_timeout_s", 300)
        expect("rollout.config.generate_timeout_s", 7200)
        expect("rollout.config.weight_update_timeout_s", 1800)
        expect("rollout.config.health_timeout_s", 60)
        expect("rollout.config.sleep_timeout_s", 180)
        expect("rollout.config.wake_up_timeout_s", 180)
        expect("rollout.config.shutdown_timeout_s", 120)

        for path in (
            "bundle.config.max_prompt_length",
            "pipeline.max_prompt_length",
            "rollout.config.max_prompt_length",
        ):
            expect(path, self.max_prompt_length)
        chat_truncation = _select(self.cfg, "rollout.config.chat_template_kwargs.truncation")
        if chat_truncation not in (None, False):
            errors.append(
                "rollout chat-template truncation must be unset; "
                "VLLMRolloutEngine applies the shared suffix limit after tokenization"
            )
        for path in (
            "rollout.config.max_new_tokens",
            "sampling.max_new_tokens",
            "algorithm.horizon",
        ):
            expect(path, self.max_new_tokens)
        expect(
            "rollout.config.engine_kwargs.max_model_len",
            self.max_prompt_length + self.max_new_tokens,
        )
        expect(
            "rollout.config.engine_kwargs.max_num_batched_tokens",
            self.max_prompt_length + self.max_new_tokens,
        )

        expect("stack.num_updates_per_batch", 1)
        expect("stack.micro_batch_size", 1)
        expect("num_rollouts", 2)
        expect("logging.report_to_wandb", False)
        expect("bundle.config.trust_remote_code", True)
        expect("rollout.config.trust_remote_code", True)
        expect("sync._target_", EXPECTED_SYNC_TARGET)
        expect("sync.flush_cache", True)
        expect("sync.host_memory_budget_mb", 8192)
        expect("sync.staging_disk_budget_mb", 71680)
        expect("backend.fsdp_cfg.activation_checkpointing", True)
        expect("bundle.config.use_gradient_checkpointing", True)
        expect("bundle.config.attn_implementation", "flash_attention_3")
        expect("sampling.samples_per_prompt", 8)
        expect("batch_size", 4)
        expect("data_source.args.algorithm.prompts_per_rollout", 4)
        expect("data_source.args.run.shuffle", True)
        expect("data_source.args.run.seed", None)

        actor_path = str(_select(self.cfg, "bundle.config.pretrained_model_ckpt_path", ""))
        rollout_path = str(_select(self.cfg, "rollout.config.pretrained_model_ckpt_path", ""))
        if not actor_path or actor_path != rollout_path:
            errors.append(
                "actor and rollout pretrained_model_ckpt_path must resolve to the same non-empty value; "
                f"got actor={actor_path!r}, rollout={rollout_path!r}"
            )
        actor_revision = str(_select(self.cfg, "bundle.config.model_revision", ""))
        rollout_revision = str(_select(self.cfg, "rollout.config.model_revision", ""))
        if actor_revision != rollout_revision:
            errors.append(f"actor and rollout model revisions differ: {actor_revision!r} vs {rollout_revision!r}")

        batch_size = _int(_select(self.cfg, "batch_size", 0))
        samples_per_prompt = _int(_select(self.cfg, "sampling.samples_per_prompt", 0))
        micro_batch_size = _int(_select(self.cfg, "stack.micro_batch_size", 0))
        total_samples = batch_size * samples_per_prompt
        if batch_size <= 0 or samples_per_prompt <= 0:
            errors.append("batch_size and sampling.samples_per_prompt must both be positive")
        elif total_samples % self.num_devices:
            errors.append(
                f"batch_size*samples_per_prompt={total_samples} is not divisible by num_devices={self.num_devices}"
            )
        elif micro_batch_size <= 0 or (total_samples // self.num_devices) % micro_batch_size:
            errors.append(
                "per-rank generated samples must be divisible by stack.micro_batch_size; "
                f"got total={total_samples}, devices={self.num_devices}, micro={micro_batch_size}"
            )

        eval_data_path = str(_select(self.cfg, "data_source.args.run.eval_data_path", ""))
        for label, raw_path in (
            ("model checkpoint", self.model_path),
            ("GSM8K training data", self.data_path),
            ("GSM8K evaluation data", eval_data_path),
        ):
            if not raw_path or not Path(raw_path).expanduser().exists():
                errors.append(f"{label} path does not exist: {raw_path!r}")
        required_unique_prompts = batch_size * _int(_select(self.cfg, "num_rollouts", 0))
        for raw_path in (self.data_path, eval_data_path):
            path = Path(raw_path).expanduser()
            if raw_path and path.is_file():
                errors.extend(_validate_gsm8k_jsonl(path, min_rows=required_unique_prompts))
        if not self.staging_dir:
            errors.append("parity.staging_dir must be explicit")
        if not self.artifact_path:
            errors.append("parity.artifact_path must be explicit")
        elif self.staging_dir:
            staging = Path(self.staging_dir).expanduser().resolve()
            artifact = Path(self.artifact_path).expanduser().resolve()
            if artifact == staging or staging in artifact.parents:
                errors.append("parity.artifact_path must not live inside the ephemeral staging_dir")

        if self.model_path and Path(self.model_path).expanduser().exists():
            try:
                from transformers import AutoConfig

                from experimental.train_inference_parity.models.qwen3_moe_30b_a3b.contract import (
                    validate_model_config,
                )

                model_config = AutoConfig.from_pretrained(
                    self.model_path,
                    revision=self.model_revision,
                    local_files_only=True,
                    trust_remote_code=True,
                )
                validate_model_config(model_config, dtype="bfloat16")
                if getattr(model_config, "model_type", None) != "qwen3_moe":
                    errors.append(
                        f"model config type={getattr(model_config, 'model_type', None)!r}; expected 'qwen3_moe'"
                    )
                architectures = tuple(getattr(model_config, "architectures", ()) or ())
                if "Qwen3MoeForCausalLM" not in architectures:
                    errors.append(f"model config architectures={architectures!r}; expected Qwen3MoeForCausalLM")
                for field in ("attention_dropout", "hidden_dropout", "hidden_dropout_prob"):
                    value = getattr(model_config, field, None)
                    if value not in (None, 0, 0.0):
                        errors.append(f"model config {field}={value!r}; exact parity requires dropout=0")
            except Exception as error:
                errors.append(f"could not validate frozen model structure: {type(error).__name__}: {error}")

        for name, expected in self.environment.items():
            actual = os.environ.get(name)
            if actual != expected:
                errors.append(f"environment {name}={actual!r}; expected {expected!r}")

        for distribution, expected in EXPECTED_VERSIONS.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"{distribution} is not installed; expected {expected}")
                continue
            if actual.split("+", 1)[0] != expected:
                errors.append(f"{distribution}=={actual}; expected public version {expected}")
        try:
            importlib.metadata.version(PLUGIN_DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{PLUGIN_DISTRIBUTION} is not installed from vllm_plugin/")

        try:
            import ray
        except ImportError:
            errors.append("ray is not installed")
        else:
            if ray.is_initialized():
                errors.append("Ray is already initialized; parity runtime_env propagation cannot be proven")

        try:
            import torch
        except ImportError:
            errors.append("torch is not importable")
        else:
            visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if visible_devices != EXPECTED_TP:
                errors.append(f"visible CUDA devices={visible_devices}; expected exactly {EXPECTED_TP}")

        if errors:
            detail = "\n".join(f"  - {error}" for error in errors)
            raise RuntimeError(f"Parity runtime contract failed:\n{detail}")


def required_environment(train_tp: int = EXPECTED_TP) -> Mapping[str, str]:
    """Expose the explicit environment for launchers and documentation tooling."""
    return {
        **_REQUIRED_ENVIRONMENT,
        "UNIRL_PARITY_TRAIN_TP": str(int(train_tp)),
    }


__all__ = [
    "EXPECTED_SYNC_TARGET",
    "EXPECTED_TP",
    "EXPECTED_VERSIONS",
    "PUBLIC_REFERENCE",
    "ParityRuntimeContract",
    "required_environment",
]
