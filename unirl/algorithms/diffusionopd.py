"""DiffusionOPD — on-policy distillation for diffusion models (teacher-anchored)."""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union

import torch
import torch.nn as nn

from unirl.algorithms.base import (
    AlgorithmStepResult,
    StageAlgorithm,
    _gaussian_kl_div,
    _transition_sigma,
    gather_sde_field,
    typed_conditions,
)
from unirl.train.lora import adapter_active, adapter_names, adapter_of_lora_key

DOMAIN_KEY = "domain"


@dataclass
class TeacherSpec:
    """One distillation teacher."""

    name: str
    guidance_scale: Optional[float] = None


class DiffusionTeacherProvider(ABC):
    """Abstract provider contract for diffusion teacher replay."""

    name: str = "teacher"

    def setup(self, *, stage: Any = None, conditions_cls: Optional[Type[Any]] = None) -> None:
        """Bind optional stage and condition context from the calling algorithm."""

    @abstractmethod
    def replay(
        self,
        conditions: Any,
        *,
        segment: Any,
        params: Any,
        step_indices: List[int],
        domain: Optional[str] = None,
    ) -> torch.Tensor:
        """Replay teacher on student trajectory; returns detached means [B, S', *latent]."""

    def wake(self) -> None:
        """Wake or onload the teacher to the execution device."""

    def offload(self) -> None:
        """Offload the teacher to CPU or sleep mode."""

    def teardown(self) -> None:
        """Release teacher resources, terminate roles, and reclaim memory."""

    def teardown_on_failure(self, exc: BaseException) -> None:
        """Exception-safe cleanup ensuring no leaked roles or memory on failure."""
        self.teardown()

    def assert_isolation(
        self,
        student_model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Assert that teacher parameters never enter student model or optimizer."""


class FrozenLoraTeacherProvider(DiffusionTeacherProvider):
    """Teacher provider backed by frozen LoRA adapters on the trainable model."""

    def __init__(
        self,
        *,
        stage: Any,
        model: torch.nn.Module,
        teachers: Mapping[str, TeacherSpec],
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        self.stage = stage
        self._model = model
        self.teachers = dict(teachers)
        self.conditions_cls = conditions_cls
        present = adapter_names(model)
        required = {spec.name for spec in self.teachers.values()}
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"FrozenLoraTeacherProvider: teacher adapter(s) {missing} not found on the trainable model "
                f"(present: {sorted(present)}). Declare them under backend.lora_cfg.frozen_adapters."
            )

    def replay(
        self,
        conditions: Any,
        *,
        segment: Any,
        params: Any,
        step_indices: List[int],
        domain: Optional[str] = None,
    ) -> torch.Tensor:
        """Replay teacher with active frozen adapter and return detached means [B, S', *latent]."""
        if domain is None:
            if len(self.teachers) == 1:
                domain = next(iter(self.teachers))
            else:
                raise RuntimeError("FrozenLoraTeacherProvider: domain is required for multi-teacher setup.")
        teacher = self.teachers.get(domain)
        if teacher is None:
            raise RuntimeError(
                f"FrozenLoraTeacherProvider.replay: batch domain {domain!r} has no configured teacher "
                f"(teachers: {sorted(self.teachers)})."
            )
        teacher_params = params
        if teacher.guidance_scale is not None:
            teacher_params = dataclasses.replace(params, guidance_scale=teacher.guidance_scale)
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad(), adapter_active(self._model, teacher.name):
            result = self.stage.replay(
                typed_conds,
                segment=segment,
                params=teacher_params,
                step_indices=step_indices,
            )
        if result.prev_sample_means is None:
            raise RuntimeError("FrozenLoraTeacherProvider.replay: stage.replay() returned prev_sample_means=None.")
        return result.prev_sample_means.detach()

    def assert_isolation(
        self,
        student_model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Assert that frozen teacher adapter parameters have requires_grad=False."""
        if self._model is not None:
            opt_p_ids = (
                {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
                if optimizer is not None
                else set()
            )
            for name, param in self._model.named_parameters():
                adapter = adapter_of_lora_key(name)
                for spec in self.teachers.values():
                    matches = (
                        (adapter == spec.name)
                        if adapter is not None
                        else (f".{spec.name}." in name or f"_{spec.name}_" in name)
                    )
                    if matches:
                        if param.requires_grad:
                            raise ValueError(
                                f"FrozenLoraTeacherProvider: teacher parameter {name!r} has requires_grad=True; "
                                "teacher adapters must be frozen."
                            )
                        if id(param) in opt_p_ids:
                            raise ValueError(
                                f"FrozenLoraTeacherProvider: teacher parameter {name!r} found in optimizer "
                                "param_groups; teacher adapters must not be in optimizer."
                            )


class FullModelTeacherProvider(DiffusionTeacherProvider):
    """Synchronous single-teacher full-model provider with explicit placement."""

    def __init__(
        self,
        *,
        model: Optional[Any] = None,
        stage: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        offload_to_cpu: bool = False,
        guidance_scale: Optional[float] = None,
        name: str = "full_teacher",
        conditions_cls: Optional[Type[Any]] = None,
        role: Optional[Any] = None,
    ) -> None:
        self.name = str(name)
        self.role = role
        self.stage = stage
        self.conditions_cls = conditions_cls
        self.guidance_scale = float(guidance_scale) if guidance_scale is not None else None
        self.dtype = dtype

        if stage is not None:
            stage_model = getattr(stage, "model", None)
            if stage_model is None:
                raise ValueError("FullModelTeacherProvider: an explicit stage must own its teacher model.")
            if model is not None and model is not stage_model:
                raise ValueError("FullModelTeacherProvider: model must be the same object as stage.model.")
            model = stage_model
        self._model = model

        if device is None:
            if model is not None and hasattr(model, "device"):
                self.device = torch.device(model.device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._freeze_parameters()

        self.offload_to_cpu = bool(offload_to_cpu)
        self._is_awake = not self.offload_to_cpu
        if self.offload_to_cpu:
            self._to_device(torch.device("cpu"))
        elif self._model is not None:
            self._to_device(self.device)

    def _freeze_parameters(self) -> None:
        """Ensure all teacher parameters have requires_grad=False."""
        for p in self.parameters():
            p.requires_grad = False

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over all teacher parameters for isolation and frozen checks."""
        if isinstance(self._model, nn.Module):
            yield from self._model.parameters()
        elif hasattr(self._model, "transformer") and isinstance(self._model.transformer, nn.Module):
            yield from self._model.transformer.parameters()

    def _to_device(self, target_device: torch.device) -> None:
        """Move underlying teacher module or wrapper to target device."""
        if self._model is None:
            return
        if isinstance(self._model, nn.Module):
            self._model.to(device=target_device, dtype=self.dtype)
        elif hasattr(self._model, "transformer") and isinstance(self._model.transformer, nn.Module):
            self._model.transformer.to(device=target_device, dtype=self.dtype)
            if hasattr(self._model, "device"):
                self._model.device = target_device

    def setup(self, *, stage: Any = None, conditions_cls: Optional[Type[Any]] = None) -> None:
        """Bind optional stage and condition context from the calling algorithm."""
        if self.conditions_cls is None:
            self.conditions_cls = conditions_cls
        if self.stage is None and stage is not None and self._model is not None:
            if hasattr(getattr(stage, "model", None), "transformer") and not hasattr(self._model, "transformer"):
                raise ValueError(
                    "FullModelTeacherProvider: this stage requires a compatible teacher bundle with its "
                    "replay configuration; a bare teacher module cannot replace the bundle."
                )
            self.stage = copy.copy(stage)
            self.stage.model = self._model

    def wake(self) -> None:
        """Wake or onload the teacher to the execution device."""
        if self.role is not None and hasattr(self.role, "wake_up"):
            self.role.wake_up()
        if self.offload_to_cpu and not self._is_awake:
            self._to_device(self.device)
            self._is_awake = True

    def offload(self) -> None:
        """Offload the teacher to CPU or sleep mode."""
        if self.role is not None and hasattr(self.role, "sleep"):
            self.role.sleep()
        if self.offload_to_cpu and self._is_awake:
            self._to_device(torch.device("cpu"))
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._is_awake = False

    def teardown(self) -> None:
        """Release teacher resources, terminate roles, and reclaim memory."""
        if self.role is not None:
            for method in ("teardown", "shutdown", "stop"):
                if hasattr(self.role, method):
                    try:
                        getattr(self.role, method)()
                    except Exception:
                        pass
                    break
            self.role = None
        if self.offload_to_cpu or self._is_awake:
            self._to_device(torch.device("cpu"))
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
        self._is_awake = False

    def teardown_on_failure(self, exc: BaseException) -> None:
        """Exception-safe cleanup ensuring no leaked roles or memory on failure."""
        self.teardown()

    def assert_isolation(
        self,
        student_model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Assert that teacher parameters never enter student model or optimizer."""
        for p in self.parameters():
            if p.requires_grad:
                raise ValueError("FullModelTeacherProvider: teacher parameter has requires_grad=True; must be frozen.")
        teacher_param_ids = {id(p) for p in self.parameters()}
        if not teacher_param_ids:
            return
        if student_model is not None:
            student_param_ids = {id(p) for p in student_model.parameters()}
            leaked = teacher_param_ids & student_param_ids
            if leaked:
                raise ValueError("FullModelTeacherProvider: teacher parameters leaked into student model parameters.")
        if optimizer is not None:
            opt_p_ids = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
            leaked = teacher_param_ids & opt_p_ids
            if leaked:
                raise ValueError("FullModelTeacherProvider: teacher parameters leaked into student optimizer groups.")

    def replay(
        self,
        conditions: Any,
        *,
        segment: Any,
        params: Any,
        step_indices: List[int],
        domain: Optional[str] = None,
    ) -> torch.Tensor:
        """Replay full-model teacher on student trajectory; returns detached means [B, S', *latent]."""
        if self.stage is None and (self.role is None or not hasattr(self.role, "replay")):
            raise RuntimeError("FullModelTeacherProvider: neither `stage` nor a runnable `role` is configured.")
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        teacher_params = params
        if self.guidance_scale is not None:
            teacher_params = dataclasses.replace(params, guidance_scale=self.guidance_scale)

        with torch.no_grad():
            if self.role is not None and hasattr(self.role, "replay"):
                result = self.role.replay(
                    typed_conds,
                    segment=segment,
                    params=teacher_params,
                    step_indices=step_indices,
                )
            else:
                result = self.stage.replay(
                    typed_conds,
                    segment=segment,
                    params=teacher_params,
                    step_indices=step_indices,
                )
        if result.prev_sample_means is None:
            raise RuntimeError("FullModelTeacherProvider: stage.replay() returned prev_sample_means=None.")
        return result.prev_sample_means.detach()


class DiffusionOPD(StageAlgorithm):
    """Teacher-anchored distillation loss over the student's own rollout."""

    requires_ema_rollout = False
    # Multi-update against the teacher anchor is unvalidated.
    supports_multi_update = False
    # Teacher adapters live on the trainable model; the trainer injects the backend.
    requires_backend = True
    # Supervision is teacher-driven; rewards (if any) are monitoring-only.
    requires_advantages = False

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
        teachers: Any = None,
        teacher_provider: Optional[DiffusionTeacherProvider] = None,
        add_kl_coefficient: bool = False,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("DiffusionOPD: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.conditions_cls = conditions_cls
        self.add_kl_coefficient = bool(add_kl_coefficient)
        if self.add_kl_coefficient and not float(getattr(params, "eta", 0.0)) > 0.0:
            raise ValueError(
                "DiffusionOPD: add_kl_coefficient=True normalizes by the SDE transition std, "
                f"which scales with sampling eta; got eta={getattr(params, 'eta', None)!r}. "
                "Use a noised rollout (eta > 0), or add_kl_coefficient=False for ODE mean-matching."
            )

        model = getattr(backend, "model", None) if backend is not None else None
        optimizer = getattr(backend, "optimizer", None) if backend is not None else None
        self._model = model

        if teacher_provider is not None:
            if not isinstance(teacher_provider, DiffusionTeacherProvider):
                raise TypeError(
                    f"DiffusionOPD: teacher_provider must be an instance of DiffusionTeacherProvider, "
                    f"got {type(teacher_provider)}."
                )
            self.teacher_provider = teacher_provider
            self.teacher_provider.setup(stage=self.stage, conditions_cls=self.conditions_cls)
            self.teacher_provider.assert_isolation(student_model=self._model, optimizer=optimizer)
            self.teachers = {}
        else:
            if model is None:
                raise ValueError(
                    "DiffusionOPD: no `backend` was injected — the default frozen-LoRA teacher "
                    "provider requires the trainable model. The v2 DiffusionTrainer injects it when the "
                    "algorithm declares requires_backend=True."
                )
            self.teachers = {}
            for entry in teachers or []:
                if isinstance(entry, TeacherSpec):
                    spec = entry
                else:
                    get = entry.get if hasattr(entry, "get") else lambda k, d=None: getattr(entry, k, d)
                    name = get("name")
                    if not name:
                        raise ValueError(f"DiffusionOPD: every teacher entry needs a 'name'; got {entry!r}.")
                    gs = get("guidance_scale")
                    spec = TeacherSpec(name=str(name), guidance_scale=None if gs is None else float(gs))
                if spec.name in self.teachers:
                    raise ValueError(f"DiffusionOPD: duplicate teacher name {spec.name!r}.")
                self.teachers[spec.name] = spec
            if not self.teachers:
                raise ValueError("DiffusionOPD: at least one teacher or a teacher_provider is required.")

            self.teacher_provider = FrozenLoraTeacherProvider(
                stage=self.stage,
                model=self._model,
                teachers=self.teachers,
                conditions_cls=self.conditions_cls,
            )
            self.teacher_provider.assert_isolation(student_model=self._model, optimizer=optimizer)

        # Set by prepare_part for the per-teacher loss metric of the current rollout.
        self._active_teacher: Optional[str] = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _resolve_target_steps(self, segment: Any) -> List[int]:
        """All SDE-recorded step indices on the segment (mirrors FlowDPPO)."""
        if segment is None or segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]

    def prepare_part(self, part: Any) -> Any:
        """Freeze the batch's teacher transition means on ``segment.sde_means``."""
        target_steps = self._resolve_target_steps(part.segment)
        if not target_steps:
            return part

        domains = {(md or {}).get(DOMAIN_KEY) for md in (part.metadata or [None] * part.batch_size)}
        domain: Optional[str] = None
        if domains != {None}:
            if len(domains) != 1:
                raise RuntimeError(
                    f"DiffusionOPD.prepare_part: mixed-domain batch {sorted(str(d) for d in domains)}; "
                    "each rollout batch must be single-domain (one teacher per batch)."
                )
            domain = str(next(iter(domains)))
        elif isinstance(self.teacher_provider, FrozenLoraTeacherProvider):
            raise RuntimeError(
                "DiffusionOPD.prepare_part: no per-row metadata[{key!r}] on the train Part. "
                "Use a domain-stamping data source (MultiDomainRLDataSource) — the trainer "
                "projects root metadata onto the train Part automatically.".format(key=DOMAIN_KEY)
            )

        try:
            self.teacher_provider.wake()
            means = self.teacher_provider.replay(
                part.conditions,
                segment=part.segment,
                params=self.params,
                step_indices=target_steps,
                domain=domain,
            )
        except Exception as exc:
            self.teacher_provider.teardown_on_failure(exc)
            raise
        finally:
            self.teacher_provider.offload()

        if means is None:
            raise RuntimeError("DiffusionOPD.prepare_part: teacher provider returned prev_sample_means=None.")

        part.segment.sde_means = means.detach().cpu()
        self._active_teacher = domain or getattr(self.teacher_provider, "name", "teacher")
        return part

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Any],
        segment: Any,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Per-step Gaussian KL between student and frozen teacher means."""
        target_steps = self._resolve_target_steps(segment)
        if not target_steps or segment.sde_means is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        student_means = replay_result.prev_sample_means  # [B, S', *latent]
        if student_means is None:
            raise RuntimeError(
                "DiffusionOPD.compute_loss_and_backward: stage.replay() returned "
                "prev_sample_means=None for the student forward."
            )

        teacher_means = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means")
        # Replay means are fp32 by contract; align the stored teacher anchor before squaring.
        teacher_f32 = teacher_means.to(device=student_means.device, dtype=torch.float32)

        sigma_t = _transition_sigma(
            self.stage,
            segment=segment,
            target_steps=target_steps,
            eta=float(getattr(self.params, "eta", 1.0)),
            device=student_means.device,
            add_coefficient=self.add_kl_coefficient,
        )
        kl_per_elem = _gaussian_kl_div(student_means, teacher_f32, sigma_t)
        kl_per_sample_step = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))  # [B, S']
        loss = kl_per_sample_step.mean()

        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {"distill_loss": float(loss.detach().item())}
        if self._active_teacher is not None:
            # Only this rollout's domain emits the key -> one wandb series per teacher.
            metrics[f"distill_loss_{self._active_teacher}"] = metrics["distill_loss"]
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )


__all__ = [
    "DiffusionOPD",
    "TeacherSpec",
    "DOMAIN_KEY",
    "DiffusionTeacherProvider",
    "FrozenLoraTeacherProvider",
    "FullModelTeacherProvider",
]
