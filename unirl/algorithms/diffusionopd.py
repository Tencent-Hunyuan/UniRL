"""DiffusionOPD — on-policy distillation for diffusion models (teacher-anchored)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.algorithms.base import (
    AlgorithmStepResult,
    StageAlgorithm,
    _gaussian_kl_div,
    _transition_sigma,
    gather_sde_field,
    typed_conditions,
)
from unirl.train.lora import adapter_active, adapter_names

DOMAIN_KEY = "domain"


@dataclass
class TeacherSpec:
    """One distillation teacher."""

    name: str
    guidance_scale: Optional[float] = None


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

        self.teachers: Dict[str, TeacherSpec] = {}
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
            raise ValueError("DiffusionOPD: at least one teacher is required.")

        model = getattr(backend, "model", None) if backend is not None else None
        if model is None:
            raise ValueError(
                "DiffusionOPD: no `backend` was injected — the teacher adapters live on the "
                "trainable model. The v2 DiffusionTrainer injects it when the algorithm "
                "declares requires_backend=True."
            )
        self._model = model
        present = adapter_names(model)
        missing = sorted(set(self.teachers) - present)
        if missing:
            raise ValueError(
                f"DiffusionOPD: teacher adapter(s) {missing} not found on the trainable model "
                f"(present: {sorted(present)}). Declare them under "
                "backend.lora_cfg.frozen_adapters as {name, path} entries."
            )
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
        if domains == {None}:
            raise RuntimeError(
                "DiffusionOPD.prepare_part: no per-row metadata[{key!r}] on the train Part. "
                "Use a domain-stamping data source (MultiDomainRLDataSource) — the trainer "
                "projects root metadata onto the train Part automatically.".format(key=DOMAIN_KEY)
            )
        if len(domains) != 1:
            raise RuntimeError(
                f"DiffusionOPD.prepare_part: mixed-domain batch {sorted(str(d) for d in domains)}; "
                "each rollout batch must be single-domain (one teacher per batch)."
            )
        domain = str(next(iter(domains)))
        teacher = self.teachers.get(domain)
        if teacher is None:
            raise RuntimeError(
                f"DiffusionOPD.prepare_part: batch domain {domain!r} has no configured teacher "
                f"(teachers: {sorted(self.teachers)})."
            )

        teacher_params = self.params
        if teacher.guidance_scale is not None:
            teacher_params = dataclasses.replace(self.params, guidance_scale=teacher.guidance_scale)

        typed_conds = typed_conditions(part.conditions, self.conditions_cls)
        with torch.no_grad(), adapter_active(self._model, teacher.name):
            result = self.stage.replay(
                typed_conds,
                segment=part.segment,
                params=teacher_params,
                step_indices=target_steps,
            )
        if result.prev_sample_means is None:
            raise RuntimeError(
                "DiffusionOPD.prepare_part: stage.replay() returned prev_sample_means=None; "
                "the distillation target requires the stage's replay to produce means."
            )
        part.segment.sde_means = result.prev_sample_means.detach().cpu()
        self._active_teacher = domain
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
        # fp32 before squaring: bf16 loses the shrinking teacher-student delta.
        student_f32 = student_means.float()
        teacher_f32 = teacher_means.to(device=student_means.device, dtype=torch.float32)

        sigma_t = _transition_sigma(
            self.stage,
            segment=segment,
            target_steps=target_steps,
            eta=float(getattr(self.params, "eta", 1.0)),
            device=student_means.device,
            add_coefficient=self.add_kl_coefficient,
        )
        kl_per_elem = _gaussian_kl_div(student_f32, teacher_f32, sigma_t)
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


__all__ = ["DiffusionOPD", "TeacherSpec", "DOMAIN_KEY"]
