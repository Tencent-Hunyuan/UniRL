"""Domain-routed AR distillation with frozen same-backbone teacher adapters."""

from dataclasses import replace
from typing import Mapping

from unirl.train.lora import adapter_names
from unirl.types.sample import Part

from .aropd import AROPD
from .base import AlgorithmStepResult


class ARMultiTeacherOPD(AROPD):
    """Select one teacher per domain-stamped rollout; see README Gotchas."""

    def __init__(self, *, teachers: Mapping[str, str], teacher: str | None = None, **kwargs) -> None:
        if teacher is not None:
            raise ValueError("ARMultiTeacherOPD uses teachers; set the inherited single teacher to null.")
        if len(teachers) < 2 or any(not domain or not teacher for domain, teacher in teachers.items()):
            raise ValueError("ARMultiTeacherOPD requires at least two non-empty domain-to-adapter entries.")
        self.teachers = dict(teachers)
        if len(set(self.teachers.values())) != len(self.teachers):
            raise ValueError("ARMultiTeacherOPD requires a different teacher adapter for each domain.")
        super().__init__(teacher=next(iter(self.teachers.values())), **kwargs)
        missing = set(self.teachers.values()) - adapter_names(self._model)
        if missing:
            raise ValueError(f"ARMultiTeacherOPD: teacher adapters not found: {sorted(missing)}.")
        self._domain = None

    def prepare_anchor(self, part: Part) -> None:
        """Choose the teacher before any micro-batch anchor replay."""
        if len(part.metadata) != part.batch_size:
            raise ValueError("ARMultiTeacherOPD requires domain metadata on every row.")
        domains = {row.get("domain") for row in part.metadata}
        if len(domains) != 1:
            raise ValueError("ARMultiTeacherOPD requires exactly one domain per rollout.")
        domain = domains.pop()
        if domain not in self.teachers:
            raise ValueError(f"ARMultiTeacherOPD: no teacher configured for domain {domain!r}.")
        self._domain = domain
        self.teacher = self.teachers[domain]

    def compute_loss_and_backward(self, **kwargs) -> AlgorithmStepResult:
        """Reuse the single-teacher objective and report metrics for the selected domain."""
        if self._domain is None:
            raise RuntimeError("ARMultiTeacherOPD requires prepare_anchor before training.")
        result = super().compute_loss_and_backward(**kwargs)
        domain_metrics = {
            f"{key}_{self._domain}": value
            for key, value in result.metrics.items()
            if key in ("distill_loss", "teacher_gap_mean", "teacher_gap_abs_mean")
        }
        return replace(result, metrics={**result.metrics, **domain_metrics})
