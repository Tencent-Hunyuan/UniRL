"""Trainer: config-driven Remote role orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from hydra.utils import get_method
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from unirl.distributed.group.remote import Remote
from unirl.reward.service import RewardService
from unirl.trainer.base import BaseTrainer, build_sampling_dict

logger = logging.getLogger(__name__)

_RESERVED_ROLE_NAMES = {
    "cfg",
    "pool",
    "roles",
    "role_specs",
    "role_device_ids",
    "role_slot_ids",
    "data_source",
    "sampling_params",
    "wandb_logger",
}


@dataclass(frozen=True)
class RoleSpec:
    """Driver-side parsed role declaration."""

    name: str
    target: str
    placement: DictConfig
    raw_cfg: DictConfig
    index: int


class Trainer(BaseTrainer):
    """Base trainer that creates recipe roles as UniRL Remote handles.

    ``Trainer`` owns the generic role lifecycle:
    parse ``cfg.roles`` → topo-sort placement dependencies → create Remote
    handles with explicit ``device_ids`` / ``slot_id`` → initialize roles.
    Recipe trainers only implement ``build_req`` and ``train_step``.
    """

    def __init__(self, *, cfg: DictConfig, logging_cfg: Optional[DictConfig] = None) -> None:
        self.cfg = cfg
        self.role_specs: List[RoleSpec] = self.parse_role_specs(cfg)
        self._role_specs_by_name: Dict[str, RoleSpec] = {spec.name: spec for spec in self.role_specs}
        self._sorted_role_specs: List[RoleSpec] = self.topological_roles()
        self._simulated_slot_ids: Dict[str, int] = self._simulate_role_slots(self._sorted_role_specs)
        self._ensure_workers_per_device(cfg, self._simulated_slot_ids)

        super().__init__(cfg=cfg, logging_cfg=logging_cfg if logging_cfg is not None else cfg.get("logging"))

        self.batch_size = int(cfg.get("batch_size", 1))
        self.data_source = self.instantiate_data_source(cfg)
        self.sampling_params = build_sampling_dict(cfg.sampling) if cfg.get("sampling") is not None else {}
        self.roles: Dict[str, Any] = {}
        self.role_device_ids: Dict[str, List[int]] = {}
        self.role_slot_ids: Dict[str, int] = {}
        self._next_colocate_slot = 1
        self._roles_initialized = False

        self.setup_roles()
        self.initialize_roles()
        self.validate_config()

    # ------------------------------------------------------------------
    # Config parsing and placement planning.
    # ------------------------------------------------------------------

    def parse_role_specs(self, cfg: DictConfig) -> List[RoleSpec]:
        roles_cfg = cfg.get("roles")
        if roles_cfg is None:
            raise ValueError("Trainer requires cfg.roles as a list of role declarations.")
        if not isinstance(roles_cfg, (list, tuple, ListConfig)):
            raise TypeError(f"cfg.roles must be a list, got {type(roles_cfg).__name__}.")

        specs: List[RoleSpec] = []
        seen: set[str] = set()
        for idx, role_cfg in enumerate(roles_cfg):
            if not OmegaConf.is_config(role_cfg):
                role_cfg = OmegaConf.create(role_cfg)
            name = str(role_cfg.get("name") or "").strip()
            if not name:
                raise ValueError(f"roles[{idx}] is missing required field `name`.")
            if name in seen:
                raise ValueError(f"Duplicate role name {name!r} in cfg.roles.")
            if name in _RESERVED_ROLE_NAMES or hasattr(self.__class__, name):
                raise ValueError(f"Role name {name!r} is reserved; choose a different role name.")
            target = str(role_cfg.get("_target_") or "").strip()
            if not target:
                raise ValueError(f"roles[{idx}] ({name}) is missing required field `_target_`.")
            placement = role_cfg.get("placement") or OmegaConf.create({})
            if not OmegaConf.is_config(placement):
                placement = OmegaConf.create(placement)
            if placement.get("share_with") and placement.get("colocate_with"):
                raise ValueError(f"Role {name!r}: placement cannot set both share_with and colocate_with.")
            specs.append(RoleSpec(name=name, target=target, placement=placement, raw_cfg=role_cfg, index=idx))
            seen.add(name)
        return specs

    def topological_roles(self) -> List[RoleSpec]:
        specs = self.role_specs
        by_name = {spec.name: spec for spec in specs}
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: List[RoleSpec] = []

        def parent_of(spec: RoleSpec) -> Optional[str]:
            parent = spec.placement.get("share_with") or spec.placement.get("colocate_with")
            return str(parent) if parent else None

        def visit(spec: RoleSpec) -> None:
            if spec.name in visited:
                return
            if spec.name in visiting:
                raise ValueError(f"Cycle detected in role placement dependencies at role {spec.name!r}.")
            visiting.add(spec.name)
            parent = parent_of(spec)
            if parent:
                if parent not in by_name:
                    raise ValueError(f"Role {spec.name!r} placement references unknown role {parent!r}.")
                visit(by_name[parent])
            visiting.remove(spec.name)
            visited.add(spec.name)
            ordered.append(spec)

        for spec in specs:
            visit(spec)
        return ordered

    def _simulate_role_slots(self, specs: Iterable[RoleSpec]) -> Dict[str, int]:
        slots: Dict[str, int] = {}
        next_colocate_slot = 1
        for spec in specs:
            if spec.placement.get("share_with"):
                slots[spec.name] = slots[str(spec.placement.share_with)]
            elif spec.placement.get("colocate_with"):
                slots[spec.name] = next_colocate_slot
                next_colocate_slot += 1
            else:
                slots[spec.name] = 0
        return slots

    def _ensure_workers_per_device(self, cfg: DictConfig, slots: Dict[str, int]) -> None:
        required = max(slots.values(), default=0) + 1
        current = int(cfg.get("workers_per_device", 1))
        if required <= current:
            return
        transport_kind = str(cfg.get("transport_kind", "colocate_store"))
        if transport_kind in ("colocate_store", "colocate"):
            raise ValueError(
                "placement.colocate_with requires multiple worker slots per GPU, but "
                f"transport_kind={transport_kind!r} only supports workers_per_device=1. "
                "Set transport_kind='gpu_store' (or another multi-slot-capable transport) "
                f"and workers_per_device>={required}."
            )
        with open_dict(cfg):
            cfg.workers_per_device = required

    # ------------------------------------------------------------------
    # Role creation.
    # ------------------------------------------------------------------

    def setup_roles(self) -> None:
        self.roles = {}
        self.role_device_ids = {}
        self.role_slot_ids = {}
        self._next_colocate_slot = 1

        for spec in self._sorted_role_specs:
            device_ids, slot_id = self.resolve_role_placement(spec)
            handle = self.create_remote_role(spec, device_ids=device_ids, slot_id=slot_id)
            self.roles[spec.name] = handle
            self.role_device_ids[spec.name] = device_ids
            self.role_slot_ids[spec.name] = slot_id
            setattr(self, spec.name, handle)

    def resolve_role_placement(self, spec: RoleSpec) -> tuple[List[int], int]:
        p = spec.placement or OmegaConf.create({})
        if p.get("share_with"):
            parent = str(p.share_with)
            parent_ids = self.role_device_ids[parent]
            n = int(p.get("n_devices") or len(parent_ids))
            return self.step_subset(parent_ids, n), self.role_slot_ids[parent]
        if p.get("colocate_with"):
            parent = str(p.colocate_with)
            parent_ids = self.role_device_ids[parent]
            n = int(p.get("n_devices") or len(parent_ids))
            return self.step_subset(parent_ids, n), self.allocate_colocate_slot(parent)

        if p.get("device_ids") is not None:
            return [int(d) for d in list(p.device_ids)], 0
        n_devices = int(p.get("n_devices") or spec.raw_cfg.get("n_devices") or self.cfg.num_devices)
        if n_devices <= 0:
            raise ValueError(f"Role {spec.name!r}: placement.n_devices must be positive, got {n_devices}.")
        return self.pool.allocate(n_devices), 0

    @staticmethod
    def step_subset(device_ids: List[int], n_devices: int) -> List[int]:
        if n_devices <= 0:
            raise ValueError(f"n_devices must be positive, got {n_devices}.")
        if n_devices > len(device_ids):
            raise ValueError(f"Cannot take {n_devices} devices from parent device slab {device_ids}.")
        if n_devices == len(device_ids):
            return list(device_ids)
        if n_devices == 1:
            return [device_ids[0]]
        last = len(device_ids) - 1
        return [device_ids[round(i * last / (n_devices - 1))] for i in range(n_devices)]

    def allocate_colocate_slot(self, parent: str) -> int:
        del parent  # The current implementation allocates globally unique colocate slots.
        slot = max(self._next_colocate_slot, max(self.role_slot_ids.values(), default=0) + 1)
        self._next_colocate_slot = slot + 1
        return slot

    def resolve_role_cls(self, target: str) -> type:
        role_cls = get_method(target)
        if not isinstance(role_cls, type) or not issubclass(role_cls, Remote):
            raise TypeError(f"Role target {target!r} must resolve to a Remote subclass, got {role_cls!r}.")
        return role_cls

    def prepare_role_cfg(self, spec: RoleSpec) -> DictConfig:
        container = OmegaConf.to_container(spec.raw_cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(f"Role {spec.name!r} config must resolve to a mapping.")
        for key in ("name", "_target_", "placement"):
            container.pop(key, None)
        return OmegaConf.create(container)

    def create_remote_role(self, spec: RoleSpec, *, device_ids: List[int], slot_id: int):
        role_cls = self.resolve_role_cls(spec.target)
        if issubclass(role_cls, RewardService):
            # RewardService takes a materialized ``backend`` (not a ``cfg`` blob):
            # pass the role's own fields as plain-dict kwargs so the worker's
            # ``_resolve_init_kwargs`` walker instantiates the nested ``_target_``
            # backend in its own CUDA context.
            init_kwargs = OmegaConf.to_container(self.prepare_role_cfg(spec), resolve=True)
        else:
            init_kwargs = {"cfg": self.prepare_role_cfg(spec)}
        return self.pool.create_remote(
            role_cls,
            device_ids=device_ids,
            slot_id=slot_id,
            role_name=spec.name,
            init_kwargs=init_kwargs,
        )

    def initialize_roles(self) -> None:
        if self._roles_initialized:
            return
        for spec in self._sorted_role_specs:
            self.roles[spec.name].initialize()
        self._roles_initialized = True

    # ------------------------------------------------------------------
    # Data, validation, training loop.
    # ------------------------------------------------------------------

    def instantiate_data_source(self, cfg: DictConfig):
        if cfg.get("data_source") is None:
            return None
        from hydra.utils import instantiate

        return instantiate(cfg.data_source)

    def validate_config(self) -> None:
        if self.data_source is None:
            raise ValueError("Trainer requires cfg.data_source.")
        if not self.sampling_params:
            raise ValueError("Trainer requires cfg.sampling.")

    def build_req(self, inputs: Any, rollout_id: int) -> Any:
        raise NotImplementedError

    def train_step(self, req: Any, *, training_progress: float = 0.0, rollout_id: int = 0) -> Dict[str, Any]:
        raise NotImplementedError

    def train(
        self,
        *,
        num_rollouts: Optional[int] = None,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: Optional[str] = None,
    ) -> None:
        num_rollouts = int(num_rollouts if num_rollouts is not None else self.cfg.get("num_rollouts", 100))
        save_interval = int(save_interval if save_interval is not None else self.cfg.get("save_interval", 0))
        save_dir = save_dir if save_dir is not None else self.cfg.get("save_dir")
        load_dir = load_dir if load_dir is not None else self.cfg.get("load_dir")
        save_mode = save_mode if save_mode is not None else self.cfg.get("save_mode", "auto")

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        data_source = self.data_source
        if data_source is None:
            raise ValueError("Trainer requires cfg.data_source.")
        for _ in range(start_rollout):
            data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = data_source.get_samples(self.batch_size)
                req = self.build_req(inputs, rollout_id)
                metrics = self.train_step(req, training_progress=training_progress, rollout_id=rollout_id)
                self.log_metrics(metrics, rollout_id=rollout_id, num_rollouts=num_rollouts)
                self.maybe_save_checkpoint(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            self._finish_wandb()

    def log_metrics(self, metrics: Dict[str, Any], *, rollout_id: int, num_rollouts: int) -> None:
        trainer_name = self.__class__.__name__.removesuffix("Trainer") or self.__class__.__name__
        metric_text = " ".join(
            f"{key}={self._format_metric_for_log(value)}"
            for key, value in metrics.items()
        )
        if metric_text:
            logger.info("%s rollout %d/%d  %s", trainer_name, rollout_id + 1, num_rollouts, metric_text)
        else:
            logger.info("%s rollout %d/%d", trainer_name, rollout_id + 1, num_rollouts)

        wb = self.wandb_logger
        if wb is not None and wb.initialized:
            wb.log_step(
                step=rollout_id + 1,
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                prefix=str((self.logging_cfg or {}).get("metric_prefix", "train/")),
            )

    @staticmethod
    def _format_metric_for_log(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            abs_value = abs(value)
            if value != 0.0 and (abs_value < 1e-3 or abs_value >= 1e4):
                return f"{value:.6e}"
            return f"{value:.4f}"
        return repr(value)

    # ------------------------------------------------------------------
    # Role-aware checkpointing.
    # ------------------------------------------------------------------

    def checkpoint_roles(self, path: str, *, step: int, mode: str) -> None:
        for name, role in self.roles.items():
            if hasattr(role, "save_checkpoint"):
                role_path = os.path.join(path, name)
                role.save_checkpoint(role_path, step=step, mode=mode)

    def load_checkpoint_roles(self, path: str) -> int:
        starts: List[int] = []
        for name, role in self.roles.items():
            if hasattr(role, "load_checkpoint"):
                role_path = os.path.join(path, name)
                if os.path.exists(role_path):
                    result = role.load_checkpoint(role_path)
                    if isinstance(result, list):
                        result = result[0] if result else 0
                    starts.append(int(result or 0))
        return max(starts, default=0)

    def maybe_save_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str = "auto",
    ) -> None:
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        logger.info("Saving role checkpoint at rollout %d/%d -> %s", step, num_rollouts, path)
        self.checkpoint_roles(path, step=step, mode=save_mode)
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts: Optional[int] = None) -> int:
        if not load_dir:
            return 0
        load_dir = os.path.abspath(load_dir)
        logger.info("Loading role checkpoint from %s", load_dir)
        start = self.load_checkpoint_roles(load_dir)
        state_path = os.path.join(load_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                self._resume_state = json.load(f)
        logger.info("Checkpoint restored; resuming at rollout %d", start)
        if num_rollouts is not None and start >= num_rollouts:
            logger.warning(
                "Checkpoint step %d >= num_rollouts %d — nothing left to train.",
                start,
                num_rollouts,
            )
        return start


__all__ = ["Trainer", "RoleSpec"]
