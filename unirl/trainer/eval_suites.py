"""Multi-reward eval suites — extra reward models scored during periodic eval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unirl.trainer.hydra import remote_hydra
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


@dataclass
class EvalRewardSuite:
    """One extra eval reward: a sibling reward remote + (optionally) its own eval set."""

    name: str
    reward: Any
    data_source: Optional[Any] = None
    num_prompts: Optional[int] = None


def build_eval_suites(
    eval_rewards_cfg: Optional[Any],
    *,
    data_source_cfg: DictConfig,
    enabled: bool = True,
) -> List[EvalRewardSuite]:
    """Instantiate the ``eval_rewards`` recipe list into :class:`EvalRewardSuite`\\ s."""
    if not eval_rewards_cfg:
        return []
    if not enabled:
        logger.warning("eval_rewards is set but eval is disabled (eval_interval=0) — no suite rewards are built.")
        return []
    suites: List[EvalRewardSuite] = []
    seen = set()
    for entry in eval_rewards_cfg:
        name = str(entry.get("name", "")).strip()
        if not name or name == "reward":
            raise ValueError(f"eval_rewards entries need a unique name (!= 'reward'); got {name!r}.")
        if name in seen:
            raise ValueError(f"duplicate eval_rewards name {name!r}.")
        seen.add(name)
        reward_cfg = entry.get("reward")
        if reward_cfg is None:
            raise ValueError(f"eval_rewards[{name}] is missing its `reward:` block.")
        eval_path = entry.get("eval_data_path")
        if entry.get("num_prompts") is not None and not eval_path:
            raise ValueError(
                f"eval_rewards[{name}] sets num_prompts without eval_data_path — a shared-set suite "
                "scores whatever the default pass generated (eval_num_prompts prompts)."
            )
        suite_source = None
        if eval_path:
            ds_cfg = OmegaConf.create(OmegaConf.to_container(data_source_cfg, resolve=True))
            ds_cfg.args.run.data_path = str(eval_path)
            ds_cfg.args.run.eval_data_path = str(eval_path)
            suite_source = instantiate(ds_cfg)
        suites.append(
            EvalRewardSuite(
                name=name,
                reward=remote_hydra(reward_cfg),
                data_source=suite_source,
                num_prompts=None if entry.get("num_prompts") is None else int(entry.get("num_prompts")),
            )
        )
    logger.info(
        "Eval reward suites: %s",
        ", ".join(f"{s.name}({'own set' if s.data_source is not None else 'default set'})" for s in suites),
    )
    return suites


def pad_eval_inputs(inputs: Sample, multiple: int) -> Sample:
    """Append replicated prompt rows until the root count divides ``multiple``."""
    n = inputs.batch_size
    if n <= 0 or multiple <= 1 or n % multiple == 0:
        return inputs
    source = inputs.slice(n - 1, n)
    source_root_id = source.parts[0].sample_ids[0]
    used_root_ids = set(inputs.parts[0].sample_ids)
    padded: list[Sample] = []
    for i in range((-n) % multiple):
        candidate = f"{source_root_id}:eval-pad:{i}"
        while candidate in used_root_ids:
            candidate += ":pad"
        used_root_ids.add(candidate)

        def replace_root(sample_id: str, *, new_root: str = candidate) -> str:
            root, separator, suffix = sample_id.partition("/")
            if root != source_root_id:
                raise ValueError(
                    f"pad_eval_inputs: selected pad tree contains unexpected root {root!r}; "
                    f"expected {source_root_id!r}."
                )
            return new_root + (f"/{suffix}" if separator else "")

        padded.append(source.map_sample_ids(replace_root))
    return Sample.concat([inputs, *padded])
