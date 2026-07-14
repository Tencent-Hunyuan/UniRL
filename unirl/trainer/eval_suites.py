"""Multi-reward eval suites — extra reward models scored during periodic eval.

The ``eval_rewards`` recipe list declares EXTRA reward models to score during
``evaluate()`` — beyond the training reward's ``eval/reward`` — so checkpoints
can be selected on independent metrics (reward hacking shows up as the training
reward climbing while the others stall)::

    eval_rewards:
      - name: hpsv2                 # unique, != "reward"; logged as eval/hpsv2
        reward:                     # full reward cfg — same schema as the
          _target_: unirl.reward.service.RewardService   # top-level `reward:`
          backend: {...}
      - name: geneval
        reward: {...}
        eval_data_path: datasets/geneval/eval.jsonl  # OPTIONAL: own prompt set
        num_prompts: 64                              # OPTIONAL: own-pass size

An entry WITHOUT ``eval_data_path`` scores the same images the default eval
pass already generated (zero extra generation — the cheapest way to compare
checkpoints on several metrics over one prompt set). An entry WITH it gets its
own generation pass over its own prompts (e.g. a GenEval manifest or an OCR
prompt set), sized by ``num_prompts`` (default: the trainer's
``eval_num_prompts``).

Placement: :func:`build_eval_suites` must be called inside the SAME placement
context that created the trainer's training reward — each suite reward becomes
a sibling remote there, so where the trainer has a ``reward_fraction`` slab
(DiffusionTrainer, ReFL) ALL eval rewards share that dedicated-GPU slab, and
elsewhere (PE, UnifiedModel) they colocate with the training reward.

Data: an own-set suite instantiates its own driver-side data source — the
trainer's ``data_source_cfg`` with ``args.run.data_path`` /
``args.run.eval_data_path`` both pointed at the suite file — so every prompt
format the trainer's data source reads (txt / JSONL / JSON manifests with
metadata) works per suite.

Scoring uses the trainer's own reward interface: composed/rollout trainers call
``suite.reward.score_and_attach``, ReFL calls ``score_differentiable`` — a
suite's backend must support whichever its trainer uses (the same contract as
the training reward).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)


@dataclass
class EvalRewardSuite:
    """One extra eval reward: a sibling reward remote + (optionally) its own eval set."""

    name: str  # wandb key: eval/<name>
    reward: Any  # reward remote, placed next to the training reward
    data_source: Optional[Any] = None  # None → scores the default eval pass
    num_prompts: Optional[int] = None  # own-pass size; None → eval_num_prompts


def build_eval_suites(
    eval_rewards_cfg: Optional[Any],
    *,
    data_source_cfg: DictConfig,
    enabled: bool = True,
) -> List[EvalRewardSuite]:
    """Instantiate the ``eval_rewards`` recipe list into :class:`EvalRewardSuite`\\ s.

    MUST be called inside the placement context that owns the training reward
    (suite rewards become sibling remotes there). Returns ``[]`` when the list
    is unset/empty — the trainer then runs its single-reward eval unchanged —
    or when ``enabled`` is False (eval off), in which case a non-empty list
    only logs a warning instead of loading reward models that would never run.
    """
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
            # Own prompt set: clone the trainer's data-source cfg with both paths
            # pointed at the suite file (a suite source only ever serves eval
            # batches; data_path merely backs the class's load-time existence check).
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
