"""Fresh rollout roots must stay unique across refills that share one rollout_id.

``AsyncAgenticTrainer._next_batch`` refills a short buffer by re-submitting under the
*same* ``rollout_id``, and a data source may restart its ids on every ``get_samples``
(``DefaultDataSource`` numbers by batch position). Without a per-drive nonce the two
drives namespace different source rows identically, and ``_GroupAssembler`` — which
buckets purely by root id — merges siblings of unrelated prompts into one GRPO group.
"""

from __future__ import annotations

from unirl.data.data_source import DefaultDataSource
from unirl.trainer.agentic_async import AsyncAgenticTrainer, _GroupAssembler

ROLLOUT_ID = 7


def _bare_trainer(*, oversample: int = 2, siblings: int = 2) -> AsyncAgenticTrainer:
    trainer = object.__new__(AsyncAgenticTrainer)
    trainer._drive_seq = 0
    trainer._oversample = oversample
    trainer._n = siblings
    trainer._gt_by_root = {}
    trainer.data_source = DefaultDataSource(None)
    trainer._stop = ["</tool_call>"]
    return trainer


def _roots(tasks) -> set[str]:
    return {task.parts[0].sample_ids[0] for task in tasks}


def _prompt(task) -> str:
    return task.parts[0].primitives["text"].texts[0]


def test_a_refill_under_the_same_rollout_id_does_not_reuse_root_ids() -> None:
    trainer = _bare_trainer()

    drained = trainer._build_tasks([], ROLLOUT_ID)
    refill = trainer._build_tasks([], ROLLOUT_ID)  # _next_batch reuses the rollout_id

    assert _prompt(drained[0]) != _prompt(refill[0])  # the data source did advance
    assert not (_roots(drained) & _roots(refill))


def test_the_assembler_never_merges_siblings_of_different_prompts() -> None:
    trainer = _bare_trainer()
    drained = trainer._build_tasks([], ROLLOUT_ID)
    refill = trainer._build_tasks([], ROLLOUT_ID)

    assembler = _GroupAssembler(2)
    assembler.add_completed([drained[0]])  # one sibling still pending from the drained drive
    assembler.add_completed([refill[0]])  # a fresh sibling for a different prompt

    assert assembler.pop_complete_groups() == []
    assert len(assembler.pending_roots()) == 2


def test_each_drive_keeps_its_own_ground_truth_entries() -> None:
    trainer = _bare_trainer()

    trainer._build_tasks([], ROLLOUT_ID)
    after_drained = dict(trainer._gt_by_root)
    trainer._build_tasks([], ROLLOUT_ID)

    # The refill must ADD entries, never overwrite the drained drive's answers.
    assert set(after_drained) < set(trainer._gt_by_root)
    assert len(trainer._gt_by_root) == 2 * len(after_drained)


def test_carried_partials_keep_the_ids_they_were_submitted_under() -> None:
    trainer = _bare_trainer()
    carried = trainer._build_tasks([], ROLLOUT_ID)[:1]
    carried_root = carried[0].parts[0].sample_ids[0]

    tasks = trainer._build_tasks(carried, ROLLOUT_ID)

    # Resubmitted as-is, so a carried trajectory still rejoins its own siblings.
    assert tasks[-1].parts[0].sample_ids[0] == carried_root
