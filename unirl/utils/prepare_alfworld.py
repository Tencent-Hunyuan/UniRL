"""Prepare ALFWorld rollout-driver rows for UniRL (LIN-519).

ALFWorld tasks live in the environment — the *prompt* is the env's reset observation,
not a jsonl field. This emits N driver rows, one per game,
``{"prompt": <placeholder>, "metadata": {"game_index": i}}``, so
``MultimodalRLDataSource`` drives N rollouts and :meth:`AlfworldEnv.reset` maps each
``game_index`` to a game. The index ordering matches
:func:`unirl.rollout.loop.alfworld_env.list_alfworld_games`, so the row and the env
agree on which game an index selects (and the ``n`` GRPO siblings of a prompt share the
same game). The prompt is a non-empty placeholder only because the data source rejects
empty-prompt rows — :meth:`AlfworldEnv.reset` discards it and uses the env observation.

  ALFWORLD_DATA=/path/to/alfworld/data python -m unirl.utils.prepare_alfworld --out-dir data/alfworld
"""

from __future__ import annotations

import argparse
import json
import os

from unirl.rollout.loop.alfworld_env import list_alfworld_games

# Non-empty so the data source keeps the row; AlfworldEnv.reset() replaces it with the
# environment's initial observation, so the text itself is never used for generation.
_PLACEHOLDER = "Begin the ALFWorld household task."


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit ALFWorld game-index driver rows.")
    ap.add_argument("--out-dir", default="data/alfworld")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="cap number of games (0 = all)")
    ap.add_argument(
        "--task-filter",
        default="",
        help="keep only games whose path contains this substring, e.g. a task type "
        "(pick_and_place_simple, look_at_obj_in_light, ...). A homogeneous, learnable "
        "set gives GRPO more mixed-outcome groups -> a stronger, cleaner learning signal.",
    )
    args = ap.parse_args()

    games = list_alfworld_games(args.split)
    if args.task_filter:
        games = [g for g in games if args.task_filter in g]
    if not games:
        raise SystemExit(
            "No ALFWorld games found (check $ALFWORLD_DATA / --task-filter). Run `alfworld-download` first."
        )
    if args.limit and args.limit < len(games):
        # Evenly spaced across the sorted games so a small fixed set spans task types
        # (sorted games cluster by type). A fixed set that a rollout FULLY covers
        # (prompts_per_rollout == #games) makes the per-rollout reward comparable across
        # rollouts — the curve then reflects learning, not which games were drawn.
        stride = max(1, len(games) // args.limit)
        games = games[::stride][: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(games)):
            # Carry the exact game FILE so the env plays precisely this row's game
            # (index alone drifts once the list is filtered/sampled).
            row = {"prompt": f"{_PLACEHOLDER} (game {i})", "metadata": {"game_index": i, "game_file": games[i]}}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(games)} rows -> {out_path}")


if __name__ == "__main__":
    main()
