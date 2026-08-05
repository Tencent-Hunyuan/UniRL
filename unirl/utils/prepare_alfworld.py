"""Prepare ALFWorld rollout-driver rows for UniRL (LIN-519).

ALFWorld tasks live in the environment — the *prompt* is the env's reset observation,
not a jsonl field. This emits N driver rows, one per game,
``{"prompt": <placeholder>, "metadata": {"game_index": i}}``, so
``MultimodalRLDataSource`` drives N rollouts and :meth:`AlfworldEnv.reset` maps each
``game_index`` to a game. The index ordering matches
:func:`unirl.rollout.env.alfworld.list_alfworld_games`, so the row and the env
agree on which game an index selects (and the ``n`` GRPO siblings of a prompt share the
same game). The prompt is a non-empty placeholder only because the data source rejects
empty-prompt rows — :meth:`AlfworldEnv.reset` discards it and uses the env observation.

  ALFWORLD_DATA=/path/to/alfworld/data python -m unirl.utils.prepare_alfworld --out-dir data/alfworld
"""

from __future__ import annotations

import argparse
import json
import os

from unirl.rollout.env.alfworld import list_alfworld_games

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
        stride = max(1, len(games) // args.limit)
        games = games[::stride][: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(games)):
            row = {"prompt": f"{_PLACEHOLDER} (game {i})", "metadata": {"game_index": i, "game_file": games[i]}}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(games)} rows -> {out_path}")


if __name__ == "__main__":
    main()
