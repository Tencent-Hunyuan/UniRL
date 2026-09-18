"""Prepare LeRobot-v3 ``lerobot/droid_100`` into Cosmos3 SFT debug samples (layout: unirl/models/cosmos3/README.md)."""

from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"prepare_droid100 needs `{module}` (pip install {module})") from exc


def download_dataset(repo: str, local_dir: Optional[str]) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, repo_type="dataset", local_dir=local_dir)


def load_meta(root: str) -> Tuple[dict, Dict[int, str], "object"]:
    pd = _require("pandas")
    with open(os.path.join(root, "meta", "info.json")) as fh:
        info = json.load(fh)

    tasks_path = os.path.join(root, "meta", "tasks.parquet")
    tasks_df = pd.read_parquet(tasks_path)
    # LeRobot v3 stores the task string either as a column or as the index.
    if "task" in tasks_df.columns:
        task_strings = tasks_df["task"].tolist()
    else:
        task_strings = tasks_df.index.tolist()
    if "task_index" in tasks_df.columns:
        tasks = {int(i): str(t) for i, t in zip(tasks_df["task_index"], task_strings)}
    else:
        tasks = {i: str(t) for i, t in enumerate(task_strings)}

    episodes_dir = os.path.join(root, "meta", "episodes")
    frames_meta = []
    for dirpath, _dirnames, filenames in os.walk(episodes_dir):
        for name in sorted(filenames):
            if name.endswith(".parquet"):
                frames_meta.append(pd.read_parquet(os.path.join(dirpath, name)))
    if not frames_meta:
        raise SystemExit(f"no episode metadata parquet under {episodes_dir}")
    episodes = pd.concat(frames_meta, ignore_index=True)
    return info, tasks, episodes


def _is_missing(value) -> bool:
    """True for absent metadata cells: None or float NaN (pandas missing marker)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _episode_video_source(episode_row, camera: str) -> Tuple[int, int, float, float]:
    """(chunk_index, file_index, from_ts, to_ts) for one episode's camera stream."""
    prefix = f"videos/{camera}/"
    row = episode_row

    def _get(*names, default=None):
        for name in names:
            if name in row and not _is_missing(row[name]):
                return row[name]
        return default

    chunk = _get(prefix + "chunk_index", "video_chunk_index", default=0)
    file = _get(prefix + "file_index", "video_file_index", default=0)
    from_ts = _get(prefix + "from_timestamp", "from_timestamp", default=0.0)
    to_ts = _get(prefix + "to_timestamp", "to_timestamp", default=None)
    return int(chunk), int(file), float(from_ts), (float(to_ts) if to_ts is not None else None)


def decode_episode_frames(
    dataset_root: str,
    info: dict,
    camera: str,
    episode_row,
    num_frames_needed: int,
) -> np.ndarray:
    """Decode one episode's ``[from_timestamp, to_timestamp)`` span for ``camera`` -> uint8 [L, H, W, 3]."""
    av = _require("av")
    chunk, file, from_ts, to_ts = _episode_video_source(episode_row, camera)
    template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = template.format(video_key=camera, chunk_index=chunk, file_index=file)
    path = os.path.join(dataset_root, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"video file not found: {path} (template={template!r})")

    frames: List[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        # Seek slightly before the episode start, then decode forward.
        seek_ts = max(from_ts - 0.5, 0.0)
        container.seek(int(seek_ts / float(stream.time_base or Fraction(1, 90000))), stream=stream, any_frame=False)
        eps = 1e-4
        for frame in container.decode(stream):
            if frame.pts is None:
                continue  # unplaceable after the early seek; keeping it would shift frame<->action pairing
            t = float(frame.pts * (stream.time_base or Fraction(1, 90000)))
            if t < from_ts - eps:
                continue
            if to_ts is not None and t >= to_ts - eps:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= num_frames_needed:
                break
    if not frames:
        raise RuntimeError(f"decoded 0 frames for episode at {path} [{from_ts}, {to_ts})")
    return np.stack(frames)


def resize_clip(clip: np.ndarray, height: int, width: int) -> torch.Tensor:
    """uint8 [T, H0, W0, 3] -> uint8 [T, 3, H, W] (bilinear, antialiased)."""
    x = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8)


def iter_episode_rows(dataset_root: str, info: dict, episodes) -> Iterator[Tuple[int, dict, "object"]]:
    """Yield (episode_index, episode_meta_row_dict, per-step dataframe)."""
    pd = _require("pandas")
    data_files: Dict[Tuple[int, int], object] = {}
    template = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    for _, row in episodes.iterrows():
        row = row.to_dict()
        ep_idx = int(row["episode_index"])
        chunk_raw = next((row[k] for k in ("data/chunk_index", "chunk_index") if not _is_missing(row.get(k))), 0)
        file_raw = next((row[k] for k in ("data/file_index", "file_index") if not _is_missing(row.get(k))), 0)
        chunk, file = int(chunk_raw), int(file_raw)
        key = (chunk, file)
        if key not in data_files:
            rel = template.format(chunk_index=chunk, file_index=file)
            data_files[key] = pd.read_parquet(os.path.join(dataset_root, rel))
        df = data_files[key]
        yield ep_idx, row, df[df["episode_index"] == ep_idx].sort_values("frame_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="lerobot/droid_100")
    parser.add_argument("--hf-local-dir", default=None, help="reuse an existing snapshot dir")
    parser.add_argument("--root", default="datasets/droid100_debug")
    parser.add_argument("--camera", default="observation.images.exterior_image_1_left")
    parser.add_argument("--num-frames", type=int, default=17, help="frames per window (chunk = num-frames - 1)")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--max-eval-windows", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=5, help="last N episodes held out")
    parser.add_argument("--action-key", default="action")
    parser.add_argument(
        "--expect-action-dim",
        type=int,
        default=7,
        help="action width the Cosmos3 recipe assumes (Cosmos3SFTConfig.raw_action_dim); "
        "prep warns if the dataset's actual dim differs so config drift is caught here.",
    )
    args = parser.parse_args()

    dataset_root = download_dataset(args.repo, args.hf_local_dir)
    info, tasks, episodes = load_meta(dataset_root)
    fps = float(info.get("fps", 15.0))
    print(f"dataset={args.repo} fps={fps} episodes={len(episodes)} camera={args.camera}")

    episode_indexes = sorted(int(i) for i in episodes["episode_index"].tolist())
    eval_set = set(episode_indexes[-args.eval_episodes :]) if args.eval_episodes else set()

    os.makedirs(os.path.join(args.root, "frames"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "actions"), exist_ok=True)

    # Pass 1: action normalization stats over train episodes.
    sums = sq_sums = None
    count = 0
    for ep_idx, _row, steps in iter_episode_rows(dataset_root, info, episodes):
        if ep_idx in eval_set:
            continue
        actions = np.stack(steps[args.action_key].to_numpy()).astype(np.float64)
        sums = actions.sum(0) if sums is None else sums + actions.sum(0)
        sq_sums = (actions**2).sum(0) if sq_sums is None else sq_sums + (actions**2).sum(0)
        count += actions.shape[0]
    if sums is None or count == 0:
        raise SystemExit(
            f"no train episodes left after holding out --eval-episodes={args.eval_episodes} "
            f"of {len(episode_indexes)} episodes; lower --eval-episodes."
        )
    mean = sums / count
    # Uniform z-normalization (gripper caveat: unirl/models/cosmos3/README.md).
    std = np.sqrt(np.maximum(sq_sums / count - mean**2, 1e-8))
    action_dim = int(len(mean))
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": int(count),
        "source": args.repo,
        "action_key": args.action_key,
        "action_dim": action_dim,
    }
    with open(os.path.join(args.root, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"action stats over {count} steps: dim={action_dim}")
    if action_dim != args.expect_action_dim:
        print(
            f"WARNING: dataset action dim {action_dim} != --expect-action-dim "
            f"{args.expect_action_dim}. Set Cosmos3SFTConfig.raw_action_dim={action_dim} "
            "in the recipe or the Cosmos3 task will raise on load."
        )

    # Pass 2: window extraction.
    manifests = {"train": [], "eval": []}
    skip_errors = (OSError, RuntimeError, _require("av").error.FFmpegError)
    skipped_episodes = 0
    for ep_idx, row, steps in iter_episode_rows(dataset_root, info, episodes):
        split = "eval" if ep_idx in eval_set else "train"
        budget = args.max_eval_windows if split == "eval" else args.max_windows
        if len(manifests[split]) >= budget:
            continue
        length = len(steps)
        if length < args.num_frames:
            continue
        task_idx = int(steps["task_index"].iloc[0]) if "task_index" in steps else 0
        instruction = tasks.get(task_idx, "").strip() or "perform the task"
        actions = np.stack(steps[args.action_key].to_numpy()).astype(np.float32)
        actions = (actions - mean.astype(np.float32)) / std.astype(np.float32)

        last_start = length - args.num_frames
        starts = list(range(0, last_start + 1, args.stride))
        max_frame = max(starts) + args.num_frames
        try:
            clip_all = decode_episode_frames(dataset_root, info, args.camera, row, max_frame)
        except skip_errors as exc:  # missing/undecodable video only; code defects propagate
            skipped_episodes += 1
            print(f"[skip] episode {ep_idx}: {exc}")
            continue
        for start in starts:
            if len(manifests[split]) >= budget:
                break
            if start + args.num_frames > len(clip_all):
                break
            sample_id = f"ep{ep_idx:04d}_f{start:05d}"
            frames = resize_clip(clip_all[start : start + args.num_frames], args.height, args.width)
            chunk = torch.from_numpy(actions[start : start + args.num_frames - 1])
            torch.save(frames, os.path.join(args.root, "frames", f"{sample_id}.pt"))
            torch.save(chunk, os.path.join(args.root, "actions", f"{sample_id}.pt"))
            manifests[split].append(
                {
                    "sample_id": sample_id,
                    "prompt": instruction,
                    "media": [
                        {
                            "modality": "video",
                            "role": "target",
                            "uri": f"frames/{sample_id}.pt",
                        },
                        {
                            "modality": "action",
                            "role": "target",
                            "uri": f"actions/{sample_id}.pt",
                        },
                    ],
                    "metadata": {
                        "fps": fps,
                        "episode_index": ep_idx,
                    },
                }
            )
        print(f"episode {ep_idx} [{split}]: total {len(manifests[split])} windows")
        if len(manifests["train"]) >= args.max_windows and len(manifests["eval"]) >= args.max_eval_windows:
            break

    if skipped_episodes:
        print(f"WARNING: skipped {skipped_episodes} episode(s) with missing/undecodable video.")
    for split, name in (("train", "manifest.jsonl"), ("eval", "eval_manifest.jsonl")):
        with open(os.path.join(args.root, name), "w") as fh:
            for record in manifests[split]:
                fh.write(json.dumps(record) + "\n")
        print(f"{name}: {len(manifests[split])} records")


if __name__ == "__main__":
    main()
