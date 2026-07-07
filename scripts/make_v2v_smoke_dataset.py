#!/usr/bin/env python
"""Generate a tiny synthetic V2V dataset for smoke-testing WAN 2.2 V2V."""

from __future__ import annotations

import argparse
import json
import os

import torch

_PROMPTS = [
    "a serene ocean at sunset, cinematic, vivid colors",
    "a cat playing in a sunny garden, high detail",
    "a busy city street at night with neon lights",
    "a snowy mountain peak under a clear blue sky",
    "a field of colorful flowers swaying in the wind",
    "a cozy cabin in a pine forest at dawn",
    "an astronaut floating above the earth",
    "a tropical beach with gently waving palm trees",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="Output directory for clips and manifest.")
    ap.add_argument("--num", type=int, default=4, help="Number of examples.")
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    vid_dir = os.path.join(out_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    gen = torch.Generator().manual_seed(int(args.seed))
    manifest_path = os.path.join(out_dir, "v2v_smoke.jsonl")
    with open(manifest_path, "w") as f:
        for i in range(int(args.num)):
            base = torch.rand(1, 3, args.height, args.width, generator=gen)
            drift = 0.05 * torch.arange(args.num_frames).view(args.num_frames, 1, 1, 1)
            frames = (base + drift).clamp(0.0, 1.0).to(torch.float32).contiguous()
            uri = os.path.join(vid_dir, f"clip_{i:03d}.pt")
            torch.save(frames, uri)
            row = {
                "prompt": _PROMPTS[i % len(_PROMPTS)],
                "prompt_id": f"v2v:{i}",
                "media": [{"modality": "video", "role": "condition", "uri": uri}],
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {args.num} clips to {vid_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
