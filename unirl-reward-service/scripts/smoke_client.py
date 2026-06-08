"""Smoke-test the running reward service.

Hits /health, /rewards, and /score against a local RewardService instance
using the canonical sample image from tests/assets/sample.jpg. Prints one
row per reward with either the sub-metric scores or the error returned by
the service.

Usage:
    python3 scripts/smoke_client.py                          # uses http://localhost:8080
    python3 scripts/smoke_client.py --url http://host:8080
    python3 scripts/smoke_client.py --prompt "a dog"         # override the prompt
    python3 scripts/smoke_client.py --rewards clip pickscore # space-separated
    python3 scripts/smoke_client.py --rewards clip,pickscore # comma-separated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from reward_service.client import RewardClient, RewardRequest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_IMAGE = _REPO_ROOT / "tests" / "assets" / "sample.jpg"


def _split_rewards(tokens: list[str] | None) -> list[str] | None:
    """Accept both `--rewards clip pickscore` and `--rewards clip,pickscore`.

    argparse's nargs="+" only handles whitespace separation, which trips up
    users who habitually comma-separate list flags. We split each token on
    commas and flatten; empty fragments (trailing comma, ",,") are dropped.
    """
    if not tokens:
        return None
    flat = [name for tok in tokens for name in tok.split(",") if name]
    return flat or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--prompt", default="a cat sitting on a table")
    ap.add_argument("--image", type=Path, default=_DEFAULT_IMAGE)
    ap.add_argument(
        "--rewards",
        nargs="+",
        default=None,
        help="subset of rewards to query; space- or comma-separated. "
             "default = all advertised by /rewards",
    )
    args = ap.parse_args()
    requested = _split_rewards(args.rewards)

    client = RewardClient(args.url)

    print(f"== {args.url} ==")
    try:
        health = client.health()
    except Exception as e:
        print(f"✗ /health failed: {e}", file=sys.stderr)
        return 1
    print("health:", json.dumps(health, indent=2))

    rewards = client.rewards()
    print(f"rewards registered ({len(rewards)}): {rewards}")

    if requested is None:
        requested = rewards
    unknown = set(requested) - set(rewards)
    if unknown:
        print(f"✗ unknown reward(s): {sorted(unknown)}", file=sys.stderr)
        return 1

    image = Image.open(args.image).convert("RGB")
    req = RewardRequest(history=[(args.prompt, image)], required_rewards=requested)

    print(f"\nscoring prompt={args.prompt!r} image={args.image.name} "
          f"size={image.size} rewards={requested}")
    (result,) = client.score([req])

    print("\n-- scores --")
    for reward in requested:
        if reward in result:
            metrics = result[reward]
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {reward:16s} {metrics_str}")
        else:
            print(f"  {reward:16s} (missing — see server logs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
