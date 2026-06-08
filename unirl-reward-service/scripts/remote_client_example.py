"""End-to-end example of calling a reward service from a different machine.

Prerequisite: ``pip install`` this repo on the caller machine, so that
``from reward_service.client import RewardClient`` resolves to the installed
package (not the source checkout). The script is designed to be relocatable
— copy it anywhere, as long as ``unirl-reward-service`` is on the Python path it
will run.

Usage:
    # from anywhere after `pip install /path/to/unirl-reward-service`
    python3 remote_client_example.py --url http://<server-ip>:8080 --image cand.jpg

    # batch: multiple candidate images scored against one prompt
    python3 remote_client_example.py --url http://<server-ip>:8080 \\
        --image cand_0.jpg cand_1.jpg cand_2.jpg cand_3.jpg

    # shell glob works too
    python3 remote_client_example.py --url http://<server-ip>:8080 --image cands/*.png

    # from a source checkout (bundled sample.jpg is auto-picked)
    python3 scripts/remote_client_example.py --url http://<server-ip>:8080

Shows three things a real caller will eventually need:
  1. How to point the client at a remote service (IP / hostname, not localhost).
  2. How to send a *batch* request (one prompt × N candidate images) and read
     the results in request-order.
  3. How to handle per-reward errors — the service reports failures per
     (request, reward) pair so the rest of the batch still succeeds.

If you see `503 Service Unavailable` with a squid error page, your shell has
HTTP_PROXY set and `requests` is honouring it. RewardClient defaults
``trust_env=False`` to bypass corporate proxies for direct intranet calls.
Only set ``--trust-env`` if you genuinely need to traverse a proxy to reach
the reward host.

If you see ``ModuleNotFoundError: No module named 'reward_service'``, the
Python you're running does not have the package installed — run
``python3 -m pip show unirl-reward-service`` to check, or install with
``python3 -m pip install /path/to/unirl-reward-service`` (the ``python3 -m pip``
prefix guarantees the same interpreter).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from reward_service.client import RewardClient, RewardRequest

# Only used when the caller doesn't pass --image; kept as an optional
# convenience so smoke-testing from the source checkout is zero-args.
# If the script is relocated (scp'd elsewhere) this file will not exist
# and the user must pass --image explicitly — that's the expected path.
_BUNDLED_SAMPLE = Path(__file__).resolve().parent.parent / "tests" / "assets" / "sample.jpg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url",
        required=True,
        help="Reward service URL, e.g. http://10.1.2.3:8080 or http://reward.lan:8080",
    )
    ap.add_argument(
        "--prompt",
        default="a cute dog running in the park",
        help="Text prompt to pair with each candidate image.",
    )
    ap.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=None,
        help="One or more candidate images to score against --prompt. Shell "
             "globs expand naturally (--image cands/*.png). If omitted, falls "
             "back to tests/assets/sample.jpg when running from a source "
             "checkout; otherwise required.",
    )
    ap.add_argument(
        "--rewards",
        nargs="+",
        default=["clip", "hpsv2", "pickscore"],
        help="Which rewards to query. Must all be listed by GET /rewards. "
             "Accepts space-separated (--rewards clip hpsv2) or comma-"
             "separated (--rewards clip,hpsv2) or a mix.",
    )
    ap.add_argument(
        "--trust-env",
        action="store_true",
        help="Honour HTTP(S)_PROXY env vars. Only set if your server is "
             "actually behind a proxy you must traverse.",
    )
    args = ap.parse_args()

    # Allow comma-separated reward names (--rewards clip,hpsv2) in addition
    # to the native nargs="+" form. Common copy-paste source is the README.
    args.rewards = [r for token in args.rewards for r in token.split(",") if r]

    # --image defaulting: only auto-pick the bundled sample when we're still
    # next to it (running from a source checkout). When relocated — e.g.
    # scp'd to a training box — force the caller to pass --image explicitly.
    image_paths: list[Path]
    if args.image:
        image_paths = list(args.image)
    elif _BUNDLED_SAMPLE.is_file():
        image_paths = [_BUNDLED_SAMPLE]
    else:
        print(
            "✗ --image is required (no bundled sample.jpg found next to this "
            "script). Pass --image /path/to/candidate.jpg",
            file=sys.stderr,
        )
        return 2

    missing_files = [p for p in image_paths if not p.is_file()]
    if missing_files:
        print(
            f"✗ image(s) not found: {[str(p) for p in missing_files]}",
            file=sys.stderr,
        )
        return 2

    client = RewardClient(args.url, trust_env=args.trust_env)

    # ── Sanity check: does the service advertise the rewards we want? ──────
    try:
        advertised = client.rewards()
    except Exception as e:
        print(f"✗ could not reach {args.url}: {e}", file=sys.stderr)
        print(
            "  tip: if the error mentions 'squid' or 503, your shell has a "
            "proxy set. RewardClient ignores it by default; make sure you're "
            "using the same client (not curl) and that --url is reachable.",
            file=sys.stderr,
        )
        return 1

    missing = set(args.rewards) - set(advertised)
    if missing:
        print(
            f"✗ service at {args.url} does not advertise reward(s): {sorted(missing)}\n"
            f"  advertised: {advertised}",
            file=sys.stderr,
        )
        return 1

    # ── Batch request: one prompt × N candidate images. ────────────────────
    candidates = [Image.open(p).convert("RGB") for p in image_paths]

    requests_ = [
        RewardRequest(history=[(args.prompt, img)], required_rewards=args.rewards)
        for img in candidates
    ]
    print(
        f"→ POST {args.url}/score  "
        f"batch={len(candidates)} prompt={args.prompt!r} rewards={args.rewards}"
    )
    results = client.score(requests_)

    # ── Pretty-print results in request-order. ─────────────────────────────
    print("\n== scores ==")
    name_w = max(max((len(p.name) for p in image_paths), default=5), 5)
    header = f"  idx  {'image':<{name_w}s}  " + "  ".join(f"{r:>12s}" for r in args.rewards)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, per_reward in enumerate(results):
        cells = []
        for reward in args.rewards:
            sub_metrics = per_reward.get(reward)
            if sub_metrics is None:
                cells.append(f"{'(missing)':>12s}")
            else:
                # Most rewards expose one sub-metric with the same name;
                # UnifiedReward exposes several (alignment / coherence / style).
                first_val = next(iter(sub_metrics.values()))
                cells.append(f"{first_val:>12.4f}")
        print(f"  {i:>3d}  {image_paths[i].name:<{name_w}s}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
