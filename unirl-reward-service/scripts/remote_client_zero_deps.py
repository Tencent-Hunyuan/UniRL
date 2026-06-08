"""Zero-dependency remote client example — only needs `requests` + `Pillow`.

Use this when you cannot / do not want to `pip install unirl-reward-service` on
the caller machine. Copy this single file over and run it. No import from
``reward_service.client`` — everything is inlined.

Usage:
    python3 remote_client_zero_deps.py --url http://<server-ip>:8080

This is intentionally one self-contained file (no local imports) so you can
scp it to any box and run without further setup. For the typed SDK version
(nicer API, typed RewardRequest), use ``scripts/remote_client_example.py``
after ``pip install unirl-reward-service`` on the caller.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import requests
from PIL import Image


def _encode_image(img: Image.Image) -> str:
    """PIL image → base64 JPEG (q=95). Matches what RewardClient sends."""
    img = img.convert("RGB") if img.mode != "RGB" else img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_session(trust_env: bool) -> requests.Session:
    """Corporate HTTP proxies (squid etc.) will return 503 for intranet
    targets — default to ignoring HTTP(S)_PROXY env vars. Pass --trust-env
    only if you genuinely must traverse a proxy to reach the reward host.
    """
    session = requests.Session()
    session.trust_env = trust_env
    return session


def score(
    session: requests.Session,
    base_url: str,
    pairs: list[tuple[str, Image.Image]],
    rewards: list[str],
    timeout: float = 120.0,
) -> tuple[list[dict], list[dict]]:
    payload = {
        "requests": [
            {
                "history": [{"text": text, "image_b64": _encode_image(img)}],
                "required_rewards": rewards,
            }
            for text, img in pairs
        ]
    }
    resp = session.post(f"{base_url.rstrip('/')}/score", json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return body["results"], body["errors"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--url", required=True,
        help="Reward service URL, e.g. http://10.1.2.3:8080",
    )
    ap.add_argument("--prompt", default="a cute dog running in the park")
    ap.add_argument(
        "--image", type=Path, required=True,
        help="Path to a candidate image (JPEG / PNG / any PIL-readable).",
    )
    ap.add_argument(
        "--rewards", nargs="+", default=["clip", "hpsv2", "pickscore"],
        help="Space-separated (clip hpsv2) or comma-separated (clip,hpsv2).",
    )
    ap.add_argument(
        "--trust-env", action="store_true",
        help="Honour HTTP(S)_PROXY env vars (default: ignore them).",
    )
    args = ap.parse_args()

    # Accept comma-separated reward names too (--rewards clip,hpsv2).
    args.rewards = [r for token in args.rewards for r in token.split(",") if r]

    session = _build_session(trust_env=args.trust_env)
    base_url = args.url.rstrip("/")

    # Sanity check: is the service up and does it advertise these rewards?
    try:
        resp = session.get(f"{base_url}/rewards", timeout=30)
        resp.raise_for_status()
        advertised = resp.json()["rewards"]
    except Exception as e:
        print(f"✗ could not reach {base_url}: {e}", file=sys.stderr)
        print(
            "  tip: if the body is an HTML squid error page, your shell has "
            "HTTP_PROXY set. Either rerun with --trust-env off (default), "
            "or add your reward host subnet to NO_PROXY.",
            file=sys.stderr,
        )
        return 1

    missing = set(args.rewards) - set(advertised)
    if missing:
        print(
            f"✗ service does not advertise reward(s): {sorted(missing)}\n"
            f"  advertised: {advertised}",
            file=sys.stderr,
        )
        return 1

    # Batch: one prompt × 4 candidates (same image reused for demo).
    img = Image.open(args.image)
    pairs = [(args.prompt, img)] * 4

    print(f"→ POST {base_url}/score  batch={len(pairs)} rewards={args.rewards}")
    results, errors = score(session, base_url, pairs, args.rewards)

    print("\n== scores ==")
    header = "  idx  " + "  ".join(f"{r:>12s}" for r in args.rewards)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, per_reward in enumerate(results):
        cells = []
        for reward in args.rewards:
            if reward in per_reward:
                first_val = next(iter(per_reward[reward].values()))
                cells.append(f"{first_val:>12.4f}")
            elif reward in errors[i]:
                cells.append(f"{'ERR':>12s}")
            else:
                cells.append(f"{'(missing)':>12s}")
        print(f"  {i:>3d}  " + "  ".join(cells))

    any_errors = any(errors)
    if any_errors:
        print("\n== per-request errors ==")
        for i, err_map in enumerate(errors):
            if err_map:
                print(f"  idx={i}: {err_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
