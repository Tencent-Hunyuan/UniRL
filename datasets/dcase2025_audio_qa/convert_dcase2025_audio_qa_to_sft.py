#!/usr/bin/env python3
"""Convert DCASE 2025 Audio QA into target-carrying UniRL SFT manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_dcase2025_audio_qa_to_unirl import convert

DEFAULT_OUT_DIR = Path("datasets/dcase2025_audio_qa_sft")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="local Hugging Face dataset snapshot")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    manifest = convert(args.snapshot, args.out_dir, supervised=True)
    print(json.dumps(manifest["stats"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
