#!/usr/bin/env bash
# DPG-Bench prompts+questions CSV (Apache-2.0, from TencentQQGYLab/ELLA).
# 7.4 MB — above this folder's vendoring cutoff, so fetched on demand.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
curl -sSL -o data/dpg_bench.csv \
  "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/dpg_bench.csv"
echo "c42a5fe03458f158303411a3c439ca33  data/dpg_bench.csv" | md5sum -c -
