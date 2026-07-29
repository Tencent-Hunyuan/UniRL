"""Isolate the EP-loader host-RAM behavior: full-read vs per-rank sliced-read.

Pure safetensors I/O (no GPU/dist). Run once per mode in a FRESH process so
ru_maxrss (process high-water mark) reflects only that mode.

Usage: python measure_ep_load.py <stacked_dir> <full|sliced> [ep_size] [ep_rank]
"""

import glob
import os
import resource
import sys

from safetensors import safe_open


def maxrss_gb() -> float:
    # Linux ru_maxrss is in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def is_expert(key: str) -> bool:
    return "mlp.experts." in key and ("gate_up_proj" in key or "down_proj" in key)


def main():
    d = sys.argv[1]
    mode = sys.argv[2]
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ep_rank = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))

    read_elems = 0
    if mode == "full":
        # Emulates the OLD loader: _read_safetensors_dir -> load_file copies every
        # tensor into RAM and keeps the whole dict resident (real host RAM).
        from safetensors.torch import load_file

        sd = {}
        for s in shards:
            sd.update(load_file(s, device="cpu"))
        read_elems = sum(t.numel() for t in sd.values())
        # ``sd`` stays resident until the function returns, so ru_maxrss captures
        # the full footprint without an extra alias.
    elif mode == "sliced":
        # NEW loader: read only THIS ep rank's expert block; .clone() forces a real
        # materialization off the mmap, then free it immediately (peak = one block).
        for s in shards:
            with safe_open(s, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    shp = sl.get_shape()
                    if is_expert(k) and ep > 1:
                        n = shp[0] // ep
                        blk = sl[ep_rank * n : (ep_rank + 1) * n].clone()
                    else:
                        blk = sl[:].clone()
                    read_elems += blk.numel()
                    del blk
    else:
        raise SystemExit("mode must be full|sliced")

    print(f"mode={mode} ep={ep} ep_rank={ep_rank} read_elems={read_elems / 1e6:.0f}M peak_host_rss={maxrss_gb():.2f}GB")


if __name__ == "__main__":
    main()
