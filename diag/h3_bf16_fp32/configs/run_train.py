"""Out-of-tree launcher: run unirl.train_diffusion and mirror every wandb.log call to $METRICS_JSONL."""

import json
import os
import runpy
import sys
import time

from wandb.sdk.wandb_run import Run

_log = Run.log


def _mirror(self, data, *args, **kwargs):
    row = {"t": time.time()}
    for key, value in dict(data).items():
        row[key] = value if isinstance(value, (int, float, str, bool)) or value is None else repr(value)[:80]
    with open(os.environ["METRICS_JSONL"], "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return _log(self, data, *args, **kwargs)


Run.log = _mirror  # wandb.init rebinds module-level wandb.log to run.log, so patch the method
sys.argv = ["unirl.train_diffusion", *sys.argv[1:]]
runpy.run_module("unirl.train_diffusion", run_name="__main__", alter_sys=True)
