# experimental/aisim — offline rollout replay adapter

> **Where it fits:** an optional, default-off offline tool. It observes a real UniRL rollout
> window, converts it into an existing AISimulate / Dynamo Replay input, and compares the
> replay against the matched real execution. It does not schedule, score, train, or feed the
> optimizer, and normal training must run with this package absent.

## Layout

| File | Role |
| --- | --- |
| `schema.py` | Versioned trace contract: manifest, workload, measured outcomes. Stdlib only. |
| `convert.py` | Workload envelope → upstream replay input. Reads no measured outcomes. |
| `runner.py` | Not present. A runner is the only file allowed to import the simulator API, and it lands once a pinned simulator revision is confirmed. |

There is deliberately no `requirements.txt`: the adapter has no dependency of its own, and the
simulator belongs to a separate environment (see the repository rule against adding
non-additive requirements under `experimental/`).

## Boundaries that are load-bearing

- **Three inputs, one direction.** `manifest.json` and `workload.jsonl` are predictor input;
  `measured_requests.jsonl` is report input only. `convert.py` cannot read a measured result
  because no function accepts one, and `assert_no_outcome_leakage` re-checks the converted rows.
- **Observed arrivals are preserved, not reconstructed.** The open-loop contract keeps the
  real arrival offsets (rebased so the first submission is 0) and rejects a window that also
  carries dependency arrivals. It never adds a `session_id` or a synthesised tool wait.
- **A missing wait is not a zero wait.** `external_delay_ms` is required for a dependency
  arrival; `0.0` has to be written explicitly.
- **Unsupported lifecycles are refused, not trimmed.** `failed` / `aborted` / `partial` /
  `retry` terminations raise, and `capture_status="incomplete"` cannot produce a fidelity run.
  Dropping the failures and calling the remainder a complete rollout is the failure mode this
  check exists to prevent.
- **A GRPO group is not a barrier, and a trajectory slot is not an in-flight request.** Neither
  is modelled here; `to_dependency` labels both `not_modelled` in its sidecar rather than
  inventing a representation.

## Status of this package

The UniRL-side contract, converters and their CPU tests are implemented. Everything that needs
the simulator itself — locating the pinned trace parser, running the CLI, request-level result
export, trajectory-slot admission and the collector barrier — is **NOT_RUN**: no AISimulate /
Dynamo package is installed in the target environment, and the docs were only read. See the
evidence directory's capability report for the exact probe results.
