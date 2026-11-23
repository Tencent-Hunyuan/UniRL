## Summary

<!--
Explain what changed, why, and where it sits in the architecture (`trainer`, `rollout`,
`reward`, `algorithms`, `train`, `models`, `sde`, `distributed`, `config`, `types`).
Keep this focused on reviewer-facing context, not a file-by-file changelog.

For `perf:` or hot-path changes, include baseline -> after numbers with hardware and shape
context (e.g. step time, throughput, peak memory).
-->

## Related Issue

<!--
Use "Fixes #123", "Closes #123", "RFC #123", or "N/A" if there is no associated issue.
-->

## Test Plan

<!--
List exact commands, environment, and observed results, or write "Not run; reason: ...".
Never report an unexecuted check as passing. Keep verification scripts out of the commit.

- Tier 0 (always):
  - `SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure`
  - Hydra resolution for touched recipes:
    `python -m unirl.<train_entry> --config-name=<domain>/<recipe> --cfg job --resolve`
- Tier 1 (hot-path / operator / sync / rollout changes):
  - Process-local profiling trace (`UNIRL_PROFILE=one-update|train` for training; engine-side trace for rollout) and Class A/B sync audit result
  - Numerical parity vs reference path (`max_diff` / `atol,rtol`, `use_reentrant=False` if AC changed)
- Tier 2 (single-GPU / simulated multi-rank, if run):
  - Exact command, rank count, `gloo` simulation + single-rank CUDA checks; note explicitly that this does
    NOT validate multi-GPU NCCL, NVLink/P2P, CUDA-IPC (`gpu_store` / `IPCWeightSync`), `DevicePool`, or multi-TP
- Tier 3 (real multi-GPU / multi-node, if applicable):
  - Model, recipe, GPU type & count, dataset/checkpoint, and exact command; or `Not run; reason: ...`
-->

## Compatibility / Risk

<!--
State impact across:
- Config keys, checkpoint/resume format, or data pipeline changes
- Training backend & rollout engine combinations (`FSDPBackend`, `VeOmniBackend` SP/EP,
  `trainside`, `vllm_omni`, `sglang`, `sglang_diffusion`, `fastvideo`, `composed`)
- Dependency extras exclusivity (e.g. `sglang` vs `vllm` in `pyproject.toml`)
- Rank lockstep, `dtype` / precision, `on-policy` replay ratio, or upstream patch `DELETE-WHEN`
Or write "N/A".
-->

## Reviewer Notes

<!--
Call out follow-up work, known limitations, AI assistance, duplicate-work checks (AGENTS.md §5),
`pr-review` Fixed/Kept summary + reviewer model, or any existing coupling debt nearby.
Use "N/A" if there is nothing special.
-->

## Checklist

- [ ] I reviewed the full diff, removed unrelated/generated artifacts, and verified modified files cleanly.
- [ ] I preserved one-line docstrings and updated affected `README.md` / `## Gotchas` / `DELETE-WHEN` notes.
- [ ] Layer ownership, composability, and hot-path sync rules match `unirl/README.md` and `.claude/skills/development/hot-path-and-backend-changes/SKILL.md`.
