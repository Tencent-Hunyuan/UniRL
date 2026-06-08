## Summary

<!--
Explain what changed and why. Keep this focused on the reviewer-facing context,
not a file-by-file changelog.
-->

## Related Issue

<!--
Use "Fixes #123", "Closes #123", or "N/A" if there is no associated issue.
-->

## Test Plan

<!--
List the exact commands or jobs you ran, or write "Not run; reason: ...".
For training or rollout changes, include the model, recipe, hardware, GPU count,
and dataset/checkpoint details that make the validation reproducible.

Examples:
- `SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure`
- `pytest`
- Hydra config validation:
  `python -m unirl.train_diffusion --config-name=<domain>/<recipe> --cfg job --resolve`
- Training / rollout smoke test:
- Not run; reason:
-->

## Compatibility / Risk

<!--
Mention config changes, checkpoint compatibility, data format changes, API changes,
GPU/resource requirement changes, migrations, risky areas, or write "N/A".
-->

## Reviewer Notes

<!--
Call out follow-up work, known limitations, AI assistance, duplicate-work checks,
or anything reviewers should inspect first. Use "N/A" if there is nothing special.
-->

## Checklist

- [ ] I reviewed the changed code and removed unrelated/generated artifacts.
- [ ] I updated tests, docs, and configs where needed, or explained why not.
