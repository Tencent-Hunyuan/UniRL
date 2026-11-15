---
name: pr-workflow
description: Create and repair UniRL pull requests. Use when creating a PR, editing a PR body, responding to PR Body or Semantic Pull Request CI failures, running gh pr create, or when the user mentions PR body, pull request template, body check, title check, or pull request formatting.
---

# PR Workflow

## Before Creating Or Updating A PR

1. Read `.github/pull_request_template.md`, `.github/workflows/pr-body.yml`, and `.github/workflows/semantic-pull-request.yml` from the current branch.
2. Inspect the full diff against the intended base branch, not just the latest commit.
3. Preserve commit history during rebases — **never squash commits on an existing PR branch when rebasing**, especially when collaborators or review bots have pushed commits (`git fetch`, then `git rebase` or `git rebase --rebase-merges` / replay commit-by-commit).
4. Keep PRs atomic — split unrelated cleanups or follow-up work into separate PRs, and check for overlapping open PRs or issues before publishing.
5. Do not include unrelated local files, generated artifacts, datasets, checkpoints, credentials, or outputs.
6. For a substantive code or config change, finish `.claude/skills/development/pr-review/SKILL.md` first (the submitting human remains accountable for the final diff).
7. For operator, backend, rollout-patch, or hot-path changes, verify compliance with `.claude/skills/development/hot-path-and-backend-changes/SKILL.md`.

## Title

Use the Semantic Pull Request types allowed by `.github/workflows/semantic-pull-request.yml`:

`fix`, `feat`, `docs`, `test`, `refactor`, `build`, `ci`, `chore`, `perf`, `revert`.

Prefer a concise title in the form `type(scope): summary` when a scope is helpful, otherwise `type: summary`.

## Body

Start from `.github/pull_request_template.md` and fill it with reviewer-facing content.

Required by the body check:

- `## Summary` must contain visible, non-comment text explaining what changed and why.
- `## Test Plan` must contain visible, non-comment text with exact validation commands/jobs, or `Not run; reason: ...`.

Recommended:

- Use `N/A` for sections that do not apply, and keep `## Checklist` boxes accurate.
- For `perf` or hot-path changes, put baseline -> after metrics with hardware/shape context in `## Summary`.
- In `## Test Plan`, distinguish single-GPU simulated multi-rank checks from real multi-GPU / IPC / NVLink validation, and write `Not run; reason: ...` for unexecuted tiers.
- Mention compatibility risks for config, checkpoint, backend/engine matrix, dependency extras, resource, or `DELETE-WHEN` upstream patches.
- Mention AI assistance and duplicate-work checks (`AGENTS.md` §5).
- Keep the body concise; do not produce a file-by-file changelog.

## Fixing PR Body Failures

If the `PR Body / Validate PR body sections` check fails:

1. Read the failure message.
2. Ensure the PR is not relying on HTML comments or untouched template examples as content.
3. Add real text under `## Summary` and `## Test Plan`.
4. Do not solve failures by weakening the workflow unless the user explicitly asked to change policy.
