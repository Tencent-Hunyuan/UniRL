# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Default code agent guidance lives here for Claude Code. Other agents, including Cursor, should reference this file with `@CLAUDE.md` and add only agent-specific differences.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Contribution Hygiene

**Avoid duplicate work, trivial PRs, and unreviewed agent output.**

Before proposing or opening a PR:
- Check the relevant issue, open PRs, and short area keywords for overlapping work.
- If another open PR already addresses the same fix, do not open a duplicate.
- If your approach is materially different from existing work, explain the difference before proceeding.

Do not create PRs for low-value busywork:
- No one-off typo fixes, isolated style churn, or mechanical cleanups without substantive work.
- Bundle mechanical cleanup only when it directly supports a meaningful change.

For AI-assisted work:
- A human submitter must understand and be able to defend every changed line.
- The submitting human should review the full diff and run relevant tests before publication.
- PR descriptions should mention AI assistance, duplicate-work checks, and test commands with results.

Fail closed when the work is not ready:
- If the change is duplicate, too trivial, missing context, or lacks a credible verification path, stop and explain what is missing.
- Do not invent process exceptions just to keep moving.

## 6. Review and Domain Guides

**Verify guidance against the current repo before applying it.**

- Treat agent or bot review comments as suggestions, not facts. Confirm they still apply to the current code before changing anything.
- Before editing specialized areas, read and follow the relevant local guide or skill.
- If a guide conflicts with the requested change, refuse that part of the change and explain the conflict.

Local skills currently in this repo:

| Area | Skill | Read before |
| --- | --- | --- |
| Model bundles | `.claude/skills/development/add-model-bundle/SKILL.md` | Adding or updating diffusion or autoregressive model pipelines, model config dataclasses, Bundle/Pipeline/Stage/Conditions implementations, LoRA targets, FSDP wrapping hints, RolloutReq/RolloutResp plumbing, or multimodal text/image/video conditioning. |
| Pull requests | `.claude/skills/development/pr-workflow/SKILL.md` | Creating or updating PRs, editing PR bodies, handling PR Body or Semantic Pull Request CI failures, or running `gh pr create`. |

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
