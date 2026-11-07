---
name: pr-review
description: Pre-push code-standard and redundancy review for UniRL changes. Use before git push or gh pr create on a substantive code or config change, when asked to review, audit, or de-bloat a PR or diff, or when a push is blocked for a missing review record.
---

# PR Review

The pushed diff should contain only code that is correct, needed, and backed by
evidence. The rules live in [CHECKLIST.md](CHECKLIST.md); this file is the procedure.
Run it in a checkout on local disk: `rg`, `git log -S`, and pre-commit over a
network filesystem are slow enough to make agents skip steps.

## 1. Scope

```bash
git fetch upstream
base=$(git merge-base upstream/main HEAD)   # stacked PR: merge-base with the parent branch
git diff --check "$base"
git diff --stat "$base"
git diff "$base"
```

- Review the whole branch diff, not the latest commit.
- In a PR stack, read the downstream PRs' diffs before refactoring (`R-STACK`).
- Review added or materially changed code and the contracts it directly touches.
  Mention unrelated dead code; don't delete it.

## 2. Author pass

1. Inventory every added helper, class, field, argument, default, conversion, copy,
   fallback, `try/except`, `getattr`/`hasattr`, validation, branch, constant, and
   config key.
2. Run the smell scan. Hits are candidates to adjudicate, not violations:

   ```bash
   smell() { git diff -U0 "$base" -- '*.py' | re="$1" awk '/^\+\+\+ /{f=substr($0,7);next} /^@@/{split($3,a,/[+,]/);n=a[2];next} /^\+/{if($0~ENVIRON["re"]){print f":"n": "substr($0,2)} n++}'; }
   smell 'getattr\(|hasattr\(|setattr\('                  # R-REQUIRED-DIRECT, R-CAPABILITY-PROTOCOL
   smell 'except( |:)'                                    # P0-OVERCATCH
   smell '\.get\([^)]*,|or (None|\{\}|\[\])|is None'      # R-DEFAULT-OWNER, P0-OVERPROTECT
   smell '(str|int|float|bool|list|dict|tuple)\('         # R-NORMALIZE-ONCE
   smell 'deepcopy|\.clone\(|\.copy\(|\.contiguous\('     # R-POINTLESS-COPY
   smell '\.item\(|\.cpu\(|\.tolist\(|\.numpy\('          # P1-HOST-SYNC
   smell 'empty_cache|gc\.collect|synchronize\('          # P1-EMPTY-CACHE
   smell 'print\(|breakpoint\(|TODO|FIXME|XXX'            # R-UNRELATED-DIFF
   ```

3. For each inventory item and scan hit, collect evidence:
   - find all uses with `rg`;
   - read the source type/default and the real construction path;
   - check whether the repo or a dependency (torch / peft / diffusers /
     huggingface_hub) already provides it, and read that source for side effects
     before swapping;
   - use `git blame` / `git log -S` when intent is unclear.
4. Apply [CHECKLIST.md](CHECKLIST.md) and fix what it flags.
5. Check every risk and claim in the PR description against live call paths.

## 3. Independent review

The author rationalizes its own code, and a reviewer on the same model shares its
blind spots. A fresh-context subagent on a different model reviews the result: a
different model family where the tool allows it (e.g. the Cursor Task `model`
option, or another agent's CLI in non-interactive mode), otherwise at least a
different model. Give it only the repo path, base SHA, intended change, and
checklist, not the implementation history:

```text
Review `git diff <base> HEAD` in <repo> as a strict reviewer.
Read .claude/skills/development/pr-review/CHECKLIST.md and apply it to added or
materially changed code and the contracts it directly touches.
Intended change: <PR summary>.
Report only findings you can support, one per line:
`<ID> <file>:<line> — problem; evidence (callers, source default, library source,
history); concrete fix`.
Skip anything pre-commit enforces and unrelated cleanup. If nothing blocks, say why
the diff is sound.
```

For each finding, confirm it against the current code, then fix it or rebut it with
evidence. A finding you cannot confirm is recorded as rejected, with the reason.
Re-run the review after fixes that change logic. After three rounds with blocking
findings still open, stop and hand the open items to the human.

## 4. Verify

```bash
SKIP=no-commit-to-branch pre-commit run --from-ref "$base" --to-ref HEAD
```

The fixer hooks rewrite files in the worktree; keep only edits that belong in the diff.

For each simplification, exercise valid/default and malformed construction, relevant
zero/negative/NaN/Inf cases, nested failures, typed-container operations, focused
behavior, and Hydra resolution for touched recipes
(`python -m unirl.<train_entry> --config-name=<domain>/<recipe> --cfg job --resolve`).

- Refactors: load the previous implementation (`git show <rev>:<path>` as a module)
  and run old and new on the same inputs.
- Loaders and conversions: compare outputs numerically against the library's own
  reference path, not only "it loads without error".
- Verification harnesses are run, not committed (AGENTS.md §5). Keep the commands and
  results for the PR's Test Plan.
- A check that needs torch, a GPU, or multiple ranks runs where that environment
  exists. If it did not run, list it in the record and Test Plan as
  `Not run; reason: ...`; never report an unexecuted check as passing.

## 5. Record

Write the record for the exact commit being pushed. Any new commit invalidates it;
review the delta and write a new record.

```bash
dir="$(git rev-parse --git-common-dir)/unirl-review"
mkdir -p "$dir" && echo "$dir/$(git rev-parse HEAD).md"
```

The first line is `Status: pass` only when no `P0`, `P1`, or `R` finding is open;
otherwise it is `Status: blocked` and the branch is not pushed.

```markdown
Status: pass

## Fixed
- R-SINGLE-CALLER unirl/x.py:42 — inlined `_foo`, its only caller is `bar`.
## Kept
- R-REQUIRED-DIRECT unirl/y.py:10 — `getattr(model, "lora", None)` is an optional capability; only `FooModel` defines it.
## Rejected findings
- P1-HOST-SYNC unirl/z.py:88 — runs once per checkpoint save, not per step.
## Verification
- `<command>` → <result>
```

Put a short Fixed/Kept summary and the reviewer's model in the PR's
`## Reviewer Notes`, and the verification in `## Test Plan`.
