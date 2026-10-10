---
name: readme-guide
description: Write and update UniRL README files. Use when creating or editing any README.md, moving a docstring constraint or Gotcha into a README, or when a change renames, moves, or removes a path, symbol, config key, or count that a README mentions.
---

# README Guide

A README is read by an engineer who just opened the directory. Every sentence
states a fact they can act on or check: a name, path, number, condition, or
consequence.

## What belongs in a README

- What the directory owns and where it sits in the pipeline.
- The non-obvious constraint or cost that shaped the design.
- Invariants that break silently: bit-identity, rank lockstep, dtype, ordering.
- Divergences from upstream: what stock upstream does, what breaks on a version
  bump, and the retirement condition (`DELETE-WHEN`).
- How to extend it, when it is meant to be extended.

Not in a README: restated signatures or docstrings, shape and dtype annotations
of individual tensors (they stay at the code site, AGENTS.md §3), file-by-file
changelogs, history of how the code got here, TODOs, or narration of what a PR
tried.

## Package README (`unirl/**`)

Exemplars: [`weight_sync`](../../../../unirl/distributed/weight_sync/README.md),
[`sde`](../../../../unirl/sde/README.md),
[`vllm_omni/patches`](../../../../unirl/rollout/engine/vllm_omni/patches/README.md).

```markdown
# <Name>

> **Where it fits:** <loop stage>. In: <input>. Out: <output>.
> Full map: [`../README.md`](../README.md).

*<One sentence: the mechanism in plain terms.>*

## What it is
<One paragraph: what this package owns, and what it does not.>

## Why it exists
<The non-obvious constraint or cost that forces this design.>

## How it works
- **<Claim in bold.>** <Mechanism, naming the exact symbols.>

**Extending it:** <what to subclass or register, and any matching receiver.>

## Gotchas
- **<Fact in bold.>** <What breaks, when, and the fix.>
```

Sections a package needs beyond these, such as a patch table or a math-to-code
map, go between `How it works` and `Gotchas`. A model package under a parent
README that already covers the family may carry only `Where it fits` and
`Gotchas` ([`hunyuan_image3`](../../../../unirl/models/hunyuan_image3/README.md)).

## Dataset README (`datasets/**`)

Exemplars: [`dapo_math`](../../../../datasets/dapo_math/README.md),
[`geo3k_mc`](../../../../datasets/geo3k_mc/README.md).

```markdown
# <Dataset name> (<dir>)

<One or two sentences: what the data is and the local format the loader reads.>

- Recipes: <links to the recipes that consume it>
- Converter: [`prepare_<dir>.py`](prepare_<dir>.py) — `--help` carries the full contract
- Loader contract: <link>

Generated <artifacts> are local artifacts and must not be committed.

## Source
<HF ids or URLs per split; mirror setting.>

## Cook
<Dependencies, one command, the files it writes.>

## Train
<Required env vars, one command.>
```

Keep only the Recipes / Converter / Loader bullets that exist for the dataset.
Top-level indexes (`README.md`, `unirl/README.md`, `datasets/README.md`) keep
their own map structure.

## Writing facts

- Name the exact symbol, config key, file, or version in backticks; link repo
  paths relatively.
- Give numbers with units.
- A Gotcha is one bold sentence stating the fact, then what breaks and how to
  avoid it. "Silently" is spelled out when the failure is silent.
- One idea per bullet; present tense; bold only the load-bearing claim.

## Keeping it true

- A PR that changes behavior a README describes updates that README in the same
  PR.
- After renaming, moving, or removing a path, symbol, or config key, or changing
  a count, run `rg -g '*.md' '<old name>'` and fix every hit.
- Delete a Gotcha once it is no longer true; don't append a correction below it.
