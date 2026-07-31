# experimental/ — the incubation tier

The **official** training flows live in core (`unirl/train_*.py` + the
trainers). This tier is the other half of a deliberate two-way flow:

- **up** — packages incubate here and, once mainstream and verified, get
  absorbed and solidified into core;
- **down** — core paths that outgrow or never fit the official
  abstractions move here (or out);
- **private** — internal packages that cannot be open-sourced live here
  *uncommitted* (`experimental/private_*/`, gitignored; this tier is not
  shipped in the wheel, so nothing private can leak into a build).

Nothing under `experimental/` carries core guarantees: no API stability,
review runs at package-owner discretion, and core CI only lint-gates it.

## What a package looks like

```
experimental/<name>/
  __init__.py  run.py        # python -m experimental.<name>.run --config-name=<cfg>
  trainer.py                 # <Name>Trainer(BaseTrainer) — the loop
  roles.py                   # package-local Remote workers (when needed)
  models/                    # mirrors unirl/models/    — graduates into the matching model packages
  reward/                    # mirrors unirl/reward/    — graduates into unirl/reward/local/
  examples/                  # mirrors top-level examples/ — graduates into examples/<name>/
  README.md                  # launch one-liners + owner + the verification table
```

**Mirror naming (促成"能上能下")**: a directory is named after the core
home its content graduates into, so promotion is a structural no-op. No
`scripts/` directories — the launch surface is one documented command in
the README. No `common/` shared space — code needed by a second package
graduates into core instead of being borrowed sideways.

## Rules (lint-enforced where possible)

1. **Import direction** (`scripts/check_experimental_boundaries.py`):
   core never imports `experimental`; packages never import each other.
2. **Additive-only requirements** (same script): reward and actor share
   one Python process, so a `requirements.txt` cannot version-"isolate" —
   it may only ADD packages absent from the core stack. Corollary:
   differentiable rewards must run on the locked core stack; a reward
   that genuinely needs a conflicting stack must be non-differentiable
   and go out-of-process (`unirl-reward-service`).
3. **Single locked stack**: code targets the versions pinned by
   `pyproject.toml` only — no version-compat branches; a wrong
   environment fails loudly and the user aligns the environment.
4. **`_target_` hygiene**: every dotpath in `experimental/**` configs
   must resolve (`scripts/check_recipe_targets.py` scans this tier).
5. **Owner + verification**: each package README carries its owner and a
   verification table (config × hardware × head × status). Unverified
   drive-by configs are rejected in review.
6. **Model/engine variation goes through `_target_` polymorphism** —
   an `if model_family == ...` branch in package code is a review reject.

## Graduation

Both directions are deliberate PRs, never drive-bys. Up: a second
consumer outside the package (or the team adopting it as recommended
practice) triggers promotion into the mirrored core home — including
deduplication against any core sibling implementation. Down: core paths
bypassed by the official abstractions are candidates to move here.

## Launch & environment

Packages run from a repo checkout (`python -m experimental.<name>.run`);
this tier is intentionally **not packaged** into the wheel. Environments
come from the locked stack (image/lockfile) — see each package README's
environment section.
