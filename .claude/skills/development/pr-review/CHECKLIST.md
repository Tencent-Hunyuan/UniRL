# PR Review Checklist

Apply to added or materially changed code and the contracts it directly touches.
Cite findings by ID. Formatting, imports, docstring length, recipe targets, and
dependency direction are owned by pre-commit; do not report what a hook enforces.

| Prefix | Area | Gate |
| --- | --- | --- |
| `P0` | Correctness | Blocking |
| `P1` | Performance on hot paths: rollout, train step, collectives, weight sync | Blocking |
| `R` | Redundant or unjustified code | Blocking |
| `P2` | Maintainability | Advisory |

## Evidence

- Keeping code needs evidence: a base class, protocol, schema, generic consumer,
  real caller, or an accepted immediate dependency. Hypothetical external or future
  users are not evidence. A Python annotation alone is not runtime normalization.
- Removing code needs evidence too: every use found with `rg`, and no interface
  contract, factory, or subclass relies on it.
- A finding without `file:line` and evidence is discarded.
- An unused exported helper is dead code to remove, not a live bug.

## P0 Correctness

- [ ] `P0-OVERCATCH` `try/except` covers a narrow region. Nested failures propagate;
      broad exception translation does not hide the original actionable error.
- [ ] `P0-OVERPROTECT` No fallback that turns a misconfiguration or impossible state
      into silently different behavior.
- [ ] `P0-VALIDATE-OWNER` Structural/numeric validation lives at the owner and
      rejects invalid or non-finite values without inventing unsupported policy
      thresholds.
- [ ] `P0-MODE-VS-GRAD` Runtime mode (`train` / `eval`) and trainability
      (`requires_grad` / optimizer membership) are checked independently.
- [ ] `P0-RANK-DIVERGENCE` Every rank reaches the same collectives in the same order;
      no rank-local condition skips or reorders one.
- [ ] `P0-ARG-MUTATION` Check whether a callee mutates its arguments (e.g. torch
      `set_model_state_dict` fills the passed dict with every model entry); copy at
      the boundary and don't read the object afterward.
- [ ] `P0-LIB-SIDE-EFFECT` Neutral-sounding library helpers may change other state
      (e.g. peft `set_adapter` / `enable_adapters` flip `requires_grad`). Read the
      source before replacing a hand-written primitive with one.
- [ ] `P0-LIB-LOADER` Build library objects with the library's own loader (e.g.
      `LoraConfig.from_pretrained`, `load_peft_weights`) instead of copying selected
      fields. Partial copies silently drop semantics (e.g. `use_rslora`,
      `alpha_pattern` change LoRA scaling).
- [ ] `P0-TYPED-CONTAINER` Typed containers preserve alignment and metadata across
      concat, slice/select, serialization, and reconstruction.
- [ ] `P0-PROVENANCE` Provenance pins hash semantic content (sorted key, dtype,
      shape, bytes), not serialized files, so re-serializing the same data doesn't
      break resume.
- [ ] `P0-COMPAT-RECORD` In compatibility records, absent and empty both mean "no
      constraint"; a default empty record must not block a later, richer
      configuration.
- [ ] `P0-RESOURCE` Files, sockets, and CUDA streams use context managers; every
      cache or buffer in a long-running loop is bounded.
- [ ] `P0-DEP-FLOOR` Dependency floors agree across `pyproject.toml` and
      `requirements.txt` before relying on a newer library API.

## P1 Performance

- [ ] `P1-HOST-SYNC` No `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, or tensor
      truthiness in per-step or per-sample paths.
- [ ] `P1-VECTORIZE` No Python loop over elements of a GPU tensor.
- [ ] `P1-FILTER-EARLY` Filter before collectives, gathers, and materialization,
      not after.
- [ ] `P1-HOIST` Resolve stable optional capabilities once outside loops rather than
      repeating `getattr` / dispatch checks per item.
- [ ] `P1-EMPTY-CACHE` No `torch.cuda.empty_cache()`, `gc.collect()`, or
      `synchronize()` without profiling evidence.
- [ ] `P1-CHEAP-FIRST` Cheap structural compatibility (media kind, key shape,
      capability) is checked before expensive execution; component-specific policy
      stays local.

## R Redundancy

### Sources and contracts

- [ ] `R-DEFAULT-OWNER` Each default has one owner; callers and implementations do
      not repeat it.
- [ ] `R-STATE-OWNER` Mutable runtime state has one owner. Derive flags/properties
      from the canonical state instead of synchronizing shadow fields.
- [ ] `R-NORMALIZE-ONCE` Normalize once at the owning boundary; remove downstream
      coercion.
- [ ] `R-REQUIRED-DIRECT` Required config keys and interface members are accessed
      directly: `get_class(cfg["_target_"])`, not
      `get_class(str(cfg.get("_target_", "")))`. Keep `getattr` / `hasattr` defaults
      only for demonstrated optional capabilities.
- [ ] `R-DEAD-FIELD` Every field and argument has a real read or evidenced
      interface/factory use.
- [ ] `R-CAPABILITY-PROTOCOL` Optional behavior uses an explicit protocol/capability
      rather than meaningless methods on every implementation.
- [ ] `R-REUSE-CANONICAL` Reuse the canonical helper or constant, in the repo or in
      a dependency (torch / peft / diffusers / huggingface_hub); `rg` for the
      literal before adding another copy.
- [ ] `R-DECLARE-ONCE` Declare each configured entity once and derive dependent
      config from it, instead of repeating the same names across config sections.
- [ ] `R-PRIVATIZE` When the last external caller moves, make the helper private and
      drop it from `__all__`.

### Abstraction and control flow

- [ ] `R-SINGLE-CALLER` Inline a helper with one caller unless it is independently
      overridden, reused, isolates a resource/ownership boundary, or materially
      clarifies complex control flow.
- [ ] `R-PASSTHROUGH-OVERRIDE` After removing the last subclass-specific behavior,
      remove pass-through overrides and local helpers; verify the inherited
      implementation and sibling subclasses.
- [ ] `R-TRIVIAL-WRAPPER` Remove wrappers around one expression, conversion, or
      constructor, and trivial iterator wrappers (`any(values)`, not
      `any(x for x in values)`). Evaluate a nontrivial expression once.
- [ ] `R-REBUILD-CONTAINER` Do not rebuild a container only to reinsert the same
      mutated members.
- [ ] `R-POINTLESS-COPY` Copies and conversions protect a real ownership or
      representation boundary.
- [ ] `R-UNREACHABLE` Remove branches made unreachable by earlier validation unless
      public mutation, deserialization, subclassing, or another API boundary can
      invalidate the state. An argument no caller overrides is a constant, and a
      check already done by the upstream normalizer is dead.
- [ ] `R-CONSTANT-KNOB` Encode a single-value implementation invariant as a
      constant, not a rejected configuration knob.
- [ ] `R-FIELD-FORWARDING` Use explicit nested config; avoid field-forwarding lists.
      Require identical keys for parallel mappings of the same components.

### Scope

- [ ] `R-SCAFFOLDING` Planned scaffolding is tied to an accepted immediate
      dependency, not generic future usefulness.
- [ ] `R-TEST-ONLY-HOOK` After deleting tests or scaffolding, re-audit private
      parameters and hooks that may have existed only for them.
- [ ] `R-UNRELATED-DIFF` The diff has no unrelated changes, generated artifacts,
      debug prints, or leftover `TODO` / `FIXME`.
- [ ] `R-STACK` In a PR stack, changes to files a downstream PR rewrites go into
      that PR or a follow-up.

## P2 Maintainability

- [ ] `P2-OWNERSHIP` Place code by lifecycle ownership: a step that must run inside
      another component's construction or checkpoint boundary belongs to that
      component; algorithm semantics stay in the algorithm. Domain-specific state
      and policy stay out of shared base classes.
- [ ] `P2-DRY` Nontrivial logic duplicated at two or more real sites moves to one
      helper. Do not extract single-use code or split functions to hit a line count.
- [ ] `P2-NAMING` Public names are unabbreviated beyond common terms (`num`, `idx`,
      `cfg`); booleans read as `is_` / `has_` / `should_`; paired verbs are
      symmetric (`start/stop`, `send/recv`); no bare `data` / `result` / `info`.
- [ ] `P2-MAGIC` A literal that repeats or encodes an invariant gets a named
      constant; a one-off literal stays inline.
- [ ] `P2-DOCS` Behavior-changing defaults and required configuration are
      documented. Module and class docstrings still describe what the file owns.
      Hard-coded module, handler, and patch counts match the registration table.
      Optional steps are described as optional once, without later caveats.
