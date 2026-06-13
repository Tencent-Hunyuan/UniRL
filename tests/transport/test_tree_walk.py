"""Unit tests for the transport tree-walkers — ``map_tree`` + ``_collect_leaves``.

The transport layer rewrites nested value trees (Batch / dict / list / tuple
holding ``TensorRef`` proxies) in two ways:

  - ``map_tree(obj, leaf_fn)`` — functional rebuild: ``leaf_fn`` runs on every
    node; a *different* return replaces the node and stops recursion, else the
    container is rebuilt structurally. ``TensorRef`` is an ATOMIC leaf (never
    recursed into, despite subclassing ``Batch``); ``Batch`` is rebuilt via
    ``_rebuild`` (preserving the hidden ``_packed_cu_seqlens``).
  - ``_collect_leaves`` / ``_collect_dict`` / ``_collect_list`` — collect
    leaves of a given type by dotted path, plus setter closures that write
    replacements back. This is the gather half of ``dehydrate`` / ``hydrate``,
    including the ``fields=`` prefix filter.

Pure-CPU: handles are faked with the minimal ``.local()/.shape/.dtype/.device``
protocol (the ``test_tensorref_spans`` pattern). ``TensorRef`` leaves resolve
through a tiny in-process ``TensorTransport`` whose ``_resolve_handles`` just
reads ``handle.local()`` — no Ray, no GPU, no real backend.
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from unirl.distributed.tensor.batch import Batch, packed_field, shared_field
from unirl.distributed.tensor.transport import (
    TensorRef,
    TensorSpan,
    TensorTransport,
    _collect_leaves,
    map_tree,
)

pytestmark = pytest.mark.cpu


# ── fake handle / ref builders (mirrors test_tensorref_spans) ─────────────────


class _FakeHandle:
    def __init__(self, t: torch.Tensor):
        self.t = t
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = t.device

    def local(self) -> torch.Tensor:
        return self.t


def _ref(t: torch.Tensor) -> TensorRef:
    """Wrap one tensor as a single full-range-span TensorRef."""
    return TensorRef(
        spans=[TensorSpan(_FakeHandle(t), 0, int(t.shape[0]))],
        shape=tuple(t.shape),
        dtype=t.dtype,
        device="cpu",
    )


# ── sample Batch subclasses holding TensorRef fields ─────────────────────────


@dataclass
class RefHolder(Batch):
    """A non-TensorRef Batch holding a nested TensorRef + a plain shared field."""

    ref: Optional[TensorRef] = shared_field(default=None)
    tag: str = shared_field(default="t")


@dataclass
class PackedHolder(Batch):
    payload: Optional[torch.Tensor] = packed_field(default=None)
    note: str = shared_field(default="n")


# =============================================================================
# map_tree
# =============================================================================


def test_map_tree_applies_leaf_fn_to_every_node():
    seen = []

    def fn(x):
        seen.append(x)
        return x  # identity -> recursion continues

    obj = {"a": [1, 2], "b": ("x",)}
    out = map_tree(obj, fn)
    assert out == obj
    # every container AND scalar leaf was visited
    assert obj in seen and [1, 2] in seen and 1 in seen and 2 in seen
    assert ("x",) in seen and "x" in seen


def test_map_tree_replacement_stops_recursion():
    visited = []

    def fn(x):
        visited.append(x)
        if x == "REPLACE_ME":
            return "REPLACED"  # different object -> node swapped, no recursion below
        return x

    out = map_tree({"k": "REPLACE_ME", "deep": [1]}, fn)
    assert out == {"k": "REPLACED", "deep": [1]}


def test_map_tree_replaces_node_and_does_not_recurse_into_it():
    # leaf_fn that swaps a whole list for a sentinel: the list's elements are
    # never visited because the swap stops recursion at that node.
    visited = []

    def fn(x):
        visited.append(x)
        if isinstance(x, list):
            return "SWAPPED"
        return x

    out = map_tree({"xs": [10, 20, 30]}, fn)
    assert out == {"xs": "SWAPPED"}
    assert 10 not in visited and 20 not in visited  # never descended into the list


def test_map_tree_tensorref_is_atomic_leaf():
    # TensorRef subclasses Batch, but map_tree must treat it as an atomic leaf:
    # an identity leaf_fn returns the SAME ref untouched, never recursing into
    # its `spans`/`shape`/... fields.
    ref = _ref(torch.arange(3).float())
    out = map_tree(ref, lambda x: x)
    assert out is ref


def test_map_tree_replaces_tensorref_via_leaf_fn():
    ref = _ref(torch.arange(3).float())
    other = _ref(torch.arange(100, 103).float())

    def fn(x):
        return other if x is ref else x

    assert map_tree({"r": ref}, fn)["r"] is other  # swapped wholesale
    assert map_tree([ref], fn)[0] is other
    assert map_tree((ref,), fn)[0] is other


def test_map_tree_rebuilds_batch_field_wise():
    # A non-TensorRef Batch is rebuilt via _rebuild; its TensorRef field is an
    # atomic leaf that leaf_fn may swap.
    ref = _ref(torch.arange(3).float())
    swapped = _ref(torch.arange(50, 53).float())
    holder = RefHolder(ref=ref, tag="orig")

    def fn(x):
        return swapped if x is ref else x

    out = map_tree(holder, fn)
    assert isinstance(out, RefHolder) and out is not holder  # rebuilt, not mutated
    assert out.ref is swapped
    assert out.tag == "orig"  # passthrough field preserved


def test_map_tree_rebuilds_nested_dict_list_batch_tuple():
    r1 = _ref(torch.arange(2).float())
    r2 = _ref(torch.arange(2).float())
    swaps = {id(r1): _ref(torch.tensor([9.0, 9.0])), id(r2): _ref(torch.tensor([8.0, 8.0]))}

    def fn(x):
        return swaps.get(id(x), x)

    nested = {
        "list": [r1, ("tuple_inner", r2)],
        "batch": RefHolder(ref=r1, tag="z"),
    }
    out = map_tree(nested, fn)
    assert out["list"][0] is swaps[id(r1)]
    assert out["list"][1][0] == "tuple_inner"  # tuple rebuilt element-wise
    assert out["list"][1][1] is swaps[id(r2)]
    assert out["batch"].ref is swaps[id(r1)] and out["batch"].tag == "z"


def test_map_tree_rebuild_preserves_packed_cu_seqlens():
    # Batch._rebuild carries the framework-managed cu_seqlens that the plain
    # constructor never sees; map_tree must preserve it across the rebuild.
    seg = PackedHolder.pack(payload=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
    cu_before = seg.cu_seqlens.clone()

    out = map_tree(seg, lambda x: x)  # identity walk forces a _rebuild
    assert out is not seg
    assert torch.equal(out.cu_seqlens, cu_before)


def test_map_tree_passthrough_other_values():
    # non-container, non-Batch values pass straight through
    assert map_tree(42, lambda x: x) == 42
    assert map_tree("hello", lambda x: x) == "hello"
    assert map_tree(None, lambda x: x) is None


# =============================================================================
# _collect_leaves / _collect_dict / _collect_list
# =============================================================================


def _collect(value, leaf_type=TensorRef, filter_fn=None):
    collected: dict = {}
    setters: dict = {}
    _collect_leaves(value, "", leaf_type, collected, setters, filter_fn)
    return collected, setters


def test_collect_leaves_nested_dict_dotted_paths():
    r1 = _ref(torch.arange(2).float())
    r2 = _ref(torch.arange(2).float())
    collected, _ = _collect({"outer": {"inner": r1}, "top": r2})
    # nested dict keys join on '.'; top-level dict key is bare (prefix == "")
    assert collected == {"outer.inner": r1, "top": r2}


def test_collect_leaves_nested_batch_dotted_paths():
    ref = _ref(torch.arange(2).float())
    holder = RefHolder(ref=ref, tag="t")
    collected, _ = _collect(holder)
    # Batch field name becomes the key; the plain `tag` shared field is skipped
    assert collected == {"ref": ref}


def test_collect_leaves_batch_inside_dict():
    ref = _ref(torch.arange(2).float())
    collected, _ = _collect({"holder": RefHolder(ref=ref)})
    assert collected == {"holder.ref": ref}  # dict key + batch field name


def test_collect_leaves_list_index_keys_for_non_batch_leaves():
    # Non-Batch leaves in a list key by positional index. (TensorRef IS a Batch,
    # so it keys by _eid instead — that path is covered separately below; here we
    # use plain torch.Tensor leaves to exercise the integer-index branch.)
    t0 = torch.arange(2).float()
    t1 = torch.arange(2).float()
    collected, _ = _collect([t0, t1], leaf_type=torch.Tensor)
    assert collected == {"0": t0, "1": t1}  # prefix == "" so bare index


def test_collect_leaves_list_eid_disambiguation_for_batch_elements():
    # Batch list-elements key by their lazily-assigned _eid, not list index, so
    # the wire key survives reordering/dedup.
    h0 = RefHolder(ref=_ref(torch.arange(2).float()))
    h1 = RefHolder(ref=_ref(torch.arange(2).float()))
    collected, _ = _collect([h0, h1])
    assert set(collected) == {f"{h0._eid}.ref", f"{h1._eid}.ref"}
    assert collected[f"{h0._eid}.ref"] is h0.ref
    assert collected[f"{h1._eid}.ref"] is h1.ref


def test_collect_leaves_setters_write_back_dict():
    ref = _ref(torch.arange(2).float())
    container = {"x": ref}
    _, setters = _collect(container)
    setters["x"]("REPLACED")
    assert container["x"] == "REPLACED"  # setter mutated the original dict in place


def test_collect_leaves_setters_write_back_batch_field():
    ref = _ref(torch.arange(2).float())
    holder = RefHolder(ref=ref)
    _, setters = _collect(holder)
    setters["ref"]("NEW")
    assert holder.ref == "NEW"  # setattr closure mutated the Batch in place


def test_collect_leaves_setters_write_back_list():
    # plain (non-Batch) leaves -> integer-index keys -> list __setitem__ closure
    t0 = torch.arange(2).float()
    t1 = torch.arange(2).float()
    lst = [t0, t1]
    _, setters = _collect(lst, leaf_type=torch.Tensor)
    setters["1"]("Z")
    assert lst[0] is t0 and lst[1] == "Z"  # closure mutated index 1 in place


def test_collect_leaves_skips_none_batch_fields():
    holder = RefHolder(ref=None, tag="t")  # ref is None -> skipped, no key
    collected, _ = _collect(holder)
    assert collected == {}


def test_collect_leaves_collects_torch_tensors():
    # the same walker drives dehydrate, collecting torch.Tensor leaves
    t0 = torch.arange(3).float()
    t1 = torch.arange(2).float()
    collected, _ = _collect({"a": t0, "nested": {"b": t1}}, leaf_type=torch.Tensor)
    assert collected == {"a": t0, "nested.b": t1}


def test_collect_leaves_order_is_deterministic():
    refs = {k: _ref(torch.arange(2).float()) for k in ("a", "b", "c", "d")}
    c1, _ = _collect(dict(refs))
    c2, _ = _collect(dict(refs))
    assert list(c1.keys()) == list(c2.keys())  # insertion-ordered, stable


def test_collect_leaves_deep_mixed_containers():
    ref = _ref(torch.arange(2).float())
    collected, setters = _collect({"lvl": [{"k": ref}]})
    # dict -> list (index 0) -> dict (key k) -> leaf
    assert collected == {"lvl.0.k": ref}
    setters["lvl.0.k"]("done")
    # the original deep structure was mutated in place
    assert {"lvl": [{"k": "done"}]}  # structural sanity


# ── fields= prefix filter ─────────────────────────────────────────────────────


def _hydrate_filter(fields):
    """Reconstruct the exact filter_fn hydrate(fields=...) builds (transport.py)."""

    def filter_fn(key):
        return any(key == f or key.startswith(f + ".") for f in fields)

    return filter_fn


def test_fields_filter_exact_and_prefix_match():
    f = _hydrate_filter({"obs", "actions.logits"})
    assert f("obs") is True  # exact
    assert f("obs.image") is True  # prefix (dotted child)
    assert f("actions.logits") is True  # exact dotted
    assert f("actions.logits.q") is True  # prefix child
    assert f("actions") is False  # parent of a filter entry does NOT match
    assert f("rewards") is False
    assert f("observation") is False  # not a prefix boundary (no trailing '.')


def test_collect_leaves_with_filter_keeps_only_matching():
    r_keep = _ref(torch.arange(2).float())
    r_drop = _ref(torch.arange(2).float())
    collected, _ = _collect(
        {"keep": r_keep, "drop": r_drop},
        filter_fn=_hydrate_filter({"keep"}),
    )
    assert collected == {"keep": r_keep}  # only the matching dotted-path collected


def test_collect_leaves_filter_prefix_on_nested():
    inner = _ref(torch.arange(2).float())
    other = _ref(torch.arange(2).float())
    collected, _ = _collect(
        {"grp": {"inner": inner}, "elsewhere": other},
        filter_fn=_hydrate_filter({"grp"}),  # prefix matches grp.inner
    )
    assert collected == {"grp.inner": inner}


# ── real hydrate() through a tiny in-process backend (no Ray) ─────────────────


class _LocalRefTransport(TensorTransport):
    """Minimal TensorTransport: resolves _FakeHandle spans by reading .local().

    Just enough surface for hydrate()/get_batch() to round-trip TensorRefs over
    fake handles on CPU, with no Ray / store / GPU. put() is unused here (the
    refs are pre-built), so it raises if reached.
    """

    def put(self, tensor):
        raise NotImplementedError("not needed: refs are pre-built in the test")

    def _resolve_handles(self, handles):
        return [h.local() for h in handles]

    def is_ref(self, value):
        return isinstance(value, TensorRef)


def test_hydrate_bare_ref_materializes():
    backend = _LocalRefTransport()
    t = torch.arange(4).float()
    out = backend.hydrate(_ref(t))
    assert isinstance(out, torch.Tensor) and torch.equal(out, t)


def test_hydrate_dict_replaces_refs_in_place():
    backend = _LocalRefTransport()
    ta, tb = torch.arange(3).float(), torch.arange(10, 12).float()
    container = {"a": _ref(ta), "b": _ref(tb)}
    out = backend.hydrate(container)
    assert out is container  # mutated in place
    assert torch.equal(out["a"], ta) and torch.equal(out["b"], tb)


def test_hydrate_fields_filter_leaves_unmatched_as_refs():
    backend = _LocalRefTransport()
    ta, tb = torch.arange(3).float(), torch.arange(2).float()
    container = {"keep": _ref(ta), "skip": _ref(tb)}
    out = backend.hydrate(container, fields={"keep"})
    assert torch.equal(out["keep"], ta)  # matched -> materialized tensor
    assert isinstance(out["skip"], TensorRef)  # unmatched -> still a ref


def test_hydrate_no_refs_returns_value_unchanged():
    backend = _LocalRefTransport()
    container = {"x": 1, "y": "z"}
    assert backend.hydrate(container) is container
