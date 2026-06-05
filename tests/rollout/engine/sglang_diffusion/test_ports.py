"""``ReservedPorts`` — typed, self-reserving port sets that engines own.

Each engine declares a frozen-dataclass subclass (one ``int`` field per port,
declaration order = reservation order) and reserves its own set at boot — no
builder injection, no per-engine ``base + rank * stride`` math. These tests pin
the validation that makes the type worth having (in-range, distinct, frozen,
arity derived from the fields) plus live reservation.
"""

from __future__ import annotations

import dataclasses
import socket

import pytest

from unirl.rollout.engine.ports import ReservedPorts
from unirl.rollout.engine.sglang_diffusion.config import SGLangDiffusionPorts


@dataclasses.dataclass(frozen=True)
class _PairPorts(ReservedPorts):
    a_port: int
    b_port: int


def test_valid_set_exposes_named_ports():
    ports = SGLangDiffusionPorts(server_port=40000, scheduler_port=40011, master_port=40023)
    assert ports.server_port == 40000
    assert ports.scheduler_port == 40011
    assert ports.master_port == 40023


def test_from_ports_maps_in_field_declaration_order():
    ports = SGLangDiffusionPorts.from_ports([41000, 41001, 41002])
    assert (ports.server_port, ports.scheduler_port, ports.master_port) == (41000, 41001, 41002)


def test_arity_derived_from_fields():
    pair = _PairPorts.from_ports([42000, 42001])
    assert (pair.a_port, pair.b_port) == (42000, 42001)
    with pytest.raises(ValueError, match="expects 2 ports"):
        _PairPorts.from_ports([1, 2, 3])
    with pytest.raises(ValueError, match="expects 3 ports"):
        SGLangDiffusionPorts.from_ports([1, 2])


@pytest.mark.parametrize("bad", [0, -1, 65536, 70000])
def test_rejects_out_of_range_port(bad):
    with pytest.raises(ValueError, match="TCP port in"):
        SGLangDiffusionPorts(server_port=bad, scheduler_port=40011, master_port=40023)


def test_rejects_non_distinct_ports():
    with pytest.raises(ValueError, match="must be distinct"):
        SGLangDiffusionPorts(server_port=40000, scheduler_port=40000, master_port=40023)


def test_set_is_frozen():
    ports = SGLangDiffusionPorts(server_port=40000, scheduler_port=40011, master_port=40023)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ports.server_port = 1  # type: ignore[misc]


def test_bare_base_rejects_instantiation():
    with pytest.raises(TypeError, match="declares no port fields"):
        ReservedPorts()


def test_reserve_returns_distinct_bindable_ports():
    ports = _PairPorts.reserve()
    # __post_init__ already guarantees in-range + distinct; check the ports are
    # actually free right now (the engine's own bind must succeed).
    socks = []
    try:
        for value in (ports.a_port, ports.b_port):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", value))
            socks.append(s)
    finally:
        for s in socks:
            s.close()
