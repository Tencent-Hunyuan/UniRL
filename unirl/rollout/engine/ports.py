"""Typed, self-reserving TCP port sets for out-of-process inference engines."""

from __future__ import annotations

import dataclasses
import socket
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReservedPorts:
    """Base for an engine-owned set of distinct TCP ports reserved together."""

    def __post_init__(self) -> None:
        fields = dataclasses.fields(self)
        if not fields:
            raise TypeError(
                f"{type(self).__name__} declares no port fields; subclass ReservedPorts with one int field per port"
            )
        for f in fields:
            value = getattr(self, f.name)
            if not isinstance(value, int) or not (1 <= value <= 65535):
                raise ValueError(f"{type(self).__name__}.{f.name} must be a TCP port in [1, 65535]; got {value!r}")
        ports = tuple(getattr(self, f.name) for f in fields)
        if len(set(ports)) != len(ports):
            raise ValueError(f"{type(self).__name__} ports must be distinct; got {ports}")

    @classmethod
    def from_ports(cls, ports: Sequence[int]) -> "ReservedPorts":
        """Build an instance from one port per field, in declaration order."""
        names = [f.name for f in dataclasses.fields(cls)]
        ports = list(ports)
        if len(ports) != len(names):
            raise ValueError(f"{cls.__name__}.from_ports expects {len(names)} ports; got {len(ports)}")
        return cls(**{name: int(port) for name, port in zip(names, ports)})

    @classmethod
    def reserve(cls) -> "ReservedPorts":
        """Reserve one free port per field on this node, right now."""
        socks = []
        try:
            for _ in dataclasses.fields(cls):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", 0))
                socks.append(s)
            return cls.from_ports([s.getsockname()[1] for s in socks])
        finally:
            for s in socks:
                s.close()


__all__ = ["ReservedPorts"]
