"""Stable hashing and seeded generators. Python's hash() is salted per process, so it is never used."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(*parts: object) -> int:
    """A 64-bit integer from the parts, identical across processes, platforms and Python versions."""
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def stable_hex(*parts: object, n: int = 8) -> str:
    """The first n hex characters of the SHA-256 of the parts."""
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:n]


def make_rng(*parts: object) -> np.random.Generator:
    """A PCG64 generator seeded from the parts. Give each subsystem its own parts so streams never interleave."""
    return np.random.Generator(np.random.PCG64(stable_hash(*parts)))
