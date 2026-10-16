"""How a determinism test compares an output: nothing is recorded.

The `test_determinism_*.py` files build an output on fixed inputs, build it again in a second process with another
`PYTHONHASHSEED`, and compare the two. That catches an output that depends on dict or set order, the clock or the
process. No value is kept between runs.

`digest` is a compact form of an output, compared within one test run, never recorded: the SHA-256 of its JSON.
"""

from __future__ import annotations

import hashlib
import json


def _plain(x):
    """JSON-ready and order-stable: tuples, ranges and sets as lists, floats rounded to 9 places."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple | range):
        return [_plain(v) for v in x]
    if isinstance(x, set | frozenset):
        return sorted((_plain(v) for v in x), key=json.dumps)
    return round(x, 9) if isinstance(x, float) else x


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(_plain(obj), sort_keys=True).encode("utf-8")).hexdigest()
