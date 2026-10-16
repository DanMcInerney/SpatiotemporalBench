"""A least-recently-used cache bounded by the size of what it holds, not by how many things it holds.

`table.Mapper` keeps what a depth image said of the floor (a handful of cells to tens of thousands), and the baselines
that run over one case in one process each read the same images; a bound on entries would hold a few small ones or
too many large ones.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable


class SizedCache[V]:
    """Entries by key, each with a size; `put` drops the least recently used until the sizes sum to at most `limit`
    (an entry larger than the limit is not kept). A value is shared with every reader that gets it, so it must not be
    changed."""

    def __init__(self, limit: int):
        self.limit, self.size = limit, 0
        self._items: OrderedDict[Hashable, tuple[V, int]] = OrderedDict()

    def get(self, key: Hashable) -> V | None:
        """The value kept under `key`, now the most recently used, or None."""
        if (found := self._items.get(key)) is None:
            return None
        self._items.move_to_end(key)
        return found[0]

    def put(self, key: Hashable, value: V, size: int) -> None:
        """Keep `value` under `key`, counted as `size`."""
        if key in self._items:
            self.size -= self._items.pop(key)[1]
        if size > self.limit:
            return
        self._items[key] = (value, size)
        self.size += size
        while self.size > self.limit:
            self.size -= self._items.popitem(last=False)[1][1]

    def clear(self) -> None:
        self._items.clear()
        self.size = 0

    def __len__(self) -> int:
        return len(self._items)
