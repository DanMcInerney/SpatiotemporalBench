"""`SizedCache`: a cache bounded by the size of what it holds."""

from __future__ import annotations

from stbench.sized_cache import SizedCache


def test_the_least_recently_used_go_first_and_a_read_counts_as_a_use():
    """Contract: `put` drops the least recently used entries until the sizes sum to at most the limit, and `get` makes
    an entry the most recent. Hand case, limit 10: a (4), b (4), then a is read, then c (4) is put: the sizes are 12,
    so b, the least recently used, goes, and a and c stay. Regression caught: a cache that evicts in insertion order
    (a read entry lost first) or never evicts."""
    cache = SizedCache[str](10)
    cache.put("a", "A", 4)
    cache.put("b", "B", 4)
    assert cache.get("a") == "A"
    cache.put("c", "C", 4)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == ("A", None, "C") and cache.size == 8 and len(cache) == 2


def test_an_entry_larger_than_the_limit_is_not_kept_and_a_key_put_again_is_replaced():
    """Contract: an entry that alone exceeds the limit is not kept (and drops nothing); a key put again takes its new
    value and size. Hand case, limit 6: keep (3); x (9) is refused; y (2) then y (1): two entries, sizes 3 + 1 = 4.
    Regression caught: one image's cells emptying the cache of everything else, or sizes counted twice for one key."""
    cache = SizedCache[str](6)
    cache.put("keep", "K", 3)
    cache.put("x", "X", 9)
    assert cache.get("x") is None and cache.get("keep") == "K" and cache.size == 3
    cache.put("y", "Y", 2)
    cache.put("y", "Z", 1)
    assert cache.get("y") == "Z" and cache.size == 4
    cache.clear()
    assert len(cache) == 0 and cache.size == 0 and cache.get("keep") is None
