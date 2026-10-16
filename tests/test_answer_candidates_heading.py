"""The candidates a best response tries over a pool of `open_heading` golds (`answer_candidates`, the heading
handler): a learned shortcut with no heading among them could not try any, and its shortcut gate would pass for
nothing."""

from __future__ import annotations

from stbench import answer_candidates


def gold(deg: float, qid: str = "g") -> dict:
    return {"qid": qid, "gold": {"type": "heading", "deg_left": deg}}


def heading(deg: float) -> dict:
    return {"type": "heading", "deg_left": deg}


def test_a_heading_pool_gives_the_compass_the_median_and_each_gold():
    """Contract (plan H1): over a pool of heading golds the candidates are `unknown`, the four compass headings (0, 90,
    180 and -90 degrees left), the pool's median, then each gold's own heading, no repeats. Hand pool (-30, 100, 40):
    the median is 40, which is also the third gold's, so it is tried once, among the summaries; the order is unknown,
    0, 90, 180, -90, 40, -30, 100. A pool of the single gold 0 repeats the compass's 0 and the median's: one 0. A gold
    of unknown adds nothing. Regression caught: a learned shortcut that never tries a heading (its gate would pass for
    nothing), a candidate list that repeats, or one that raises on the new type."""
    pool = [gold(-30.0), gold(100.0), gold(40.0), {"qid": "u", "gold": {"type": "unknown"}}]
    got = answer_candidates.candidates(pool)
    assert got == [{"type": "unknown"}, *(heading(d) for d in (0.0, 90.0, 180.0, -90.0, 40.0, -30.0, 100.0))]
    assert answer_candidates.candidates([gold(0.0)]) == [{"type": "unknown"},
                                                         *(heading(d) for d in (0.0, 90.0, 180.0, -90.0))]
