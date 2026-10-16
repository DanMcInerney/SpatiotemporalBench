"""The event rules `changed` is scored by (`change_events`), by hand on things in a 4 x 3 m room, the gold's and the
table systems' one function."""

from __future__ import annotations

from stbench.change_events import Seen, events, swallow
from stbench.rules import MATCH_M


def inside(p):
    """The room: x 0..4, y 0..3."""
    return 0.0 <= p[0] <= 4.0 and 0.0 <= p[1] <= 3.0


def free_j(p, radius):
    """Seen empty in pass j: the box's old place for a 0.15 m radius."""
    return (p, radius) == ((3.0, 2.0), 0.15)


def free_i(p, radius):
    """Seen empty in pass i: the toy's new place for a 0.1 m radius."""
    return (p, radius) == ((3.5, 0.5), 0.1)


def read(*things):
    return events(list(things), inside, free_i, free_j)


def test_events_by_hand():
    """Contract: the event rules by class, each thing alone in a room (x 0..4, y 0..3), a removal or an addition an
    event only where its place was seen empty in the other pass for the thing's own radius:
    - a bag moved 1.0 m inside the room: an event where it is now, a don't-care place where it was;
    - a box gone from (3, 2), a place pass j saw empty: an event there; one gone from (0.5, 0.5), a place nobody saw
      empty: a don't-care place, no event;
    - a chair shifted 0.3 m (between the thresholds 0.15 and 0.45): don't-care places at both ends, no event; one
      shifted 0.1 m is the same place, nothing;
    - a toy added at (3.5, 0.5), seen empty in pass i: an event; one added where pass i did not see: a don't-care place;
    - a thing moved out of the room is a removal where it was, one moved in an addition where it is now, one that
      never entered the room nothing;
    - a thing with `unsure` places (a twin, a glimpse, a person): those of them inside the room are don't-care places,
      and it gives no event whatever else it did.
    Regression caught: a change the robot could not have seen scored (an unseen place called empty), one it saw left
    out, or a thing that cannot be told from another reported as a change."""
    cases = {
        "moved": (Seen((1.0, 1.0), (2.0, 1.0), 0.15), [(2.0, 1.0)], [(1.0, 1.0)]),
        "removed, place seen empty": (Seen((3.0, 2.0), None, 0.15), [(3.0, 2.0)], []),
        "removed, place never seen": (Seen((0.5, 0.5), None, 0.15), [], [(0.5, 0.5)]),
        "shifted": (Seen((1.0, 2.5), (1.3, 2.5), 0.2), [], [(1.0, 2.5), (1.3, 2.5)]),
        "at rest": (Seen((1.0, 1.0), (1.1, 1.0), 0.2), [], []),
        "added, place seen empty": (Seen(None, (3.5, 0.5), 0.1), [(3.5, 0.5)], []),
        "added, place not seen": (Seen(None, (1.5, 2.5), 0.1), [], [(1.5, 2.5)]),
        "moved out": (Seen((3.5, 1.0), (5.0, 1.0), 0.15), [(3.5, 1.0)], []),
        "moved in": (Seen((5.0, 1.0), (3.0, 1.0), 0.15), [(3.0, 1.0)], []),
        "never in the room": (Seen((5.0, 1.0), (6.5, 1.0), 0.15), [], []),
        "removed outside": (Seen((5.0, 1.0), None, 0.15), [], []),
        "unsure": (Seen((2.0, 0.5), (3.9, 0.5), 0.1, ((2.5, 2.8), (9.0, 9.0))), [], [(2.5, 2.8)]),
    }
    for name, (thing, want_events, want_care) in cases.items():
        assert read(thing) == (want_events, want_care), name
    both = read(*(thing for thing, _, _ in cases.values()))
    assert both == ([p for _, ev, _ in cases.values() for p in ev], [p for _, _, dc in cases.values() for p in dc])


def test_moved_thresholds_hold_through_float_noise():
    """Contract: a pair of sightings exactly `MOVED_NO_M` (0.15) apart is a shift (don't-care places), and exactly
    `MOVED_YES_M` (0.45) apart a move, though 1.15 - 1.0 is 0.1499999999999999 and 1.45 - 1.0 is 0.44999999999999996
    in floats, each just under its threshold. Regression caught: a class flipped by rounding at the very distance a
    hand-built world puts things at."""
    shifted = read(Seen((1.0, 1.0), (1.15, 1.0), 0.2))
    assert shifted == ([], [(1.0, 1.0), (1.15, 1.0)])
    moved = read(Seen((1.0, 1.0), (1.45, 1.0), 0.2))
    assert moved == ([(1.45, 1.0)], [(1.0, 1.0)])


def test_the_swallow_rule():
    """Contract: an event within `match_m` of a don't-care place becomes a don't-care place, and then its neighbours
    are tested against it, until none moves: with a don't-care place at (0, 0), an event 0.4 m away is swallowed and
    one 0.6 m away stays; at exactly 0.5 m (the scorer drops an answer item within the radius, inclusive) it is
    swallowed too; events at 0.4, 0.8 and 1.9 m on one line go two by two (0.4, then 0.8 from the one just
    swallowed) and leave 1.9. The output is sorted to 4 decimals, don't-care places within 0.01 m one place. Regression
    caught: a right answer the scorer drops (an event the system could report only inside a don't-care radius, never
    matched), or a chain left half swallowed, or an order that depends on how events were listed."""
    assert swallow([(0.4, 0.0)], [(0.0, 0.0)], MATCH_M) == ([], [(0.0, 0.0), (0.4, 0.0)])
    assert swallow([(0.6, 0.0)], [(0.0, 0.0)], MATCH_M) == ([(0.6, 0.0)], [(0.0, 0.0)])
    assert swallow([(0.5, 0.0)], [(0.0, 0.0)], MATCH_M) == ([], [(0.0, 0.0), (0.5, 0.0)])
    kept, care = swallow([(1.9, 0.0), (0.8, 0.0), (0.4, 0.0)], [(0.0, 0.0)], MATCH_M)
    assert (kept, care) == ([(1.9, 0.0)], [(0.0, 0.0), (0.4, 0.0), (0.8, 0.0)])
    assert swallow([(3.0, 3.0), (1.69996, 2.00004)], [(2.0, 2.0), (2.005, 2.0), (2.5, 0.12344)], MATCH_M) == (
        [(3.0, 3.0)], [(1.7, 2.0), (2.0, 2.0), (2.5, 0.1234)])
