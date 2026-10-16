"""The look-alike screen on real asked things (0.5.0's pilot fix round; `questions_where.lookalike_sighted`, `askable`):
`object_where`, `object_was` and `distance_between` (both things of a pair) do not ask a thing when another thing of a
kind it can be mistaken for (`descriptions.CONFUSABLE`: a bag with a tote, a bin and a box) and of its colour has a
sighting in the window, and do ask it when the look-alike has another colour or is never sighted. Hand-built worlds of
the kits; the sets of asked things are written by hand."""

from __future__ import annotations

import family_kit as fk
import things_kit as tk
from change_kit import thing, window
from space_kit import BAG_SIZE, vacuum_run
from stbench.questions_change import object_was
from stbench.questions_things import distance_between
from stbench.questions_where import pools


def put(s, eid, kind, colour, xy, seen=(1,)):
    """A carried thing standing at (x, y), sighted (1,500 px, recognisable) at the keyframes `seen`: none, never."""
    s.item(eid, kind, {"color": colour}, [(0, None, (*xy, 0.0))], size=BAG_SIZE if kind == "bag" else (0.1, 0.1, 0.1))
    if seen:
        s.see(list(seen), eid, 1500)


def entities(cands) -> set[str]:
    """The ids of the world's things the candidates name (a description the world lacks is not an entity)."""
    return {x[2:] for c in cands for x in c.subjects if x.startswith("e:")}


def test_object_where_asks_neither_pool_about_a_thing_a_sighted_lookalike_hides():
    """Contract: `object_where` leaves a real thing out of both pools (answerable and unknown) when another thing of a
    confusable kind and of its colour has a sighting anywhere in the window, whenever in it, and asks it when the
    look-alike has another colour, is never sighted, or the kind has no look-alike. Hand-worked on one task: a blue bag
    beside a blue box, a green bag beside a green tote (sighted only at keyframe 20) and a brown bag beside a brown bin
    are all hidden; a red bag (with a white box and an orange tote beside it), the white box, a pink bag whose pink
    tote no frame shows and black shoes are answerable; of the unseen things the yellow box (a sighted yellow tote) is
    hidden and the orange toy is the unknown pool's one real thing. Regression caught: a blue skip asked beside a blue
    dumpster (here a bag beside a box of its colour), a bag treated as no look-alike of a tote, bin or box, a screen
    that reads when the look-alike was seen, or one that hides the answerable pool alone and still asks the unseen
    thing."""
    s = vacuum_run()
    put(s, "E0001", "bag", "blue", (1.0, 1.0))
    put(s, "E0002", "box", "blue", (3.0, 1.0))
    put(s, "E0003", "bag", "green", (1.0, 2.0))
    put(s, "E0004", "tote", "green", (3.0, 2.0), seen=(20,))
    put(s, "E0005", "bag", "brown", (5.0, 1.0))
    put(s, "E0006", "bin", "brown", (5.0, 2.0))
    put(s, "E0007", "bag", "red", (1.0, 0.5))
    put(s, "E0008", "box", "white", (3.0, 0.5))
    put(s, "E0009", "tote", "orange", (4.0, 0.5))
    put(s, "E0010", "bag", "pink", (5.0, 0.5))
    put(s, "E0011", "tote", "pink", (6.0, 0.5), seen=())
    put(s, "E0012", "shoes", "black", (6.0, 1.5))
    put(s, "E0013", "box", "yellow", (7.0, 1.0), seen=())
    put(s, "E0014", "tote", "yellow", (7.0, 2.0))
    put(s, "E0015", "toy", "orange", (7.0, 0.5), seen=())
    answerable, unknown = pools(fk.world(s))
    assert entities(answerable) == {"E0007", "E0008", "E0010", "E0012"}
    assert entities(unknown) == {"E0015"}


def test_object_was_asks_a_moved_thing_a_sighted_lookalike_does_not_hide():
    """Contract: `object_was` asks the moved things (a place in one task, 2.5 m on in the next) and, as `object_where`
    does, not one that a sighted look-alike of its colour hides, at either state. Hand-worked on two tasks of four moved
    bags: a blue one beside a static blue box and a brown one beside a brown bin are hidden, a red one (a white box
    beside it) and a green one (a green tote that no frame shows) are in both pools of a draw. Regression caught: a
    moved blue bag asked beside a blue box (the agent holds the box where the bag was), or the screen applied to
    `object_where` and forgotten in the family that asks the moved things."""
    s = vacuum_run(2)
    for eid, kind, colour, x in (("E0001", "bag", "blue", 1.0), ("E0003", "bag", "red", 2.0),
                                 ("E0005", "bag", "green", 3.0), ("E0007", "bag", "brown", 4.0)):
        thing(s, eid, kind, colour, [(x, 0.5), (x, 3.0)], seen=[1, 29])
    put(s, "E0002", "box", "blue", (5.0, 0.5))
    put(s, "E0004", "box", "white", (5.5, 0.5))
    put(s, "E0006", "tote", "green", (6.0, 0.5), seen=())
    put(s, "E0008", "bin", "brown", (6.5, 0.5))
    w = window(s, 0, 1)
    (draw, *_) = object_was(w)
    assert entities(draw.answerable) == {"E0003", "E0005"} and entities(draw.unknown) == {"E0003", "E0005"}


def test_distance_between_names_neither_of_a_pairs_things_if_a_lookalike_hides_it():
    """Contract: a `distance_between` row, answerable or unknown, names no real thing that a sighted look-alike of its
    colour hides (either side of a pair), and still pairs the others. Hand-worked on one task: a blue bag and a blue
    box (each the other's look-alike) are in no row; a red bag, a white box and an orange bag (its orange tote never
    sighted) are paired (bag with bag, bag with box); of the unseen boxes the yellow one (a sighted yellow tote) is in
    no unknown row and the green one is the unknown rows' second thing. Regression caught: a pair of a blue bag and
    a blue box asked as two things (the answer is the look-alikes' distance), or the screen skipped on the unseen
    side of an unknown row."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "blue", (1.0, 1.0), recognised=[10])
    tk.put(s, "E0002", "box", "blue", (4.0, 1.0), recognised=[12])
    tk.put(s, "E0003", "bag", "red", (1.0, 3.0), recognised=[10])
    tk.put(s, "E0004", "box", "white", (4.0, 3.0), recognised=[12])
    tk.put(s, "E0005", "bag", "orange", (2.0, 5.0), recognised=[10])
    tk.put(s, "E0006", "tote", "orange", (2.0, 6.0))
    tk.put(s, "E0007", "box", "yellow", (6.0, 1.0))
    tk.put(s, "E0008", "tote", "yellow", (6.0, 2.0), recognised=[14])
    tk.put(s, "E0009", "box", "green", (6.0, 3.0))
    found = distance_between(tk.world(s))
    answerable = [c for m in found for c in m.answerable]
    unknown = [c for m in found for c in m.unknown]
    assert {frozenset(c.entities) for c in answerable} == {
        frozenset(pair) for pair in (("E0003", "E0004"), ("E0003", "E0005"), ("E0004", "E0005"))}
    assert len(answerable) == 3
    assert entities(unknown) == {"E0003", "E0004", "E0005", "E0009"}
