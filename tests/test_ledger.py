"""The ledger's interval rule: what `Ledger.place`, `remove` and `set_state` do when a call lands at the instant the
open interval began (`validate` rejects an interval with no length, `bad_interval`)."""

from __future__ import annotations

import math

import pytest

from conftest import script_on_map
from stbench.ledger import Ledger
from stbench.timeline import Placement, State
from stbench.validate import validate
from validate_kit import generic_rules_only  # noqa: F401  (a fixture)

LIVING_ROOM_FLOOR = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 0.0)]  # open floor, x 2.4-3.2
FACING_0 = (1.0, 0.0, 0.0, 0.0)  # the rotation about z for yaw 0 and for yaw pi, as [w, x, y, z]
FACING_PI = (0.0, 0.0, 0.0, 1.0)


def test_a_placement_placed_again_at_its_own_start_is_replaced_not_closed_into_an_empty_interval():
    """Contract (the ledger, M1 plan: a script re-places an entity at t = 0 that `generate.build_world` placed there;
    `validate` rejects an interval [t, t)): `Ledger.place` at the instant the entity's open placement began replaces it,
    so one open placement remains, with the new pose and support. Hand case: a box placed at t = 0 on the floor and
    placed again at t = 0 on a table at another spot leaves exactly that second placement. Regression caught: the first
    placement closed at [0, 0) beside the second (a `bad_interval` in every script that moves light furniture or drops
    an item at the start)."""
    ledger = Ledger()
    ledger.place("box", 0.0, (1.0, 2.0, 0.0), 0.0, None)
    ledger.place("box", 0.0, (3.0, 4.0, 0.5), math.pi, "table")
    placements, states = ledger.finish()
    assert placements == [Placement("box", 0.0, None, (3.0, 4.0, 0.5), FACING_PI, "table")]
    assert states == []


def test_a_placement_placed_again_later_still_closes_the_first_at_that_time():
    """Contract (the ledger's rule: "open intervals close when the next one starts", which the world processes of
    `changes` rely on): `Ledger.place` at a later instant closes the open placement there and opens the new one. Hand
    case: placed at t = 0, placed again at t = 5.0 gives [0, 5.0) at the first pose and [5.0, open) at the second.
    Regression caught: the same-instant rule widened to any later call (the first placement dropped or left open, so a
    shifted chair is never seen where it stood before the shift)."""
    ledger = Ledger()
    ledger.place("box", 0.0, (1.0, 2.0, 0.0), 0.0, None)
    ledger.place("box", 5.0, (3.0, 4.0, 0.5), math.pi, "table")
    placements, _ = ledger.finish()
    assert placements == [Placement("box", 0.0, 5.0, (1.0, 2.0, 0.0), FACING_0, None),
                          Placement("box", 5.0, None, (3.0, 4.0, 0.5), FACING_PI, "table")]


def test_a_placement_removed_at_its_own_start_leaves_none_and_keeps_the_ones_before_it():
    """Contract (the ledger, as above): `Ledger.remove` at the instant the open placement began drops it, so the entity
    has no placement from that instant on, and the placements that closed before stay. Hand case: a box placed at
    t = 0, placed again at t = 5.0 and removed at t = 5.0 leaves only [0, 5.0); a crate placed and removed at t = 2.0
    leaves nothing. Regression caught: an empty [t, t) placement left behind (`bad_interval`), or the removal taking the
    earlier, closed placement with it."""
    ledger = Ledger()
    ledger.place("box", 0.0, (1.0, 2.0, 0.0), 0.0, None)
    ledger.place("box", 5.0, (3.0, 4.0, 0.5), math.pi, "table")
    ledger.remove("box", 5.0)
    ledger.place("crate", 2.0, (6.0, 7.0, 0.0), 0.0, None)
    ledger.remove("crate", 2.0)
    placements, _ = ledger.finish()
    assert placements == [Placement("box", 0.0, 5.0, (1.0, 2.0, 0.0), FACING_0, None)]


def test_a_state_set_twice_at_one_instant_keeps_the_last_value_and_a_later_set_still_closes_it():
    """Contract (the ledger, as above): `Ledger.set_state` at the instant the open state began replaces it (the last
    value set at an instant wins) and at a later instant closes it there; other keys are not touched. Hand case: a
    door's `open` set to False and then to True at t = 0, then to False at t = 30.0, with its `locked` set once at
    t = 0, gives `open` True over [0, 30.0) and False from 30.0, and `locked` open from 0. Regression caught: a [0, 0)
    state beside the second (`bad_interval`), the first value winning, or the later set no longer closing the value
    before it."""
    ledger = Ledger()
    ledger.set_state("door", "open", 0.0, False)
    ledger.set_state("door", "locked", 0.0, True)
    ledger.set_state("door", "open", 0.0, True)
    ledger.set_state("door", "open", 30.0, False)
    _, states = ledger.finish()
    assert states == [State("door", "locked", 0.0, None, True),
                      State("door", "open", 0.0, 30.0, True),
                      State("door", "open", 30.0, None, False)]


@pytest.mark.usefixtures("generic_rules_only")
def test_a_world_that_places_light_furniture_again_at_t_zero_validates(apt):
    """Contract (M1 plan: a script's light-furniture starts and floor drops re-place an entity at t = 0 after the
    generator placed it; `validate` must accept the world): a timeline whose chair's placements come from a ledger that
    placed it twice at t = 0 has one placement of it, at the second pose, and no problem. Hand case: the apartment's
    seed-1 world with its first chair first placed at t = 0 far outside the map and then at t = 0 at its own pose; had
    the first placement stayed, `validate` would flag it (`bad_interval`, `outside_map`).
    Regression caught: the first placement kept beside the second, which makes every such script fail validation."""
    tl = script_on_map("apartment_a", 1, "vacuum", LIVING_ROOM_FLOOR)
    chair = next(e for e in tl.entities if e.kind == "chair")
    own = tl.placement_of(chair.id, 0.0)
    ledger = Ledger()
    ledger.place(chair.id, 0.0, (-50.0, -50.0, 0.0), 0.0, None)
    ledger.place(chair.id, 0.0, own.pos, own.yaw, None)
    replaced = tl.with_placements({chair.id: ledger.finish()[0]})
    (only,) = replaced.placements_of(chair.id)
    assert (only.t0, only.t1, only.pos) == (0.0, None, own.pos)
    assert validate(replaced, apt) == []
