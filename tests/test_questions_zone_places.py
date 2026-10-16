"""`zone_clear` on the two zone sources that are no piece's footprint, by hand on `apartment_a` (`zone_kit`'s
apartment): the way of an interior opening (the vacuum's doorway) and the window of the surface the base faces (the
helper's spot). The states are hand-set (`Views`, `paint`) except where a test ray-casts the tote camera's view of a
way."""

from __future__ import annotations

import math

import pytest

import zone_kit as zk
from shape_kit import APARTMENT, NORTH
from space_kit import box
from stbench import questions, questions_zone, zones
from stbench.questions_text import FAMILIES
from zone_kit import REACHED, Views, paint

K = zk.DECISION_K
AT_D1, TOWARD_D1 = (3.9, 1.1), math.pi / 4   # D1's way is the nearest, and the camera sees all of it
AT_ARCH = (3.6, 2.2)                         # the arch O1's way is the nearest
AT_TABLE = (2.1, 0.9)                        # facing north the dining table, facing south the sink unit
D1_WAY = zones.zone_of(APARTMENT, zk.WAY, AT_D1, TOWARD_D1)
O1_WAY = zones.zone_of(APARTMENT, zk.WAY, AT_ARCH, 0.0)
TABLE_WINDOW = zones.zone_of(APARTMENT, zk.SPOT, AT_TABLE, NORTH)
SINK_WINDOW = zones.zone_of(APARTMENT, zk.SPOT, AT_TABLE, -NORTH)
WAY_CELLS = zones.zone_cells(D1_WAY)


def way(base=AT_D1, yaw: float = TOWARD_D1, views: Views | None = None):
    """`zone_clear`'s candidate at the kit's decision, a `before_room` of the vacuum's kind, over `views`."""
    s = zk.apartment_script("before_room", base, yaw)
    return questions_zone.zone_clear(zk.world(s, views or Views(), job=zk.WAY_JOB, m=APARTMENT), K)


def spot(yaw: float, views: Views):
    """`zone_clear`'s candidate at the kit's decision, a `before_place` at (2.1, 0.9) heading `yaw`."""
    s = zk.apartment_script("before_place", AT_TABLE, yaw)
    return questions_zone.zone_clear(zk.world(s, views, job=zk.SPOT_JOB, m=APARTMENT), K)


def test_a_doorway_is_asked_about_its_way():
    """Contract: a `DOORWAY` rule asks `zone_clear` of the way of the interior opening nearest the base (the way
    `zone_shapes.doorway_way` gives, on the floor, judged from 0.02 m up), by the same look window-only gold; the
    question names the door's entity, an arch (it has none) only the time. Hand values on `apartment_a` from (3.9, 1.1)
    facing north-east, where D1's way (x 4.6..4.95, y 1.575..2.225: centres x 4.65..4.95, y 1.65..2.15, 4 x 6 = 24
    cells) is the nearest:
    - all 24 free: yes, evidence (17,), the question of the door D1 (entity ED1) at 17, text 'Is the doorway ahead of
      you clear of anything taller than 0.02 m right now?'; 20 free (0.833): yes; 19 (0.792, under the 19.2 the 80%
      needs): unknown; one occupied among 24 free: no;
    - the way seen whole at 12 (an earlier look window) and not in the decision's: unknown, no evidence;
    - every request is for ground 0 and band (0.02, 3.5);
    - from (3.6, 2.2) the nearest way is the arch O1's: the same gold, the subjects the time alone (w:17).
    Ray-cast from the tote camera at (3.9, 1.1): an empty way is clear, a 0.1 m tall thing across it blocks, and a
    0.015 m mat (under the 0.02 m the band starts at, which a vacuum drives over) does not.
    Regression caught: a doorway judged on the robot's 0.3 m band (a low thing in the way read as clear), the nearest
    piece of a kind looked up for a rule that has none, a door's entity looked up for the arch, or an earlier frame
    read."""
    assert len(WAY_CELLS) == 24 and D1_WAY.piece == "D1" and O1_WAY.piece == "O1"
    spec = FAMILIES["zone_clear"]
    c = way(views=Views({17: paint(D1_WAY)}))
    assert c.slots == {"zone": "the doorway ahead of you", "height": 0.02}
    assert c.gold == {"type": "yesno", "value": True} and c.evidence_k == (17,) and c.decision_k == K
    assert c.subjects == (f"w:{K}", "e:ED1")
    assert spec.render(c.slots) == "Is the doorway ahead of you clear of anything taller than 0.02 m right now?"
    assert way(views=Views({17: paint(D1_WAY, n=20)})).gold["value"] is True
    assert way(views=Views({17: paint(D1_WAY, n=19)})).gold == {"type": "unknown"}
    assert way(views=Views({17: {**paint(D1_WAY), WAY_CELLS[7]: "occupied"}})).gold["value"] is False
    old = way(views=Views({12: paint(D1_WAY)}))
    assert (old.gold, old.slice, old.evidence_k) == ({"type": "unknown"}, "unknown", ())
    views = Views(zone={17: paint(D1_WAY)}, robot={17: paint(D1_WAY, "occupied")})
    assert way(views=views).gold["value"] is True and set(views.calls) == {(0.0, (0.02, 3.5))}
    arch = way(AT_ARCH, 0.0, Views({17: paint(O1_WAY)}))
    assert (arch.gold["value"], arch.subjects) == (True, (f"w:{K}",))
    states = {name: zk.look(D1_WAY, boxes, AT_D1, TOWARD_D1) for name, boxes in (
        ("empty", []), ("tall", [box(4.7, 4.85, 1.8, 2.0, 0.0, 0.1)]), ("mat", [box(4.65, 4.9, 1.7, 2.1, 0.0, 0.015)]))}
    assert [way(views=Views({17: cells})).gold["value"] for cells in states.values()] == [True, False, True]


def test_a_spot_is_asked_about_the_window_the_base_faces():
    """Contract: a `SPOT` rule asks `zone_clear` of the window of the surface the base faces along its delivered
    heading, on that surface, judged from 0.01 to 0.5 m above it; the heading, not the nearest piece, picks the zone,
    and a base facing nothing has no zone (a dead band, not asked). Hand values on `apartment_a` from (2.1, 0.9):
    - facing north the dining table's window, x 1.9..2.3, y 1.56..2.24 at 0.75 m (centres x 1.95..2.25, y 1.65..2.15:
      4 x 6 = 24 cells): all free is yes, one occupied among them no, evidence (17,), the question of the table
      (entity EF07), text 'Is the spot in front of you where you are putting it down clear of anything taller than
      0.01 m right now?'; every request is for ground 0.75 and band (0.01, 0.5);
    - the same cells painted free, facing south: the sink unit's window (its counter at 0.9 m) is a different place, so
      the answer is unknown (nothing of it seen) and the question is of the sink unit (EF03);
    - facing east: nothing within 1.5 m is faced, so there is no question.
    `clearance_then` finds the same window from the pose at its event (the helper's place): the first at-station step,
    keyframe 18, at (2.1, 0.9) facing north, with the table's window free in the look window 18-25 and not in the end
    look window: yes, the question of the table.
    Regression caught: a spot chosen by the nearest piece of the base (the sink unit is nearer than the table), a
    heading read from the camera or left out (`ValueError`), a window judged on the floor's ground, or a moment that
    drops the heading."""
    assert len(zones.zone_cells(TABLE_WINDOW)) == 24
    assert (TABLE_WINDOW.piece, TABLE_WINDOW.base_z, TABLE_WINDOW.band) == ("F07", 0.75, (0.01, 0.5))
    assert (SINK_WINDOW.piece, SINK_WINDOW.base_z) == ("F03", 0.9)
    c = spot(NORTH, Views({17: paint(TABLE_WINDOW)}))
    assert c.slots == {"zone": zk.SPOT.words, "height": 0.01}
    assert c.gold == {"type": "yesno", "value": True} and c.evidence_k == (17,)
    assert c.subjects == (f"w:{K}", "e:EF07")
    assert FAMILIES["zone_clear"].render(c.slots) == (
        "Is the spot in front of you where you are putting it down clear of anything taller than 0.01 m right now?")
    cells = zones.zone_cells(TABLE_WINDOW)
    assert spot(NORTH, Views({17: {**paint(TABLE_WINDOW), cells[5]: "occupied"}})).gold["value"] is False
    south = spot(-NORTH, Views({17: paint(TABLE_WINDOW)}))
    assert (south.gold, south.subjects) == ({"type": "unknown"}, (f"w:{K}", "e:EF03"))
    assert spot(0.0, Views({17: paint(TABLE_WINDOW)})) is None
    views = Views({17: paint(TABLE_WINDOW)})
    spot(NORTH, views)
    assert set(views.calls) == {(0.75, (0.01, 0.5))}

    s = zk.apartment_script("before_place", AT_TABLE, NORTH)
    s.poses[18], s.yaws[18] = AT_TABLE, NORTH
    w = zk.world(s, Views({18: paint(TABLE_WINDOW)}), job=zk.SPOT_JOB, m=APARTMENT)
    (moment,) = questions_zone.clearance_then(w)
    (then,) = moment.answerable
    assert moment.key == f"0|{REACHED}"
    assert (then.gold, then.subjects) == ({"type": "yesno", "value": True}, ("w:18", "e:EF07"))


@pytest.mark.parametrize("kind, base, yaw, job, subjects", [
    ("before_room", AT_D1, TOWARD_D1, zk.WAY_JOB, ["w:17", "e:ED1"]),
    ("before_room", AT_ARCH, 0.0, zk.WAY_JOB, ["w:17"]),
    ("before_place", AT_TABLE, NORTH, zk.SPOT_JOB, ["w:17", "e:EF07"])])
def test_the_new_zone_rows_compile(kind, base, yaw, job, subjects):
    """Contract: the doorway and spot questions go through the compiler as `now` rows that need the picture and the pose
    alone, with the subjects the record validators accept (a time, and the piece's entity when it has one): a door,
    an arch (the time alone) and a table. Regression caught: a subject the rows cannot hold (a map id in place of an
    entity), a scope other than `now`, or a `needs` the camera-only channel cannot meet."""
    s = zk.apartment_script(kind, base, yaw)
    zone = zones.zone_of(APARTMENT, job.zones[kind], base, yaw)
    qs, golds = questions.compile_inputs(zk.inputs(s, Views({17: paint(zone)}), job, APARTMENT), zk.record(s))
    ((q, g),) = [(q, g) for q, g in zip(qs, golds, strict=True) if g["family"] == "zone_clear"]
    assert (q["template"], q["needs"], q["decision_k"]) == ("zone_clear", ["see", "pose"], K)
    assert (g["scope"], g["gold"], g["subjects"]) == ("now", {"type": "yesno", "value": True}, subjects)
