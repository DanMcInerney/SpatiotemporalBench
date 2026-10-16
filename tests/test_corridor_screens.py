"""The ground corridor's screens (`corridor_screens`), on `clear_kit`'s tote AMR: base (0.6, 0) heading +x, front edge
(1.0, 0), body 0.58 m wide (half 0.29), climb 0.01 m, band top 0.675 m, C 0.45 m (C/2 0.225), camera 0.5 m up at x 0.98
with fy 381.36 px; the look point is keyframe 19 (look window 16 to 19). A tie's one-pixel error is (d^2 + H^2) / (H
fy)."""

from __future__ import annotations

import pytest

import space_kit
from clear_kit import LAST, Owned, drive_run, thing, wall, world
from space_kit import box
from stbench import corridor_screens as screens
from stbench.geometry import quat_z
from stbench.questions_world import World
from stbench.robots import ROBOTS, blind_m
from stbench.timeline import Entity, Placement

AMR = ROBOTS["amr"]
START, KS = (1.0, 0.0), range(16, 20)
GOLD = {"type": "number", "value": 2.0}


def tied(w, gold=GOLD) -> bool:
    return screens.stopper_tied(w, LAST, START, 0.0, gold, w.latest_cells(KS))


# -- the stopper ------------------------------------------------------------------------------------------------
def seen_things(script, *ids, frames=KS):
    for eid in ids:
        script.see(list(frames), eid, 1500)
    return script


def test_a_box_on_the_floor_is_tied_and_a_tables_overhang_is_not():
    """Contract (review 2.1, the stopper screen): the stopper is tied when a floor contact of the same thing lies within
    C/2 beyond the stop and reads to within C/2 at one pixel. Hand scenes, the corridor stopping at 2.0 (a face at x
    3.01): (a) a 0.5 m tall box on the floor, its face's foot at 3.01, d 2.03 m from the camera at x 0.98 and H 0.5:
    (2.03^2 + 0.5^2) / (0.5 x 381.36) = 0.023 m, under 0.225: tied; (b) a table top at 0.45 to 0.50 m whose legs stand
    at x 3.41, 0.4 m behind its edge (more than C/2 = 0.225): no contact within reach, not tied; (c) the same table with
    its legs at the edge, x 3.01 to 3.06: tied. Regression caught: an overhang read as a stop (a gold the picture cannot
    place, since the floor under the edge is out of reach), or a floor box refused."""
    crate = Owned(box(3.01, 3.51, -0.25, 0.25, 0.0, 0.5), "E0001")
    w = world(seen_things(thing_run(crate), "E0001"), [crate])
    assert tied(w)

    def table(leg_x: float):
        top = Owned(box(3.01, 4.01, -0.4, 0.4, 0.45, 0.5), "E0002")
        legs = [Owned(box(leg_x, leg_x + 0.05, y, y + 0.05, 0.0, 0.45), "E0002") for y in (-0.4, 0.35)]
        script = drive_run(boxes=[top.at, *(leg.at for leg in legs)])
        thing(script, "E0002", (1.0, 0.8, 0.5), (3.51, 0.0, 0.0))
        return world(seen_things(script, "E0002"), [top, *legs])

    assert not tied(table(3.41))
    assert tied(table(3.01))


def thing_run(*owned: Owned):
    """A script whose boxes are the owned ones, each a carried entity of the same id standing at its centre."""
    script = drive_run(boxes=[o.at for o in owned])
    for o in owned:
        lo, hi = o.at
        size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        thing(script, o.who, size, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
    return script


def test_the_stopper_is_read_in_the_frame_that_last_observed_the_stopping_cell():
    """Contract (review 2.1): the tie is read in the keyframe of the look window that last observed the gold's stopping
    fine cell, not in the look window's first or last frame showing anything. Hand scene: a floor box in the cells at x
    3.01 in keyframes 16 to 18 and, in keyframe 19, a table (top at 0.45 to 0.50, legs 0.4 m behind its edge) in its
    place, so the stopping cell [3.0, 3.1) was last observed in 19 and its stopper is the table: not tied. The mirror
    (the table in 16 to 18, the box in 19) is tied. Regression caught: a tie read in an earlier (or a later) frame than
    the one the gold rests on."""
    def pieces(frames: tuple[int, ...]):
        crate = Owned(box(3.01, 3.51, -0.25, 0.25, 0.0, 0.5), "E0001", frames)
        top = Owned(box(3.01, 4.01, -0.4, 0.4, 0.45, 0.5), "E0002", frames)
        legs = [Owned(box(3.41, 3.46, y, y + 0.05, 0.0, 0.45), "E0002", frames) for y in (-0.4, 0.35)]
        return [crate], [top, *legs]

    def run(early: list[Owned], late: list[Owned]):
        script = drive_run(frame_boxes={**dict.fromkeys(range(16, 19), [o.at for o in early]),
                                        19: [o.at for o in late]})
        thing(script, "E0001", (0.5, 0.5, 0.5), (3.26, 0.0, 0.0))
        thing(script, "E0002", (1.0, 0.8, 0.5), (3.51, 0.0, 0.0))
        return world(seen_things(script, "E0001", "E0002"), [*early, *late])

    early_crate, early_table = pieces((16, 17, 18))
    late_crate, late_table = pieces((19,))
    assert not tied(run(early_crate, late_table))
    assert tied(run(early_table, late_crate))


def test_the_stopper_must_be_visible():
    """Contract (review 2.1): a stopper is asked only if a reader of the frame can see it: structure (no entity) and
    `fixed` or `structure` entities always, a thing only if it is recognisable in the frame. Hand scene: the tied floor
    box of the first test as E0001, once seen at 1,500 px (recognisable), once at 100 px (a glimpse, under the 1,024 px
    a recognition needs): visible, then not; as a `fixed` piece at 100 px, visible again; and the box as structure (a
    wall with no entity), visible. Regression caught: a gold stopped by a thing no frame makes out."""
    crate = Owned(box(3.01, 3.51, -0.25, 0.25, 0.0, 0.5), "E0001")

    def run(px: int, mobility: str = "carried"):
        script = drive_run(boxes=[crate.at])
        thing(script, "E0001", (0.5, 0.5, 0.5), (3.26, 0.0, 0.0), mobility)
        script.see(list(KS), "E0001", px)
        return world(script, [crate])

    assert tied(run(1500)) and not tied(run(100)) and tied(run(100, "fixed"))
    assert tied(world(drive_run(boxes=[wall(3.01).at]), [wall(3.01)]))


def test_a_gold_at_the_cap_needs_no_stopper():
    """Contract: a corridor seen empty to the cap names no stopper, so it needs no frame; an answerable gold under the
    cap on inputs that carry no frame raises rather than passing for nothing. Regression caught: a screen that passes
    silently when it cannot read."""
    w = world(drive_run(boxes=[wall(3.01).at]), None)
    assert tied(w, {"type": "number", "value": 5.0})
    with pytest.raises(ValueError, match="frame_of"):
        tied(w)


# -- the side ---------------------------------------------------------------------------------------------------
def side_world(offset: float):
    """A 0.5 m box whose side lies `offset` m north of the axis (x 3.01 to 3.51, y offset to offset + 0.5, 0.5 m tall)
    and a wall at x 5.51, structure, 4.5 m from the front edge."""
    script = drive_run(boxes=[box(3.01, 3.51, offset, offset + 0.5, 0.0, 0.5), wall(5.51).at])
    return world(script, None)


def test_a_stopper_grazing_the_bodys_side_is_not_asked():
    """Contract (review D2, `CLEAR_SIDE_M` 0.01): an obstacle within 1 cm of the body's side may fall either way, so the
    question is not asked when the corridor 1 cm narrower on each side gives a gold more than C/2 from it. Hand scenes
    (half-width 0.29): a box whose side is at 0.285, 0.5 cm inside: its side's sub-cell (centre 0.29) leaves the
    corridor narrowed to half 0.28, the box stops nothing and the corridor runs to the wall, 4.5 m: moved by 2.5 m, not
    asked; a box at 0.26, 3 cm inside (sub-cell centre 0.27): in the narrowed and in the widened corridor alike, the
    gold stays 2.0: asked. Regression caught: a grazing gold asked, which a pixel-perfect reader loses on a coin of the
    2 cm."""
    for offset, asked in ((0.285, False), (0.26, True)):
        w = side_world(offset)
        assert screens.side_stable(w, START, 0.0, GOLD, w.latest_cells(KS), blind_m(AMR)) is asked, offset


# -- the height -------------------------------------------------------------------------------------------------
def height_world(size_z: float, mobility: str = "carried", under_m: float = 0.0, y: float = 0.0):
    """The tote AMR's corridor with a 1 m square thing across it from x 2.5 to 3.5 (centre (3.0, y)), `size_z` tall."""
    script = drive_run()
    thing(script, "E0001", (1.0, 1.0, size_z), (3.0, y, 0.0), mobility, under_m)
    return world(script, None)


def clear(w, value: float = 4.0) -> bool:
    return screens.height_clear(w, LAST, START, 0.0, {"type": "number", "value": value})


def test_a_top_a_camera_cannot_tell_from_the_climb_is_not_asked():
    """Contract (review D3, `CLEAR_HEIGHT_M` 0.005): not asked when a thing that meets the corridor before the stop has
    its top within 0.5 cm of the climb (0.01 m): a mat 1.5 cm tall (0.5 cm over) or 1.3 cm; asked for one 2.0 cm tall (1
    cm over). A fixed floor covering whose top is under the climb plus the 3 mm quantum (1.0 cm, 1.3 cm) is floor by
    construction and exempt; one 1.4 cm is not. The thing is skipped when it lies beside the corridor (1 m to the side)
    or beyond the stop (the stop at 1.0, the mat from 2.5). Regression caught: a gold decided by a height no frame
    shows."""
    assert not clear(height_world(0.015)) and not clear(height_world(0.013)) and clear(height_world(0.02))
    assert clear(height_world(0.01, "fixed")) and clear(height_world(0.013, "fixed"))
    assert not clear(height_world(0.014, "fixed"))
    assert clear(height_world(0.015, y=1.0)) and clear(height_world(0.015), 1.0)


def test_an_underside_a_camera_cannot_tell_from_the_bands_top_is_not_asked():
    """Contract (review D3): not asked when a thing over the corridor has its underside (`Entity.under_m`) within 0.5 cm
    of the band's top. The tote AMR's band top is 0.65 + 0.05 / 2 = 0.675: a table 0.68 over the floor (0.5 cm over) is
    not asked, one at 0.70 is. The vacuum's is 0.09 + 0.01 / 2 = 0.095: a coffee table's shelf at 0.10 (the same 0.5 cm)
    is not asked, a shelf at 0.12 is. Regression caught: the vacuum under a shelf that is 0.5 cm over its band, decided
    by a height no frame shows."""
    assert not clear(height_world(0.75, "fixed", 0.68)) and clear(height_world(0.75, "fixed", 0.70))
    def shelf(under: float) -> bool:
        s = space_kit.vacuum_run()
        s.entities.append(Entity("F0001", "coffee_table", {}, "fixed", (0.6, 0.6, 0.4), under_m=under))
        s.placements.append(Placement("F0001", 0.0, None, (1.0, 0.0, 0.0), quat_z(0.0), None))
        w = World(space_kit.inputs(s), space_kit.record(s))
        return screens.height_clear(w, 21, (0.0, 0.0), 0.0, {"type": "number", "value": 2.0})

    assert not shelf(0.10) and shelf(0.12)


def test_the_height_screen_is_centred_on_the_bands_floor_not_the_climb():
    """Contract (final review, nit): a camera cannot tell a top from the band's floor (the climb plus the 3 mm
    quantum, `grid.band`: 0.013 for the tote AMR) within 0.5 cm, so the screen reads the floor, which is what the gold
    does. Hand values: a thing 0.017 tall is 4 mm over the floor (7 mm over the climb): not asked; one 0.019 is 6 mm
    over: asked; one 0.0075 is 5.5 mm under the floor (2.5 mm under the climb): asked. Regression caught: a screen
    centred on the climb, which asks the 0.017 thing and refuses the 0.0075 one."""
    assert not clear(height_world(0.017)) and clear(height_world(0.019)) and clear(height_world(0.0075))


def hanging(bottom: float, mobility: str = "carried"):
    """The tote AMR's corridor with a 1 m square thing 0.3 m thick hanging across it from `bottom` m over the floor."""
    script = drive_run()
    thing(script, "E0001", (1.0, 1.0, 0.3), (3.0, 0.0, bottom), mobility)
    return world(script, None)


def test_the_bottom_edge_of_a_hanging_thing_is_screened_against_the_bands_top():
    """Contract (final review, nit): a thing that hangs over the corridor (a picture frame, a shelf's lip) has a lower
    edge that is read against the band's top (0.675 for the tote AMR) as an underside is: not asked within 0.5 cm of
    it. Hand values: a thing whose bottom is at 0.68 (5 mm over) or 0.67 (5 mm under) is not asked, one at 0.70 or 0.60
    is, a thing standing on the floor (bottom 0) is not judged by its bottom. Regression caught: a gold decided by the
    lower edge of a frame no picture shows (helper case b458860ca444f88a, k160: 1.433 m against the band's 1.435)."""
    assert not clear(hanging(0.68)) and not clear(hanging(0.67)) and not clear(hanging(0.68, "fixed"))
    assert clear(hanging(0.70)) and clear(hanging(0.60)) and clear(height_world(0.3))
