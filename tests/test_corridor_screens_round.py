"""The ground corridor's screens for a round body (`corridor_screens`; final review B2): the side, stopper and height
screens read a disc's own sweep. The tote AMR's scene (`clear_kit`: base (0.6, 0), front edge (1.0, 0), camera 0.5 m up,
C 0.45 m) with its body made a disc as wide as it is (0.58 m across, radius 0.29), and the vacuum's run (`space_kit`:
radius 0.17, band floor 0.023)."""

from __future__ import annotations

from dataclasses import replace

import space_kit
from clear_kit import Owned, drive_run, thing, wall, world
from space_kit import box
from stbench import corridor_screens as screens
from stbench import grid
from stbench.geometry import quat_z
from stbench.questions_world import World
from stbench.robots import ROBOTS, blind_m
from stbench.timeline import Entity, Placement
from test_corridor_screens import KS, START, seen_things, tied

AMR = ROBOTS["amr"]
VACUUM = ROBOTS["vacuum"]


def round_body(w: World) -> World:
    """The same world for a disc as wide as the tote AMR (0.58 m across, radius 0.29): the camera, the band and the
    start are the AMR's, only the body's shape differs."""
    body = replace(w.robot.body, shape="disc", length_m=0.58, width_m=0.58)
    return World(replace(w.inp, robot=replace(w.robot, body=body)), w.rec)


def gold_of(w: World) -> dict:
    return grid.corridor_clear(w.latest_cells(KS), w.robot, START, 0.0, 5.0, blind_m(AMR))[0]


def side_world(offset: float) -> World:
    """A 0.5 m box whose side lies `offset` m north of the axis (x 3.01 to 3.51, y offset to offset + 0.5, 0.5 m tall)
    and a wall at x 5.51, structure, 4.5 m from the front edge."""
    return world(drive_run(boxes=[box(3.01, 3.51, offset, offset + 0.5, 0.0, 0.5), wall(5.51).at]), None)


def test_the_side_screen_judges_a_disc_by_its_own_radius():
    """Contract (review B2): the side screen narrows and widens the body by `CLEAR_SIDE_M` each side, a disc's radius
    included. Hand scenes for the disc (radius 0.29): a box whose side is at 0.285 (sub-cell centre 0.29) is touched
    only at the disc's widest point, so the disc narrowed to 0.28 misses it and the gold moves to the wall (4.5 m
    on): not asked; a box at 0.26 (centre 0.27) is touched after 2.02 + 0.29 - sqrt(0.29^2 - 0.27^2) = 2.204 and by
    the disc 0.01 narrower or wider after 2.226 or 2.189, within C/2 (0.225): asked. Regression caught: a disc's side
    screen decided by the strip's width alone."""
    for offset, asked in ((0.285, False), (0.26, True)):
        w = round_body(side_world(offset))
        gold = gold_of(w)
        assert gold["type"] == "number", offset
        assert screens.side_stable(w, START, 0.0, gold, w.latest_cells(KS), blind_m(AMR)) is asked, offset


def test_the_stopper_of_a_disc_is_the_thing_it_touches_first():
    """Contract (review B2): the stopper screen reads the obstacle the body touches first, not the nearest ahead. Hand
    scene for the disc (radius 0.29, front edge at x 1.0): a floor box A at x 3.01 to 3.2, y 0.25 to 0.30 (nearest
    ahead, 2.02, but touched at its side: after about 2.20) and a table B on the axis from x 3.12 (its top at 0.45 to
    0.50, legs 0.4 m behind its edge), touched after about 2.13. B stops the disc and is not tied (its floor contact is
    beyond C/2), whatever A is; with B a floor box the stop is tied. Regression caught: the tie read on the nearest
    thing ahead (A, a box on the floor: tied)."""
    def scene(table: bool) -> World:
        a = Owned(box(3.01, 3.2, 0.25, 0.30, 0.0, 0.5), "E0001")
        if table:
            top = Owned(box(3.12, 3.62, -0.1, 0.1, 0.45, 0.5), "E0002")
            legs = [Owned(box(3.52, 3.57, y, y + 0.05, 0.0, 0.45), "E0002") for y in (-0.1, 0.05)]
            b = [top, *legs]
        else:
            b = [Owned(box(3.12, 3.62, -0.1, 0.1, 0.0, 0.5), "E0002")]
        script = drive_run(boxes=[o.at for o in (a, *b)])
        thing(script, "E0001", (0.19, 0.05, 0.5), (3.105, 0.275, 0.0))
        thing(script, "E0002", (0.5, 0.2, 0.5), (3.37, 0.0, 0.0))
        return round_body(world(seen_things(script, "E0001", "E0002"), [a, *b]))

    for table, is_tied in ((True, False), (False, True)):
        w = scene(table)
        gold = gold_of(w)
        assert gold["type"] == "number" and 2.05 < gold["value"] < 2.14, (table, gold)
        assert tied(w, gold) is is_tied, table


def vacuum_world(*things: tuple[tuple[float, float, float], tuple[float, float, float]], robot=None) -> World:
    """The vacuum's run with `things` (a size and a base centre each, carried, ids F0001 on) across its corridor from
    the origin, as a `World`; `robot` replaces the vacuum."""
    s = space_kit.vacuum_run()
    for n, (size, at) in enumerate(things, 1):
        s.entities.append(Entity(f"F000{n}", "pallet", {}, "carried", size))
        s.placements.append(Placement(f"F000{n}", 0.0, None, at, quat_z(0.0), None))
    inp = space_kit.inputs(s)
    return World(inp if robot is None else replace(inp, robot=robot), space_kit.record(s))


def test_the_height_screen_sweeps_the_discs_path_not_the_strip():
    """Contract (review B2): an entity the body sweeps is judged for its height edges; for a disc that is the stadium of
    its path (its sides beside the front arc, behind the front edge, and a round tip), not the rectangle from the
    front edge. Hand scenes, the vacuum (radius 0.17, band floor 0.023), a mat 0.02 tall (3 mm under the floor, an edge
    no picture shows): (a) the mat at x -0.08 to -0.02, y 0.12 to 0.16, beside the front arc (0.192 from the disc's
    centre: outside the body, swept as it moves on) on a 2.0 m corridor: the disc is refused; the strip, from the edge,
    does not meet it: asked; (b) the mat at x 3.95 to 4.0, y 0.15 to 0.17 at the far corner of a 4.0 m corridor, 0.192
    from the end of the path: the disc is asked; the strip's corner meets it: refused. Regression caught: the strip's
    footprint test for a round body."""
    box_vacuum = replace(VACUUM, body=replace(VACUUM.body, shape="rect"))

    def asked(mat, value, robot=None) -> bool:
        w = vacuum_world(mat, robot=robot)
        return screens.height_clear(w, 21, (0.0, 0.0), 0.0, {"type": "number", "value": value})

    beside_arc = ((0.06, 0.04, 0.02), (-0.05, 0.14, 0.0))
    far_corner = ((0.05, 0.02, 0.02), (3.975, 0.16, 0.0))
    assert not asked(beside_arc, 2.0) and asked(beside_arc, 2.0, box_vacuum)
    assert asked(far_corner, 4.0) and not asked(far_corner, 4.0, box_vacuum)
