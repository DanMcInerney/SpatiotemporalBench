"""The free-space rule on ray-cast depth (`space_kit.render_depth`) and on a script's own files (`freespace`): scenes
whose cells are worked out by hand from the geometry, seen by the vacuum's real camera (80 degrees, 0.08 m up, pitched 5
degrees up, on a base at (-0.17, 0) heading +x, so its front edge is at x = 0 and the camera at x = -0.04)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from builders import keyframe
from space_kit import SpaceScript, box, render_depth
from stbench import freespace, grid, questions_world
from stbench.robots import ROBOTS, blind_m
from synth import HOME, camera

VACUUM = ROBOTS["vacuum"]
BAND = grid.band(VACUUM)
CAM = camera()
FRAME = keyframe("vacuum", 0, 0.0, (-0.17, 0.0, 0.0), 0.0, task=0, phase="docked")


def seen_from_frame(boxes, depth_edit=None):
    """The fine-cell states the frame's depth gives for the scene."""
    depth = render_depth(CAM, FRAME.pos, FRAME.rot, boxes)
    if depth_edit:
        depth = depth_edit(depth)
    return grid.observe(grid.points(depth, CAM, FRAME.pos, FRAME.rot), 0.0, BAND)


def cells_in(states, state, us, vs):
    return {(u, v) for u in us for v in vs if states.get((u, v)) == state}


def test_points_by_hand():
    """Contract: `points` unprojects every pixel with depth above 0 at its centre (j + 0.5, i + 0.5) through its planar
    depth in the camera's unit and moves it to the map. Hand-worked for a camera at (1, 2, 0.5) with the identity
    rotation (looking along +x, y left, z up) and fx = fy = 320 / tan 40 degrees = 381.363: the pixel at row 240, column
    320 (centre offset 0.5 from the principal point) at 2000 mm is the point 2 m ahead, 0.5 x 2 / 381.363 = 0.002622
    to the right (so y 2 - 0.002622) and the same below the axis (z 0.5 - 0.002622): (3, 1.997378, 0.497378). A pixel
    of depth 0 adds no point. Regression caught: pixel corners instead of centres, depth read in the wrong unit, or a
    zero read as a surface at the camera."""
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[240, 320] = 2000
    pts = grid.points(depth, CAM, (1.0, 2.0, 0.5), (1.0, 0.0, 0.0, 0.0))
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [3.0, 1.997378, 0.497378], atol=1e-5)
    assert grid.points(np.zeros((480, 640), dtype=np.uint16), CAM, (0, 0, 0), (1.0, 0.0, 0.0, 0.0)).shape == (0, 3)


def test_freespace_matches_a_geometric_oracle():
    """Contract: a 0.3 m box (x 1.43 to 1.73, y -0.15 to 0.15, 0.3 m high) seen from the vacuum gives exactly
    four occupied cells (its front face x = 1.43 lies in column 14 and spans y -0.15 to 0.15, rows -2 to 1; its top
    is overhead), the floor in front free (columns 2 to 13, rows -2 to 1), and its shadow absent: the cells straight
    behind it (columns 16 to 19, rows -1 and 0) are free when the box is not there and unobserved when it is, since
    every ray over its top keeps rising. Regression caught: a rule that certifies space it never saw, or that reads a
    box's top as an obstacle or its shadow as floor."""
    scene = seen_from_frame([box(1.43, 1.73, -0.15, 0.15, 0.0, 0.3)])
    assert {c for c, s in scene.items() if s == "occupied"} == {(14, v) for v in range(-2, 2)}
    assert cells_in(scene, "free", range(2, 14), range(-2, 2)) == {(u, v) for u in range(2, 14) for v in range(-2, 2)}
    shadow = [(u, v) for u in range(16, 20) for v in (-1, 0)]
    assert not any(c in scene for c in shadow)
    empty = seen_from_frame([])
    assert all(empty.get(c) == "free" for c in shadow)


def test_depth_zero_certifies_nothing():
    """Contract: a pixel of depth 0 adds no point, so a cell seen only through such pixels is absent. Hand-worked on
    the empty floor: image column j sees y = -(j + 0.5 - 320) x z / 381.36, so the left half (columns 0 to 319) sees
    y > 0 and the right half y < 0; zeroing the left half leaves no cell of row 0 or above (y >= 0) and keeps the
    floor of rows -1 and below, free from column 2 to 13; zeroing everything leaves nothing. Regression caught: 'no
    surface' read as clear (map X-H5), which would certify the space past `far_m` and behind the robot's own body."""
    left_gone = seen_from_frame([], lambda d: np.where(np.arange(d.shape[1]) < 320, 0, d).astype(np.uint16))
    assert left_gone and all(v <= -1 for _, v in left_gone)
    assert cells_in(left_gone, "free", range(2, 14), (-1,)) == {(u, -1) for u in range(2, 14)}
    assert seen_from_frame([], lambda d: np.zeros_like(d)) == {}


def test_overhead_is_ignored():
    """Contract: a point above the band's top (0.095 m for the vacuum) is overhead and marks nothing; the floor under
    it is seen where the camera sees it. Hand-worked:
    - a table (top slab x 1.03 to 1.88, y -0.4 to 0.4, z 0.72 to 0.75; four 0.05 m legs at its corners, z 0 to 0.72, at
      x 1.03 to 1.08 and 1.81 to 1.86) marks exactly the legs' cells (columns 10 and 18, rows -4 and 3) and the floor
      between the legs is free (columns 11 to 16, rows -1 and 0);
    - a shelf at 0.10 m (x 1.03 to 1.43, y -0.3 to 0.3, z 0.10 to 0.12), whose underside and front face are a hair above
      the top, marks nothing, and the floor under it (columns 11 to 13, rows -1 and 0) is free.
    Regression caught: a band that includes overhead surfaces, so a vacuum could not go under a table or a shelf."""
    legs = [box(x, x + 0.05, y, y + 0.05, 0.0, 0.72) for x in (1.03, 1.81) for y in (-0.4, 0.35)]
    table = seen_from_frame([box(1.03, 1.88, -0.4, 0.4, 0.72, 0.75), *legs])
    assert {c for c, s in table.items() if s == "occupied"} == {(u, v) for u in (10, 18) for v in (-4, 3)}
    assert cells_in(table, "free", range(11, 17), (-1, 0)) == {(u, v) for u in range(11, 17) for v in (-1, 0)}
    shelf = seen_from_frame([box(1.03, 1.43, -0.3, 0.3, 0.10, 0.12)])
    assert "occupied" not in shelf.values()
    assert cells_in(shelf, "free", range(11, 14), (-1, 0)) == {(u, v) for u in range(11, 14) for v in (-1, 0)}


def test_an_overhang_in_the_band_marks_its_front_and_the_floor_under_it_free():
    """Contract (the rule's documented limit): a slab whose underside is at 0.05 m, inside the band, seen from the
    camera's 0.08 m, marks its front cells occupied (the front face from 0.05 to 0.095 m lies in the band: x 1.03,
    column 10, y -0.4 to 0.4, rows -4 to 3) and sees the floor beneath it, which is marked free (columns 11 to 17, rows
    -1 and 0): the sofa's base. A corridor stops at the front: `path_ahead` answers 1.02 (the sub-cell [1.02, 1.04) that
    holds the face, centre 1.03, less 0.01) from the cells before it. Regression caught: a change that claims either
    more (the underside seen) or less, silently moving every sofa-like gold."""
    states = seen_from_frame([box(1.03, 1.83, -0.4, 0.4, 0.05, 0.40)])
    assert {c for c, s in states.items() if s == "occupied"} == {(10, v) for v in range(-4, 4)}
    assert cells_in(states, "free", range(11, 18), (-1, 0)) == {(u, v) for u in range(11, 18) for v in (-1, 0)}
    answer, evidence = grid.path_ahead(grid.latest({0: states}), VACUUM, (-0.17, 0.0), 0.0, 3.0)
    assert (answer, evidence) == ({"type": "number", "value": 1.02}, [0])


def two_frames(base_z: float = 0.0) -> SpaceScript:
    """A script of three keyframes in the kitchen: 0 and 1 face +x from (-0.17, 0), 2 faces +y from (0, -0.17); a box
    1.43 m ahead of the front edge in each heading; only keyframes 0 and 2 have depth."""
    stage = replace(HOME, base_z=base_z)
    return SpaceScript(tasks=[[("kitchen", 3)]], stage=stage, poses={0: (-0.17, 0.0), 1: (-0.17, 0.0), 2: (0.0, -0.17)},
                       yaws={0: 0.0, 1: 0.0, 2: math.pi / 2}, depth_ks={0, 2},
                       boxes=[box(1.43, 1.73, -0.15, 0.15, base_z, base_z + 0.3),
                              box(-0.15, 0.15, 1.43, 1.73, base_z, base_z + 0.3)])


def test_script_states_reads_the_delivered_files(tmp_path):
    """Contract: `script_states` gives, per requested keyframe, the rule's cells from the script's own depth PNG, the
    pose row's camera and the robot file, with the pose row's base height as the floor: only the keyframes asked for,
    and the same cells when the whole script stands 0.5 m higher. Hand-worked from the box scene: keyframe 0 (heading
    +x) has occupied cells (14, -2..1); keyframe 2 (heading +y, base (0, -0.17)) has (-2..1, 14). Regression caught: a
    rule that reads the timeline's own floats instead of the delivered rows, uses a floor of z 0 whatever the base
    height, or reads depth the script has not got for a keyframe it was not asked about."""
    low = freespace.script_states(two_frames().write(tmp_path / "low"), [0, 2])
    assert set(low) == {0, 2}
    assert {c for c, s in low[0].items() if s == "occupied"} == {(14, v) for v in range(-2, 2)}
    assert {c for c, s in low[2].items() if s == "occupied"} == {(u, 14) for u in range(-2, 2)}
    high = freespace.script_states(two_frames(0.5).write(tmp_path / "high"), [0, 2])
    assert high == low


def ahead(boxes, cap, base=(-0.17, 0.0), yaw=0.0):
    """What `grid.path_ahead` answers over the one frame at `base` heading `yaw` (the corridor starts at the front
    edge, 0.17 m ahead of the base), and the frame's states."""
    frame = keyframe("vacuum", 0, 0.0, (*base, 0.0), yaw, task=0, phase="docked")
    depth = render_depth(CAM, frame.pos, frame.rot, boxes)
    states = grid.observe(grid.points(depth, CAM, frame.pos, frame.rot), 0.0, BAND)
    return grid.path_ahead(grid.latest({0: states}), VACUUM, base, yaw, cap), states


def doorway(gap, left=True, right=True):
    """A wall 0.2 m deep (x 1.03 to 1.23, 2 m tall) whose jambs stand `gap` m either side of the axis."""
    return ([box(1.03, 1.23, gap, 1.5, 0.0, 2.0)] if left else []) + (
        [box(1.03, 1.23, -1.5, -gap, 0.0, 2.0)] if right else [])


def test_a_door_frame_outside_the_bodys_width_does_not_stop_the_corridor():
    """Contract (F1 of M1's review): only what stands within the body's width (the vacuum's 0.34 m, half-width 0.17 m
    from the axis) stops `path_ahead`, so a door frame beside a doorway does not, however near the 0.1 m cells that
    hold it have their centres. Hand-worked from the geometry, a wall 0.2 m deep (x 1.03 to 1.23) with jambs at y =
    +/-g from the axis, the front edge at x 0 and cap 2 (the floor is seen to 2.3 m, so the corridor is read as empty):
    - g 0.19, the frame 2 cm outside the body's side: its inner faces (y 0.19 and -0.19, x 1.03 to 1.23) make cells
      (10..12, 1) and (10..12, -2) occupied, whose centres (y +/-0.15) lie in the corridor, and so the old rule stopped
      at x 1.0; their points are all outside the width, so the answer is 2 (the cap), seen from frame 0 alone;
    - g 0.15, the frame 2 cm inside: the front faces at x 1.03 are in the corridor, and the vacuum is a disc, so the
      jamb's sub-cell centred (1.03, 0.15), 0.15 to the side, is touched after 1.03 + 0.17 - sqrt(0.17^2 - 0.15^2) =
      1.03 + 0.17 - 0.08 = 1.12 (the strip said 1.03), less 0.01: 1.11, the evidence frame 0;
    - one jamb only, at 0.19 and at 0.15: 2 and 1.11 (the side does not matter).
    Regression caught: the corridor judged by the 0.1 m cells' centres, which stopped a robot that drives through a
    doorway 0.5 m ahead (its gold 0.255 m against 3 m) or, the other way, a frame that stood in its way and was
    ignored."""
    (answer, evidence), states = ahead(doorway(0.19), 2.0)
    framed = {(u, v) for u in (10, 11, 12) for v in (-2, 1)}
    assert all(states.get(c) == "occupied" for c in framed)
    assert framed <= {(u, v) for u, v, _ in grid.corridor((0.0, 0.0), 0.0, VACUUM.body.width_m, 2.0)}
    assert (answer, evidence) == ({"type": "number", "value": 2.0}, [0])
    assert ahead(doorway(0.15), 2.0)[0] == ({"type": "number", "value": 1.11}, [0])
    assert ahead(doorway(0.19, right=False), 2.0)[0][0] == {"type": "number", "value": 2.0}
    assert ahead(doorway(0.19, left=False), 2.0)[0][0] == {"type": "number", "value": 2.0}
    assert ahead(doorway(0.15, right=False), 2.0)[0][0] == {"type": "number", "value": 1.11}
    assert ahead(doorway(0.15, left=False), 2.0)[0][0] == {"type": "number", "value": 1.11}


def test_a_corridor_at_45_degrees_judges_its_width_by_the_axis():
    """Contract: the width is measured across the heading, not the map's axes: at 45 degrees a box whose nearest corner
    stands 1.0 m ahead of the front edge and 0.13 m to the left of the axis (inside the 0.17 m half-width) stops the
    corridor, and one whose corner is 0.21 m to the left (0.04 m outside) does not. Hand-worked: the base at (1, 2)
    heading 45 degrees has its front edge at s = (1 + 0.17 c, 2 + 0.17 c), c = cos 45 degrees; the point `along` ahead
    and `a` to the left of the axis is s + c (along - a, along + a), and a box lying left of the axis with that point as
    its corner (its x to the corner's, its y from it) has no point nearer the axis, so the answer is the cap
    (2) for a = 0.21. For a = 0.13 the box's nearest point inside the corridor is where its south face (y = cy = 2.1202
        + 1.13 c = 2.9193) meets the corridor's left edge, at 0.96 along and x = 1.1202 + 0.79 c = 1.679. The vacuum is
        a disc (radius 0.17), which touches a point `p` along and `w` to the left after p + 0.17 - sqrt(0.17^2 - w^2),
        so the box's corner (1.0 along, 0.13 left) is touched after 1.0 + 0.17 - sqrt(0.17^2 - 0.13^2) = 1.0605, and no
        point of its two faces (w = 0.13 + |p - 1|) is touched sooner; the sub-cells that hold it are within a
        centimetre of that, so the answer is 1.0605 less half a sub-cell, 1.05 to a centimetre (the strip said 0.937,
        the 0.1 m cell's near edge 0.911). A box on the right (x from the corner's, y up to it) at -0.13 stops it at
        the same distance, and at -0.21 neither does. Regression caught: the width judged along the map's axes, or a
        corridor that leans, whose answers are wrong only at the headings the axis-aligned cases never meet."""
    c = math.sqrt(0.5)
    s = (1.0 + 0.17 * c, 2.0 + 0.17 * c)

    def corner(a):
        return s[0] + c * (1.0 - a), s[1] + c * (1.0 + a)

    def answer(a):
        x, y = corner(a)
        wall = box(x - 1.0, x, y, y + 1.0, 0.0, 2.0) if a > 0 else box(x, x + 1.0, y - 1.0, y, 0.0, 2.0)
        return ahead([wall], 2.0, (1.0, 2.0), math.pi / 4)[0][0]

    assert answer(0.21) == answer(-0.21) == {"type": "number", "value": 2.0}
    stop = answer(0.13)
    assert stop == answer(-0.13) and stop["type"] == "number" and abs(stop["value"] - 1.05) < 0.015


def test_an_open_floor_is_answered_the_cap_where_the_floor_is_seen_far_enough():
    """Contract: an empty floor to the cap is `cap`, not `unknown`, when at least 80% of the counted cells were seen
    free, and the camera sees the floor far enough for that at most positions of the corridor: measured over 72
    headings and positions on a flat floor the share is 79% to 90% (mean 84%) and 69 of 72 answer 3, not "about 79%
    and `unknown`" (docs/questions.md). Hand-worked from the vacuum's 0.08 m camera: its floor rows thin out with
    distance, a 0.1 m cell holding four floor points out to about 2.3 m. The strip of the corridor has 116 cells (u 1
    to 29, four rows; u 0 is inside `blind_m`); the vacuum is a disc, which never sweeps the two cells of the last
    column at |y| 0.15 (their centres, 2.95 along, are touched after 2.95 + 0.17 - sqrt(0.17^2 - 0.15^2) = 3.04, past
    the cap 3), so 114 are counted. The camera sees 23 free columns (92 cells, 80.7%: 3.0) with the front edge at x 0,
    and 24 (96 cells, 84.2%) with it at x 0.07; two cells fewer seen (90 of 114, 78.9%, under 80%) is `unknown`.
    Regression caught: a corridor that can never be read as empty at the cap, the rectangle's corners counted against a
    disc, or a 0.8 that moves by a column without anyone seeing the yield change."""
    for base_x, free_cols in ((-0.17, 23), (-0.10, 24)):
        (answer, evidence), states = ahead([], 3.0, (base_x, 0.0))
        counted = [(u, v) for u, v, d in grid.corridor((base_x + 0.17, 0.0), 0.0, VACUUM.body.width_m, 3.0)
                   if d > blind_m(VACUUM)]
        free = [c for c in counted if states.get(c) == "free"]
        assert len(counted) == 116 and len(free) == 4 * free_cols
        assert (answer, evidence) == ({"type": "number", "value": 3.0}, [0])
        thinner = {c: s for c, s in states.items() if c not in free[: 4 * free_cols - 90]}   # 90 of the 114 are free
        unknown = grid.path_ahead(grid.latest({0: thinner}), VACUUM, (base_x, 0.0), 0.0, 3.0)
        assert unknown == ({"type": "unknown"}, [0])


def low_box_script(base_z: float = 0.0) -> SpaceScript:
    """One keyframe in the kitchen at (-0.17, 0) facing +x, with a 0.2 m box 1.43 m ahead of the front edge and a render
    only there (a second keyframe has none)."""
    return SpaceScript(tasks=[[("kitchen", 2)]], stage=replace(HOME, base_z=base_z), poses={0: (-0.17, 0.0)},
                       yaws={0: 0.0}, depth_ks={0}, boxes=[box(1.43, 1.73, -0.15, 0.15, base_z, base_z + 0.2)])


def test_a_zone_band_has_its_own_states(tmp_path):
    """Contract: `script_states` observes a keyframe with the floor and band it is given (a zone's ground and band;
    the pose row's base height and the robot's band when None) and keeps only the cells in `keep`. Hand values for the
    0.2 m box (its front face x 1.43, cells (14, -2..1), points 0 to 0.2 m up): in the vacuum's band (0.02, 0.095) it
    marks those four cells occupied; in a zone band (0.3 m, 3 m) the box is under the 0.3 m, so they are free and
    nothing is occupied; `keep` {(14, -1), (14, 0), (3, 0)} leaves exactly those three, as the unkept states have
    them. With the base 0.5 m up, a floor of 0.5 m (or None, the base height) sees the same free floor and a floor
    of 0 m sees every floor point 0.5 m up, over the band's 0.3 m: occupied. Regression caught: a zone judged on the
    robot's band or the base's floor, or a cache that keeps whole drone frames."""
    paths = low_box_script().write(tmp_path / "low")
    robot = freespace.script_states(paths, [0])[0]
    zone = freespace.script_states(paths, [0], 0.0, (0.3, 3.0))[0]
    face = {(14, v) for v in range(-2, 2)}
    assert {c for c, s in robot.items() if s == "occupied"} == face
    assert "occupied" not in zone.values() and all(zone[c] == "free" for c in face)
    keep = {(14, -1), (14, 0), (3, 0)}
    assert freespace.script_states(paths, [0], 0.0, (0.3, 3.0), keep=keep)[0] == {c: zone[c] for c in keep}
    high = low_box_script(0.5).write(tmp_path / "high")
    from_base = freespace.script_states(high, [0], band=(0.3, 3.0))[0]
    assert from_base == freespace.script_states(high, [0], 0.5, (0.3, 3.0))[0] == zone
    assert set(freespace.script_states(high, [0], 0.0, (0.3, 3.0))[0].values()) == {"occupied"}


def test_the_state_cache_keys_by_keyframe_floor_and_band():
    """Contract: `cached_states` computes a keyframe's cells once for each (floor, band) asked, hands every later
    asker the same dict, passes `floor_z` and `band` to the wrapped function only when given, and keeps serving a
    function of the keyframes alone. Hand values: asking [1, 2], then [2, 3] computes 1, 2 then 3 only; asking
    [1] on ground 0.15 and band (0.3, 3) computes 1 again, with both given, and once; another ground or band is
    another entry. Regression caught: one cache entry for two bands (a zone read as the robot's band), a band
    recomputed per question, or a kit's one-argument function broken by the new arguments."""
    calls = []

    def states_of(ks, **given):
        calls.append((list(ks), given))
        return {k: {(k, 0): "free"} for k in ks}

    cached = questions_world.cached_states(states_of)
    first = cached([1, 2])
    cached([2, 3])
    assert cached([1])[1] is first[1]
    assert calls == [([1, 2], {}), ([3], {})]
    cached([1], 0.15, (0.3, 3.0))
    cached([1, 1], 0.15, (0.3, 3.0))
    cached([1], 0.0, (0.3, 3.0))
    cached([1], 0.15, (0.5, 3.0))
    cached([1], band=(0.3, 3.0))
    assert calls[2:] == [([1], {"floor_z": 0.15, "band": (0.3, 3.0)}), ([1], {"floor_z": 0.0, "band": (0.3, 3.0)}),
                         ([1], {"floor_z": 0.15, "band": (0.5, 3.0)}), ([1], {"band": (0.3, 3.0)})]
    assert questions_world.cached_states(lambda ks: {k: {} for k in ks})([4]) == {4: {}}
