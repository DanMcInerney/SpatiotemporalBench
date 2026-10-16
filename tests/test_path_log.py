"""`path_log`: the strongest system that reads no pictures. Hand-drawn maps and hand-placed poses; every expected value
is worked out in the test (a 0.1 m fine cell, the vacuum's 0.34 m disc, its front edge 0.17 m ahead of its base)."""

from __future__ import annotations

import math

import pytest

from baseline_kit import EVENTS, at, det, gold, number, record, started, step, where
from baseline_kit import ahead as ahead_question
from stbench import baselines_path_space, grid, mapspec, score
from stbench.baselines_path import PATH_MODULES, PathLog
from stbench.baselines_path_space import DOOR_LEAF_M, detour, obstacles
from stbench.blind import Blind
from stbench.generate import DOOR_THICKNESS
from stbench.geometry import Rect
from stbench.grid import raycast
from stbench.places import dense_places
from stbench.robots import ROBOTS
from test_architecture import MODULES, import_closure

VACUUM = ROBOTS["vacuum"]
WALL = {"thickness": 0.1, "height": 2.5, "exterior": True}


def hand_map(under_m: float = 0.7, furniture: bool = True):
    """Two 4 x 3 m rooms side by side (x 0 to 8), the wall between them on x = 4.03 (faces 3.98 and 4.08) with an
    interior door 0.9 m wide centred on y = 1.5 (hinge at y = 1.05, swinging west) and an exterior door in the south
    wall at x = 6 (closed). In the west room: a fixed table 1.0 x 0.8 m centred on (2.5, 2.0), `under_m` high
    underneath, a rug and a light chair on the floor west of it."""
    def piece(fid, kind, mobility, center, size, blocks, under):
        return {"id": fid, "name": kind, "kind": kind, "mobility": mobility, "center": center, "size": size,
                "yaw": 0.0, "color": "oak", "blocks": blocks, "surfaces": [], "under_m": under}

    return mapspec.parse_map({
        "name": "hand", "bounds": [0, 0, 8, 3], "wall_thickness": 0.1, "wall_height": 2.5,
        "regions": [{"id": "R1", "name": "west", "anchor": [2, 1.5], "level": 0, "z0": 0.0, "z1": 2.5,
                     "polygon": [[0.05, 0.05], [3.98, 0.05], [3.98, 2.95], [0.05, 2.95]]},
                    {"id": "R2", "name": "east", "anchor": [6, 1.5], "level": 0, "z0": 0.0, "z1": 2.5,
                     "polygon": [[4.08, 0.05], [7.95, 0.05], [7.95, 2.95], [4.08, 2.95]]}],
        "walls": [{"id": "W01", "a": [0, 0], "b": [8, 0], **WALL}, {"id": "W02", "a": [8, 0], "b": [8, 3], **WALL},
                  {"id": "W03", "a": [0, 3], "b": [8, 3], **WALL}, {"id": "W04", "a": [0, 0], "b": [0, 3], **WALL},
                  {"id": "W05", "a": [4.03, 0], "b": [4.03, 3], **{**WALL, "exterior": False}}],
        "openings": [{"id": "D1", "type": "door", "wall": "W05", "at": 1.5, "width": 0.9, "height": 2.0, "hinge": "a",
                      "swing": "left", "interior": True, "initial_open": False, "between": ["R1", "R2"]},
                     {"id": "E1", "type": "door", "wall": "W01", "at": 6.0, "width": 0.9, "height": 2.0, "hinge": "a",
                      "swing": "left", "interior": False, "initial_open": False, "between": ["R2", "outside"]}],
        "windows": [],
        "furniture": [piece("F1", "table", "fixed", [2.5, 2.0], [1.0, 0.8, 0.75], True, under_m),
                      piece("F2", "rug", "fixed", [1.2, 2.0], [1.0, 1.0, 0.01], False, 0.0),
                      piece("F3", "chair", "light", [1.8, 2.0], [0.4, 0.4, 0.9], True, 0.0)] if furniture else [],
        "ceiling_lights": [{"region": "R1", "pos": [2, 1.5, 2.45]}]})


def ray(m, x: float, y: float, yaw_deg: float = 0.0, cap: float = 3.0) -> float:
    """The raycast from the base (x, y) heading `yaw_deg`: front edge, corridor, cells."""
    yaw = math.radians(yaw_deg)
    return raycast(obstacles(m, VACUUM), grid.front_edge(VACUUM, (x, y), yaw), yaw, VACUUM.body, cap)


def test_path_log_raycast_by_hand():
    """Contract (plan §6 N4): from a hand-placed pose the raycast gives the distance, from the body's front edge, to the
    first 0.02 m sub-cell whose centre lies in the corridor and in the map's wall pieces, fixed pieces or a door's leaf,
    less half a sub-cell, `cap` for none (the sub-cell rule: `test_answers_rear.py`; a centre on an edge is in).
    Hand values, heading +x from a base 0.17 m behind its front edge: the wall's face at 3.98 puts the sub-cell centred
    3.99 at 3.99 - 1.67 = 2.32 m, so 2.31; the open leaf (y 1.03-1.07, x 3.13-4.03) the sub-cell centred 3.13, on its
    near edge, at 3.13 - 1.67 = 1.46, so 1.45; the table's leg (x 2.0-2.15, y 1.6-1.75) the sub-cell centred 2.01 at
    2.01 - 0.67 = 1.34, so 1.33; the closed exterior leaf (y -0.02-0.02) heading south from (6, 1) the sub-cell centred
    0.01 at 0.83 - 0.01 = 0.82, so 0.81; a corridor between the legs passes under the top, over the rug and round the
    light chair, so `cap` (3.0), while a table too low to pass under stops it at its edge (1.33). Regression caught:
    the layout shortcut left weaker than the agent it models (a door leaf, a leg or a closed door that the raycast
    cannot see), or the table's top, the rug and the chair counted as walls."""
    m = hand_map()
    assert ray(m, 1.5, 2.7) == 2.31
    assert ray(m, 1.5, 1.05) == 1.45
    assert ray(m, 0.5, 1.7) == 1.33
    assert ray(m, 6.0, 1.0, yaw_deg=-90.0) == 0.81
    assert ray(m, 0.5, 2.0) == 3.0
    assert ray(hand_map(under_m=0.05), 0.5, 2.0) == 1.33
    assert ray(m, 0.5, 2.0, cap=1.0) == 1.0   # nothing within the cap: the cap, not a distance beyond it


def test_the_raycast_sees_the_apartments_open_door_leaf():
    """Contract: on the committed apartment the kitchen door D1's open leaf (the wall x = 4.6, hinge at y 1.475, the
    leaf running west to x 3.75 and 0.04 m thick, so y 1.455-1.495: `builders.py`'s hand values) stops a robot at (3.8,
    1.0) heading north: its first sub-cell row (centred y 1.47; the one at 1.45 lies outside the leaf) at 1.47 - 1.17 =
    0.30 m, so 0.29. No fine cell centre lies in the leaf, so a test of centres alone would miss it. Regression caught:
    the door leaf invisible to a raycast over the real map, the shortcut weaker than the layout-memorising agent it
    models."""
    assert ray(mapspec.load_map("apartment_a"), 3.8, 1.0, yaw_deg=90.0) == 0.29


def test_the_raycast_sees_a_long_wall_whose_centre_is_beyond_the_cap():
    """Contract: the raycast's prefilter keeps every rect that could reach the corridor, a long one included, so the
    distance is the first obstructed sub-cell's whatever the rect's centre. Hand case: a wall 10 m long and 0.1 m thick
    on y = 0 from x = 0.5 to 10.5 (centre 5.5, beyond the cap and the corridor's reach of 3.34 m), a corridor from (0,
    0) along +x, 0.34 m wide, cap 3: the first sub-cell centre inside the wall is x = 0.51 (the one at 0.49 is outside
    it), so 0.51 - 0.01 = 0.5. The same wall starting at x = 4 is beyond the cap:
    3. Regression caught: a prefilter that tests the rect's centre alone, so a wall that runs along the corridor
    is invisible to the layout-memorising shortcut."""
    for sx, cx, want in ((10.0, 5.5, 0.5), (10.0, 9.0, 3.0)):
        assert raycast([Rect(cx, 0.0, sx, 0.1, 0.0)], (0.0, 0.0), 0.0, VACUUM.body, 3.0) == want


def test_the_leaf_thickness_is_the_generators():
    """Contract: `baselines_path_space.DOOR_LEAF_M` equals `generate.DOOR_THICKNESS`, which a baseline may not import
    (the generator is truth side). Regression caught: the generator's door thickened and the raycast's leaf not."""
    assert DOOR_LEAF_M == DOOR_THICKNESS == 0.04


def poses(*xy):
    """Steps along the base positions `xy` in the phase 'transit', heading +x."""
    return [step(k, x, y, phase="transit") for k, (x, y) in enumerate(xy)]


def ask_ahead(tmp_path, steps, prior=()):
    """What `path_log` on the hand map without furniture answers to `path_ahead` (cap 3) after `steps`."""
    system = started(PathLog, tmp_path, map=hand_map(furniture=False), events=EVENTS, prior=list(prior))
    system.observe(steps)
    return system.ask("during", {"k": steps[-1]["k"], "clock": ""}, [ahead_question()])["q1"]


def test_path_log_reads_its_own_detours(tmp_path):
    """Contract (design §8.1, "paths leak"): the cells its body swept count as seen free, and a detour is evidence: an
    entry that bent round a box where the leaving corridor later runs gives the distance to where it stopped being
    swept. Hand case: the base ends at (0.5, 1.5) heading +x (front edge 0.67), after coming from the east through (3.5,
    1.5), (2.0, 1.5), (1.9, 0.9), (1.2, 0.9), (1.1, 1.5) round a box at x 1.5-1.7, y 1.4-1.6. The axis cells at 0.72,
    0.82 ... 1.22 are swept (centres within 0.17 of the last leg or its end), those at 1.32 to 1.72 are not, and 1.82 on
    are (within 0.17 of (2.0, 1.5)): the first unswept axis point is at s = 0.65 from the edge, so the answer is 0.6
    with no prior and no map. The gold, the box's face at cell 15 (centre 1.55), is 1.55 - 0.67 - 0.05 = 0.83: 0.23 off,
    full credit on the 0.25 m rung, against 0 for the raycast's 3.0. After a straight entry the same corridor gets the
    raycast, 3.0. Regression caught: a family that a detour gives away passing its shortcut gate."""
    bent = poses((3.5, 1.5), (2.0, 1.5), (1.9, 0.9), (1.2, 0.9), (1.1, 1.5), (0.5, 1.5))
    answer = ask_ahead(tmp_path / "bent", bent)
    assert answer == number(0.6)
    assert score.penalised(answer, gold(number(0.83))) == 1.0
    assert score.penalised(number(3.0), gold(number(0.83))) == -1.0
    assert ask_ahead(tmp_path / "straight", poses((3.5, 1.5), (0.5, 1.5))) == number(3.0)


def test_a_detour_needs_a_swept_start_a_gap_and_a_swept_cell_beyond_it():
    """Contract (design §8.1): `detour` reads the corridor's axis cells in order, s = 0.05, 0.15, ... below the cap, and
    answers only when the first is swept, a later one is not, and a cell beyond that gap is swept again (the robot went
    round something), `s - 0.05` at the first unswept cell. Hand cells on the axis from (0, 0.05) along +x, cap 1: the
    ten cells (0, 0) to (9, 0). Swept 0, 1, 2, 6, 7: the first gap is cell 3, at s = 0.35, so 0.3; swept 0, 1, 4, 5, 8,
    9: its first gap is cell 2, so 0.2. A path that crossed the corridor only further ahead (5 and 6 swept, the first
    cell not) is no detour, nor is one that swept the start and stopped (0, 1, 2), nor one that swept every cell, nor
    a robot that swept none: None, so the raycast answers. Regression caught: a start that need not be swept (a path
    crossing far ahead makes it answer 0 m), a gap that need not be left behind (a sweep that ends reads as an
    obstacle), either of which makes `path_log` answer short and pass its shortcut gate for nothing."""
    def cells(*us):
        return {(u, 0) for u in us}

    axis = ((0.0, 0.05), 0.0, 1.0)
    assert detour(cells(0, 1, 2, 6, 7), *axis) == 0.3
    assert detour(cells(0, 1, 4, 5, 8, 9), *axis) == 0.2
    assert detour(cells(5, 6), *axis) is None
    assert detour(cells(0, 1, 2), *axis) is None
    assert detour(cells(*range(10)), *axis) is None
    assert detour(cells(), *axis) is None


def test_path_log_answers_the_raycast_unless_its_own_path_went_round_something(tmp_path):
    """Contract (design §8.1): the same rule from real poses. The base ends at (0.5, 1.5) heading +x on the hand map
    without furniture (front edge 0.67; its axis cells 0.1 m wide from x = 0.7 on, cap 3 to x = 3.67, where the raycast
    finds nothing: 3). (a) It came by (2.5, 0.5), (2.5, 2.5), (0.5, 2.5): its body crossed the axis only at x = 2.3 to
    2.7, far ahead, and swept no cell of the first stretch: no detour, 3. (b) It backed from (1.5, 1.5) to (0.5, 1.5):
    it swept the axis cells to x = 1.67 and no more, nothing swept again: 3. Regression caught: a path that crosses the
    corridor far ahead, or a sweep that merely ends, read as a detour round something at 0 or 1 m."""
    crossed = poses((2.5, 0.5), (2.5, 2.5), (0.5, 2.5), (0.5, 1.5))
    backed = poses((1.5, 1.5), (0.5, 1.5))
    assert ask_ahead(tmp_path / "crossed", crossed) == number(3.0)
    assert ask_ahead(tmp_path / "backed", backed) == number(3.0)


def test_path_ahead_blends_the_pool_with_the_raycast(tmp_path):
    """Contract (plan §4.7): without a detour the answer is the best response over the records that share its
    (template, phase, pose) key, when five of them from three scripts do, with the raycast as an extra candidate, the
    raycast alone when there is no pool. Hand case: the raycast from (1.5, 2.7) is 2.31. Five records at that phase and
    pose whose golds are all `unknown` (the corridor is never seen empty there) make `unknown` score 1 against the
    raycast's -1: `unknown`; three of them are no pool, so 2.31. Golds at 2.2 make the median and the raycast (0.11
    off, inside the 0.25 m rung) tie at 1 and the median, first, win with 2.2. Golds 2.1, 2.1, 2.1, 2.5 and 2.5 give
    the raycast 1 on each (0.21 and 0.19 off), the median 2.1 (3 + 2 x 0.5) / 5 = 0.8 (0.4 off is three rungs of four)
    and 2.5 0.7, so the raycast wins with 2.31. Regression caught: a pose prior that is never consulted, one that is
    consulted on a handful of records, or a raycast dropped once a pool exists."""
    steps = poses((1.0, 2.7), (1.5, 2.7))
    q = ahead_question()
    here = {"phase": "transit", "base": (1.5, 2.7), "yaw": 0.0, "when": "during"}

    def pool(*values):
        return [record(q, v, script=f"s{i}", **here) for i, v in enumerate(values)]

    unknown = {"type": "unknown"}
    assert ask_ahead(tmp_path / "none", steps) == number(2.31)
    assert ask_ahead(tmp_path / "unknown", steps, prior=pool(*[unknown] * 5)) == unknown
    assert ask_ahead(tmp_path / "thin", steps, prior=pool(*[unknown] * 3)) == number(2.31)
    assert ask_ahead(tmp_path / "near", steps, prior=pool(*[number(2.2)] * 5)) == number(2.2)
    spread = pool(*map(number, (2.1, 2.1, 2.1, 2.5, 2.5)))
    assert ask_ahead(tmp_path / "spread", steps, prior=spread) == number(2.31)


def test_path_ahead_moved_unchanged(tmp_path):
    """Contract (E3a's move): `path_log`'s `path_ahead` is `baselines_path_space.path_ahead`, found through
    `PATH_ANSWERERS` and bound to the instance (`PathLog.answerers`), so it answers as it did when it lived in the
    class, from its own map and log and nothing shared. Hand case, the cases above: on the hand map with its furniture
    a base at (0.5, 1.7) heading +x, entered straight, has the table's leg 1.33 m ahead
    (`test_path_log_raycast_by_hand`); on the map without it, the same pose has 3.0. Two systems asked in turn, A (no
    furniture), B (furniture), A again, answer 3.0, 1.33, 3.0 (twice round); and the bound function is the module's,
    called with the instance. Regression caught: the move changing an answer, or its raycast's obstacles kept on the
    class or the module, so that the second system answers the first's map."""
    steps = poses((3.5, 1.7), (0.5, 1.7))
    bare = started(PathLog, tmp_path / "bare", map=hand_map(furniture=False), events=EVENTS)
    furnished = started(PathLog, tmp_path / "furnished", map=hand_map(), events=EVENTS)
    for system in (bare, furnished):
        system.observe(steps)
    rounds = [[system.ask("during", {"k": 1, "clock": ""}, [ahead_question()])["q1"] for system in (bare, furnished)]
              for _ in range(2)]
    assert rounds == [[number(3.0), number(1.33)]] * 2
    bound = bare.answerers["path_ahead"]
    assert (bound.func, bound.args) == (baselines_path_space.path_ahead, (bare,))
    assert {"path_ahead": baselines_path_space.path_ahead,
            "rear_clear": baselines_path_space.rear_clear} == baselines_path_space.PATH_ANSWERERS


def test_path_log_reads_no_pictures(tmp_path):
    """Contract: its import closure, the family modules of `PATH_MODULES` that exist included (they are found by name,
    not imported), holds no module that reads an image (`render_encode`, the table's mapper) and its answers are the
    same with every step's `rgb`, `depth` and `detections` removed: here they name files that do not exist and
    detections of things that are not there. Regression caught: a 'no-image' baseline that sees."""
    closure = import_closure(("baselines_path", *PATH_MODULES), MODULES)
    assert {"baselines_path", "baselines_path_space", "blind", "table_log"} <= closure
    assert not {"render_encode", "table", "perception"} & closure
    clean = poses((3.5, 1.5), (0.5, 1.5))
    seen = [{**s, "rgb": f"inbox/rgb/{s['k']:06d}.jpg", "depth": f"inbox/depth/{s['k']:06d}.png",
             "detections": [det("a", "box", "red", 1.0, 1.5)]} for s in clean]
    assert ask_ahead(tmp_path / "seen", seen) == ask_ahead(tmp_path / "clean", clean) == number(3.0)


def test_path_log_keys_on_what_it_was_sent(tmp_path):
    """Contract (`answer_keys`): the record it keys on holds the question, `when`, the clock, phase, base, heading and
    subgoal (None: this step has none) of the ask's step, and the rooms its briefs name among the map's region kinds,
    whole words only. Hand case: the brief 'clean the west room, not the easterly one' names `west`, not `east`, and a
    step at (1.5, 2.7) heading 90 degrees in the phase 'transit'. A system without the map or the job's events is
    refused. Regression caught: a record of another shape than the harness writes, which would make a prior that never
    matches."""
    system = started(PathLog, tmp_path, map=hand_map(), events=EVENTS)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Please clean the west room, not the easterly one.")
    system.observe([step(0, 1.5, 2.7, yaw_deg=90.0, phase="transit")])
    q = where("box", "red", "when you started cleaning")
    assert system.ask_record(q, {"k": 0, "clock": ""}, when="end") == {
        "template": "object_where", "slots": q["slots"], "when": "end", "clock": "2026-03-02T08:00:00",
        "phase": "transit", "base": (1.5, 2.7), "yaw": 90.0, "subgoal": None, "rooms": ("west",)}
    opts = {"map": hand_map(), "events": EVENTS}
    for missing in opts:
        with pytest.raises(ValueError, match=f"needs the option '{missing}'"):
            started(PathLog, tmp_path / missing, **{k: v for k, v in opts.items() if k != missing})


# -- object_where --------------------------------------------------------------------------------------------------
KITCHEN, BED = (2.0, 2.0), (7.5, 5.0)   # where the brown box was in the prior's kitchen and bedroom scripts
START = "when you started cleaning"     # the first 'cleaning' step: 3 in `scripted_steps`


def scripted_steps(x: float, y: float, yaw_deg: float, *, after: tuple | None = None):
    """Six steps in the phases docked, undocking, undocking, cleaning, cleaning, docking, the base at (x, y) heading
    `yaw_deg` for the first four (up to the event, step 3) and at `after` (x, y, yaw) for the last two."""
    phases = ["docked", "undocking", "undocking", "cleaning", "cleaning", "docking"]
    return [step(k, *((x, y, yaw_deg) if k <= 3 or after is None else after), phase=phase)
            for k, phase in enumerate(phases)]


def ask_where(system_cls, tmp_path, steps, prior, room: str, **opts):
    """`system_cls` told to clean `room`, fed `steps`, asked where the brown box was when it started cleaning."""
    system = started(system_cls, tmp_path, **opts, prior=prior)
    system.task(0, "2026-03-02T08:00:00", 0.0, f"Its current assignment is to clean the {room}.")
    system.observe(steps)
    return system.ask("end", {"k": steps[-1]["k"], "clock": ""}, [where("box", "brown", START)])["q1"]


def room_prior():
    """Ten other scripts: in five that cleaned the kitchen the brown box was at KITCHEN, in five that cleaned the
    bedroom at BED."""
    q = where("box", "brown", START)
    return [record(q, at(*place), script=f"{room}{i}", rooms=(room,))
            for room, place in (("kitchen", KITCHEN), ("bedroom", BED)) for i in range(5)]


def test_path_log_is_never_weaker_than_blind(tmp_path):
    """Contract (VISION rule 5): `path_log` knows everything `blind` does and more, so on a prior where the briefs'
    rooms carry the signal its mean penalised score on `object_where` is at least `blind`'s. Hand case: the prior holds
    the brown box at (2, 2) in five kitchen scripts and at (7.5, 5) in five bedroom ones. `blind`'s pool is all ten,
    where answering either place scores (5 - 5) / 10 = 0 like abstaining, so `unknown` (first) wins: 0 on either test
    script. `path_log` keys on the room its brief names, five golds on one place, which its camera faced before the
    event (1.5 m ahead in the kitchen, 1.0 m in the bedroom): 1 each. Regression caught: an end-question `path_log`
    that is only a coarser `blind`, so a no-image shortcut goes ungated."""
    apartment = mapspec.load_map("apartment_a")
    scripts = {"kitchen": scripted_steps(0.5, 1.9, 0.0), "bedroom": scripted_steps(6.5, 5.0, 0.0)}
    ans = {(name, room): ask_where(cls, tmp_path / f"{name}{room}", steps, room_prior(), room, **opts)
           for name, cls, opts in (("blind", Blind, {}), ("path_log", PathLog, {"map": apartment, "events": EVENTS}))
           for room, steps in scripts.items()}
    assert [ans["blind", room] for room in scripts] == [{"type": "unknown"}] * 2
    assert [ans["path_log", room] for room in scripts] == [at(*KITCHEN), at(*BED)]
    golds = {"kitchen": gold(at(*KITCHEN)), "bedroom": gold(at(*BED))}
    mean = {name: sum(score.penalised(ans[name, room], golds[room]) for room in scripts) / 2
            for name in ("blind", "path_log")}
    assert mean["path_log"] >= mean["blind"] and mean == {"blind": 0.0, "path_log": 1.0}


def golds_at(x: float, y: float) -> list[dict]:
    """Three gold rows of other scripts, each a place at (x, y)."""
    return [gold(at(x, y), qid=f"g{i}") for i in range(3)]


def test_dense_places_read_position_golds():
    """Contract (`places.dense_places`): the places that hold a thing in many scripts are found from `position`
    golds (an `object_where`'s) as from `positions` ones, densest first, each the mean of the things within 0.5 m
    it gathered, `unknown` golds holding none. Hand case: golds at (2, 2), (2.2, 2) and (2, 2), two at (7.5, 5) and
    one `unknown` give (6.2 / 3, 2, 0) and then (7.5, 5, 0). Regression caught: a pool of `position` golds that
    yields no place (the old reader took `positions` only), so `path_log` would never have a candidate place."""
    pool = [gold(at(*xy), qid=f"g{i}") for i, xy in enumerate([(2.0, 2.0), (2.2, 2.0), (2.0, 2.0), (7.5, 5.0),
                                                              (7.5, 5.0)])] + [gold({"type": "unknown"})]
    assert dense_places(pool) == [(pytest.approx(6.2 / 3), 2.0, 0.0), (7.5, 5.0, 0.0)]


def test_path_log_answers_unknown_when_it_faced_no_candidate_place(tmp_path):
    """Contract (plan §4.7): it answers `unknown` when its camera faced none of its candidate places (the kind's
    receptacle and the pool's dense places) at a logged step up to the event. Hand case, the kitchen script of the prior
    above: the robot facing away from the places (heading 180 degrees from x = 0.5), or looking at them only after the
    event (steps 4 and 5), answers `unknown` though five golds agree. With the camera facing +x, the candidates are the
    dense place (2, 2, 0) and the dining table's surface (2.1, 1.9, 0.75); for a pool of golds at (0, 1.9), behind the
    camera (0.63 m back from it), the dense place is not faced and only the table is offered. Regression caught: a guess
    at a place it never looked at, which `unknown` beats whenever the thing was seen."""
    opts = {"map": mapspec.load_map("apartment_a"), "events": EVENTS}
    away = scripted_steps(0.5, 1.9, 180.0)
    late = scripted_steps(0.5, 1.9, 180.0, after=(0.5, 1.9, 0.0))
    assert ask_where(PathLog, tmp_path / "away", away, room_prior(), "kitchen", **opts) == {"type": "unknown"}
    assert ask_where(PathLog, tmp_path / "late", late, room_prior(), "kitchen", **opts) == {"type": "unknown"}
    facing = started(PathLog, tmp_path / "facing", **opts, prior=[])
    facing.observe(scripted_steps(0.5, 1.9, 0.0))
    assert facing.faced_places("box", golds_at(2.0, 2.0), until=3) == [(2.0, 2.0, 0.0), (2.1, 1.9, 0.75)]
    assert facing.faced_places("box", golds_at(0.0, 1.9), until=3) == [(2.1, 1.9, 0.75)]
