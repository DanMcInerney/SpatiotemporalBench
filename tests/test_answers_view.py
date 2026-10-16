"""The answerers of `out_of_view` (`table_view` for the table systems, `baselines_path_view` for `path_log`) on
`view_kit`'s run: the decision at keyframe 21 has the look window 19-21 and the base (-0.17, 0), radius 3 m, height 0.3
m (a second task's is at 49, look window 47-49). Expected places are worked out from the scripts; a hand store stands
for the table's own rules (`Store.add_step`, detections with a chair's or a vase's extent), a compiled case for the
harness's."""

from __future__ import annotations

import pytest

import view_kit as vk
from answers_kit import compiled_case, scored
from baseline_kit import EVENTS, ROBOT, UNKNOWN, det, question, record, rows, started, step
from space_kit import vacuum_run
from stbench import mapspec, questions_view, score
from stbench.baselines import CurrentView, MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.stream import delivered, run_case
from stbench.table import Store
from stbench.table_view import out_of_view
from view_kit import BAG, GLIMPSE_PX, SHOES, VASE, looking, person, thing


@pytest.fixture(autouse=True, scope="module")
def primary_height():
    """These tests are worked out by hand at the text's primary height, `HEIGHT_M` (0.3 m); the height as built is the
    fallback (M2's fast gate check), pinned by `test_questions_view.test_the_built_height_is_the_fallback`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(questions_view, "HEIGHT", questions_view.HEIGHT_M)
        yield



SLOTS = {"height": 0.3, "radius": 3.0}
BEIGE = {"color": "beige"}


def listed(answer: dict) -> list[list[float]]:
    """The places of a `positions` answer, as written."""
    assert answer["type"] == "positions"
    return [i["xy"] for i in answer["items"]]


APARTMENT = mapspec.load_map("apartment_a")   # the public map the scripted systems are given (its fixed kinds)


def windowed(system_class, case):
    """`system_class` (a table that keeps its steps: the oracle, `memory_table`) told the case's look windows. The
    harness sends one frame an `observe`, and the gold of `out_of_view` reads the look window that ends at its decision
    while the table reads the steps of its last `observe`, so this test adapter lets each frame be taken as it comes
    (the inbox holds that frame's files alone) and, when a look window ends (`stream.delivered(...).windows`, the
    harness's own cuts), makes the window's steps the table's last `observe`. The asks come right after the frame at
    their step, which ends a window."""
    ends = {hi - 1 for _, hi in delivered(case, "detections").windows}

    class Windowed(system_class):
        held: list[int] = []

        def observe(self, steps):
            super().observe(steps)
            self.held = [*self.held, *(s["k"] for s in steps)]
            if self.held[-1] in ends:
                self.latest, self.held = self.held, []
                self.store.add_window(self.latest)

    return Windowed


def played(system_class, case, **opts):
    """`run_case` of `case` on `system_class` over the `detections` channel (fed the look windows, `windowed`), given
    the public map, complete and with no failure."""
    opts.setdefault("map", APARTMENT)
    system_class = windowed(system_class, case)
    run = run_case(lambda state, work: system_class(state, work, events=EVENTS, **opts), case, channel="detections")
    assert run.status == "complete" and run.failures == {}, system_class.__name__
    return run


def hand_world():
    """Things around the base (-0.17, 0), each seen at keyframe 3 (before the look window 19-21): C1 and C2 chairs 1.0 m
    ahead and 1.2 m to the side (the two items); SH shoes (0.1 m); BG a 0.314 m bag at (-1.2, -1) (inside the height
    band); RG a chair at 2.8 m (inside the radius band); FAR one at 3.6 m; VS a 0.4 m vase on a 0.8 m table; IN a chair
    at (-1.17, 0) that a 100 px row at keyframe 20 puts in view now; P1 a person at (-0.17, 2)."""
    s = vacuum_run()
    thing(s, "C1", (0.83, 0.0))
    thing(s, "C2", (-0.17, -1.2))
    thing(s, "SH", (-0.17, 1.0), SHOES, kind="shoes", mobility="carried")
    thing(s, "BG", (-1.2, -1.0), BAG, kind="bag", mobility="carried")
    thing(s, "RG", (2.63, 0.0))
    thing(s, "FAR", (3.43, 0.0))
    thing(s, "VS", (0.83, 1.5), VASE, kind="vase", mobility="carried", z=0.8)
    thing(s, "IN", (-1.17, 0.0))
    s.see(20, "IN", GLIMPSE_PX)
    person(s, "P1", (-0.17, 2.0))
    return s


def gap_world():
    """Two tasks, the gap at t 3,600 s, and the base at (-0.17, 0) in the look windows 19-21 and 47-49, with a look at
    keyframe 38 (the robot at (-0.5, 0) heading +x, a render of bare floor: the only render between a thing's
    sightings; a render of task 0's look window, with the things standing, would show bare floor too, so no other
    keyframe has one). Each thing is seen at keyframes 3 and 40 unless said:
    - A stands at (-0.17, -1.2); B moves 0.6 m from (0.83, 0) to (0.83, 0.6), the old place looked at (38);
    - C stands at (1.4, -0.3) and is gone in the gap, never seen again, the place looked at (38) and found free;
    - D moves 0.6 m from (-1.0, 1.5) to (-1.0, 2.1), the old place never looked at again;
    - T and U are identical beige chairs, T standing at (1.5, -1.5), U moving 0.6 m from (-1.2, -0.9) to (-1.2, -0.3).
    A to D have a colour each (`view_kit.thing`), so no description is shared among them."""
    s = vacuum_run(2)
    s.depth_ks = {47, 48, 49}
    moves = {"B": ((0.83, 0.0), (0.83, 0.6)), "D": ((-1.0, 1.5), (-1.0, 2.1)), "U": ((-1.2, -0.9), (-1.2, -0.3))}
    thing(s, "A", (-0.17, -1.2), at=(3, 40))
    thing(s, "C", None, at=(3,), where=[(0, 3600.0, (1.4, -0.3, 0.0))])
    thing(s, "T", (1.5, -1.5), attrs=BEIGE, at=(3, 40))
    for eid, (old, new) in moves.items():
        thing(s, eid, None, at=(3, 40), attrs=BEIGE if eid == "U" else None,
              where=[(0, 3600.0, (*old, 0.0)), (3600.0, None, (*new, 0.0))])
    looking(s, 38, (-0.5, 0.0))
    return s


@pytest.fixture(scope="module")
def hand_case(tmp_path_factory):
    """`hand_world` as the case "c0": its one `out_of_view` question (the decision at 21) and gold: `(case, qs, golds)`.
    """
    case = compiled_case(tmp_path_factory.mktemp("view"), "c0", hand_world(), vk.VIEW_JOB, 0, 0)
    return (case, *rows(case))


@pytest.fixture(scope="module")
def gap_case(tmp_path_factory):
    """`gap_world` as the case "c0" (two questions, the decisions at 21 and 49) and the oracle's and `memory_table`'s
    runs of it: `(qs, golds, {system name: run})`."""
    case = compiled_case(tmp_path_factory.mktemp("gap"), "c0", gap_world(), vk.VIEW_JOB, 0, 1)
    return (*rows(case), {cls.__name__: played(cls, case) for cls in (Oracle, MemoryTable)})


def test_the_oracle_answers_out_of_view(hand_case):
    """Contract (E3d §1, §3): through `run_case` on `detections`, the oracle and `memory_table` score 1 on the question
    of a world with every case the gold's bands exist for, and list what the table holds, not what the gold does: the
    two items C1 (0.83, 0) and C2 (-0.17, -1.2) and the places the gold makes don't-care, which the scorer drops: the
    bag (-1.2, -1) in the height band, the chair at 2.8 m in the radius band, the chair a glimpse in the look window
    puts in view, the person; never the shoes (0.1 m), the vase on the table, or the chair at 3.6 m. Regression caught:
    the table's rule differing from the gold's beyond the don't-care bands."""
    case, qs, golds = hand_case
    assert [(q["template"], q["decision_k"]) for q in qs] == [("out_of_view", vk.K)]
    assert golds[0]["gold"]["items"] == [{"xy": [-0.17, -1.2]}, {"xy": [0.83, 0.0]}]
    held = [[-1.2, -1.0], [-1.17, 0.0], [-0.17, -1.2], [-0.17, 2.0], [0.83, 0.0], [2.63, 0.0]]
    for cls in (Oracle, MemoryTable):
        run = played(cls, case)
        assert listed(run.answers[qs[0]["qid"]]) == held, cls.__name__
        assert scored(golds, run.answers) == {qs[0]["qid"]: 1.0}, cls.__name__


def chair(key: str, x: float, y: float, colour: str = "red", size=(0.4, 0.4, 0.9), z: float = 0.0,
          kind: str = "chair") -> dict:
    """A detection of a thing `size` big standing on a surface `z` high at (x, y): its box centre is half its height
    up."""
    return {**det(key, kind, colour, x, y, z + size[2] / 2), "extent_m": list(size)}


def asked(store: Store, window: list[int]) -> dict:
    """What `table_view` answers at the store's last step with `window` the last observe."""
    return out_of_view(store, SLOTS, {"now": store.steps[max(store.steps)], "latest": window, "robot": ROBOT,
                                      "map": APARTMENT})


def fed(*frames: list[dict]) -> Store:
    """A store fed steps 0, 1, ... with the base at the origin, step k carrying the detections of `frames[k]`."""
    store = Store()
    for k, dets in enumerate(frames):
        store.add_step(step(k, 0.0, 0.0, detections=dets))
    return store


def test_the_table_by_hand():
    """Contract (E3d §1 items 1 and 3): from a store fed by hand, at step 5 with the look window 4-5, the table lists a
    track's last sighting before the look window when it stood on the floor, is taller than 0.3 m and is within 3 m of
    the base, and nothing else. Worked out: the red chair seen once 1.0 m ahead (1.0, 0) is listed; the yellow chair
    seen at (-1, 1) and again, 0.4 m on, at (-1, 1.4) is listed at its last place only; a chair at 2.9 m is, one at 3.1
    m is not; the blue chair seen at step 1 and again at step 4, in the look window, is in view and is not; the vase on
    a 0.8 m table (its box bottom 0.8 m over the base) is not; shoes (0.1 m) are not; and a chair first seen in the look
    window has no sighting before it and is not. Regression caught: a thing in view, off the floor, short or beyond the
    radius listed, or a track listed at its first place."""
    shoes, vase = (0.25, 0.1, 0.1), (0.1, 0.1, 0.4)
    store = fed(
        [chair("a", -1.0, 1.0, "yellow")],
        [chair("b", 1.0, 0.0), chair("c", 0.0, -1.0, "blue"), chair("d", 0.0, 1.5, size=vase, z=0.8, kind="vase"),
         chair("e", 2.9, 0.0, "green"), chair("f", 3.1, 0.0, "grey"), chair("g", -1.0, 0.0, size=shoes, kind="shoes")],
        [chair("a", -1.0, 1.4, "yellow")], [],
        [chair("c", 0.0, -1.0, "blue"), chair("h", 1.0, -1.0, "pink")], [])
    assert listed(asked(store, [4, 5])) == [[-1.0, 1.4], [1.0, 0.0], [2.9, 0.0]]
    # a look window of step 5 alone: the blue and pink chairs, last seen at step 4, were out of view
    assert listed(asked(store, [5])) == [[-1.0, 1.4], [0.0, -1.0], [1.0, -1.0], [1.0, 0.0], [2.9, 0.0]]


FREE = {(u, v): "free" for u in range(5, 16) for v in range(-5, 6)}   # fine cells over x 0.5..1.6, y -0.5..0.6


def test_a_fixed_pieces_track_is_never_listed():
    """Contract: the table never lists a track whose kind is a fixed piece's kind on the public map (the question asks
    for movable things), and a table built without the map raises. By hand at step 5, the look window 4-5: a bed (a
    fixed kind of `apartment_a`, 1.6 x 2.0 x 0.5 m) and a wardrobe (0.6 x 1.2 x 2.0 m) seen at step 1, 1.0 m and 2.0 m
    ahead, are not listed; a red chair (light) beside them is. Regression caught: the oracle listing the bed, the
    wardrobe, the desk and the hall console as movable things (M2's smoke build: 0.667 on a medium case)."""
    bed, wardrobe = (1.6, 2.0, 0.5), (0.6, 1.2, 2.0)
    store = fed([], [chair("b", 1.0, 0.0, "white", size=bed, kind="bed"),
                     chair("w", 2.0, 0.0, "white", size=wardrobe, kind="wardrobe"), chair("c", 0.0, 1.0)],
                [], [], [], [])
    assert listed(asked(store, [4, 5])) == [[0.0, 1.0]]
    with pytest.raises(ValueError, match="public map"):
        out_of_view(store, SLOTS, {"now": store.steps[5], "latest": [4, 5], "robot": ROBOT, "map": None})


def test_a_doors_track_is_never_listed():
    """Contract: a door is a fixed piece of the public map though no furniture (the map names it among its openings, and
    the timeline holds it as a fixed entity the gold never lists), so the table never lists a door's track. By hand at
    step 5, the look window 4-5, on `apartment_a` (five doors): a door (0.9 x 0.04 x 2.05 m, standing on the floor) seen
    at step 1, 1.0 m ahead, is not listed, and nor is one 2.0 m to the side; a red chair beside them is. Regression
    caught: the oracle listing the hall's door as a movable thing (M3's helper look-dev check: 0.908 on 19 questions,
    each of the four shortfalls one door place the gold lacks)."""
    door = (0.9, 0.04, 2.05)
    store = fed([], [chair("d", 1.0, 0.0, "white", size=door, kind="door"),
                     chair("e", 0.0, 2.0, "white", size=door, kind="door"), chair("c", 0.0, -1.0)],
                [], [], [], [])
    assert listed(asked(store, [4, 5])) == [[0.0, -1.0]]


def test_a_superseded_track_is_dropped(gap_case):
    """Contract (E3d §1 item 2, the plan review's m5): a track whose place the steps after its last sighting and
    before the look window certify empty is not listed, and one whose place was only looked at before, or at the same
    step, or after, or where something stands, is. By hand, a chair seen at (1, 0) at step 1 with the look window 4-5:
    free cells observed at step 2 drop it; the same cells at step 1 (the sighting's own), at step 4 (the look window) or
    with one cell of the place occupied at step 2 keep it. Through `run_case` on `gap_world` at the second task's
    decision: C, gone from (1.4, -0.3), whose place keyframe 38 saw free, is not listed (the gold has a don't-care place
    there and no item), while B (moved, re-seen at (0.83, 0.6), its old place looked at) and D (moved, re-seen at (-1.0,
    2.1), its old place never looked at again) are listed at their new places only, so the oracle and `memory_table`
    list the same and score 1 on both questions. Regression caught: a track of a thing the robot saw gone listed (m5),
    or a place called empty that was looked at too early or too late."""
    for cells, kept in (({2: FREE}, False), ({1: FREE}, True), ({4: FREE}, True),
                        ({2: {**FREE, (10, 0): "occupied"}}, True)):
        store = fed([], [chair("a", 1.0, 0.0)], [], [], [], [])
        store.cells.update(cells)
        assert listed(asked(store, [4, 5])) == ([[1.0, 0.0]] if kept else []), cells
    qs, golds, runs = gap_case
    assert [q["decision_k"] for q in qs] == [vk.K, vk.K2]
    assert [[i["xy"] for i in g["gold"]["items"]] for g in golds] == [
        [[-1.0, 1.5], [-0.17, -1.2], [0.83, 0.0], [1.4, -0.3]], [[-1.0, 2.1], [-0.17, -1.2], [0.83, 0.6]]]
    assert [r["xy"] for r in golds[1]["gold"]["dont_care"]] == [[-1.2, -0.9], [-1.2, -0.3], [1.4, -0.3], [1.5, -1.5]]
    for name, run in runs.items():
        later = listed(run.answers[qs[1]["qid"]])
        assert [-1.0, 2.1] in later and [0.83, 0.6] in later and [-1.0, 1.5] not in later, name
        assert [0.83, 0.0] not in later and [1.4, -0.3] not in later, name
        assert scored(golds, run.answers) == dict.fromkeys((q["qid"] for q in qs), 1.0), name


def test_twins_are_never_an_item(gap_case):
    """Contract (E3d §1, the gold's twin rule): two identical chairs have no item, and every place either was sighted
    at is a don't-care place, so whatever a table lists of them costs nothing. In `gap_world` T stands at (1.5, -1.5)
    and U moves from (-1.2, -0.9) to (-1.2, -0.3): the gold of the second question has the three items A, D and B and
    the twins' three places (with C's) among its don't-care places; the oracle and `memory_table` list T and U's new
    place (the newest tracks of their description, a tie; U's old track is a ghost and is left out), and an answer that
    lists all three twin places, or none, scores 1 too, while one that adds a place nothing explains scores under 1.
    Regression caught: a twin of a moved thing made an item, or its ghost made a false positive."""
    qs, golds, runs = gap_case
    twins = [[-1.2, -0.9], [-1.2, -0.3], [1.5, -1.5]]
    assert not any(place in [i["xy"] for i in g["gold"]["items"]] for g in golds for place in twins)
    assert all(place in [r["xy"] for r in golds[1]["gold"]["dont_care"]] for place in twins)
    for name, run in runs.items():
        assert listed(run.answers[qs[1]["qid"]]) == [[-1.2, -0.3], [-1.0, 2.1], [-0.17, -1.2], [0.83, 0.6],
                                                     [1.5, -1.5]], name

    def score_of(extra: list[list[float]]) -> float:
        items = [[-1.0, 2.1], [-0.17, -1.2], [0.83, 0.6], *extra]
        return score.penalised({"type": "positions", "items": [{"xy": p} for p in items]}, golds[1])

    assert (score_of([]), score_of(twins)) == (1.0, 1.0)
    assert score_of([[2.0, 1.0]]) < 1.0


def test_only_the_newest_track_of_a_description_is_listed():
    """Contract (E3d §1): of the tracks of one description only the most recently sighted is listed (those last
    sighted at the same step all are), since a thing is in one place and an older track of its description is a ghost
    the gold, which lists a thing at its last seen place, does not have. Hand store, the look window 4-5: two red chairs
    seen at step 1 at (1, 0) and (-1, 1) are both listed (a tie), as is the blue chair at (0, -1); a red chair at (2.9,
    0), 1.9 m or more from both, at step 2 is a third track, and then only it is listed of the red chairs. Regression
    caught: a ghost track listed beside the thing's newest place (a false positive nothing absorbs for a thing without
    a twin), or the newest track of a description dropped with the old."""
    frames = [[], [chair("a", 1.0, 0.0), chair("b", -1.0, 1.0), chair("c", 0.0, -1.0, "blue")]]
    tie = fed(*frames, [], [], [], [])
    assert listed(asked(tie, [4, 5])) == [[-1.0, 1.0], [0.0, -1.0], [1.0, 0.0]]
    newest = fed(*frames, [chair("d", 2.9, 0.0)], [], [], [])
    assert listed(asked(newest, [4, 5])) == [[0.0, -1.0], [2.9, 0.0]]


def test_current_view_lists_nothing(tmp_path):
    """Contract (E3d §1): `current_view` keeps the latest look window alone, so every track it holds has a sighting in
    the look window and its answer is the empty list, a memoryless baseline; `memory_table` fed the same two look
    windows lists the chair seen in the first (1, 0). Hand case: steps 0-1 detect a chair at (1, 0), steps 2-3 detect a
    stool at (0, -1), asked at 3. Regression caught: a memoryless baseline that remembers (it would hold the
    memory-needed gate's floor above the empty answer's), or a table that forgets."""
    first, second = [chair("a", 1.0, 0.0)], [chair("b", 0.0, -1.0, "oak", (0.3, 0.3, 0.45), kind="stool")]
    question_ = question("out_of_view", SLOTS, answer_type="positions")
    answers = {}
    for cls in (CurrentView, MemoryTable):
        system = started(cls, tmp_path / cls.__name__, events=EVENTS, map=APARTMENT)
        system.observe([step(0, 0.0, 0.0, detections=first), step(1, 0.0, 0.0)])
        system.observe([step(2, 0.0, 0.0, detections=second), step(3, 0.0, 0.0)])
        answers[cls.__name__] = system.ask("during", {"k": 3, "clock": "2026-03-02T08:00:03"}, [question_])["q1"]
    assert answers == {"CurrentView": {"type": "positions", "items": []},
                       "MemoryTable": {"type": "positions", "items": [{"xy": [1.0, 0.0]}]}}


def test_path_log_reads_no_pictures(hand_case):
    """Contract (E3d §2): `path_log` answers `out_of_view` from the pool of other scripts' records alone, the best
    response of the first level of its key that has the support (the template, the phase and the pose key of the ask's
    step), `unknown` with none, and whatever it is sent: every step's pictures, depth and detections removed (and
    counted, so the strip is not vacuous) give the same answer. Hand case: five records of three scripts, phase
    `transit` at (-0.17, 0), each with the gold place (2, 2), which no thing of the world stands at: the answer is that
    one place; with no prior it is `unknown`. Regression caught: a 'no-image' baseline that sees, or one that answers
    with no support."""
    case, qs, golds = hand_case
    pool = [{**record(question("out_of_view", SLOTS, answer_type="positions"), {"type": "unknown"}, script=f"s{n % 3}",
                      when="during", phase="transit", base=(-0.17, 0.0)),
             "gold": {**golds[0], "gold": {"type": "positions", "items": [{"xy": [2.0, 2.0]}]}}} for n in range(5)]
    seen = set()

    class Blindfolded(PathLog):
        def observe(self, steps):
            seen.update(key for s in steps for key in ("rgb", "depth", "detections") if key in s)
            super().observe([{k: v for k, v in s.items() if k not in ("rgb", "depth", "detections")} for s in steps])

    def play(cls, prior):
        run = run_case(lambda state, work: cls(state, work, map=mapspec.load_map("apartment_a"), events=EVENTS,
                                               prior=prior), case, channel="detections")
        assert run.status == "complete" and run.failures == {}
        return run.answers[qs[0]["qid"]]

    plain, blind = play(PathLog, pool), play(Blindfolded, pool)
    assert seen == {"depth", "detections"}
    assert plain == blind == {"type": "positions", "items": [{"xy": [2.0, 2.0]}]}
    assert score.penalised(plain, golds[0]) < 1.0   # the pool's place is not where the things were
    assert play(PathLog, []) == UNKNOWN
