"""`current_view`, the best memoryless responder, and `current_view_prior`, which adds other scripts' records keyed on
its answer. Hand-fed frames (one step an `observe`) for the latest-frame rule; compiled cases for the agreement
with `memory_table`."""

from __future__ import annotations

from baseline_kit import EVENTS, at, compiled_case, det, play, record, rows, started, step, where
from stbench.baselines import CurrentView, CurrentViewPrior, MemoryTable
from stbench.events import EventRule

MUG = ("mug", "red", "when you left the dock")


def feed_frames(system):
    """Five frames of a stream, one `observe` each: the red mug seen at (2, 0.5) in step 1; a blue bag at (3, 1) in step
    3 (the mug out of view); the mug at (2.2, 0.5) in step 4, where it was moved to. Steps 0-1 are 'undocking', the rest
    'cleaning'. It yields 1, 3 and 4 after those frames, the steps a test asks after."""
    system.observe([step(0, 1.0, 0.0, phase="undocking")])
    system.observe([step(1, 1.0, 0.0, phase="undocking", detections=[det("a", "mug", "red", 2.0, 0.5)])])
    yield 1
    system.observe([step(2, 1.0, 0.0)])
    system.observe([step(3, 1.0, 0.0, detections=[det("b", "bag", "blue", 3.0, 1.0)])])
    yield 3
    system.observe([step(4, 1.0, 0.0, detections=[det("c", "mug", "red", 2.2, 0.5)])])
    yield 4


def test_current_view_answers_from_the_latest_frame(tmp_path):
    """Contract (plan §4.7, VISION rule 4): a thing in view in the latest frame gets its place there, whatever moment
    the text names; a thing out of view gets `unknown`, though an earlier frame showed it; a thing that moved gets its
    current place, not the place at the time named; nothing is read from an earlier frame. Hand case: the mug is seen
    at (2, 0.5) in step 1 and the blind-spot bag at (3, 1) in step 3, and after step 3 the mug is `unknown` and the bag
    (3, 1), whatever event the text names (one that nothing defines too); after step 4 the mug is at (2.2, 0.5), where
    `memory_table`, which resolves 'when you left the dock' to step 1, still says
    (2, 0.5). Regression caught: a memoryless baseline weaker than a memoryless agent, so the memory-needed gate would
    pass vacuously."""
    view = started(CurrentView, tmp_path / "view")
    table = started(MemoryTable, tmp_path / "table", events={MUG[2]: EventRule("last", "undocking"),
                                                             "when you finished": EventRule("last", "cleaning")})
    asked, both = {}, [where(*MUG, qid="mug"), where("bag", "blue", "when you finished", qid="bag")]
    for system, name in ((view, "view"), (table, "table")):
        for k in feed_frames(system):
            asked[name, k] = system.ask("end", {"k": k, "clock": ""}, both)
    assert asked["view", 3] == {"mug": {"type": "unknown"}, "bag": at(3.0, 1.0)}
    assert asked["view", 1]["mug"] == at(2.0, 0.5) and asked["view", 4]["mug"] == at(2.2, 0.5)
    assert asked["view", 4]["bag"] == {"type": "unknown"}
    assert view.ask("end", {"k": 4, "clock": ""}, [where("mug", "red", "when the moon rose")]) == {"q1": at(2.2, 0.5)}
    assert asked["table", 4] == {"mug": at(2.0, 0.5), "bag": at(3.0, 1.0)}


def test_current_view_equals_memory_table_on_now_questions(tmp_path):
    """Contract (plan §4.7): on every `now` question the two systems give the same answer, since a `now` question is
    about the latest frame both read; on the memory questions they differ, current_view answering only what its last
    frame shows. Regression caught: a `now` tag that is not what the latest frame shows, or a `current_view` that has
    lost the mapper."""
    case = compiled_case(tmp_path)
    _, gold = rows(case)
    view, table = (play(case, cls, **opts)[0].answers for cls, opts in ((CurrentView, {}),
                                                                       (MemoryTable, {"events": EVENTS})))
    now = {g["qid"] for g in gold if g["scope"] == "now"}
    memory = {g["qid"] for g in gold if g["scope"] == "memory"}
    assert now and {q: view[q] for q in now} == {q: table[q] for q in now}
    assert [q for q in memory if view[q] != table[q]] and all(view[q] == {"type": "unknown"} for q in memory)


def test_current_view_prior_keys_on_its_own_answer(tmp_path):
    """Contract (plan §4.7, `answer_keys.levels`): the record it keys on holds the question, `when`, the clock, phase,
    base and heading of the ask's step and `nm`, `current_view`'s answer there; the pool is the other scripts' records
    that share (template, phase, pose, the answer's cell) and the answer is the best response over it with `nm` an
    extra candidate. Hand case: after step 4 the mug is in view at (2.2, 0.5), cell `4,1`. Five records of
    that cell have golds (2, 0.5) three times and (3.2, 0.5) twice on the 0.25, 0.5, 1, 2 ladder: (2, 0.5) scores (3 -
    2 x 0.5) / 5 = 0.4, the centroid (2.48, 0.5) (3 x 0.5 + 0) / 5 = 0.3 and `nm` (3 + 0) / 5 = 0.6 (the 1 m miss on
    (3.2, 0.5) is the 1 and 2 rungs, half, so 0), the best, so `nm` is the answer. Five records of another phase, at
    the same pose and answer cell, share only the coarser (template, answer cell) level and, holding golds at (9, 9),
    give that answer; the same records of another phase and cell share no level that is not the bare template, which
    holds as many, and so give it too, while three of them are no pool and leave `nm`. Regression caught: a prior that
    never matches (keys of another rounding), one that ignores the current view, or one that answers from a handful of
    records."""
    q = where(*MUG)
    prior = [record(q, g, script=f"s{i}", phase="cleaning", base=(1.0, 0.0), nm=at(2.3, 0.6))
             for i, g in enumerate([at(2.0, 0.5)] * 3 + [at(3.2, 0.5)] * 2)]
    system = started(CurrentViewPrior, tmp_path, prior=prior)
    list(feed_frames(system))
    now = {"k": 4, "clock": ""}
    want = {"template": "object_where", "slots": q["slots"], "when": "end", "clock": "2026-03-02T08:00:04",
            "phase": "cleaning", "base": (1.0, 0.0), "yaw": 0.0, "subgoal": None, "nm": at(2.2, 0.5)}
    assert system.ask_record(q, now, when="end") == want

    def ask(name, prior):
        system = started(CurrentViewPrior, tmp_path / name, prior=prior)
        list(feed_frames(system))
        return system.ask("end", now, [q])

    def elsewhere(n, nm):
        return [record(q, at(9.0, 9.0), script=f"s{i}", phase="docking", base=(1.0, 0.0), nm=nm) for i in range(n)]

    assert ask("here", prior) == {"q1": at(2.2, 0.5)}
    assert ask("same_cell", elsewhere(5, at(2.3, 0.6))) == {"q1": at(9.0, 9.0)}   # (template, answer's cell) matches
    assert ask("other_cell", elsewhere(5, at(7.0, 7.0))) == {"q1": at(9.0, 9.0)}   # only the template matches
    assert ask("thin", elsewhere(3, at(2.3, 0.6))) == {"q1": at(2.2, 0.5)}


def test_current_view_prior_falls_back_to_its_own_answer(tmp_path):
    """Contract: with no record at all (an empty prior) the answer is `current_view`'s own. Regression caught: a prior
    system that answers `unknown` for want of a pool, which would hand the memory-needed gate a weaker responder than
    the best memoryless one."""
    system = started(CurrentViewPrior, tmp_path, prior=[])
    list(feed_frames(system))
    assert system.ask("end", {"k": 4, "clock": ""}, [where(*MUG)]) == {"q1": at(2.2, 0.5)}
