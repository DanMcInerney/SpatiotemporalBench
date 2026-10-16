"""`latest_table` and `first_table`, the time-blind notebooks that the `time_needed` gate holds a temporal family
against: the oracle's table keeping one sighting a track (the latest, the first) and reading no event table. Hand-fed
steps, as `test_current_view` feeds them."""

from __future__ import annotations

from baseline_kit import at, det, question, started, step, where
from stbench.baselines import BASELINES, FirstStore, FirstTable, LatestStore, LatestTable, Oracle
from stbench.events import EventRule

DOCK, DONE = "when you left the dock", "when you finished"
EVENTS = {DOCK: EventRule("last", "undocking"), DONE: EventRule("last", "cleaning")}


def seen(k: int, phase: str, *things: tuple[str, str, str, float, float]) -> dict:
    """The step `k` in `phase` with the oracle's privileged identity, in which the robot sees each of `things`,
    `(entity, kind, colour, x, y)`."""
    return step(k, 1.0, 0.0, phase=phase, detections=[det(f"d{k}{i}", kind, colour, x, y)
                                                      for i, (_, kind, colour, x, y) in enumerate(things)],
                identity={f"d{k}{i}": entity for i, (entity, *_) in enumerate(things)})


def feed(system) -> None:
    """The red mug E1 seen at (2, 0.5) while the robot undocks (steps 0-1: the dock is left at step 1), a blue bag E2 at
    (3, 1) at step 2, then the mug, moved in a gap of unseen steps, at (5, 2) at step 4 (steps 2-4 are cleaning): the
    mug's place at 'when you left the dock' is (2, 0.5), its latest place (5, 2)."""
    system.observe([seen(0, "undocking"), seen(1, "undocking", ("E1", "mug", "red", 2.0, 0.5))])
    system.observe([seen(2, "cleaning", ("E2", "bag", "blue", 3.0, 1.0)), seen(3, "cleaning")])
    system.observe([seen(4, "cleaning", ("E1", "mug", "red", 5.0, 2.0))])


def test_latest_table_keeps_the_latest_place_of_each_thing_and_ignores_the_moment(tmp_path):
    """Contract (FD-0.5.0-impl §2, `time_needed`): `latest_table` is the oracle's table with perfect identity but one
    sighting a track, the latest, and no event table, so it answers any question that names a moment with the thing's
    latest place and cannot tell where the thing was. Hand stream: the mug is seen at (2, 0.5) at step 1 and, after a
    gap in which it was moved, at (5, 2) at step 4. Asked where the mug was 'when you left the dock' (step 1) the
    oracle says (2, 0.5); `latest_table` says (5, 2), and says it again for a wording the job never defined (it reads no
    event table, so it has none to refuse it with); the bag, which did not move, is at (3, 1) for both. Its store holds
    one sighting a track, the mug's latest and the bag's only, so it cannot say which of the two was seen first (the
    oracle: the mug, at step 1 against the bag's 2; `latest_table`: the bag, the mug's one sighting being step 4), and
    it is built without the option `events`, which the oracle refuses. Regression caught: a notebook with the history
    behind it, which answers the moment and proves nothing about whether time is needed, or one that resolves the
    moment and so is `memory_table`."""
    oracle = started(Oracle, tmp_path / "oracle", events=EVENTS)
    latest = started(LatestTable, tmp_path / "latest")
    for system in (oracle, latest):
        feed(system)
    asked = [where("mug", "red", DOCK, qid="mug"), where("bag", "blue", DONE, qid="bag")]
    now = {"k": 4, "clock": ""}
    assert oracle.ask("end", now, asked) == {"mug": at(2.0, 0.5), "bag": at(3.0, 1.0)}
    assert latest.ask("end", now, asked) == {"mug": at(5.0, 2.0), "bag": at(3.0, 1.0)}
    assert latest.ask("end", now, [where("mug", "red", "when the moon rose")]) == {"q1": at(5.0, 2.0)}
    assert sorted((t.ident, len(t.sightings), t.last[1:3]) for t in latest.store.tracks) == [
        ("E1", 1, [5.0, 2.0]), ("E2", 1, [3.0, 1.0])]
    first = question("order", {"first": {"kind": "mug", "attributes": {"color": "red"}},
                               "second": {"kind": "bag", "attributes": {"color": "blue"}}}, answer_type="yesno")
    assert oracle.ask("end", now, [first]) == {"q1": {"type": "yesno", "value": True}}
    assert latest.ask("end", now, [first]) == {"q1": {"type": "yesno", "value": False}}
    assert sorted(len(t.sightings) for t in oracle.store.tracks) == [1, 2]   # the oracle keeps the mug's two places


def test_latest_table_gives_two_moments_of_a_moved_thing_one_place(tmp_path):
    """Contract: a time-blind notebook gives a thing one place whatever two moments are asked, so a question that wants
    to know whether a thing was in the same place at two moments reads the same place twice. Hand stream as above, the
    mug moved between 'when you left the dock' (step 1) and 'when you finished' (step 4): the oracle places it at
    (2, 0.5) then and at (5, 2) later; `latest_table` at (5, 2) at both. Regression caught: a `latest_table` that reads
    the moment, which would answer a change question that a notebook must fail."""
    oracle = started(Oracle, tmp_path / "oracle", events=EVENTS)
    latest = started(LatestTable, tmp_path / "latest")
    for system in (oracle, latest):
        feed(system)
    both, now = [where("mug", "red", DOCK, qid="then"), where("mug", "red", DONE, qid="later")], {"k": 4, "clock": ""}
    assert oracle.ask("end", now, both) == {"then": at(2.0, 0.5), "later": at(5.0, 2.0)}
    assert latest.ask("end", now, both) == {"then": at(5.0, 2.0), "later": at(5.0, 2.0)}


def test_latest_table_is_registered_as_the_oracles_table_without_events():
    """Contract: `latest_table` is a scripted system of `BASELINES` that `check` and `eval` can run: an `Oracle` that
    reads no event table (`reads_events` false, so the registry gives it the map and no events) and keeps its stream in
    a `LatestStore`. Regression caught: a notebook registered with the oracle's flags, which would be handed an event
    table it must not read, or one that keeps the oracle's store and so its history."""
    b = BASELINES["latest_table"]
    assert b.cls is LatestTable and issubclass(LatestTable, Oracle) and b.needs_map and not b.needs_events
    assert not LatestTable.reads_events and Oracle.reads_events
    assert LatestTable.store_class is LatestStore and Oracle.store_class is not LatestStore


def test_first_table_keeps_the_first_place_of_each_thing_and_ignores_the_moment(tmp_path):
    """Contract (DECISIONS, "0.5.0's code review", `time_needed`): `first_table` is `latest_table`'s mirror, the
    oracle's table with perfect identity but one sighting a track, the first, and no event table, so it answers any
    question that names a moment with the place the thing was first seen. Hand stream as above (the mug at (2, 0.5) at
    step 1 and, after a gap in which it was moved, at (5, 2) at step 4; the bag at (3, 1) at step 2): asked where the
    mug was 'when you finished' (step 4, where the oracle says (5, 2)) `first_table` says (2, 0.5), and says it again
    for a wording the job never defined (it reads no event table); the bag, which did not move, is at (3, 1) for both.
    Its store holds one sighting a track, the mug's first and the bag's only, so it can say which was seen first (the
    mug, step 1 against 2: yes, where `latest_table` says no), and it is built without the option `events`.
    Regression caught: a notebook with the history behind it, which answers a moved thing's later state and proves
    nothing about whether time is needed, or one that keeps the latest sighting and so is `latest_table`."""
    oracle = started(Oracle, tmp_path / "oracle", events=EVENTS)
    first = started(FirstTable, tmp_path / "first")
    for system in (oracle, first):
        feed(system)
    asked = [where("mug", "red", DONE, qid="mug"), where("bag", "blue", DONE, qid="bag")]
    now = {"k": 4, "clock": ""}
    assert oracle.ask("end", now, asked) == {"mug": at(5.0, 2.0), "bag": at(3.0, 1.0)}
    assert first.ask("end", now, asked) == {"mug": at(2.0, 0.5), "bag": at(3.0, 1.0)}
    assert first.ask("end", now, [where("mug", "red", "when the moon rose")]) == {"q1": at(2.0, 0.5)}
    assert sorted((t.ident, len(t.sightings), t.last[1:3]) for t in first.store.tracks) == [
        ("E1", 1, [2.0, 0.5]), ("E2", 1, [3.0, 1.0])]
    order = question("order", {"first": {"kind": "mug", "attributes": {"color": "red"}},
                               "second": {"kind": "bag", "attributes": {"color": "blue"}}}, answer_type="yesno")
    assert first.ask("end", now, [order]) == {"q1": {"type": "yesno", "value": True}}


def test_each_notebook_is_right_on_one_state_of_a_moved_thing_and_wrong_on_the_other(tmp_path):
    """Contract: the two time-blind notebooks are one-sided in opposite ways, which is why `time_needed` reads both: a
    thing that moved is given at its old place by `first_table` and at its new place by `latest_table`, whatever moment
    the question names. Hand stream as above, the mug moved from (2, 0.5) to (5, 2): asked where it was 'when you left
    the dock' (step 1, the earlier state, gold (2, 0.5)) `first_table` is right and `latest_table` wrong; asked 'when
    you finished' (step 4, the later state, gold (5, 2)) `latest_table` is right and `first_table` wrong; and the
    oracle is right on both. Regression caught: a pair of notebooks that agree (a family asked at one state only would
    be answered by one of them and read as time-needing by the other)."""
    systems = {"oracle": started(Oracle, tmp_path / "oracle", events=EVENTS),
               "first": started(FirstTable, tmp_path / "first"), "latest": started(LatestTable, tmp_path / "latest")}
    for system in systems.values():
        feed(system)
    now, then, later = {"k": 4, "clock": ""}, at(2.0, 0.5), at(5.0, 2.0)
    asked = [where("mug", "red", DOCK, qid="then"), where("mug", "red", DONE, qid="later")]
    got = {name: system.ask("end", now, asked) for name, system in systems.items()}
    assert got == {"oracle": {"then": then, "later": later}, "first": {"then": then, "later": then},
                   "latest": {"then": later, "later": later}}


def test_first_table_is_registered_as_the_oracles_table_without_events():
    """Contract: `first_table` is a scripted system of `BASELINES` that `check` and `eval` pick up: an `Oracle` that
    reads no event table (`reads_events` false, so the registry gives it the map and no events) and keeps its stream in
    a `FirstStore`, which is not `LatestStore`. Regression caught: a notebook missing from the registry (the gate then
    reads no rows for it), registered with the oracle's flags (handed an event table it must not read), or one that
    keeps the latest sighting."""
    b = BASELINES["first_table"]
    assert b.cls is FirstTable and issubclass(FirstTable, Oracle) and b.needs_map and not b.needs_events
    assert not FirstTable.reads_events and FirstTable.store_class is FirstStore
    assert FirstStore is not LatestStore and not issubclass(FirstStore, LatestStore)
