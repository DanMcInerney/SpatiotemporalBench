"""The registry of the scripted systems and what each class declares for the harness (V0 §10, plan §4.7)."""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest

from baseline_kit import EVENTS, UNKNOWN, ahead, det, started, step, where
from stbench.baselines import (
    BASELINES,
    NM_SYSTEM,
    AllUnknown,
    CurrentView,
    CurrentViewPrior,
    MemoryTable,
    Oracle,
    in_pool_twins,
    lookup,
)
from stbench.gates import FORM, LEARNED, in_pool_name
from stbench.stream_protocol import CHANNELS


def test_the_registry_says_what_options_each_system_is_given():
    """Contract (plan §4.7): `Baseline`'s flags are what `suite_priors.options_for` reads, so they say exactly
    which systems get the other scripts' records (and which of those records carry `current_view`'s answers), the
    public map, the job's events and another case's stream. By the plan's table: the prior systems `blind`,
    `questionnaire_blind`, `current_view_prior`, `path_log` and the reference `kind_range` get records, only
    `current_view_prior` with the first wave's answers; the table systems (`oracle`, `memory_table`, `current_view`,
    `current_view_prior`, `kind_range`, `other_case`, `latest_table`, `first_table`), `path_log` and the reference
    `map_reader` (0.8.0) the map, for the families that name a room (agents get none: D1);
    `oracle`, `memory_table`, `path_log`, `map_reader` and `other_case` the events; only `other_case` is fed another
    stream; only `path_log` and `map_reader` model an agent (on a held-out layout they get the region kinds and no map,
    as an agent that learned the public layouts has none of an unseen one); `NM_SYSTEM` is
    `current_view`; `latest_table` and `first_table` are the oracle's table without its events: they get the map and
    no events.
    Regression caught: a flag changed without the options builder knowing, which would hand a system more than it may
    have or less than it needs, or a table system with no room to read."""
    flags = {name: tuple(flag for flag in ("needs_prior", "prior_nm", "needs_map", "needs_events", "other_stream",
                                           "models_agent") if getattr(b, flag)) for name, b in BASELINES.items()}
    assert flags == {"oracle": ("needs_map", "needs_events"), "memory_table": ("needs_map", "needs_events"),
                     "current_view": ("needs_map",), "current_view_prior": ("needs_prior", "prior_nm", "needs_map"),
                     "path_log": ("needs_prior", "needs_map", "needs_events", "models_agent"),
                     "blind": ("needs_prior",),
                     "other_case": ("needs_map", "needs_events", "other_stream"),
                     "questionnaire_blind": ("needs_prior",), "all_unknown": (), "latest_table": ("needs_map",),
                     "first_table": ("needs_map",), "kind_range": ("needs_prior", "needs_map"),
                     "map_reader": ("needs_prior", "needs_map", "needs_events", "models_agent")}
    assert NM_SYSTEM == "current_view" and BASELINES[NM_SYSTEM].cls is CurrentView


def test_each_learned_shortcut_has_an_in_pool_twin_that_check_runs_and_run_does_not():
    """Contract (VISION rule 5): `in_pool_twins` names `<name>_in_pool` (`gates.in_pool_name`) for each baseline that
    is in `gates.LEARNED` (`blind`, `path_log`, `questionnaire_blind`, `current_view_prior`; with the form adversary,
    which is no baseline, the term's one owner); `lookup` resolves a twin to its original's class and flags
    with `in_pool` set, and nothing else differs; the registry `stbench run` lists its thirteen systems. A system with
    no prior has no twin, nor have `kind_range` and `map_reader`, which take one but are the declared references of
    `ego_now` and of the structure family and are no shortcut, and a name that is none resolves to nothing. Regression
    caught: a twin that drops a flag of its original (the first wave of `current_view_prior`, the map of `path_log`),
    so a different system is scored under the other rule, a learned shortcut with no twin, whose gate then reads only
    the held-out rule, or a twin of the reference, which `check` would run and no gate reads."""
    learned = {name for name in BASELINES if name in LEARNED}
    assert learned == {"blind", "path_log", "questionnaire_blind", "current_view_prior"}
    assert set(LEARNED) == learned | {FORM} and len(BASELINES) == 13
    assert {name for name, b in BASELINES.items() if b.needs_prior} == learned | {"kind_range", "map_reader"}
    for reference in ("kind_range", "map_reader"):
        assert reference not in LEARNED
        assert lookup(in_pool_name(reference)) is None and in_pool_name(reference) not in in_pool_twins()
    assert in_pool_twins() == [in_pool_name(name) for name in BASELINES if name in learned]
    for name in learned:
        twin = lookup(in_pool_name(name))
        assert twin == replace(BASELINES[name], in_pool=True) and twin.cls is BASELINES[name].cls and twin.in_pool
        assert BASELINES[name].in_pool is False and lookup(name) is BASELINES[name]
    assert lookup(in_pool_name("oracle")) is None and lookup(in_pool_name("all_unknown")) is None
    assert lookup("blind_in_pool_in_pool") is None and lookup("nothing") is None


def test_each_class_declares_what_the_harness_reads():
    """Contract (`stream_protocol.System`): the harness reads `channels`, `privileged` and `structured` from the
    class. The oracle and its two time-blind notebooks (`latest_table`, `first_table`) alone are privileged; every
    system but `all_unknown` takes the structured view; the classes that read detections or depth (the table and its
    subclasses) run on the `detections` channel only, the others on every channel. Regression caught: `stbench run
    oracle` refused for want of a flag, a prior system handed the text view, or a table system run on a channel with no
    detections."""
    classes = {name: b.cls for name, b in BASELINES.items()}
    assert [name for name, cls in classes.items() if cls.privileged] == ["oracle", "latest_table", "first_table"]
    assert [name for name, cls in classes.items() if not cls.structured] == ["all_unknown"]
    table = {"oracle", "memory_table", "current_view", "current_view_prior", "kind_range", "other_case", "latest_table",
             "first_table"}
    assert {name: cls.channels for name, cls in classes.items()} == {
        name: ("detections",) if name in table else tuple(CHANNELS) for name in classes}


def test_baseline_classes_pickle_by_reference():
    """Contract (B.2): the suite's spawn pool is sent the class, which pickles by reference only when it is
    module-level. Regression caught: a baseline defined in a function, which would fail only in a pooled `check`."""
    for b in BASELINES.values():
        assert pickle.loads(pickle.dumps(b.cls)) is b.cls


def test_identity_is_the_oracles_alone(tmp_path):
    """Contract (VISION rule 6): the oracle refuses a step that lacks the stream's privileged `identity`, and the plain
    table never uses one even if it is sent: two identical mugs never seen together are two tracks to the oracle and one
    to `memory_table` (a unique description re-identifies a moved item). Regression caught: a ceiling that is not
    privileged, or a table that quietly uses the oracle's input."""
    oracle = started(Oracle, tmp_path / "oracle", events=EVENTS)
    with pytest.raises(ValueError, match="identity"):
        oracle.observe([step(0, 1.0, 1.0)])
    table = started(MemoryTable, tmp_path / "table", events=EVENTS)
    for system in (oracle, table):
        system.observe([step(0, 1.0, 1.0, detections=[det("a", "mug", "red", 2.0, 0.0)], identity={"a": "E1"})])
        system.observe([step(1, 1.0, 1.0, detections=[det("b", "mug", "red", 5.0, 2.0)], identity={"b": "E2"})])
    assert (len(oracle.store.tracks), len(table.store.tracks)) == (2, 1)


def test_the_table_systems_need_the_job_events_only_where_they_resolve_them(tmp_path):
    """Contract: `memory_table` and the oracle resolve an end question's moment and so refuse to be built without the
    option `events` (silently answering from the latest sighting would be `current_view`); `current_view` and its prior
    system take none. Regression caught: a table baseline that answers every moment as 'now' for want of an option the
    harness forgot."""
    for cls in (MemoryTable, Oracle):
        with pytest.raises(ValueError, match="'events'"):
            started(cls, tmp_path / cls.__name__)
    for cls in (CurrentView, CurrentViewPrior):
        started(cls, tmp_path / cls.__name__)


def test_all_unknown_is_the_floor(tmp_path):
    """Contract: `all_unknown` answers every question `unknown`, whatever it is asked and sent. Regression caught:
    a floor that is not the floor, which every gate's margin is measured from."""
    system = started(AllUnknown, tmp_path)
    system.observe([step(0, 1.0, 1.0)])
    assert system.ask("end", {"k": 0, "clock": ""}, [where("mug", "red", "e", qid="a"), ahead(qid="b")]) == {
        "a": UNKNOWN, "b": UNKNOWN}
