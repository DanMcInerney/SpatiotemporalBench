"""The shortcut systems on a held-out layout (`suite_priors`, `form_check`, `suite_report`, `PathLog`): an agent that
learned the public layouts has never seen an unseen one, so `path_log` gets the region kinds and no map, a family that
is a fact of the layout (reference `other_case`) is learned from public layouts only, and the reach rule spares what
that leaves empty. The apartment stands in for a held-out layout by patching its name into `map_files.HELD_OUT` (a
held-out map itself is never a fixture)."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, question, started, step
from check_kit import hand_synth
from run_kit import golds
from score_kit import result_row
from stbench import baselines, gates, map_files, suite_priors, suite_report
from stbench.baselines import Baseline
from stbench.baselines_path import PathLog
from stbench.form_check import Asked, form_adversary
from stbench.mapspec import load_map
from stbench.questions_text import FAMILIES
from stbench.stream import CaseRun
from stbench.stream_protocol import System
from stream_kit import case_on, end_q, gold_row, qid

LAYOUT = "apartment_a"
INSTRUMENTS = ("oracle", "memory_table", "current_view", "current_view_prior")   # they keep the map on every layout
ADJACENT = {"space": "room", "a": "kitchen", "b": "bedroom"}
SQUARE = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]


def held_out(monkeypatch) -> None:
    """The apartment is a held-out layout for this test."""
    monkeypatch.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, LAYOUT))


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Three one-task cases of three scripts on the apartment, each asking `adjacent` (reference `other_case`, credit
    `region`) and `object_where` at the end, with their golds."""
    root = tmp_path_factory.mktemp("pools")
    out = []
    for seed in (1, 2, 3):
        qs = [end_q(1, template="adjacent", slots=ADJACENT, answer_type="position", text="Give a point."), end_q(2)]
        case = case_on(root, hand_synth(seed), qs)
        golds(case, rows=[gold_row(qs[0], reference="other_case", credit="region", region=SQUARE,
                                   tolerances={"erode_m": 0.2}), gold_row(qs[1])])
        out.append(case)
    return out


def options(name: str, case, cases_) -> dict:
    return suite_priors.options_for(name, case, cases_, {}).opts


def test_path_log_has_kinds_but_no_geometry_on_a_held_out_layout(cases, tmp_path, monkeypatch):
    """Contract (E §2, rule 5): on a public layout `path_log` is given the case layout's map and no kinds; on a held-out
    layout it is given the layout's region kinds in map order (hall, living room, kitchen, bedroom, bathroom here) and
    no map; the instruments (`oracle`, `memory_table`, `current_view`, `current_view_prior`) keep the map on both. The
    record it keys on names the same rooms either way: the brief "clean the kitchen and the hall" names both. Regression
    caught: `path_log` answering from an unseen layout's map, an instrument handed less than a perfect perceiver's
    knowledge, or `path_log` losing the rooms its briefs name (keys that disagree with the records)."""
    case = cases[0]
    public = {n: options(n, case, [case]) for n in ("path_log", *INSTRUMENTS)}
    assert public["path_log"]["map"].name == LAYOUT and "kinds" not in public["path_log"]
    held_out(monkeypatch)
    held = {n: options(n, case, [case]) for n in ("path_log", *INSTRUMENTS)}
    kinds = ("hall", "living room", "kitchen", "bedroom", "bathroom")
    assert tuple(r.kind for r in load_map(LAYOUT).regions) == kinds
    assert "map" not in held["path_log"] and held["path_log"]["kinds"] == kinds
    assert all(o[n]["map"].name == LAYOUT and "kinds" not in o[n] for o in (public, held) for n in INSTRUMENTS)
    records = []
    for i, opts in enumerate((public["path_log"], held["path_log"])):
        system = started(PathLog, tmp_path / str(i), **opts)
        system.task(0, "2026-03-02T08:00:00", 0.0, "Please clean the kitchen and the hall.")
        system.observe([step(0, 2.0, 1.5)])
        records.append(system.ask_record(question("adjacent", ADJACENT), {"k": 0, "clock": ""}, when="end"))
    assert records[0] == records[1] and records[0]["rooms"] == ("hall", "kitchen")


def test_the_kinds_go_to_the_flag_not_to_a_name(cases, monkeypatch):
    """Contract (plan §4.8, `baselines.py`'s rule): `options_for` reads the `models_agent` flag and nothing else, so on
    a held-out layout a system of any name that carries it is given the region kinds and no map, and one that does not
    keeps the map; on a public layout both get the map. Hand registry of two systems named for neither `path_log` nor an
    instrument. Regression caught: the agent models listed by name in the options builder, so a new one added to the
    registry with the flag would be handed an unseen layout's map."""
    class Plain(System):
        """A system the options are built for and never run."""

    monkeypatch.setattr(baselines, "BASELINES", {"agent": Baseline(Plain, needs_map=True, models_agent=True),
                                                 "instrument": Baseline(Plain, needs_map=True)})
    case = cases[0]
    public = {n: options(n, case, [case]) for n in baselines.BASELINES}
    assert all(o["map"].name == LAYOUT and "kinds" not in o for o in public.values())
    held_out(monkeypatch)
    held = {n: options(n, case, [case]) for n in baselines.BASELINES}
    assert "map" not in held["agent"] and held["agent"]["kinds"] == tuple(r.kind for r in load_map(LAYOUT).regions)
    assert held["instrument"]["map"].name == LAYOUT and "kinds" not in held["instrument"]


# one question of every family, with the slots its answerer reads
MUG, BOWL = {"kind": "mug", "attributes": {"color": "red"}}, {"kind": "bowl", "attributes": {"color": "red"}}
ASKED = {
    "path_ahead": {"cap": 3.0}, "rear_clear": {"cap": 3.0},
    "object_where": {**MUG, "event": "when you left the dock", "task": None},
    "where_was_i": {"clock": "2026-03-02T08:00:01"}, "uncovered": {}, "out_of_view": {"height": 0.3, "radius": 3.0},
    "changed": {"room": "kitchen", "tasks": [0, 1]}, "when_seen": MUG, "order": {"first": MUG, "second": MUG},
    "distance_between": {"first": MUG, "second": BOWL},
    "zone_clear": {"zone": "bay", "height": 0.3, "view": "in view"},
    "clearance_then": {"zone": "bay", "height": 0.3, "event": "when you left the dock", "task": 0},
    "slots_empty": {"places": "lane", "height": 0.3}, "adjacent": ADJACENT,
    "object_was": {**MUG, "task": 0}, "same_place": {**MUG, "tasks": [0, 1]},
    "person_where": {"kind": "person", "attributes": {}, "event": "when you left the dock", "task": None},
    "ego_now": MUG, "ego_mem": MUG, "clear_path": {"mode": "drive", "direction": "ahead", "cap": 3.0},
    "task_things": {"phrase": "within 1 m of the way you docked", "task": 0},
    "closest_person": {"task": 0}, "person_mem": {}, "open_heading": {"cap": 3.0},
    "closest_approach": {}, "approach_mem": {}, "drop_zone_people": {}, "tangle_ahead": {},
    "way_to": {**MUG, "reach_m": 1.0, "room_m": 0.49}}


def answers(tmp_path, **opts) -> dict[str, dict]:
    """`path_log` over four steps in the kitchen, asked one question of every family in turn."""
    system = started(PathLog, tmp_path, events=EVENTS, **opts)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Please clean the kitchen.")
    system.observe([step(k, 2.0 + 0.1 * k, 1.5) for k in range(4)])
    return {name: system.ask(FAMILIES[name].when, {"k": 3, "clock": ""},
                             [question(name, slots, "q1", FAMILIES[name].answer_type)])["q1"]
            for name, slots in ASKED.items()}


def test_every_path_answer_runs_without_a_map(tmp_path):
    """Contract (E §2): `PathLog(kinds=...)` with no map answers one question of every family in the registry without
    raising, and where the map was its only source it says `unknown` (with no prior here: the corridor ahead and behind,
    which it would otherwise cast a ray through the map for, and the cells of the room that its brush left, which need
    the room's polygon), while with the map it answers the first and the last; the answers that come from the log alone
    are the same either way (`where_was_i`: the base at the clock asked, (2.1, 1.5)). Regression caught: a held-out
    gate check crashing on the first map-based answer, a family nobody asked of `path_log` without a map, or a
    candidate that needs the map invented without one."""
    assert set(ASKED) == set(FAMILIES)   # a family added to the registry must be asked here
    none = answers(tmp_path / "none", kinds=("hall", "kitchen"))
    some = answers(tmp_path / "some", map=load_map(LAYOUT))
    unknown = {"type": "unknown"}
    assert none["path_ahead"] == none["rear_clear"] == none["uncovered"] == unknown
    assert some["path_ahead"]["type"] == "number" and some["uncovered"]["type"] == "cells"
    assert none["where_was_i"] == some["where_was_i"] == {"type": "position", "xy": [2.1, 1.5]}


def test_a_layout_constant_family_pools_public_layouts_only(cases, monkeypatch):
    """Contract (E §3): a family whose reference is `other_case` is learned from public layouts only. The prior of a
    case on the public apartment holds the other two scripts' `adjacent` and `object_where` records (four); with the
    apartment held out it holds their `object_where` records (two) and none of `adjacent`, in `prior_records` and so in
    the options of `blind` and `path_log`; the `other_case` partner is the same case either way (it is found by script,
    layout and length, not through `prior_records`). Regression caught: a prior that learns an unseen layout's topology
    from its other cases, the filter applied to a public layout (which would empty the gate of a family asked there),
    or an `other_case` partner moved by the filter."""
    def families(records) -> list[str]:
        return sorted(r["gold"]["family"] for r in records)

    def pools() -> list[list[str]]:
        return [families(suite_priors.prior_records(cases[0], cases, {}, nm=False)),
                *(families(options(name, cases[0], cases)["prior"]) for name in ("blind", "path_log"))]

    partner = suite_priors.other_case_for(cases[0], cases).id
    assert pools() == [["adjacent", "adjacent", "object_where", "object_where"]] * 3
    held_out(monkeypatch)
    assert pools() == [["object_where", "object_where"]] * 3
    assert suite_priors.other_case_for(cases[0], cases).id == partner


def asked(script: str, layout: str, family: str, gold: dict, **row) -> Asked:
    """One delivered question of the family on the layout, with its gold row; every question has the same feature."""
    q = {"qid": qid(1), "when": "end", "text": "Where?", "template": family, "slots": {}, "answer_type": "position"}
    info = {"case": f"c-{script}", "script": script, "layout": layout, "robot": "vacuum", "job": "vacuum/1",
            "length": "short"}
    return Asked(f"c-{script}", script, q, gold_row(q, family=family, gold=gold, **row), info, {"form": "same"})


def test_the_form_adversary_pools_a_layout_constant_family_from_public_layouts(monkeypatch):
    """Contract (E §3): for a question of a family whose reference is `other_case` the form adversary learns from the
    questions of public layouts only, and for one of any other family from every other script. Six public scripts and
    twenty on the held-out apartment, every question of one form; `adjacent`'s golds are the centre of one square on the
    public layout and of another on the held-out one (credit `region`), `object_where`'s the point (1, 1) and (5, 5).
    By hand: an `adjacent` question on the held-out layout is answered from the six public golds, the first square's
    point, and is wrong (-1); one on the public layout from the five other public golds, and is right (+1), the ten
    held-out golds not learned from. An `object_where` question is answered from all 25 other golds, the held-out
    point being the majority for either layout (19 of 25, 20 of 25) and holding on the script left out (a held-out
    question's pool leaves out a held-out script 18-6 for (5, 5) and a public one 19-5, so +0.52 against abstaining's
    0; a public question's 19-5 and 20-4, +0.6; and a resample of 25 scripts holds the six or five public ones thirteen
    times or more, a gain of 0 or under, in under 1% of the draws, so the interval is above 0): right for the held-out
    questions (+1), wrong for the public ones (-1). With ten held-out scripts the interval would reach under 0 and the
    answer be `unknown`.
    Regression caught: a form adversary that learns an unseen layout's topology from its other cases, or one that stops
    pooling a family that is not a fact of the layout."""
    held_out(monkeypatch)
    near, far = SQUARE, [[10.0, 10.0], [14.0, 10.0], [14.0, 14.0], [10.0, 14.0]]
    region = {"credit": "region", "tolerances": {"erode_m": 0.2}, "reference": "other_case"}
    questions: list[Asked] = []
    for layout, n, square, centre, spot in (("pub", 6, near, [2.0, 2.0], [1.0, 1.0]),
                                            (LAYOUT, 20, far, [12.0, 12.0], [5.0, 5.0])):
        for i in range(n):
            questions += [asked(f"{layout}-{i}", layout, "adjacent", {"type": "position", "xy": centre}, region=square,
                                **region),
                          asked(f"{layout}-{i}", layout, "object_where", {"type": "position", "xy": spot})]
    rows, _ = form_adversary(questions)
    means = {}
    for key in {(r["layout"], r["family"]) for r in rows}:
        mine = [r["score"] for r in rows if (r["layout"], r["family"]) == key]
        means[key] = sum(mine) / len(mine)
    assert means == {("pub", "adjacent"): 1.0, (LAYOUT, "adjacent"): -1.0, ("pub", "object_where"): -1.0,
                     (LAYOUT, "object_where"): 1.0}


def hand_rows(*groups: tuple[str, str, str | None]) -> dict[str, list[dict]]:
    """Result rows of each system for each (layout, family, reference) in `groups`: four scripts of two questions, the
    second unknown; the oracle and `memory_table` exact, `blind` at 0.4 and the rest as all_unknown."""
    scores = {name: [[0, 1]] * 4 for name in (*gates.BASELINE_NAMES, gates.FORM)}
    scores |= {gates.ORACLE: [[1, 1]] * 4, gates.TABLE: [[1, 1]] * 4, gates.BLIND: [[0.4, 0.4]] * 4}
    return {name: [result_row(f"{family}-c{i}", f"{family}-q{i}{j}", s, "correct" if s >= 1 else "wrong" if s < 0
                              else "abstained", script=f"s{i}", family=family, scope="memory", reference=reference,
                              slice_="unknown" if j else "answerable", layout=layout)
                   for layout, family, reference in groups for i, per in enumerate(per_script)
                   for j, s in enumerate(per)] for name, per_script in scores.items()}


def never(reference: str | None) -> dict:
    """What `group_numbers` holds of a family whose priors find no record: every reach rate 0."""
    zero = dict.fromkeys(("blind", "path_log", "current_view_prior"), 0.0)
    return {"reference": reference, "hit_rates": zero, "reach_rates": zero}


def test_the_reach_rule_spares_a_layout_constant_family(monkeypatch):
    """Contract (E §4): a prior with reach 0 is BROKEN (a vacuous gate) except where its pool is empty by rule: a
    family whose reference is `other_case`, on a held-out layout. Hand rows with every prior's reach 0: `adjacent` on
    the held-out apartment is not named; an `object_where` beside it on the same layout is, once for each of the three
    priors; and so is `adjacent` on a public layout, where the pool is not empty by rule. Regression caught: the
    exemption swallowing a real broken instrument (another family, or the same family on a public layout)."""
    held_out(monkeypatch)
    report = gates.evaluate(hand_rows((LAYOUT, "adjacent", "other_case"), (LAYOUT, "object_where", None),
                                      ("pub", "adjacent", "other_case")), n_boot=500)
    numbers = {(LAYOUT, "vacuum/1"): {"families": {"adjacent": never("other_case"), "object_where": never(None)}},
               ("pub", "vacuum/1"): {"families": {"adjacent": never("other_case")}}}
    lines = suite_report.broken_instruments(report, numbers)
    assert sorted(line.split(":")[0] for line in lines) == sorted(
        [f"BROKEN {LAYOUT} vacuum/1 object_where"] * 3 + ["BROKEN pub vacuum/1 adjacent"] * 3)


def test_the_report_says_why_a_layout_constant_family_has_no_pool(cases, monkeypatch):
    """Contract (E §4): the group's numbers carry the family's reference and, where its priors abstain by design (a
    layout-constant family on a held-out layout), the reason (`suite_report.ABSTAINS`). Over the three-case group, with
    `current_view`'s answers hand-fed: on the public apartment `adjacent`'s hit and reach rates are not 0 (the other
    scripts ask it too) and no note is made; with the apartment held out they are 0 and the note is made, while
    `object_where`'s rates and silence do not change. Regression caught: a reach rate of 0 reported with no reason, or
    a reason given for a family that has a pool."""
    answers_ = {qid(1): {"type": "position", "xy": [1.0, 1.0]}, qid(2): {"type": "position", "xy": [3.2, 1.5]}}
    runs = {("current_view", c.id): CaseRun(answers_, "complete", {}, {}, {}, None, "0" * 16) for c in cases}

    def families() -> dict:
        return suite_report.group_numbers(cases, runs, {gates.CURRENT: []})["families"]

    public = families()
    assert public["adjacent"]["reference"] == "other_case" and "note" not in public["adjacent"]
    assert all(rate > 0 for rate in public["adjacent"]["reach_rates"].values())
    held_out(monkeypatch)
    held = families()
    assert held["adjacent"]["note"] == suite_report.ABSTAINS
    assert set(held["adjacent"]["hit_rates"].values()) == set(held["adjacent"]["reach_rates"].values()) == {0.0}
    assert held["object_where"] == public["object_where"] and "note" not in held["object_where"]
