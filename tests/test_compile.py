"""The compile (`questions`): rows that the validators read, wording rendered from slots, determinism, no truth in
what a system is asked, golds that nothing unobserved moves, and the command that writes a dataset's cases."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

import family_kit as fk
from family_kit import fake_spec, register
from space_kit import JOB, SpaceScript, box, inputs, record, things_run, use_job, vacuum_run
from stbench import cli, job_registry, leak, questionnaire, questions, questions_world
from stbench.case import CasePaths
from stbench.geometry import quat_z
from stbench.job_registry import Relevance
from stbench.questions_select import Cand
from stbench.questions_swap import gold_changes_under_swap, swap_placements
from stbench.questions_text import FAMILIES
from stbench.questions_world import World, load_inputs, scope_of
from stbench.records import GOLD_KEYS, GOLD_OPTIONAL, QUESTION_KEYS, check_family, check_gold, check_question
from stbench.seeding import stable_hex
from stbench.stream_protocol import CAPABILITIES, ask_view
from stbench.timeline import Placement

CASE_IDS = [stable_hex("compile", i, n=16) for i in range(6)]


PATH_KEY = json.dumps(["path_ahead", {"cap": 3.0}, 21], sort_keys=True, separators=(",", ":"))  # the draw's point


def compile_ids(script=None, ids=CASE_IDS):
    """Every case id's questions and gold over one run, in memory."""
    script = script or things_run()
    inp = inputs(script)
    return [questions.compile_inputs(inp, record(script, cid=cid)) for cid in ids]


def without_qids(rows):
    return [{k: v for k, v in row.items() if k != "qid"} for row in rows]


def test_text_is_rendered_from_slots():
    """Contract: every compiled question's `text` is its family's render of its slots, and the renders are, by hand:
    `path_ahead` with cap 3 reads 'How far could you move straight ahead now before reaching anything? Answer 3 if
    nothing is within 3 m.' (cap 4.5 reads 4.5, and `{cap:g}` drops no digit); `object_where` reads 'Where did you last
    see the red bag? Answer in map coordinates.' (kind and attributes, no moment and no task), a unit kind reads 'the
    pair of black shoes', and a thing with no describable attribute is just its kind; `object_was` names a task, 'Where
    was the red bag during your third task? Answer in map coordinates.' (task 2). Regression caught: wording that
    carries more than the structured view (map X-H19), so a baseline that reads the slots knows less than one that reads
    the words, or an event or a task back in `object_where`'s text."""
    path = FAMILIES["path_ahead"].render
    assert path({"cap": 3.0}) == ("How far could you move straight ahead now before reaching anything? Answer 3 if "
                                  "nothing is within 3 m.")
    assert path({"cap": 4.5}).endswith("Answer 4.5 if nothing is within 4.5 m.")
    where = FAMILIES["object_where"].render
    tail = " Answer in map coordinates."
    assert where({"kind": "bag", "attributes": {"color": "red"}}) == "Where did you last see the red bag?" + tail
    assert where({"kind": "shoes", "attributes": {"color": "black"}}) == (
        "Where did you last see the pair of black shoes?" + tail)
    assert where({"kind": "toy", "attributes": {}}) == "Where did you last see the toy?" + tail
    was = FAMILIES["object_was"].render
    assert was({"kind": "bag", "attributes": {"color": "red"}, "task": 2}) == (
        "Where was the red bag during your third task?" + tail)
    seen = {"path_ahead": 0, "object_where": 0}
    for qs, _ in compile_ids():
        for q in qs:
            assert q["text"] == FAMILIES[q["template"]].render(q["slots"])
            seen[q["template"]] += 1
    assert all(seen.values())


def test_compiled_rows_pass_the_validators():
    """Contract: every question and gold row of a compile passes `check_question` and `check_gold`, a question and its
    gold share their qid (16 hex), the question rows have exactly `QUESTION_KEYS` and the gold rows `GOLD_KEYS` then
    `GOLD_OPTIONAL` in order, each family passes `check_family`, a during row comes before every end row, an answerable
    gold names its evidence and its entity as a subject, an unknown gold carries its tolerances too, and every need is a
    capability.
    Regression caught: a record the harness or the gates cannot read, or a family whose rows do not look alike."""
    assert {"path_ahead", "object_where"} <= set(FAMILIES)
    assert {name: check_family(spec) for name, spec in FAMILIES.items()} == {name: [] for name in FAMILIES}
    rows = [row for qs, gs in compile_ids() for row in zip(qs, gs, strict=True)]
    assert len(rows) >= 12
    for q, g in rows:
        assert check_question(q) == [] and check_gold(g) == []
        assert q["qid"] == g["qid"] and re.fullmatch(r"[0-9a-f]{16}", q["qid"])
        assert tuple(q) == QUESTION_KEYS and tuple(g) == (*GOLD_KEYS, *GOLD_OPTIONAL)
        tol = g["tolerances"]   # a ladder's first rung is 5 cm at least; a graded row's resolution is 1 cm at least
        finest = tol["ladder"][0] if "ladder" in tol else min(tol.get("resolutions_m") or [tol["resolution_m"]])
        assert set(q["needs"]) <= set(CAPABILITIES) and finest >= (0.05 if "ladder" in tol else 0.01)
        assert (g["slice"] == "answerable") == bool(g["evidence_k"]) or g["family"] == "path_ahead"
        assert all(f"e:{e}" in g["subjects"] for e in g["entities"])
    for qs, _ in compile_ids():
        assert [q["when"] for q in qs] == sorted((q["when"] for q in qs), key={"during": 0, "end": 1}.get)
    path = next(g for _, g in rows if g["family"] == "path_ahead")
    assert (path["scope"], path["credit"], path["metric"], path["critical"]) == ("now", "ladder", None, True)
    where = next(g for _, g in rows if g["family"] == "object_where")
    assert (where["scope"], where["credit"], where["metric"], where["region"], where["reference"]) == (
        "memory", "graded", "xy", None, None)


def test_compile_is_deterministic_but_qids():
    """Contract: two compiles of one case agree on everything but the random qids, so the draws depend on the case and
    never on run order or process state. Regression caught: a draw that depends on run order (an unseeded shuffle, a
    set's iteration order)."""
    first, second = compile_ids(), compile_ids()
    for (q1, g1), (q2, g2) in zip(first, second, strict=True):
        assert without_qids(q1) == without_qids(q2) and without_qids(g1) == without_qids(g2)
    assert {row["qid"] for qs, _ in first for row in qs}.isdisjoint({row["qid"] for qs, _ in second for row in qs})


def test_delivered_files_hold_no_gold():
    """Contract: what a system is asked (`ask_view`: qid, text and answer type) holds no entity id, region id or the
    word truth (`leak.RULES`), none of the case's entity ids, and no coordinate of any gold, and the structured view's
    slots hold the cap or a description and an event wording, never an id, a keyframe or a place. Regression caught:
    truth in a question."""
    script = things_run()
    ids = {e.id for e in script.entities}
    for qs, gs in compile_ids(script):
        for q, g in zip(qs, gs, strict=True):
            asked = json.dumps(ask_view(q)) + json.dumps(q["slots"])
            assert list(ask_view(q)) == ["qid", "text", "answer_type"]
            assert not [r for r, rx in leak.RULES.items() if rx.search(asked)]
            assert not any(i in asked for i in ids)
            if g["gold"]["type"] == "position":
                assert not any(str(v) in q["text"] for v in g["gold"]["xy"])
            assert set(q["slots"]) == set(FAMILIES[q["template"]].slots)


def test_hidden_changes_never_change_a_gold():
    """Contract (VISION, truth rule 2: nothing unobserved changes a kept gold):
    (a) exchanging the placements and the sightings of two things from a moment after their last sighting changes no
        candidate's gold (E0001's gold stays where it was last seen, (2, 1), though the exchange puts it at E0004's
        (3.5, 1) at the moment the question names: a gold read from the placement at the named time would have moved);
    (b) a box added in the corridor where no frame looked, with the depth images unchanged, changes no candidate's gold,
        the `path_ahead` gold included (2.2);
    (c) the same box where a frame saw it, its depth rendered, moves the `path_ahead` gold to 1.0 (its face at 1.01 is
        the sub-cell centred 1.01, less 0.01);
    (d) a thing moved far (2 m and more) after its last sighting, where no frame looked, changes no candidate's gold,
        nor does it become a moved thing the questionnaire leaves out: E0001 (last recognisable at keyframe 10) moves
        to (5, 2.8) at t 30 s and E0004 (last at 12) to (6, 2.5), and every candidate, 'got home' ones included, keeps
        its gold: stale and moved are read from sightings alone.
    Regression caught: a free-space gold that reads footprints instead of depth (design §8.1), a position read at the
    named time instead of the last sighting, a thing's staleness read from where it really is, or a check that cannot
    fail."""
    script = things_run()
    inp, rec = inputs(script), record(script)
    t = 30.0  # keyframe 20: after E0001's and E0004's last sightings (10 and 12), before the moment 'got home' (26)
    assert gold_changes_under_swap(inp, rec, "E0001", "E0004", t) == []
    moved = swap_placements(inp.tl, "E0001", "E0004", t)
    assert moved.placement_of("E0001", inp.tl.keyframe(26).t).pos[:2] == (3.5, 1.0)
    assert inp.tl.placement_of("E0001", inp.tl.keyframe(26).t).pos[:2] == (2.0, 1.0)
    before = {c.key: c.gold for c in questionnaire.candidates(World(inp, rec))}
    assert before[PATH_KEY] == {"type": "number", "value": 2.2}
    hidden = things_run()
    hidden.item("E0050", "box", {"color": "green"}, [(0, None, (1.23, 0.0, 0.0))], size=(0.4, 0.3, 0.267))
    after = {c.key: c.gold for c in questionnaire.candidates(World(inputs(hidden), rec))}
    assert {k: after[k] for k in before} == before
    seen = things_run(boxes=[*things_run().boxes, box(1.01, 1.43, -0.15, 0.15, 0.0, 0.267)])
    seen.item("E0050", "box", {"color": "green"}, [(0, None, (1.23, 0.0, 0.0))], size=(0.4, 0.3, 0.267))
    changed = {c.key: c.gold for c in questionnaire.candidates(World(inputs(seen), rec))}
    assert changed[PATH_KEY] == {"type": "number", "value": 1.0} and before[PATH_KEY] != changed[PATH_KEY]
    gone = things_run()
    gone.placements = [replace(p, t1=30.0) if p.entity in ("E0001", "E0004") else p for p in gone.placements]
    gone.placements += [Placement("E0001", 30.0, None, (5.0, 2.8, 0.0), quat_z(0.0), None),
                        Placement("E0004", 30.0, None, (6.0, 2.5, 0.0), quat_z(0.0), None)]
    after = {c.key: c.gold for c in questionnaire.candidates(World(inputs(gone), rec))}
    assert after == before
    got_home = [c for c in questionnaire.candidates(World(inputs(gone), rec)) if "e:E0001" in c.subjects]
    assert got_home and {tuple(c.gold["xy"]) for c in got_home} <= {(1.1, 1.0), (2.0, 1.0)}


def test_write_compiles_a_case_from_its_files(tmp_path, monkeypatch):
    """Contract: `write` compiles from the script's files, writes `questions.jsonl` (sorted keys, ask order) and
    `gold.jsonl` (the same order) beside `case.json`, and returns the questions asked per family. Hand-worked on the run
    with things: one `path_ahead` (keyframe 21, gold 2.2) and, by the draw of this case id, end questions that follow
    the compiled rows; a gold row carries no stamp of what made it. Regression caught: rows out of step between the
    two files, or a `compiler` key written into a gold row again."""
    use_job(monkeypatch)
    case = things_run().write_case(tmp_path)
    counts = questions.write(case)
    qs = [json.loads(line) for line in case.questions.read_text(encoding="utf-8").splitlines()]
    gs = [json.loads(line) for line in case.gold.read_text(encoding="utf-8").splitlines()]
    assert [q["qid"] for q in qs] == [g["qid"] for g in gs]
    assert all(list(q) == sorted(q) for q in qs)
    assert counts["path_ahead"] == 1 and sum(counts.values()) == len(qs) and set(counts) <= set(FAMILIES)
    assert (gs[0]["family"], gs[0]["gold"], gs[0]["evidence_k"]) == ("path_ahead", {"type": "number", "value": 2.2},
                                                                     [21])
    assert all("compiler" not in g for g in gs)


def test_the_registered_vacuum_job_compiles_a_vacuum_run(tmp_path):
    """Contract: a script of the vacuum job as the registry holds it (`vacuum/1`, no stand-in) compiles from its files:
    every row passes the validators, the case asks no during question (the job asks no family at its decision points in
    M1) and every end question is `object_where`'s, with the slots `kind` and `attributes` and no event. Regression
    caught: the compiler and the job drifting apart (a decision kind, an event or the relevance list the compiler cannot
    read), which would surface only at the integration, or a family asked at the doorway that the job dropped."""
    job = job_registry.get("vacuum/1")
    case = things_run().write_case(tmp_path)
    questions.write(case)
    qs, gs = [[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] for p in
              (case.questions, case.gold)]
    assert qs and [check_question(q) + check_gold(g) for q, g in zip(qs, gs, strict=True)] == [[]] * len(qs)
    assert {q["when"] for q in qs} == {"end"} and {q["template"] for q in qs} == {"object_where"}
    assert all(set(q["slots"]) == {"kind", "attributes"} for q in qs) and job.ends is None


def test_the_command_compiles_a_dataset_and_refuses_partial_renders(tmp_path, monkeypatch, capsys):
    """Contract: `stbench compile DATASET` compiles every case (or those named by `--cases`) and prints each case's
    yields and the total; a script whose renders are missing or partial (`render_cli.render_binding_problems`) is
    refused, the problems named, with exit 2 and no gold written, the other cases still compiled; an unknown case id
    is exit 2. Regression caught: gold compiled over missing or partial renders, which is wrong without saying so."""
    use_job(monkeypatch)
    one = things_run().write_case(tmp_path)
    two = things_run(seed=2).write_case(tmp_path)
    problems = {one.script.id: ["the render log covers keyframes [0, 2] of 3: a partial render; render every keyframe"]}
    monkeypatch.setattr("stbench.render_cli.render_binding_problems", lambda paths: problems.get(paths.id, []),
                        raising=False)
    assert cli.main(["compile", str(tmp_path), "--workers", "1"]) == 2
    out, err = capsys.readouterr()
    assert f"{two.id}: " in out and "path_ahead=1" in out and "all: 1 cases" in out
    assert one.script.id in err and "a partial render" in err and not one.gold.exists() and two.gold.exists()
    problems.clear()
    assert cli.main(["compile", str(tmp_path), "--cases", one.id, "--workers", "1"]) == 0
    assert "all: 1 cases" in capsys.readouterr().out and one.gold.exists()
    assert cli.main(["compile", str(tmp_path), "--cases", "nosuchcase", "--workers", "1"]) == 2
    assert "no such case" in capsys.readouterr().err
    assert isinstance(CasePaths.from_id(tmp_path, one.id), CasePaths)


def test_perceive_refuses_renders_that_are_not_whole(tmp_path, monkeypatch, capsys):
    """Contract: `stbench perceive` runs only when `render_cli.render_binding_problems` is clean; otherwise it names the
    problems on stderr, exits 2 and writes no visibility log or oracle channel. Regression caught: a visibility log made
    over missing or partial renders, which puts wrong evidence under every gold."""
    script = SpaceScript(tasks=[[("kitchen", 3)]])
    paths = script.write(tmp_path, perception=False)
    paths.visibility.unlink()
    monkeypatch.setattr("stbench.render_cli.render_binding_problems", lambda p: ["the render covers 2 of 3 keyframes"],
                        raising=False)
    assert cli.main(["perceive", str(paths.root)]) == 2
    assert "covers 2 of 3 keyframes" in capsys.readouterr().err
    assert not paths.visibility.exists() and not paths.oracle.exists()


def test_evidence_scope_is_resolved_per_question(monkeypatch):
    """Contract: a family whose spec scope is `evidence` writes `now` exactly when its evidence is non-empty and every
    evidence keyframe is the decision's own frame (`questions_world.scope_of`: the latest frame, which a system that
    gets one frame at a time holds when it is asked), else `memory`, and a candidate that sets no scope stops the
    compile, naming the family. Hand-worked on `vacuum_run`, decision 21 (its look window is keyframes 19 to 21;
    keyframe 14 is in the one before): evidence (21,) is `now`; (20,) `memory` (a frame of the same look window is an
    earlier frame); (14,) `memory`; (20, 21) `memory`; none, `memory`. Regression caught: a `now` tag the latest frame
    does not support, so a system holding only that frame is asked a `now` question it cannot answer."""
    register(monkeypatch, fake_spec("fake_ev", "during", "evidence", "number"), lambda w, k: None)
    w = fk.world(vacuum_run())

    def cand(evidence, scope):
        gold = {"type": "number", "value": 1.0} if evidence else {"type": "unknown"}
        return Cand("fake_ev", {"n": list(evidence)}, gold, "answerable" if evidence else "unknown", evidence, (),
                    {"ladder": [0.25, 0.5, 1.0, 2.0]}, decision_k=21, subjects=("w:21",), scope=scope)

    for evidence, want in (((21,), "now"), ((20,), "memory"), ((14,), "memory"), ((20, 21), "memory"),
                           ((), "memory")):
        assert scope_of(w, 21, evidence) == want
        _, (g,) = questions.records([cand(evidence, scope_of(w, 21, evidence))])
        assert g["scope"] == want
    with pytest.raises(ValueError, match="fake_ev"):
        questions.records([cand((20,), None)])


def test_a_region_gold_is_written(monkeypatch):
    """Contract: a candidate with a polygon and `{"erode_m": 0.2}` of a `region`-credit family writes a gold row that
    `check_gold` accepts, its region the polygon and its metric `xy` (a position answer); a family without a region
    writes null, and a number answer's metric is null; a region-credit candidate with no polygon stops the compile,
    naming the family. Regression caught: region families unwritable (`region` always null), or a metric on a row that
    has no position."""
    register(monkeypatch, fake_spec("fake_region", credit="region"), lambda w: [])
    polygon = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    where = Cand("fake_region", {"n": 1}, {"type": "position", "xy": [2.0, 1.5]}, "answerable", (3,), (),
                 {"erode_m": 0.2}, subjects=("r:R1",), region=polygon)
    _, (g,) = questions.records([where])
    assert check_gold(g) == [] and (g["credit"], g["region"], g["metric"], g["tolerances"]) == (
        "region", polygon, "xy", {"erode_m": 0.2})
    ahead = Cand("path_ahead", {"cap": 3.0}, {"type": "number", "value": 1.0}, "answerable", (21,), (),
                 {"ladder": [0.25, 0.5, 1.0, 2.0]}, decision_k=21, subjects=("w:21",))
    _, (g,) = questions.records([ahead])
    assert (g["region"], g["metric"]) == (None, None)
    with pytest.raises(ValueError, match="fake_region"):
        questions.records([replace(where, region=None)])


def test_the_window_helpers_by_hand():
    """Contract: the helpers every family reads describe the window as the stream delivers it. Hand values on three
    tasks of `vacuum_run` (28 keyframes each, 1.5 s apart) and the window of the last two (keyframes 28 to 83, its clock
    starting 08:00:00): `tasks` is (0, 28, 56) and (1, 56, 84), `multi` is true (false for one task), `task_index` of
    keyframes 28, 55 and 56 is 0, 0 and 1; keyframe 30 is 3.0 s in, 08:00:03.0, and 31 is 4.5 s, 08:00:04.5 (delivered
    to the tenth of a second); a thing stands on the floor when its base height is within the vacuum's 0.02 m climb of
    the robot's (0 and 0.02 yes, 0.021 and 0.75 no, and no before it exists); `in_ks` counts a visibility row of any
    pixel count in the
    window's keyframes only (a glimpse at 60 yes, a row at keyframe 3 before the window no); `sighted` keeps 300 px and
    100 px sightings (57 and 58) and `recognised` only the 1,500 px one (60); and a job whose relevance differs by task
    gives `relevance_of` each task its own, `relevance` the last's. Regression caught: a helper that reads the script's
    numbering where the case's is meant, or a sighting from before the window."""
    s = vacuum_run(3)
    for eid, z in (("A", 0.0), ("B", 0.02), ("C", 0.021), ("D", 0.75)):
        s.item(eid, "bag", {"color": eid}, [(0, None, (1.0, 1.0, z))])
    s.item("E", "bag", {"color": "E"}, [(7230.0, None, (1.0, 1.0, 0.0))])
    s.see([3, 60], "A", 100)
    s.see(57, "F", 300)
    s.see(58, "F", 100)
    s.see([3, 60], "F", 1500)
    s.item("F", "bag", {"color": "F"}, [(0, None, (2.0, 1.0, 0.0))])
    w = fk.world(s, 1, 2, "c1", "medium")
    assert w.tasks == [(0, 28, 56), (1, 56, 84)] and w.multi and not fk.world(s).multi
    assert (w.task_index(28), w.task_index(55), w.task_index(56)) == (0, 0, 1)
    assert (w.clock(30), w.clock(31)) == ("2026-03-02T08:00:03.0", "2026-03-02T08:00:04.5")
    assert [w.on_floor(e, 30) for e in "ABCDE"] == [True, True, False, False, False] and w.on_floor("E", 50)
    assert w.in_ks("A", [59, 60]) and not w.in_ks("A", [61, 62]) and not w.in_ks("A", [3]) and not w.in_ks("A", [900])
    assert (w.sighted("F"), w.recognised("F")) == ([57, 58, 60], [60])
    kinds = {"kitchen": "mug", "living room": "toy"}
    two = SpaceScript(tasks=[[("kitchen", 6, "cleaning")], [("living room", 6, "cleaning")]])
    job = replace(JOB, relevance=lambda plan: Relevance(frozenset({kinds[plan.params["room"]]}), frozenset(),
                                                        frozenset()))
    w = fk.world(two, 0, 1, job=job, length="medium")
    assert (w.relevance_of(0).kinds, w.relevance_of(1).kinds) == (frozenset({"mug"}), frozenset({"toy"}))
    assert w.relevance == w.relevance_of(1)


def test_the_world_reads_the_public_map(tmp_path, monkeypatch):
    """Contract: `World.map` is the inputs' map when they hold one (a hand map's rooms here), else the layout's through
    `mapspec.load_map` of the timeline's map name, never the package's file directly (a held-out map is found by
    `map_files`), and `load_inputs` passes that map. Hand-worked: a world over `fk.inputs` has the hand map's kitchen
    and living room and loads nothing; one over `space_kit.inputs` (no map) loads 'apartment_a' once, as does
    `load_inputs` of a written case. Regression caught: a family reading a room's polygon from the wrong map, or from
    the package behind `map_files`' back."""
    script = vacuum_run()
    assert [r.name for r in fk.world(script).map.regions] == ["kitchen", "living room"]
    case = script.write_case(tmp_path)
    loaded, sentinel = [], object()
    monkeypatch.setattr(questions_world, "load_map", lambda name: loaded.append(name) or sentinel)
    use_job(monkeypatch)
    assert fk.world(script).map is not sentinel and loaded == []
    assert World(inputs(script), record(script)).map is sentinel and loaded == ["apartment_a"]
    assert load_inputs(case).map is sentinel and loaded == ["apartment_a", "apartment_a"]
