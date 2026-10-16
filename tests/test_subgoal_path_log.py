"""`path_log` and the subgoal (`answer_keys`, `blind.build_record`, `suite_priors`): a during question is keyed on the
subgoal of its ask step, the records the harness writes and the lookups a system builds make the same keys, and a
subgoal that told a coin's outcome would let `path_log` beat the floor (so its shortcut gate would catch the job)."""

from __future__ import annotations

import pytest

from baseline_kit import EVENTS, at, play, question, record, rows, scores, started, step
from check_kit import hand_synth, synthetic_dataset
from run_kit import golds
from stbench import answer_keys, score, suite_priors
from stbench import case as case_mod
from stbench.baselines_path import PathLog
from stbench.blind import build_record, pool_at
from stbench.mapspec import load_map
from stbench.stream import run_case
from stbench.stream_protocol import GATE_CHANNEL
from stream_kit import case_on, during_q, end_q, gold_row, qid
from subgoal_kit import with_subgoals

DURING = {"template": "zone_clear", "when": "during", "slots": {"zone": "the bay ahead of you", "height": 0.3},
          "phase": "waiting", "base": (2.0, 1.0), "yaw": 0.0, "rooms": ()}
CLEAR, BLOCKED = "docking at the bay", "waiting at the bay: it is blocked"


def test_a_during_question_is_keyed_on_the_subgoal_of_its_ask_step():
    """Contract (FD-0.5.0-control-impl §2): `path_log`'s during levels are the template, phase, pose key and subgoal;
    the template, phase and pose key; the template and phase; the template. Two records that differ in their subgoal
    alone differ in the first level and share the rest; a record with no subgoal (None, or the key absent) has None in
    that level, so every record has the same four levels, which line up by depth in `pool_at`. An end question's levels
    are `blind`'s with the rooms and read no subgoal. Hand values: base (2.0, 1.0) heading 0 is cell `4,2,0`.
    Regression caught: a subgoal no level keys on (a leaky subgoal the gate never sees), or levels of two lengths whose
    depths no longer match."""
    first = answer_keys.levels("path_log", {**DURING, "subgoal": CLEAR})
    assert first == [("zone_clear", "waiting", "4,2,0", CLEAR), ("zone_clear", "waiting", "4,2,0"),
                     ("zone_clear", "waiting"), ("zone_clear",)]
    other = answer_keys.levels("path_log", {**DURING, "subgoal": BLOCKED})
    assert other[0] != first[0] and other[1:] == first[1:]
    none = answer_keys.levels("path_log", DURING)
    assert none == answer_keys.levels("path_log", {**DURING, "subgoal": None})
    assert none[0] == ("zone_clear", "waiting", "4,2,0", None) and none[1:] == first[1:]
    end = {"template": "object_where", "when": "end", "slots": {"kind": "mug"}, "rooms": ("kitchen",)}
    assert answer_keys.levels("path_log", {**end, "subgoal": CLEAR}) == answer_keys.levels("path_log", end)
    assert answer_keys.levels("blind", {**DURING, "subgoal": CLEAR}) == answer_keys.levels("blind", DURING)
    assert "subgoal" in answer_keys.RECORD_KEYS


def test_a_lookup_without_the_subgoal_still_finds_the_pool_of_its_pose():
    """Contract: a system sent steps without a subgoal (the neutral brief, or a job that states none) builds a record
    whose subgoal is None, and `pool_at` finds the records of other scripts at the pose level, since the levels line up
    by depth. Hand case: five records of five scripts at one pose, each with a subgoal, are the pool of a lookup with
    none, and of one with a subgoal none of them has, and of one with their own; a lookup in another phase finds only
    the five of the bare template, and with no records of the template finds nothing. Regression caught: a level present
    only when the subgoal is, which would shift a lookup's pose key onto the subgoal's depth and leave the pool of
    another level (or none)."""
    q = question("zone_clear", DURING["slots"], answer_type="yesno")
    prior = [{**record(q, at(0, 0), script=f"s{i}", when="during", phase="waiting", base=(2.0, 1.0)),
              "subgoal": CLEAR} for i in range(5)]
    base = {"template": "zone_clear", "slots": DURING["slots"], "when": "during", "phase": "waiting",
            "base": (2.0, 1.0), "yaw": 0.0, "rooms": ()}
    for subgoal in (None, "driving into another bay"):
        assert len(pool_at("path_log", {**base, "subgoal": subgoal}, prior)) == 5, subgoal
    assert len(pool_at("path_log", {**base, "subgoal": CLEAR}, prior)) == 5
    assert len(pool_at("path_log", {**base, "template": "slots_empty", "subgoal": None}, prior)) == 0


def test_build_record_carries_the_subgoal_of_the_step():
    """Contract: the record a system builds at an ask holds the subgoal of the step the ask is about, None when the step
    carries none; `Blind`, which sees no step, builds a record without it. Regression caught: a lookup that cannot see
    the subgoal, so `path_log` could never find the records the harness keyed on it."""
    q = question("zone_clear", DURING["slots"], answer_type="yesno")
    sent = step(3, 2.0, 1.0, phase="waiting")
    assert build_record(q, "during", {**sent, "subgoal": CLEAR})["subgoal"] == CLEAR
    assert build_record(q, "during", sent)["subgoal"] is None
    assert "subgoal" not in build_record(q, "during", clock="2026-03-02T08:00:03")


@pytest.fixture
def cases(tmp_path, monkeypatch):
    """Two compiled cases of the synthetic script, written for a vacuum job that states a subgoal."""
    with_subgoals(monkeypatch)
    return synthetic_dataset(tmp_path, 2)


def looked_up(case, brief: str) -> dict[str, dict]:
    """The record `path_log` builds at each ask of the case streamed under the brief `brief` (`ask_record`)."""
    seen: dict[str, dict] = {}
    opts = suite_priors.options_for("path_log", case, [case], {}).opts

    class Spy(PathLog):
        def ask(self, when, now, questions):
            for q in questions:
                seen[q["qid"]] = self.ask_record(q, now, when=when)
            return super().ask(when, now, questions)

    run_case(lambda state, work: Spy(state, work, **opts), case, channel=GATE_CHANNEL, brief=brief)
    return seen


def test_records_and_lookups_build_the_same_levels(cases):
    """Contract (FD-0.5.0-control-impl §2): at every ask of a case, the levels `path_log` keys the record the harness
    writes (`suite_priors.record_for`, from the script's pose rows) equal the levels of the record the system builds
    from the steps it was sent; for the during question the subgoal in the first level is the text of the decision's
    keyframe ("cleaning the kitchen, stage 1" at keyframe 21 of the kit). Under the neutral brief, which withholds the
    subgoal, the lookup's first level holds None and its other levels are the record's (a during question only: the
    neutral brief names no room either). Regression caught: a record keyed on a subgoal the stream never sends (a prior
    that never matches), or a lookup that loses more than the subgoal level."""
    for case in cases:
        questions, golds = rows(case)
        seen = {brief: looked_up(case, brief) for brief in ("correct", "neutral")}
        for q, g in zip(questions, golds, strict=True):
            made = answer_keys.levels("path_log", suite_priors.record_for(case, q, g, None))
            assert answer_keys.levels("path_log", seen["correct"][q["qid"]]) == made, q["qid"]
            neutral = answer_keys.levels("path_log", seen["neutral"][q["qid"]])
            if q["when"] == "during":
                assert made[0][-1] == f"cleaning the kitchen, stage {q['decision_k'] % 4}"
                assert neutral[0][-1] is None and neutral[1:] == made[1:]


def yesno(value: bool) -> dict:
    """A gold row for a yes/no question, scored by exact match."""
    return {"qid": "g", "family": "zone_clear", "gold": {"type": "yesno", "value": value}, "slice": "answerable",
            "credit": "exact", "region": None, "tolerances": {}, "metric": None, "scope": "now", "reference": None,
            "critical": False, "subjects": [], "evidence_k": [], "entities": []}


def pooled(coin: bool, text: str, i: int) -> dict:
    """The record of another script's zone question: clear (`coin`) or blocked, the subgoal of the step `text`."""
    q = question("zone_clear", DURING["slots"], answer_type="yesno")
    return {**record(q, at(0, 0), script=f"s{i}", when="during", phase="waiting", base=(2.0, 1.0)),
            "subgoal": text, "gold": yesno(coin)}


def asked(tmp_path, prior: list[dict], text: str | None) -> dict:
    """`path_log` over the prior, shown a step with the subgoal `text` and asked whether the bay is clear."""
    system = started(PathLog, tmp_path, map=load_map("apartment_a"), events=EVENTS, prior=prior)
    system.task(0, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the kitchen.")
    system.observe([{**step(0, 2.0, 1.0, phase="waiting"), **({"subgoal": text} if text else {})}])
    q = question("zone_clear", DURING["slots"], answer_type="yesno")
    return system.ask("during", {"k": 0, "clock": ""}, [{**q, "qid": qid(1)}])[qid(1)]


def test_a_subgoal_that_tells_the_coin_makes_path_log_beat_the_floor(tmp_path):
    """Contract (FD-0.5.0-control-impl §2, the gate's reason): `path_log` answers a during question from the records of
    other scripts that share its template, phase, pose and subgoal. Hand case: ten other scripts stood at the same pose
    in the same phase, five with the bay clear and five blocked.
    - With a subgoal that tells the coin ("docking at the bay" when it is clear, "waiting at the bay: it is blocked"
      when not), the lookup finds the five records of its own text and answers by them: yes for the one, no for the
      other, both right, a mean penalised score of 1 against the 0 of abstaining (`all_unknown`), which a shortcut
      gate fails.
    - With a subgoal that does not (the same text in all ten), the pool is the ten, split five and five, and it
      abstains: 0, the floor.
    - A lookup whose subgoal no record has falls back to the pose and abstains.
    Regression caught: a `path_log` that ignores the subgoal (the leak would pass the gate), or one that answers from a
    pool the coin splits evenly."""
    leaky = [pooled(i % 2 == 0, CLEAR if i % 2 == 0 else BLOCKED, i) for i in range(10)]
    clean = [pooled(i % 2 == 0, CLEAR, i) for i in range(10)]
    answers = {text: asked(tmp_path / f"leaky-{n}", leaky, text) for n, text in enumerate((CLEAR, BLOCKED))}
    assert answers == {CLEAR: {"type": "yesno", "value": True}, BLOCKED: {"type": "yesno", "value": False}}
    mean = sum(score.penalised(answers[text], yesno(coin)) for text, coin in ((CLEAR, True), (BLOCKED, False))) / 2
    assert mean == 1.0
    unknown = {"type": "unknown"}
    assert asked(tmp_path / "clean", clean, CLEAR) == unknown
    assert asked(tmp_path / "other", leaky, "driving into another bay") == unknown
    assert asked(tmp_path / "none", leaky, None) == unknown
    assert score.penalised(unknown, yesno(True)) == 0.0   # the floor: `all_unknown` scores 0 on an answerable question


def coin_cases(root, monkeypatch, tells: bool) -> list[case_mod.CasePaths]:
    """Twelve one-task cases of twelve scripts, each asking whether the bay is clear after step 12, the same phase and
    pose in all; the bay is clear in the six scripts of even seed and blocked in the six of odd seed (the gold, by
    hand). The job's subgoal is the same text everywhere, or, with `tells`, "docking at the bay" in the clear scripts
    and "waiting at the bay: it is blocked" in the blocked ones: a job that leaks its coin."""
    def subgoals(tl, m):
        clear = tl.episode.seed % 2 == 0
        return [CLEAR if clear or not tells else BLOCKED] * tl.n_keyframes

    with_subgoals(monkeypatch, subgoals)
    cases = []
    for seed in range(1, 13):
        q = during_q(1, 12, template="zone_clear", slots=DURING["slots"], answer_type="yesno",
                     text="Is the bay clear?")
        case = case_on(root, hand_synth(seed), [q, end_q(2)])
        gold = gold_row(q, gold={"type": "yesno", "value": seed % 2 == 0}, credit="exact", tolerances={})
        golds(case, rows=[gold, gold_row(end_q(2))])
        cases.append(case)
    return cases


def mean_during(cases: list[case_mod.CasePaths]) -> float:
    """The mean penalised score of `path_log` on the during question of each case, its prior the other scripts' records
    (`suite_priors.options_for`): the score of `all_unknown` there is 0."""
    total = 0.0
    for case in cases:
        run, _ = play(case, PathLog, **suite_priors.options_for("path_log", case, cases, {}).opts)
        total += scores(case, run)[qid(1)]
    return total / len(cases)


def test_a_job_whose_subgoal_tells_its_coin_fails_the_path_log_floor(tmp_path, monkeypatch):
    """Contract (FD-0.5.0-control-impl §2): a job whose subgoal tells the outcome of its coin lets `path_log` beat
    `all_unknown`, so the shortcut gate that runs it fails the job; a job whose subgoal does not leaves it at the floor.
    Hand case, through the files: twelve scripts of a synthetic job, the bay clear in the six of even seed. With a
    subgoal that says "docking at the bay" when it is clear and "waiting at the bay: it is blocked" when not, each
    case's prior holds five scripts of its own text with its own gold, and `path_log` answers every one right: a mean
    penalised score of 1.0, where `all_unknown` has 0. With one text for all, the pool is the eleven others, five and
    six to a side, and it abstains: 0.0. Regression caught: a gate that cannot see a subgoal that leaks (the key not
    built from the pose rows, the stream's steps or the records)."""
    assert mean_during(coin_cases(tmp_path / "leaky", monkeypatch, tells=True)) == 1.0
    assert mean_during(coin_cases(tmp_path / "clean", monkeypatch, tells=False)) == 0.0
