"""The scripted systems' output in a second process (`determinism_kit.py`: nothing is recorded, the two are compared).

`baselines_output` below holds the constants of the scripted systems (the mapper's and the free-space rule's, the
corridor's sub-cell, the priors' support floor, `answer_keys`' quantisers and level lists, the raycast's, the
candidates' and the places'), the two registries' module lists and the families they answer
(`table_answers.ANSWER_MODULES`, `baselines_path.PATH_MODULES`), the nested-description levels of `answer_keys`, the
map every table system is given, and what every system answers to the questions of one fixed synthetic case: the
oracle, the table, the memoryless pair, `path_log`, `blind`, `other_case`, the questionnaire system and the floor, with
a prior, a map and an event table written by hand. The prior holds six records of six scripts for each question, so the
pools clear the support floor and the systems answer from them, not only from the fallbacks. The case is built on the
frozen vacuum of `test_determinism_records.py` and a fixed case id, so a robot tune moves nothing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from baseline_kit import CID, EVENTS, UNKNOWN, at, number, play, record
from determinism_kit import digest
from space_kit import things_run
from stbench import (
    answer_candidates,
    answer_keys,
    baselines,
    baselines_path,
    baselines_path_space,
    blind,
    grid,
    mapspec,
    places,
    robots,
    rules,
    table_answers,
)
from stbench import case as case_mod
from stbench import episode as ep
from stbench.baselines import BASELINES
from stream_kit import during_q, end_q
from test_determinism_records import FROZEN

START = "when you started cleaning"
# The golds the six other scripts hold for each question of `questions()`, in its order: all six agree (a pool the
# systems answer from, its held-out gain over abstaining sure: 2.2, then the bag at (2, 1); with one script out of six
# in disagreement the gain is uncertain and they abstain), and two splits that are no better than abstaining
# (`unknown`).
PRIOR_GOLDS = [[number(2.2)] * 6, [at(2.0, 1.0)] * 6,
               [at(2.0, 1.0)] * 3 + [at(3.5, 1.0)] * 2 + [UNKNOWN], [at(2.0, 1.0)] * 2 + [at(3.5, 1.0)] + [UNKNOWN] * 3]


def questions() -> list[dict]:
    """What the case is asked: `path_ahead` after keyframe 21, and where the red bag, the brown box and a blue toy that
    no frame shows were at three of the job's moments."""
    def where(n, kind, color, event):
        return end_q(n, slots={"kind": kind, "attributes": {"color": color}, "event": event},
                     needs=("see", "pose", "status"))

    return [during_q(1, 21, slots={"cap": 3.0}), where(2, "bag", "red", "when you got back to the dock"),
            where(3, "box", "brown", "when you finished cleaning"), where(4, "toy", "blue", START)]


def prior(qs: list[dict]) -> list[dict]:
    """Six records of six other scripts for each question, as the harness would write them, with golds that differ
    (`PRIOR_GOLDS`): enough support (`blind.MIN_POOL_GOLDS` golds of `blind.MIN_POOL_SCRIPTS` scripts) at the
    question's own key, so the systems meet a pool, answer from it where it beats abstaining and say `unknown` where it
    does not."""
    out = []
    for q, values in zip(qs, PRIOR_GOLDS, strict=True):
        siblings = [{"template": o["template"], "slots": o["slots"]} for o in qs if o is not q and o["when"] == "end"]
        out += [record(q, v, script=f"s{i}", when=q["when"], phase="transit", base=(-0.17, 0.0), rooms=("kitchen",),
                       nm=v if v["type"] != "unknown" else None, siblings=siblings if q["when"] == "end" else ())
                for i, v in enumerate(values)]
    return out


def answers_of_every_system(dataset: Path) -> dict:
    """What every system answers to the fixed case's questions, by system."""
    with mock.patch.dict(robots.ROBOTS, {"vacuum": FROZEN}), mock.patch.object(case_mod, "claim_id",
                                                                                lambda dataset, entry: CID):
        case = things_run().write_case(dataset)
    qs = questions()
    ep.write_jsonl(case.questions, qs)
    given = {"prior": prior(qs), "map": mapspec.load_map("apartment_a"), "events": EVENTS}
    answers = {}
    for name, b in BASELINES.items():
        opts = {key: given[key] for key, wanted in (("prior", b.needs_prior), ("map", b.needs_map),
                                                    ("events", b.needs_events)) if wanted}
        run, _ = play(case, b.cls, **opts, **({"stream_from": case, "ask_map": {21: 21}} if b.other_stream else {}))
        answers[name] = {"status": run.status, "answers": [run.answers[q["qid"]] for q in qs]}
    return answers


def constants() -> dict:
    """The constants and tables the scripted systems read."""
    sample = {"template": "path_ahead", "slots": {"cap": 3.0}, "when": "during", "phase": "transit", "base": (0.2, 0.7),
              "yaw": 100.0, "rooms": ("kitchen",), "nm": number(1.2)}
    shoe = {"kind": "shoe", "attributes": {"color": "red"}}
    nested = {"template": "order", "slots": {"first": shoe, "second": {**shoe, "attributes": {"color": "blue"}},
                                             "task": 2, "tasks": [0, 2]}, "when": "end", "rooms": ("hall",)}
    return {
        "grid": {"min_floor_px": grid.MIN_FLOOR_PX, "free_share": grid.FREE_SHARE, "fine": grid.FINE_M,
                 "sub": grid.SUB_M, "cell": grid.CELL_M, "eps": grid.EPS},
        "support": {"scripts": blind.MIN_POOL_SCRIPTS, "golds": blind.MIN_POOL_GOLDS},
        "keys": {"record": answer_keys.RECORD_KEYS, "drop": answer_keys.SLOT_DROP_ORDER,
                 "clocks": answer_keys.CLOCK_SLOTS, "edges": answer_keys.NUMBER_EDGES,
                 "buckets": answer_keys.COUNT_BUCKETS, "headings": answer_keys.HEADING_BINS, "eps": answer_keys.EPS,
                 "none": answer_keys.NO_ANSWER,
                 "levels": {name: answer_keys.levels(name, sample) for name in ("blind", "path_log",
                                                                                "current_view_prior")},
                 "nested": {name: answer_keys.levels(name, nested) for name in ("blind", "path_log")}},
        "raycast": {"door_leaf": baselines_path_space.DOOR_LEAF_M,
                    "interior_open": baselines_path_space.INTERIOR_DOORS_OPEN},
        "candidates": {"max": answer_candidates.MAX_CANDIDATES, "shares": [str(s) for s in
                                                                            answer_candidates.CELL_SHARES],
                       "grid": answer_candidates.GRID},
        "places": {"max": places.MAX_PLACES, "heights": places.HEIGHTS, "match_m": rules.MATCH_M},
        "registries": {"table_modules": list(table_answers.ANSWER_MODULES),
                       "path_modules": list(baselines_path.PATH_MODULES),
                       "table": sorted(table_answers.ANSWERERS), "path": sorted(baselines_path.PATH_ANSWERERS)},
        "nm": baselines.NM_SYSTEM,
        "flags": {name: [b.needs_prior, b.prior_nm, b.needs_map, b.needs_events, b.other_stream]
                  for name, b in BASELINES.items()}}


def baselines_output(dataset: Path) -> str:
    """The digest of the scripted systems' output: its constants and every system's answers on the fixed case."""
    return digest({"constants": constants(), "answers": answers_of_every_system(dataset)})


def test_the_baselines_output_is_deterministic(tmp_path):
    """Contract: the scripted systems' output (the mapper's and the free-space rule's constants, `answer_keys`'
    quantisers and levels, the raycast's, the candidates' and places' constants, the priors' support floor, and every
    system's answers on one synthetic case) is the same in a second process (another `PYTHONHASHSEED`). Regression
    caught: an answer that depends on dict or set order or on the process."""
    here = baselines_output(tmp_path / "here")
    out = subprocess.run([sys.executable, __file__, str(tmp_path / "there")], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout
    assert json.loads(out) == here


def test_the_fixture_prior_clears_the_support_floor():
    """Contract: every fixed question has a pool at its own `blind` key, so the compared answers include the systems
    answering from other scripts' golds and not only the fallbacks. Regression caught: a fixture prior thinner than
    `blind.MIN_POOL_GOLDS` golds of `blind.MIN_POOL_SCRIPTS` scripts, which every prior system would meet with
    `unknown` (or a coarser level), leaving the pool paths out of the comparison."""
    qs = questions()
    records = prior(qs)
    for q in qs:
        assert blind.pool_at("blind", blind.build_record(q, q["when"]), records), q["qid"]


if __name__ == "__main__":  # the second process of the test above
    print(json.dumps(baselines_output(Path(sys.argv[1]))))
