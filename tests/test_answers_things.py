"""The answers to `when_seen`, `order` and `distance_between` (`table_things`, `baselines_path_things`): the table
systems from the sightings of their tracks, `path_log` from its pool, its camera and the public map. Hand-worked values;
the oracle runs on a window of `things_kit`, whose golds `test_questions_things` works out by hand."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import things_kit as tk
from answers_kit import compiled_case
from baseline_kit import det, play, question, record, rows, scores, started, step
from space_kit import EVENTS
from stbench import mapspec, score, table_answers
from stbench.baselines import MemoryTable, Oracle
from stbench.baselines_path import PathLog
from stbench.blind import Blind, Prior
from stbench.table import Store

THREE = ("when_seen", "order", "distance_between")
# Case ids that ask each family answerable and `when_seen` unknown too, over `things_window(2)` and `movers()`.
WINDOW_CID, MOVERS_CID = "1da095819c2bb777", "9cf868f2b426aaf4"


def clock(minute: int, second: int) -> str:
    return f"2026-03-02T08:{minute:02d}:{second:02d}"


def thing(kind: str, **attributes: str) -> dict:
    return {"kind": kind, "attributes": attributes}


def movers():
    """One flat task of things that each move between their first and last sightings, so that no first sighting stands
    where the last did, nor a first for a last in time (keyframes 1.5 s apart; the end look window is 52 to 59): red bag
    (1, 1) seen at k 10, at (1, 4) from 30 s and seen at 30; blue bag (6, 1) seen at 12, at (2, 5) from 30 s, seen at
    35; green bag glimpsed at 50 after its one sighting at 40 (never asked); red shoes (2, 2) seen at 8, at (5, 5) from
    45 s, seen at 49; blue shoes (5, 3) seen at 38, at (5, 0.5) from 60 s, seen at 48; white shoes (5, 4) detected at 3
    (400 px), at (6.5, 4) from 60 s and seen at 47."""
    s = tk.flat()
    tk.put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10, 30], moves=(30.0, (1.0, 4.0)))
    tk.put(s, "E0002", "bag", "blue", (6.0, 1.0), recognised=[12, 35], moves=(30.0, (2.0, 5.0)))
    tk.put(s, "E0003", "bag", "green", (3.0, 5.0), recognised=[40], glimpsed=[50])
    tk.put(s, "E0004", "shoes", "red", (2.0, 2.0), recognised=[8, 49], moves=(45.0, (5.0, 5.0)))
    tk.put(s, "E0005", "shoes", "blue", (5.0, 3.0), recognised=[38, 48], moves=(60.0, (5.0, 0.5)))
    tk.put(s, "E0006", "shoes", "white", (5.0, 4.0), detected=[3], recognised=[47], moves=(60.0, (6.5, 4.0)))
    return s


def test_the_oracle_answers_the_three(tmp_path):
    """Contract: through `run_case` on `detections`, the oracle and `memory_table` score 1 on every question of two
    cases that ask the three families answerable and `when_seen` unknown too: a window of two tasks of `things_kit`
    (`things_window(2)`, case `WINDOW_CID`) and one flat task of `movers()` (`MOVERS_CID`), where each thing's first and
    last sightings differ in place and in time by more than a ladder rung or 5% of a distance, so only the sighting the
    gold reads scores 1: a thing's last recognisable detection for `when_seen` and `distance_between`, its first for
    `order`; a description no track matches is `unknown`. Regression caught: an answerer reading another sighting than
    the gold (the first for the last, the last for the first, a glimpse for a recognisable one), or one that answers
    where the memory has nothing."""
    window = compiled_case(tmp_path / "window", WINDOW_CID, tk.things_window(2), tk.THINGS_JOB, 0, 1)
    flat = compiled_case(tmp_path / "flat", MOVERS_CID, movers(), tk.THINGS_JOB, 0, 0)
    for case in (window, flat):
        asked = Counter((g["family"], g["slice"]) for g in rows(case)[1] if g["family"] in THREE)
        assert all(asked[family, "answerable"] for family in THREE) and asked["when_seen", "unknown"]
        for cls in (Oracle, MemoryTable):
            run, _ = play(case, cls, events=EVENTS)
            assert run.status == "complete" and run.failures == {}
            assert set(scores(case, run).values()) == {1.0}, (case.id, cls.__name__)


def sighted(store: Store, k: int, *things: tuple[str, str, float, float]) -> None:
    """Step `k` (clock 08:00:<k>) of the vacuum delivered with a detection of each (kind, colour, x, y)."""
    store.add_step(step(k, 0.0, 0.0, detections=[det(f"d{i}", *t) for i, t in enumerate(things)]))


def ask(store: Store, family: str, slots: dict) -> dict:
    return table_answers.answer(store, question(family, slots), {})


def test_the_table_by_hand():
    """Contract: from a hand store, `when_seen` is the clock of the latest sighting of the thing, over every track its
    description matches; `order` compares the earliest sightings, yes when the first named thing's is the earlier and
    `unknown` when both lie at one step; `distance_between` is the distance, 3 places, between the places of the latest
    sightings (a thing seen elsewhere later is where it was last seen); and each is `unknown` when a description matches
    no track. Hand-worked: a red bag at (0, 0) at k 3 and at (1, 1) at k 12 and 30, a blue bag at (6, 2) at k 40; a red
    pair of shoes at k 5 and 50, a blue pair at k 9; a pair of white shoes and a grey one, both first seen at k 20; a
    brown box at (4, 5) at k 8.
    - `when_seen`: the red bag, step 30 (08:00:30); any bag, step 40, over both tracks; the red box, never seen:
      `unknown`;
    - `order`: red shoes before blue, first sightings at k 5 and 9 (the red one's later sighting at 50 is not read):
      yes, and blue before red no; white before grey, both at k 20: `unknown`; red shoes before a green pair never
      seen: `unknown`;
    - `distance_between`: the red bag's last place (1, 1) and the box's (4, 5): 3-4-5, so 5.0 (not the 6.403 from the
      bag's first place), in either named order; the red bag with the blue bag (6, 2): sqrt(26) = 5.099; with a green
      bag never seen: `unknown`.
    Regression caught: an answerer that reads the first sighting where the gold reads the last (or the reverse for
    `order`), the two places mismatched, or an answer from a description the memory lacks."""
    store = Store()
    sighted(store, 3, ("bag", "red", 0.0, 0.0))
    sighted(store, 5, ("shoes", "red", 3.0, 3.0))
    sighted(store, 8, ("box", "brown", 4.0, 5.0))
    sighted(store, 9, ("shoes", "blue", 2.0, 3.0))
    sighted(store, 12, ("bag", "red", 1.0, 1.0))
    sighted(store, 20, ("shoes", "white", 7.0, 1.0), ("shoes", "grey", 7.0, 2.0))
    sighted(store, 30, ("bag", "red", 1.0, 1.0))
    sighted(store, 40, ("bag", "blue", 6.0, 2.0))
    sighted(store, 50, ("shoes", "red", 3.0, 3.0))
    assert ask(store, "when_seen", thing("bag", color="red")) == {"type": "time", "clock": clock(0, 30)}
    assert ask(store, "when_seen", thing("bag")) == {"type": "time", "clock": clock(0, 40)}
    assert ask(store, "when_seen", thing("box", color="red")) == {"type": "unknown"}

    def before(first, second):
        return ask(store, "order", {"first": first, "second": second})

    red, blue = thing("shoes", color="red"), thing("shoes", color="blue")
    assert (before(red, blue), before(blue, red)) == ({"type": "yesno", "value": True},
                                                      {"type": "yesno", "value": False})
    assert before(thing("shoes", color="white"), thing("shoes", color="grey")) == {"type": "unknown"}
    assert before(red, thing("shoes", color="green")) == {"type": "unknown"}

    def apart(first, second):
        return ask(store, "distance_between", {"first": first, "second": second})

    bag, box = thing("bag", color="red"), thing("box", color="brown")
    assert apart(bag, box) == apart(box, bag) == {"type": "number", "value": 5.0}
    assert apart(bag, thing("bag", color="blue")) == {"type": "number", "value": 5.099}
    assert apart(bag, thing("bag", color="green")) == {"type": "unknown"}


# -- path_log -------------------------------------------------------------------------------------------------------
CREDIT = {"when_seen": ("ladder", {"ladder": [15.0, 30.0, 60.0, 120.0]}), "order": ("exact", {}),
          "distance_between": ("relative", {})}
TASKS = {"when_seen": thing("box", color="brown"),
         "order": {"first": thing("shoes", color="red"), "second": thing("shoes", color="blue")},
         "distance_between": {"first": thing("shoes", color="red"), "second": thing("box", color="brown")}}
OTHERS = {"when_seen": thing("box", color="grey"),   # what the prior's scripts were asked: other colours
          "order": {"first": thing("shoes", color="grey"), "second": thing("shoes", color="white")},
          "distance_between": {"first": thing("shoes", color="grey"), "second": thing("box", color="white")}}


def answer_row(family: str, value: dict) -> dict:
    """A gold row of the family as the compiler writes one: its credit and tolerances."""
    credit, tolerances = CREDIT[family]
    return {**record(question(family, {}), value, script="g")["gold"], "family": family, "credit": credit,
            "tolerances": tolerances}


ASK = clock(0, 50)   # where the hand case of `path_log` is asked: the clock of its last step


def prior_of(family: str, kitchen: list[dict], bedroom: list[dict]) -> list[dict]:
    """Other scripts' records of the family, one for each of the golds `kitchen`, from scripts that cleaned the kitchen,
    and of `bedroom`'s. Each script ran on a day of its own (a case's clock starts on a day drawn per case): the n-th
    was asked at `ASK` n days later, and a time gold is moved with it."""
    out = []
    scripts = [(room, i, value) for room, golds in (("kitchen", kitchen), ("bedroom", bedroom))
               for i, value in enumerate(golds)]
    for n, (room, i, value) in enumerate(scripts, 1):
        later = timedelta(days=n)
        if value["type"] == "time":
            value = {**value, "clock": (datetime.fromisoformat(value["clock"]) + later).isoformat()}
        clock_ = (datetime.fromisoformat(ASK) + later).isoformat()
        out.append({**record(question(family, OTHERS[family]), value, script=f"{room}{i}", rooms=(room,), clock=clock_),
                    "gold": answer_row(family, value)})
    return out


def numbers(*values: float) -> list[dict]:
    return [{"type": "number", "value": v} for v in values]


def test_path_log_is_never_weaker_than_blind(tmp_path):
    """Contract (VISION rule 5): `path_log` knows everything `blind` does and more, so on a prior where the rooms the
    brief names and the kinds' receptacles carry signal its penalised score on each of the three families is at least
    `blind`'s, both keyed by `answer_keys`' nested levels (the kind, or the kind pair, for the prior's things are of
    other colours). Hand case: the robot is told to clean the kitchen; its camera faces the dining table (the box's
    receptacle, (2.1, 1.9) on the apartment) from (0.5, 1.9) at steps 0 to 3 (clocks 08:00:00, :10, :20, :30) and away
    at steps 4 and 5, asked at 08:00:50. Ten other scripts hold five golds from kitchen scripts and five from bedroom
    ones, the bedroom's far from the truth, so `blind`'s pool of all ten scores no answer above abstaining; each ran on
    a day of its own and was asked at 08:00:50 of it, so a time gold is read as the seconds it lies behind that.
    - `when_seen` of the brown box: the kitchen golds lie 50, 30, 10, -10 and -30 s behind the ask, whose best own
      answer (the median, 10 s behind: 08:00:40) has mean penalised score .4, and 08:00:30 (the last step that faced the
      table, not the first; 20 s behind) .6, so `path_log` answers it, and the truth 08:00:30 scores 1 against `blind`'s
      `unknown` 0;
    - `order` of red shoes and blue: kitchen golds yes, bedroom golds no; `path_log` answers yes (the truth), `blind`
      (yes and no tie at 0 with `unknown` first) `unknown`; no receptacle candidate here, the two shoes share theirs;
    - `distance_between` red shoes (rack (4.8, 0.5)) and the brown box (table (2.1, 1.9)): sqrt(7.29 + 1.96) = 3.041.
      Against kitchen golds 2.95, 3.0, 3.1, 3.15, 3.19, each within 5% of it (the credit's top rung: +1 on all five),
      while the median 3.1 is 5.1% off 2.95 (credit .9, so +.8 there and a mean of .96), `path_log` answers 3.041, the
      truth, 1 against `blind`'s 0. Leaving a script out the rule still lands within 6% of the gold left out (+.8 at
      the least), so it holds on the next script and is kept. Against kitchen golds 2.9, 2.9, 4.2, 4.6, 5.0, whose best
      own answer (the median 4.2) has mean penalised score .2 and 3.041 .28, it answers nothing: left out in turn, the
      two 2.9s are answered 4.4 by the other four with 3.041 (-1 each), 4.2 is answered 3.041 (0), 4.6 (-.2) and 5.0
      (-.4), a mean of -.52 on scripts it did not learn from, under abstaining's 0, so `path_log` is `unknown`, as
      `blind` is. That is the cost of the held-out rule: its in-pool twin, which keeps the pool's best
      response, still answers 3.041 there.
    Regression caught: an end-question `path_log` that is only a coarser `blind`, or one that drops its own candidates,
    so a no-image shortcut goes ungated or answers worse than the prior it holds."""
    unknown = {"type": "unknown"}
    time, yes = {"type": "time", "clock": clock(0, 30)}, {"type": "yesno", "value": True}
    pool = {"type": "number", "value": 3.041}
    cases = [("when_seen", [{"type": "time", "clock": clock(*t)} for t in ((0, 0), (0, 20), (0, 40), (1, 0), (1, 20))],
              {"type": "time", "clock": clock(10, 0)}, time, time, time),
             ("order", [yes] * 5, {"type": "yesno", "value": False}, yes, yes, yes),
             ("distance_between", numbers(2.95, 3.0, 3.1, 3.15, 3.19), {"type": "number", "value": 9.0}, pool, pool,
              pool),
             ("distance_between", numbers(2.9, 2.9, 4.2, 4.6, 5.0), {"type": "number", "value": 9.0}, pool, unknown,
              pool)]
    apartment = mapspec.load_map("apartment_a")
    steps = [step(k, 0.5, 1.9, 0.0 if k < 4 else 180.0, clock=clock(0, 10 * k)) for k in range(6)]
    for n, (family, kitchen, bedroom, truth, held_out, in_pool) in enumerate(cases):
        prior = prior_of(family, kitchen, [bedroom] * 5)
        answers = {}
        for name, cls, opts, records in (("blind", Blind, {}, prior),
                                         ("path_log", PathLog, {"map": apartment, "events": EVENTS}, prior),
                                         ("in_pool", PathLog, {"map": apartment, "events": EVENTS},
                                          Prior(prior, in_pool=True))):
            system = started(cls, tmp_path / f"{n}-{name}", prior=records, **opts)
            system.task(0, "2026-03-02T08:00:00", 0.0, "Its current assignment is to clean the kitchen.")
            system.observe(steps)
            answers[name] = system.ask("end", {"k": 5, "clock": ASK}, [question(family, TASKS[family])])["q1"]
        gold = answer_row(family, truth)
        assert answers == {"blind": unknown, "path_log": held_out, "in_pool": in_pool}, (n, family)
        assert score.penalised(answers["path_log"], gold) >= score.penalised(answers["blind"], gold)
