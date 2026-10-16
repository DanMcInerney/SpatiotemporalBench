"""The scripted systems and their registry (V0 §10). They read only what a system may read:
the steps the stream delivers, the briefs, the robot and the options the harness gives them.

- `oracle`: the memory table with perfect identity (the stream's privileged `identity`). Ceiling.
- `latest_table` and `first_table`: the oracle's table keeping only each track's latest (first) sighting and reading no
  event table: a time-blind notebook of the last (first) place each thing was seen, whatever moment a question names.
  The references of the `time_needed` gate (the larger of their two scores is read) and shortcuts of every temporal
  family's `separation`.
- `memory_table`: detections matched by description and position, the mapper's free-space cells, answers by template
  (`table_answers`; the families a module of `table_answers.ANSWER_MODULES` answers included). The honest scripted
  memory. Every table system is given the public map of the case's layout, for the families that name a room (agents get
  no floor plan: D1).
- `current_view`: `memory_table` on the latest frame alone, whatever time a question names: the best memoryless
  responder, the reference of every `now` question. `current_view_prior` adds the best response over other scripts'
  records keyed on its answer, the phase and the pose.
- `kind_range` (`baselines_kind_range`): `ego_now`'s declared reference. `current_view`'s true bearing to the thing, put
  at the typical floor range of its kind over other scripts' `ego_now` golds; `unknown` on every other template. A
  reference and not a shortcut: no gate lists it among the shortcuts, and it has no in-pool twin.
- `path_log` (`baselines_path`): a system that learned the public layouts and keeps its own log, no pictures.
- `map_reader` (`baselines_map`): a structure family's declared reference (0.8.0, `clear_path`). `path_log`'s system
  with the public map's raycast and the pose-keyed pool of other scripts' golds, answering that one template and
  `unknown` on every other. A reference and not a shortcut: the gates leave it out of the shortcut gate and of
  `separation`, and the report prints its score beside the family's. It has no in-pool twin.
- `blind`, `questionnaire_blind` (`blind`): other scripts' records alone. `other_case`: the table fed another case's
  stream, asked a clock- or task-named question at the same moment of that stream (`MemoryTable.translated`) and its
  time answer put back on the asking case's clock (`MemoryTable.untranslated`). `all_unknown`: always `unknown`. Floor.

`BASELINES` names them as `gates.BASELINE_NAMES` does; each `Baseline` says which options the harness gives its system
(the flags are all `suite_priors.options_for` reads). `in_pool_twins` names a second run of each learned shortcut
(`gates.LEARNED`, the one owner of the term) under the in-pool rule (`blind.learned_answer`), which `check` scores
beside the first and the gates read the larger of.
`Baseline.reads_scorer` says whether a system's answers pass through the scorer (`blind.learned_answer` and the held-out
rounds of the own-log rules choose by it): `stbench check --reuse` reloads the stored answers of the systems that do not
and runs the ones that do (`check_answers`).
This module imports no truth module (tests/test_architecture.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import gates, table_answers
from .answer_candidates import UNKNOWN
from .baselines_kind_range import TEMPLATE as KIND_RANGE_TEMPLATE
from .baselines_kind_range import Ranges, placed
from .baselines_map import MapReader
from .baselines_path import PathLog
from .blind import Blind, QuestionnaireBlind, build_record, learned_answer, pool_at
from .camera import Camera
from .robots import from_doc
from .stream_protocol import System
from .table import Mapper, Store

NM_SYSTEM = "current_view"          # the system whose answers `prior_nm` systems' records carry


class MemoryTable(System):
    """Tracks detections into a store (`table.Store`) and answers each question from it with the template's answerer.
    The job's event table is the option `events` (required: an end question names a moment of the job); `map` (the case
    layout's public `MapSpec`) reaches the answerers as `ctx["map"]`, and `clock_map` and `task_map` (`other_case`'s)
    move a question's clock and task onto the stream it was fed (`translated`) and `clock_back` a time answer back onto
    the asking case's clock (`untranslated`). `zones` and `slot_zones` (the job's zone
    rules) go to the mapper with the map, so it observes each zone's own band; a system built without them answers the
    zone families `unknown`. `task_things` (the job's `task_predicate.TaskThings`, None for a job that asks none)
    reaches the answerers as `ctx["task_things"]`."""

    channels = ("detections",)   # it reads detections, and the depth the mapper turns into free space
    structured = True
    reads_events = True
    store_class: type[Store] = Store   # what it keeps the stream in

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        if self.reads_events and "events" not in opts:
            raise ValueError(f"{type(self).__name__} needs the job's event table as the option 'events'")
        self.store = self.store_class()
        self.latest: list[int] = []   # the `k`s of the last observe: the latest frame, which a `now` question is about

    def start(self, session: str, robot: dict, channel: str) -> None:
        super().start(session, robot, channel)
        self.model = from_doc(robot)
        self.mapper = Mapper(self.model, Camera.load(self.public / "camera.json"), self.workdir, self.opts.get("map"),
                             self.opts.get("zones", {}).values(), self.opts.get("slot_zones", ()))
        self.store = self.store_class(self.mapper)

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        self.store.begin_task(index, clock, gap_s, brief)

    def observe(self, steps: list[dict]) -> None:
        for step in steps:
            self.store.add_step(self.prepare(step))
        self.latest = [s["k"] for s in steps]
        self.store.add_window(self.latest)

    def prepare(self, step: dict) -> dict:
        """The step as the store sees it; the plain table never uses identity, even if one were present."""
        return {k: v for k, v in step.items() if k != "identity"}

    def answer_at(self, question: dict, now: dict) -> dict:
        """The table's answer to a question asked at the step `now` (`{k, clock}`), the question first translated by
        `translated` and the answer put back by `untranslated` (the identity unless the options carry maps)."""
        ctx = {"now": self.store.steps[now["k"]], "latest": self.latest, "robot": self.model,
               "events": self.opts["events"] if self.reads_events else None, "map": self.opts.get("map"),
               "task_things": self.opts.get("task_things")}
        return self.untranslated(table_answers.answer(self.store, self.translated(question), ctx))

    def translated(self, question: dict) -> dict:
        """A copy of the question as the stream this table was fed would be asked it (`other_case`): `slots["clock"]`
        through `opts["clock_map"]`, `slots["task"]` and each of `slots["tasks"]` through `opts["task_map"]`, a value
        the map lacks kept. Both maps are empty for a system fed its own case's stream, so it answers as asked."""
        clocks, tasks = self.opts.get("clock_map", {}), self.opts.get("task_map", {})
        slots = dict(question["slots"])
        if "clock" in slots:
            slots["clock"] = clocks.get(slots["clock"], slots["clock"])
        if "task" in slots:
            slots["task"] = tasks.get(slots["task"], slots["task"])
        if "tasks" in slots:
            slots["tasks"] = [tasks.get(i, i) for i in slots["tasks"]]
        return {**question, "slots": slots}

    def untranslated(self, answer: dict) -> dict:
        """`translated`'s inverse for an answer: a time answer's clock, one the fed stream delivered, through
        `opts["clock_back"]` onto the clock of the case that is asked, so it is judged against that case's gold and not
        against another day's; a clock the map lacks, and any other answer, kept."""
        if answer.get("type") != "time" or "clock" not in answer:
            return answer
        return {**answer, "clock": self.opts.get("clock_back", {}).get(answer["clock"], answer["clock"])}

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        return {q["qid"]: self.answer_at(q, now) for q in questions}


class Oracle(MemoryTable):
    """The table with perfect identity: each detection is joined to its entity through the stream's privileged input."""

    privileged = True

    def prepare(self, step: dict) -> dict:
        if "identity" not in step:
            raise ValueError("oracle needs the stream's privileged `identity` input (run it with privileged=True)")
        return step


class LatestStore(Store):
    """A store whose tracks keep one sighting each, the latest (and its floor hull): all it holds of a thing is where it
    was last seen."""

    def add_step(self, step: dict) -> None:
        super().add_step(step)
        for track in self.tracks:
            del track.sightings[:-1]
            track.hulls = {k: h for k, h in track.hulls.items() if k == track.sightings[0][0]}


class LatestTable(Oracle):
    """A time-blind notebook (the `time_needed` gate's reference): the oracle's table with perfect identity, but one
    sighting a track, the latest, and no event table. It answers a question that names a moment (an event, a task) from
    that one place whatever the moment is: a thing that moved in a gap is given at its new place, and two moments of one
    thing are given the same place. A family that a notebook of last places answers is not one of change over time."""

    reads_events = False
    store_class = LatestStore


class FirstStore(Store):
    """`LatestStore`'s mirror: tracks that keep one sighting each, the first (and its floor hull): all it holds of a
    thing is where it was first seen."""

    def add_step(self, step: dict) -> None:
        super().add_step(step)
        for track in self.tracks:
            del track.sightings[1:]
            track.hulls = {k: h for k, h in track.hulls.items() if k == track.sightings[0][0]}


class FirstTable(Oracle):
    """`LatestTable`'s mirror, the other time-blind notebook of the `time_needed` gate: the oracle's table with perfect
    identity, but one sighting a track, the first, and no event table. It answers a question that names a moment from
    the place each thing was first seen, whatever the moment is: a thing that moved in a gap is given at its old place.
    A family that either notebook answers is not one of change over time, whichever way it asks."""

    reads_events = False
    store_class = FirstStore


class CurrentView(MemoryTable):
    """The best memoryless responder: a fresh store at every observe, so it answers every question from the latest
    frame alone, whatever time the question names and whatever event its text gives (it reads no event table). A thing
    seen in that frame gets its current place; any other, and a corridor it did not see empty, is `unknown`."""

    reads_events = False

    def observe(self, steps: list[dict]) -> None:
        self.store = self.store_class(self.mapper)
        super().observe(steps)


class CurrentViewPrior(CurrentView):
    """`current_view` plus other scripts' records: the learned answer (`blind.learned_answer`) over the pool at the
    first non-empty level of `answer_keys.levels("current_view_prior", ...)`, keyed on the phase, the base pose and its
    own answer (`nm`), which is also a candidate; its own answer when no record matches. `opts["prior"]` is the records
    (default none)."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.prior: list[dict] = opts.get("prior", [])

    def ask_record(self, question: dict, now: dict, *, when: str) -> dict:
        """The record it keys on: `template`, `slots`, `when`, the phase, base and heading of the ask's step, and `nm`,
        what `current_view` answers there."""
        return build_record(question, when, self.store.steps[now["k"]], nm=self.answer_at(question, now))

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        out = {}
        for q in questions:
            rec = self.ask_record(q, now, when=when)
            pool = pool_at("current_view_prior", rec, self.prior)
            out[q["qid"]] = learned_answer(pool, [rec["nm"]]) if pool else rec["nm"]
        return out


class KindRange(CurrentView):
    """`ego_now`'s reference (`baselines_kind_range`): `current_view`'s answer to the question, an offset to the thing
    in view, kept only for its bearing and put at the typical floor range of the thing's kind (`opts["prior"]`: other
    scripts' records, default none); `unknown` where `current_view` has no offset, there is no typical range, or the
    template is any other. It never reads the question's range, which only the gold holds."""

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.ranges = Ranges(opts.get("prior", ()))

    def answer_at(self, question: dict, now: dict) -> dict:
        if question["template"] != KIND_RANGE_TEMPLATE:
            return dict(UNKNOWN)
        return placed(super().answer_at(question, now), self.ranges.of(question["slots"].get("kind")))


class AllUnknown(System):
    reads_files = False

    def observe(self, steps: list[dict]) -> None:
        pass

    def answer(self, question: dict) -> dict:
        return dict(UNKNOWN)


@dataclass(frozen=True)
class Baseline:
    """What `suite_priors.options_for` reads of a system, and nothing else."""

    cls: type[System]
    needs_prior: bool = False    # `opts["prior"]`: other scripts' records (`answer_keys.RECORD_KEYS`)
    prior_nm: bool = False       # those records carry `NM_SYSTEM`'s answers (a first wave runs it)
    needs_map: bool = False      # `opts["map"]`: the case layout's public `MapSpec`
    needs_events: bool = False   # `opts["events"]`: the job's `{wording: EventRule}`
    other_stream: bool = False   # fed another case's stream: `RunOptions.stream_from` and `ask_map`
    models_agent: bool = False   # an agent that learned the public layouts only: on a held-out one, kinds and no map
    in_pool: bool = False        # a learned shortcut that keeps its pool's own best response (`in_pool_twins`)
    reads_scorer: bool = False   # its answers pass through the scorer (`score.penalised`): a scorer change moves them


BASELINES = {
    "oracle": Baseline(Oracle, needs_map=True, needs_events=True),
    "memory_table": Baseline(MemoryTable, needs_map=True, needs_events=True),
    "current_view": Baseline(CurrentView, needs_map=True),
    "current_view_prior": Baseline(CurrentViewPrior, needs_prior=True, prior_nm=True, needs_map=True,
                                   reads_scorer=True),
    "path_log": Baseline(PathLog, needs_prior=True, needs_map=True, needs_events=True, models_agent=True,
                         reads_scorer=True),
    "blind": Baseline(Blind, needs_prior=True, reads_scorer=True),
    "other_case": Baseline(MemoryTable, needs_map=True, needs_events=True, other_stream=True),
    "questionnaire_blind": Baseline(QuestionnaireBlind, needs_prior=True, reads_scorer=True),
    "all_unknown": Baseline(AllUnknown),
    "kind_range": Baseline(KindRange, needs_prior=True, needs_map=True),
    # on a held-out layout (no map) it answers from the pose pool through `blind.learned_answer`, which reads the scorer
    "map_reader": Baseline(MapReader, needs_prior=True, needs_map=True, needs_events=True, models_agent=True,
                           reads_scorer=True),
    "latest_table": Baseline(LatestTable, needs_map=True),
    "first_table": Baseline(FirstTable, needs_map=True),
}


def in_pool_twins() -> list[str]:
    """The names (`gates.in_pool_name`) of the twins of the learned shortcuts of `BASELINES` (`gates.LEARNED`;
    `kind_range` takes a prior but is a reference that keeps no best response, and `map_reader` is a reference, so
    neither has one): each is the same
    system keeping its pool's best response without `blind.learned_answer`'s held-out round.
    That round abstains on a small pool near the boundary where a fair learner would answer, so a gate reads the larger
    score of a shortcut under the two rules, and `check` runs both. The twins are for the gates, not for `run`."""
    return [gates.in_pool_name(name) for name in BASELINES if name in gates.LEARNED]


def lookup(name: str) -> Baseline | None:
    """The baseline `name` names: one of `BASELINES`, or the twin of a learned one (`in_pool_twins`); None if none."""
    if name in BASELINES:
        return BASELINES[name]
    base = name.removesuffix(gates.IN_POOL)
    if base != name and base in BASELINES and base in gates.LEARNED:
        return replace(BASELINES[base], in_pool=True)
    return None
