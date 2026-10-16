"""The job registry: what a job is, and where to find one by its id.

A job (`vacuum/1`) is a module of `JOB_MODULES` with a `SPEC`, and a robot may have several jobs beside each other
(`vacuum/1` in `job_vacuum`, `vacuum/2` in `job_vacuum2`): `get` finds a job by its id over those modules. Other lanes
read jobs only through here: the compiler the decision kinds, events and relevance, the harness the task kinds, domain
and assignment texts, `validate` the job's own checks. `get` imports a module on first use, so `validate` ->
`job_registry` -> `job_vacuum` never cycles at import (`validate.Problem`, `Timeline` and `MapSpec` are imported for
type checking only), and a job that is not written yet costs nothing: `get` and `all_jobs` skip its module.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .events import EventRule
from .geometry import Point
from .questions_words import CLEAR_DIRECTIONS, DISTANCE_LADDER, FLY_DIRECTIONS
from .robots import ROBOTS
from .task_predicate import TaskThings
from .timeline_ego import Ego, TaskPlan
from .zones import ZoneRule

if TYPE_CHECKING:
    from .mapspec import MapSpec
    from .timeline import Timeline
    from .validate import Problem


@dataclass(frozen=True)
class Relevance:
    kinds: frozenset[str]
    regions: frozenset[str]
    phases: frozenset[str]
    floor: bool = False  # a job whose robot works on the floor: critical things stood on the robot's floor
    near: tuple[Point, ...] = ()  # with `near_m`: a place is critical only within `near_m` of one of these points
    near_m: float | None = None  # None: no near rule (the drone's: the ground its camera sees from the drop point)


@dataclass(frozen=True)
class JobSpec:
    id: str  # "vacuum/1"
    robot: str  # a ROBOTS key
    domain: str  # "home", "warehouse" or "delivery": the brief's place
    task_kinds: tuple[str, ...]
    decisions: dict[str, tuple[str, ...]]  # decision kind -> the during families asked there
    events: dict[str, EventRule]  # an end question's event wording -> its rule
    plan: Callable[[MapSpec, int, int], list[TaskPlan]]  # (map, seed, n_tasks) -> world-independent assignments
    script: Callable[..., Ego]  # the path, status and decision points over a built world
    relevance: Callable[[TaskPlan], Relevance]  # what is task-critical for an assignment
    checks: Callable[[Timeline, MapSpec], list[Problem]]  # the job's own validate rules, in its own module
    anchors: frozenset[str]  # entity kinds the robot may touch, never overlap: the vacuum's dock
    assignments: dict[str, str]  # task kind -> the brief's assignment template
    swaps: tuple[tuple[str, str], ...] = ()  # path-compatible task-kind pairs for the brief swap
    continues: frozenset[str] = frozenset()  # task kinds that carry on an earlier task: no case window starts at one
    ends: tuple[str, ...] | None = None  # the end families the job asks; None: the questionnaire's default (M1's)
    zones: dict[str, ZoneRule] = field(default_factory=dict)  # decision kind (zone_clear) or event wording -> its zone
    slot_zones: tuple[ZoneRule, ...] = ()  # the rules slots_empty asks about over a whole window
    # The text the robot is working on at each keyframe, from the timeline alone and causal (a keyframe's text reads
    # the phases and decisions up to it and the responses before it, never a coin decided at or after it). None: the
    # job states none.
    subgoals: Callable[[Timeline, MapSpec], list[str]] | None = None
    ladder: tuple[float, ...] = DISTANCE_LADDER  # the rungs of the distance credit for the job's place, in metres
    # The families asked at look points, in order (`ego_now`, `ego_mem`; empty: the job asks none, as every job before
    # 0.6.0) and the predicate of the job's `task_things` question (None: it asks none); both are the lanes of 0.6.0's.
    looks: tuple[str, ...] = ()
    task_things: TaskThings | None = None
    # The decision kinds (keys of `decisions`) whose frames come first in the draw of the look points (0.10.0,
    # `questions_looks.points`): the moments the job's task turns on, where a question about the surroundings is
    # decision-relevant. Only the draw reads it; no family is mapped to a decision (`questionnaire.check_decisions`).
    look_decisions: tuple[str, ...] = ()
    # The job's own numbers for the clearance credit (0.7.0, docs/questions.md "Credit"): the margin C its avoidance and
    # approach keep, and the re-find radius R of a place it drives back to (None: R is `match_m`). Both are declared
    # per job by lane D1; None for C until then, and `clearance_m` refuses to answer while it is unset.
    clearance: float | None = None
    reacquire: float | None = None
    # The ego families (`ego_now`, `ego_mem`) whose questions the resolvability screen would drop are kept on the
    # rungs of `nearest_part.fallback_ladder`: the declared remedy for a job whose supply of a family falls under
    # `abilities.MIN_ABILITY_N`. Empty on every job until a dataset's supply asks for it (DECISIONS).
    screen_fallback: frozenset[str] = frozenset()
    # The slot table of the look points (0.8.0, docs/questions.md "What a case asks"): per slot, the (family, percent)
    # pairs it draws from; any number of slots (0.10.0 uses three: harm, situation and memory, the rest). A point asks
    # one question a slot, the family being the one whose cumulative range holds `stable_hash(case id, k, f"slot{i}")
    # % 100` (`questions_looks.slot_family`), none when the hash falls past the slot's total. None: one slot per
    # family of `looks`, each at 100, which is 0.7.0's draw. A family must be in `looks` and every family of `looks`
    # in some slot; a slot's percents sum to at most 100 (`ValueError` naming the job otherwise, at construction).
    look_shares: tuple[tuple[tuple[str, int], ...], ...] | None = None
    # Per look family, the percent of its `unknown` golds that are kept (a hash of the case, the family and the point,
    # `questionnaire.looks`); a family not named keeps all. Refused for a `SLICED` family, whose share is its slice.
    look_unknown_keep: Mapping[str, int] = field(default_factory=dict)
    # The status phases in which `clear_path` is asked (the robot is driving; elsewhere its status answers), and a
    # flier's least base height for it (its fixed structure of a place the corridor must clear). Empty and None: the
    # job asks no corridor, or (for the floor) nothing is too low.
    clear_phases: frozenset[str] = frozenset()
    # The status phases a look family is asked in (0.10.0), per family of `looks`, as `clear_phases` is for
    # `clear_path`: the hazard families ask about the ground below or the floor ahead, which means nothing while the
    # robot is docked or flying elsewhere. A family not named here has no phase rule of its own; `drop_zone_people`
    # and `tangle_ahead` refuse to be asked without one (`questions_hazard`). Validated at construction (families in
    # `looks`, phases among the robot's own).
    look_phases: Mapping[str, frozenset[str]] = field(default_factory=dict)
    fly_floor_m: float | None = None
    # The job's world holds at most one person (the helper's resident): "the person" names one (`person_mem`, which no
    # job asks now).
    one_person: bool = False
    # The moves a `clear_path` question may ask (0.9.0): each look point draws one of these, equally, by a hash of the
    # case, the point and "direction". A ground robot's are among ahead, back, left and right (the last two turn in
    # place and drive on); a flier's among ahead and back. Refused at construction, naming the job, otherwise.
    clear_directions: tuple[str, ...] = ("ahead",)

    def __post_init__(self) -> None:
        _check_looks(self)
        _check_phases(self)
        _check_directions(self)

    @property
    def look_slots(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        """The slot table the look points draw from: `look_shares`, else one slot per family of `looks` at 100."""
        if self.look_shares is None:
            return tuple(((family, 100),) for family in self.looks)
        return self.look_shares

    @property
    def unit_m(self) -> float:
        """The job's place unit u: the first rung of its ladder, 0.25 m in a home, 0.5 in a warehouse, 1.0 in the city.
        It sets the band edges, the change thresholds and the predicate bands."""
        return self.ladder[0]

    @property
    def match_m(self) -> float:
        """The matching radius of the job's set answers, twice its unit (0.5 m in a home, 1.0 warehouse, 2.0 city)."""
        return 2 * self.unit_m

    @property
    def clearance_m(self) -> float:
        """The job's clearance C, metres: the margin its avoidance and approach keep (`rules.clearance_ladder` builds
        the credit's rungs from it). ValueError naming the job while none is declared."""
        if self.clearance is None:
            raise ValueError(f"job {self.id!r} declares no clearance")
        return self.clearance

    @property
    def reacquire_m(self) -> float:
        """The job's re-find radius R, metres: how close a place must be named for the robot to find the thing again
        on arrival. The job's own `reacquire` when it declares one, else its `match_m`."""
        return self.match_m if self.reacquire is None else self.reacquire


def _check_directions(job: JobSpec) -> None:
    """ValueError naming the job for `clear_directions` that is empty, repeats a direction, or holds one its body cannot
    take: a ground robot's are among `CLEAR_DIRECTIONS`, a flier's among `FLY_DIRECTIONS`."""
    flies = job.robot in ROBOTS and ROBOTS[job.robot].flies
    allowed = FLY_DIRECTIONS if flies else CLEAR_DIRECTIONS
    if not job.clear_directions or len(set(job.clear_directions)) != len(job.clear_directions):
        raise ValueError(f"the job {job.id} declares clear_directions {list(job.clear_directions)}: at least one, and "
                         f"each once")
    if stray := [d for d in job.clear_directions if d not in allowed]:
        body = "flier" if flies else "ground"
        raise ValueError(f"the job {job.id} declares clear_directions {stray}, which its {body} body cannot take; "
                         f"the directions are {list(allowed)}")


def _check_phases(job: JobSpec) -> None:
    """ValueError naming the job for a `look_phases` entry that cannot be asked: a family not in `looks`, no phase at
    all, or a phase the robot's status does not have (when the robot is a registered one)."""
    robot = ROBOTS.get(job.robot)
    for family, phases in job.look_phases.items():
        if family not in job.looks:
            raise ValueError(f"the job {job.id} declares look_phases for {family!r}, which is not in its looks "
                             f"{list(job.looks)}")
        if not phases:
            raise ValueError(f"the job {job.id} declares no phase for {family!r} in look_phases: at least one")
        if robot is not None and (stray := sorted(set(phases) - set(robot.status.phases))):
            raise ValueError(f"the job {job.id} declares look_phases {stray} for {family!r}, which its {job.robot} "
                             f"does not have; its phases are {list(robot.status.phases)}")


def _check_looks(job: JobSpec) -> None:
    """ValueError naming the job for a slot table, an unknown keep or a priority decision that cannot be drawn: a family
    not in `looks`, a family of `looks` in no slot, a percent outside 1 to 100, a slot whose percents sum past 100, a
    keep outside 0 to 100 or for a family not in `looks`, or for a `SLICED` family (its share is the slice); a
    `look_decisions` kind that repeats, that is no decision of the job, or that a job with no `looks` names."""
    if job.look_decisions and not job.looks:
        raise ValueError(f"the job {job.id} names look_decisions {list(job.look_decisions)} but asks no look family")
    if len(set(job.look_decisions)) != len(job.look_decisions):
        raise ValueError(f"the job {job.id} names a decision kind twice in look_decisions {list(job.look_decisions)}")
    if stray := [kind for kind in job.look_decisions if kind not in job.decisions]:
        raise ValueError(f"the job {job.id} names look_decisions {stray}, which are no decision of the job "
                         f"{list(job.decisions)}")
    for family, keep in job.look_unknown_keep.items():
        if family not in job.looks:
            raise ValueError(f"the job {job.id} keeps unknown golds of {family!r}, which is not in its looks "
                             f"{list(job.looks)}")
        if not 0 <= keep <= 100:
            raise ValueError(f"the job {job.id} keeps {keep}% of the unknown golds of {family!r}; a percent is 0 to "
                             f"100")
    if job.look_unknown_keep:
        from .questions_text import SLICED  # the registry imports the family modules, which import this one
        if sliced := sorted(set(job.look_unknown_keep) & SLICED):
            raise ValueError(f"the job {job.id} keeps a share of the unknown golds of {', '.join(sliced)}, a sliced "
                             f"family whose unknown share is its slice")
    if job.look_shares is None:
        return
    named = {family for slot in job.look_shares for family, _ in slot}
    if stray := sorted(named - set(job.looks)):
        raise ValueError(f"the job {job.id} draws {', '.join(stray)} in its look slots, which are not in its looks "
                         f"{list(job.looks)}")
    if unslotted := sorted(set(job.looks) - named):
        raise ValueError(f"the job {job.id} asks {', '.join(unslotted)} at look points but draws it in no slot")
    for i, slot in enumerate(job.look_shares):
        if any(not 1 <= percent <= 100 for _, percent in slot):
            raise ValueError(f"the job {job.id} gives a family of look slot {i} a percent outside 1 to 100: {slot}")
        if sum(percent for _, percent in slot) > 100:
            raise ValueError(f"the job {job.id}'s look slot {i} sums to more than 100%: {slot}")


JOB_MODULES = ("stbench.job_vacuum", "stbench.job_vacuum2", "stbench.job_helper", "stbench.job_amr",
               "stbench.job_drone")


def _import(module: str):
    """The module, or None when it does not exist yet (ModuleNotFoundError naming that module); any other import
    error is raised."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return None


def get(job_id: str) -> JobSpec:
    """The job with this id: the `SPEC` of the first module of `JOB_MODULES` that holds it (a module not
    written yet is skipped, any other import error raised); KeyError names the job when none does."""
    for name in JOB_MODULES:
        spec = getattr(_import(name), "SPEC", None)
        if spec is not None and spec.id == job_id:
            return spec
    raise KeyError(f"no job {job_id!r}")


def all_jobs() -> list[JobSpec]:
    """Every registered job, in `JOB_MODULES` order; a module not written yet is skipped."""
    return [module.SPEC for name in JOB_MODULES if (module := _import(name)) is not None]
