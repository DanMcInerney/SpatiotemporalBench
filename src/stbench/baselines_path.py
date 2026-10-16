"""`path_log`: the strongest system that reads no pictures (recommendations §2.3 item 3; design §8.1, "paths leak").

It models an agent that learned the public layouts and keeps its own log. It is sent, and keeps, the `k`, clock, pose,
camera, status and subgoal of every step and the briefs of the tasks, and it holds the public map of the case's layout,
the job's event table and the other scripts' records; it never reads an rgb image, a depth image or a detection (a test
holds its import closure to that). The robot comes from the `start` message (`robots.from_doc`). On a held-out layout
it holds the layout's region kinds and no map (`suite_priors.options_for`): an agent that learned the public layouts
cannot have learned an unseen one, but it reads "clean the study" in its briefs, so it keeps the kinds its briefs can
name. Every answer that works from the map then goes without the candidate that needs it.

`object_where` is built in: `blind`'s pool with the rooms its briefs named (`answer_keys.rooms_named`) as the key's
first levels, and as candidates the places its camera faced at a logged step up to the question's moment: the window's
end for a question with no event ("where did you last see"), else the step its event resolves to over the steps of the
task the question names (`slots["task"]`). The candidates are the kind's receptacle in the public map, the pool's dense
places and the map pieces its subgoals name (`named_pieces`: a piece's name in the subgoal text of a logged step, with
an ordinal after a name that repeats on the map). It answers `unknown` when it faced none of them. It is never weaker
than `blind` (VISION rule 5): everything `blind` knows is in its levels, and the map, the subgoals and the places it
faced are more.

Every other family is answered by a module of `PATH_MODULES` (`baselines_path_<group>`, paired with
`questions_text.FAMILY_MODULES` and `table_answers.ANSWER_MODULES`) that defines `PATH_ANSWERERS: dict[str,
PathAnswerer]`, found by `answer_registry.collect` and bound to the instance in `PathLog.answerers`. An answerer reads
the instance's `model`, `map` (None on a held-out layout), `kinds`, `events`, `task_things`, `prior`, `log` and
`cached`; a module imports `table_log`, `grid`, `blind`, `places`, `answer_*` and neutral modules, never this one
(except for `PathAnswerer` and `PathLog` under `TYPE_CHECKING`), and never a picture reader. `baselines_path_space`
answers `path_ahead`: its own path (the fine cells its body swept between its delivered base poses count as seen free,
a detour is evidence), else a raycast over the public map (when it has one) with the pose pool.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar, cast

from . import answer_registry
from . import events as events_mod
from .answer_candidates import UNKNOWN
from .answer_keys import rooms_named
from .blind import build_record, learned_answer, pool_at
from .camera import Camera
from .catalog import for_map
from .mapspec import Furniture, MapSpec, surface_world_rect
from .places import Viewpoints, dense_places
from .robots import from_doc
from .stream_protocol import System
from .table_log import Log

PATH_MODULES = ("baselines_path_self", "baselines_path_space", "baselines_path_cover", "baselines_path_view",
                "baselines_path_changed", "baselines_path_things", "baselines_path_route", "baselines_path_zone",
                "baselines_path_adjacent", "baselines_path_change", "baselines_path_ego", "baselines_path_task",
                "baselines_path_clear", "baselines_path_people", "baselines_path_heading", "baselines_path_approach",
                "baselines_path_hazard", "baselines_path_way")
PathAnswerer = Callable[["PathLog", dict, dict], dict]   # (the system, a question, the record of its ask) -> answer
T = TypeVar("T")


def path_registry(modules: tuple[str, ...] = PATH_MODULES) -> dict[str, PathAnswerer]:
    """The answerers of the modules among `modules` that exist, family by family; `object_where` is the system's own."""
    return answer_registry.collect(modules, "PATH_ANSWERERS", taken=("object_where",))


PATH_ANSWERERS: dict[str, PathAnswerer] = path_registry()


def at(place: tuple[float, float, float]) -> dict:
    return {"type": "position", "xy": [round(place[0], 4), round(place[1], 4)]}


def named_pieces(m: MapSpec, texts: Iterable[str]) -> list[Furniture]:
    """The map pieces whose names occur in the texts (the subgoals of the logged steps) as whole words, any case, in map
    order. A name that repeats on the map is followed by an ordinal in the text, "kitchen counter 2": the second piece
    of that name in map order (1-based; an ordinal past the last names none); a name with no ordinal names every piece
    of the name."""
    by_name: dict[str, list[Furniture]] = {}
    for f in m.furniture:
        by_name.setdefault(f.name.lower(), []).append(f)
    found: set[str] = set()
    for name, pieces in by_name.items():
        for text in texts:
            for hit in re.finditer(rf"(?<!\w){re.escape(name)}(?: (\d+))?(?!\w)", text.lower()):
                n = None if hit[1] is None else int(hit[1])
                found.update(f.id for f in (pieces if n is None else pieces[n - 1:n] if n >= 1 else []))
    return [f for f in m.furniture if f.id in found]


def piece_place(f: Furniture) -> tuple[float, float, float]:
    """The centre of the piece's largest surface at that surface's height, else of its footprint at the piece's
    height."""
    best = None
    for s in f.surfaces:
        r = surface_world_rect(f.center, f.yaw, s)
        if best is None or r.sx * r.sy > best[0]:
            best = (r.sx * r.sy, (r.cx, r.cy, s.z))
    return best[1] if best else (f.center[0], f.center[1], f.size[2])


class PathLog(System):
    """The log-keeping, picture-free system. Options: `events` (the job's `{wording: EventRule}`) and `map` (the case
    layout's public `MapSpec`) are required, `prior` (other scripts' records) is optional. On a held-out layout the
    option `kinds` (the layout's region kinds, in map order) stands in for `map`, which it is then not given; a system
    given neither a map's regions nor kinds names no room of its briefs."""

    structured = True
    reads_files = False   # it keeps the log of the steps and opens no picture

    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        for name in ("map", "events"):
            if name not in opts and not (name == "map" and "kinds" in opts):
                raise ValueError(f"path_log needs the option {name!r}")
        self.map: MapSpec | None = opts.get("map")
        self.kinds: tuple[str, ...] = tuple(r.kind for r in self.map.regions) if self.map else tuple(
            opts.get("kinds", ()))
        self.events: dict = opts["events"]
        self.task_things = opts.get("task_things")   # the job's `TaskThings`, or None (it asks no `task_things`)
        self.prior: list[dict] = opts.get("prior", [])
        self.log = Log()
        self._cache: dict[str, object] = {}
        self.answerers: dict[str, Callable[[dict, dict], dict]] = {   # template -> (question, record) -> answer
            "object_where": self._object_where, **{f: partial(fn, self) for f, fn in PATH_ANSWERERS.items()}}

    def start(self, session: str, robot: dict, channel: str) -> None:
        super().start(session, robot, channel)
        self.model = from_doc(robot)
        self.cam = Camera.load(self.public / "camera.json")

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        self.log.begin_task(index, clock, gap_s, brief)

    def observe(self, steps: list[dict]) -> None:
        for step in steps:
            self.log.add(step)

    def cached(self, key: str, make: Callable[[], T]) -> T:
        """What an answerer works out once a case from the public map and the robot (a raycast's obstacles):
        `make()` the first time `key` is asked, the same object after."""
        if key not in self._cache:
            self._cache[key] = make()
        return cast(T, self._cache[key])

    # -- the record, the pool ---------------------------------------------------------------------------------
    def ask_record(self, question: dict, now: dict, *, when: str) -> dict:
        """The record it keys on: `template`, `slots`, `when`, and the phase, base pose and heading of the step the ask
        is about (the log's entry at `now["k"]`) and the rooms its briefs named (`answer_keys.rooms_named` over the
        region kinds of the layout)."""
        rooms = rooms_named(self.log.briefs(), self.kinds)
        return build_record(question, when, self.log.steps[now["k"]], rooms=rooms)

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        out = {}
        for q in questions:
            if q["template"] not in self.answerers:
                raise KeyError(f"path_log has no answerer for the template {q['template']!r}; "
                               f"the answerers are {sorted(self.answerers)}")
            out[q["qid"]] = self.answerers[q["template"]](q, self.ask_record(q, now, when=when))
        return out

    # -- object_where -----------------------------------------------------------------------------------------
    def _spot(self, kind: str) -> tuple[float, float, float] | None:
        """The centre of the largest surface of the kind's most common receptacle in the public map, or None: the
        catalog lists a kind's receptacles with the most typical first. None on a layout with no map."""
        if self.map is None:
            return None
        kinds = for_map(self.map).kinds
        for receptacle in kinds[kind].receptacles if kind in kinds else ():
            best = None
            for f in self.map.furniture:
                for s in f.surfaces:
                    if s.kind == receptacle:
                        r = surface_world_rect(f.center, f.yaw, s)
                        if best is None or r.sx * r.sy > best[0]:
                            best = (r.sx * r.sy, (r.cx, r.cy, s.z))
            if best:
                return best[1]
        return None

    def named_places(self, until: int) -> list[tuple[float, float, float]]:
        """The places of the map pieces that the subgoals of the logged steps up to `until` name (`named_pieces`): none
        on a layout with no map, or when no logged step carries a subgoal."""
        texts = {step["subgoal"] for step in self.log.steps_until(until) if step.get("subgoal")}
        return [piece_place(f) for f in named_pieces(self.map, texts)] if self.map is not None and texts else []

    def faced_places(self, kind: str, pool: list[dict], until: int) -> list[tuple[float, float, float]]:
        """The candidate places its camera faced at a logged step up to `until`: the kind's receptacle in the public
        map, the pool's dense places and the pieces its subgoals name, those in view (`places.Viewpoints`) at some such
        step."""
        spot = self._spot(kind)
        view = Viewpoints([s["camera"] for s in self.log.steps_until(until)], self.cam)
        return [p for p in [*dense_places(pool), *([spot] if spot else []), *self.named_places(until)]
                if view.sees(p)]

    def _object_where(self, question: dict, rec: dict) -> dict:
        slots = question["slots"]
        if slots.get("event") is None:   # "where did you last see": every logged step, to the window's end
            until = max(self.log.steps, default=None)
        elif slots["event"] not in self.events:
            raise ValueError(f"the job's events do not name {slots['event']!r}; they are {sorted(self.events)}")
        else:
            until = events_mod.resolve(self.events[slots["event"]], self.log.task_steps(slots.get("task")))
        pool = pool_at("path_log", rec, self.prior)
        golds = [r["gold"] for r in pool]
        faced = self.faced_places(slots["kind"], golds, until) if pool and until is not None else []
        return learned_answer(pool, [at(p) for p in faced]) if faced else dict(UNKNOWN)
