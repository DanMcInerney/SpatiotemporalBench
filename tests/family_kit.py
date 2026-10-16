"""What the family lanes' compiler tests share, on top of `space_kit`: a script of several tasks of the vacuum, things
of every fate in each, a hand map, and fake families to register for a test of the questionnaire.

`multi_run(n)` is `space_kit.vacuum_run(n)` (28 keyframes a task: docked 0, undocking 1-2, transit 3-6, cleaning 7-18,
transit 19-22, docking 23-25, docked 26-27; decision points at 5, 18 and 21; `JOB`'s events at 2, 7, 18 and 26, each
plus 28 a task) with the brush on exactly in `cleaning`; `things_by_task(s, n)` adds five things a task, each with a
description of its own (the colour names the task, the kind the fate; a box takes a colour no bag has, since a sighted
look-alike of a thing's colour hides it from the questions, `questions_where.askable`):
- `S<t>` a bag recognisable at the task's first step and still there: answerable at every event of task t;
- `L<t>` a box recognisable only from offset 12: answerable at the last two events of task t;
- `G<t>` shoes seen at 500 px only, a glimpse: skipped;
- `U<t>` a toy no frame shows: unknown at every event of task t;
- `C<t>` a bag recognisable at offset 1 of task t and never again: in a later task, skipped (its last sighting lies in
  an earlier task).
`hand_map(s)` is a `MapSpec` whose regions are the script's synthetic rooms (`synth.RECTS`; the bundled map's rooms
are elsewhere), and `inputs`, `world` and `compiled` give a family a window of it with that map. `register` adds a fake
family to the registry for one test (a gold function of the test's own, a spec of `fake_spec`).
"""

from __future__ import annotations

from dataclasses import replace

from space_kit import BAG_SIZE, JOB, LEGS, SpaceScript, vacuum_run
from space_kit import inputs as space_inputs
from space_kit import record as space_record
from stbench import questionnaire, questions, questions_text
from stbench.job_registry import JobSpec
from stbench.mapspec import MapSpec, Region
from stbench.questions_select import Cand, Moment
from stbench.questions_world import Inputs, World
from stbench.records import FamilySpec
from synth import CENTRE_XY, RECTS

TASK_FRAMES = sum(leg[1] for leg in LEGS)  # the keyframes of one task: 28
COLOURS = ("red", "blue", "green", "black", "white", "brown", "grey", "orange", "yellow", "pink", "purple", "cyan",
           "teal", "beige", "navy", "olive", "maroon", "gold")   # the last six: a task's box, none a bag's colour


def brush_on(n_tasks: int) -> dict[int, dict]:
    """`status` overrides that turn the brush on at every keyframe of a `cleaning` leg."""
    out: dict[int, dict] = {}
    k = 0
    for _ in range(n_tasks):
        for _room, n, phase in LEGS:
            for _ in range(n):
                if phase == "cleaning":
                    out[k] = {"brush": True}
                k += 1
    return out


def multi_run(n_tasks: int = 3, *, seed: int = 1, **fields) -> SpaceScript:
    """`n_tasks` tasks of the vacuum with the brush on in `cleaning`; `fields` replace `SpaceScript`'s."""
    return vacuum_run(n_tasks, seed=seed, **{"status": brush_on(n_tasks), **fields})


def things_by_task(s: SpaceScript, n_tasks: int) -> SpaceScript:
    """Add each task's five things (the module docstring); all stand still at (0.5 + 0.4 t, y) with a y a fate."""
    for t in range(n_tasks):
        o, x = TASK_FRAMES * t, 0.5 + 0.4 * t
        s.item(f"S{t}", "bag", {"color": COLOURS[t]}, [(0, None, (x, 1.0, 0.0))], size=BAG_SIZE)
        s.see(o + 1, f"S{t}", 1500)
        s.item(f"L{t}", "box", {"color": COLOURS[12 + t]}, [(0, None, (x, 1.5, 0.0))])
        s.see(o + 12, f"L{t}", 1500)
        s.item(f"G{t}", "shoes", {"color": COLOURS[t]}, [(0, None, (x, 2.0, 0.0))])
        s.see(o + 3, f"G{t}", 500)
        s.item(f"U{t}", "toy", {"color": COLOURS[t]}, [(0, None, (x, 2.5, 0.0))])
        s.item(f"C{t}", "bag", {"color": COLOURS[6 + t]}, [(0, None, (x, 0.5, 0.0))], size=BAG_SIZE)
        s.see(o + 1, f"C{t}", 1500)
    return s


def hand_map(s: SpaceScript) -> MapSpec:
    """A map of the script's two synthetic rooms, as `synth.Synth.timeline` names them, and nothing else."""
    regions = tuple(Region(f"R{i + 1}", name, poly, CENTRE_XY[i], 0, 0.0, 2.5, name)
                    for i, (name, poly) in enumerate(zip(s.stage.rooms, RECTS, strict=True)))
    return MapSpec(s.stage.map_name, (0.0, 0.0, 8.0, 3.0), 0.1, 2.5, regions, (), (), (), (), ())


def inputs(s: SpaceScript, job: JobSpec = JOB) -> Inputs:
    """`space_kit.inputs` with the hand map."""
    return replace(space_inputs(s, job), map=hand_map(s))


def world(s: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short",
          job: JobSpec = JOB) -> World:
    """The window of tasks a..b of the script (b defaults to a) as the compiler's `World`, with the hand map."""
    return World(inputs(s, job), space_record(s, a=a, b=b, cid=cid, length=length))


def compiled(s: SpaceScript, a: int = 0, b: int | None = None, cid: str = "c0", length: str = "short",
             job: JobSpec = JOB) -> tuple[list[dict], list[dict]]:
    """The window's questions and gold rows, compiled in memory (qids random)."""
    return questions.compile_inputs(inputs(s, job), space_record(s, a=a, b=b, cid=cid, length=length))


# -- fake families ----------------------------------------------------------------------------------------------
def fake_spec(name: str, when: str = "end", scope: str = "memory", answer_type: str = "position",
              credit: str = "ladder") -> FamilySpec:
    """A family that passes `check_family`, one slot `n`, rendering "<name> <n>"."""
    return FamilySpec(name, when, answer_type, credit, ("see", "pose", "status"), scope, None, False, ("n",),
                      lambda slots: f"{name} {slots['n']}")


def fake_cand(family: str, n, subjects=(), *, unknown: bool = False, decision_k: int | None = None,
              answer_class: str | None = None, range_m: float | None = None) -> Cand:
    """A candidate of a fake family: a position gold (or `unknown`), the distance ladder, the given subjects, and the
    class and range a gold row carries (`records.ANSWER_CLASSES`; null where the family declares none)."""
    gold = {"type": "unknown"} if unknown else {"type": "position", "xy": [1.0, 1.0]}
    return Cand(family, {"n": n}, gold, "unknown" if unknown else "answerable", () if unknown else (1,), (),
                {"ladder": [0.25, 0.5, 1.0, 2.0]}, decision_k=decision_k, subjects=tuple(subjects),
                answer_class=answer_class, range_m=range_m)


def look_moment(k: int, answerable: list[Cand], unknown: list[Cand] | None = None) -> Moment:
    """What a look family's gold gives at the look point `k`: the point's two pools, its key `str(k)`."""
    return Moment(str(k), answerable, unknown or [])


def camera_of(w: World, k: int) -> tuple[tuple[float, float, float], float]:
    """The delivered camera of script keyframe `k` read straight from the pose row (position, base heading in
    degrees), for a test that checks a helper against the file and not against the helper."""
    row = w.rows[k]
    return tuple(row["camera"]["pos"]), row["pose"]["yaw_deg"]


def register(monkeypatch, spec: FamilySpec, gold, *, sliced: bool = False, coin: bool = False, look: bool = False,
             capped: bool = False, relax=None) -> None:
    """The family in the registry for this test: its spec, its gold function, and (for a sliced or a coin end family)
    its name in `SLICED` or `COINS`, (for a during family asked at look points) in `LOOKS`, (for an end family whose
    empty golds the draw may thin) in `CAPPED`, and its `relax` function in `RELAX`; a fake lands after the real
    families, so the registry's order is the real ones' first."""
    if sliced and coin:
        raise ValueError(f"{spec.name} cannot be both sliced and a coin")
    for module in (questions_text, questionnaire):   # a module that reads the hook from the registry sees it too
        for name, on in (("LOOKS", look), ("CAPPED", capped)):
            if on and hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name) | {spec.name})
        if relax is not None and hasattr(module, "RELAX"):
            monkeypatch.setitem(module.RELAX, spec.name, relax)
    monkeypatch.setitem(questionnaire.FAMILIES, spec.name, spec)
    monkeypatch.setitem(questionnaire.GOLDS, spec.name, gold)
    if sliced:
        monkeypatch.setattr(questionnaire, "SLICED", questionnaire.SLICED | {spec.name})
    if coin:
        monkeypatch.setattr(questionnaire, "COINS", questionnaire.COINS | {spec.name})
