"""What the tests of the look points and the questionnaire's order share (0.6.0 lane A2, redrawn for 0.10.0), on
`space_kit` and `family_kit`.

A look point is any frame at least `LOOK_MIN_STEPS` (4) into its task and not the task's last, at least
`LOOK_GAP_STEPS` (4) from every point kept, up to `ceil(frames / LOOK_PER_FRAMES)` (6). The vacuum's task of 28
keyframes has the candidate frames 4 to 26 (`FRAMES`) and asks up to 5 points (`VACUUM_CAP`); its stream is cut after
the decision points at 5, 18 and 21 and after its last step (27) and every eighth step otherwise (looks 5, 13, 18, 21,
27 end a decision window), which no longer decides anything for a look point.

`long_task(n)` is one task of `n` keyframes with no decision point of the vacuum's own, or those given: a frame number
(a `before_doorway`) or a `(frame, kind)` pair. `look_job` is `space_kit.JOB` asking the fake look family `fake_look`
at look points (and, unless told, no decision point); `register_look` puts that family in the registry for a test, with
a gold function of the test's own (default: one answerable candidate a point, naming `r:look<k>`).
"""

from __future__ import annotations

from dataclasses import replace

from family_kit import fake_cand, fake_spec, look_moment, register
from space_kit import JOB, SpaceScript, inputs, record, vacuum_run
from stbench.questions_world import World
from stbench.seeding import stable_hash

FRAMES = list(range(4, 27))   # the vacuum task's candidate look frames: 4 or more into the task, not its last (27)
VACUUM_CAP = 5               # ceil(28 / 6): the most points a case over the vacuum's task asks


def look_job(*looks: str, decisions: dict | None = None, **over):
    """`JOB` with these look families; `decisions` replaces its decision mapping (default: none asks)."""
    return replace(JOB, looks=looks or ("fake_look",), decisions={} if decisions is None else decisions, **over)


def long_task(frames: int = 90, decisions=()) -> SpaceScript:
    """One task of `frames` keyframes (kitchen, cleaning), with decision points at `decisions`: a frame (a
    `before_doorway`) or a `(frame, kind)` pair."""
    points = [d if isinstance(d, tuple) else (d, "before_doorway") for d in decisions]
    return SpaceScript(tasks=[[("kitchen", frames, "cleaning")]], decisions=points, depth_ks=set())


def world(script: SpaceScript, cid: str = "c0", length: str = "short", job=JOB, a: int = 0, b: int | None = None):
    return World(inputs(script, job), record(script, a=a, b=b, cid=cid, length=length))


def worlds(script: SpaceScript, cids, length: str = "short", job=JOB):
    """The window over the script's first task for each case id, over one `Inputs` (its free-space cells are computed
    once, whatever the number of ids)."""
    inp, rec = inputs(script, job), record(script, length=length)
    return (World(inp, replace(rec, case=cid)) for cid in cids)


def look_gold(w, k):
    """One answerable candidate at every look point, asked at `k`, naming a subject of its own."""
    return look_moment(k, [fake_cand("fake_look", f"k{k}", (f"r:look{k}",), decision_k=k)])


def register_look(monkeypatch, gold=look_gold, name: str = "fake_look", *, sliced: bool = False, scope: str = "now"):
    """The look family `name` in the registry for this test (a during family, in `LOOKS`)."""
    register(monkeypatch, fake_spec(name, "during", scope), gold, look=True, sliced=sliced)


def hand_points(cid: str, candidates: list[int] = FRAMES, cap: int = VACUUM_CAP, gap: int = 4,
                first: frozenset[int] | set[int] = frozenset()) -> list[int]:
    """The look points worked out by hand: the candidates with the `first` ones leading, each group in `stable_hash(case
    id, k, "ask")` order, each kept when it is `gap` or more from every point kept, up to `cap`; in keyframe order."""
    kept: list[int] = []
    for k in sorted(candidates, key=lambda k: (k not in first, stable_hash(cid, k, "ask"), k)):
        if len(kept) < cap and all(abs(k - o) >= gap for o in kept):
            kept.append(k)
    return sorted(kept)


__all__ = ["FRAMES", "VACUUM_CAP", "hand_points", "long_task", "look_gold", "look_job", "register_look", "vacuum_run",
           "world", "worlds"]
