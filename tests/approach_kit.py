"""What the `closest_approach` tests share: `person_kit`'s one-task vacuum script with the robot driving north at the
look point, and people put in by hand, so every distance is worked out in the test.

The look point is `ego_kit.K` = 13. The base is at (3.0, 1.0) heading north there and at (3.0, 0.55) at keyframe 12
(`DRIVING`), so the camera, 0.13 m ahead of the base, is at (3.0, 1.13) at 13 and the robot has moved 0.45 m north. The
clock the stream delivers is to the tenth of a second: keyframe 12 is at 18.0 s and 13 at 19.5 s, delivered as 18.0 and
19.5, so the robot's velocity is 0.45 m over 1.5 s, 0.3 m/s north. A person's two sightings are keyframes 12 and 13,
1.5 s apart. `EGO_JOB`'s clearance is 2.0 m (C/2 = 1.0 m), so a person is evidence out to about 5.5 m.

`point` is a person whose pixels are one floor point (a hull of one vertex: its distance is the point's), for the cases
where the arithmetic is the point of the test; `put_person` with no `pts` is the 0.34 x 0.56 m footprint.

`approach_mem`'s tests (0.11.0) put the person's one sighting at keyframe `J` = 10 (15.0 s), 4.5 s before the ask: the
base is at (6.0, 1.5) there, so the 8 cm camera is at about (5.9, 1.6), and a person standing at `AT` = (3.5, 2.0) is
about 2.2 m from it, well inside the 5.5 m the contact tie allows. At the look point the camera floor point is
(3.0, 1.13) and moves north at 0.3 m/s, so the person's footprint (x 3.33-3.67, y 1.72-2.28) lies 0.33 m east of its
path from 1.97 s to 3.83 s on. `mem` is the candidate or dead band at a look point; `case_of` compiles it into a case.
"""

from __future__ import annotations

import numpy as np

import person_kit as pk
from ego_kit import K
from stbench import episode as ep
from stbench import questions, questions_approach

DRIVING = {12: (3.0, 0.55)}   # the base at keyframe 12; at K it is (3.0, 1.0): north at 0.3 m/s (0.45 m in 1.5 s)
J = 10                        # `approach_mem`'s last sighting: 15.0 s, 4.5 s before the ask at K (19.5 s)
AT = (3.5, 2.0)               # where its person stands


def scene(**fields):
    """The vacuum's script with the robot driving north into the look point `K`."""
    return pk.scene(poses={**DRIVING, **fields.pop("poses", {})}, **fields)


def point(xy: tuple[float, float]) -> np.ndarray:
    """The points of a person whose pixels are one floor point: their floor hull has one vertex at `xy`."""
    return np.array([(xy[0], xy[1], 0.0), (xy[0], xy[1], 0.9)])


def walker(s, pid: str, before: tuple[float, float], now: tuple[float, float], seen=(12, K)) -> str:
    """A person at `before` until 18.5 s and at `now` after (so at `before` at every keyframe up to 12, 18.0 s, and at
    `now` at 13, 19.5 s), a floor point sighted at the keyframes `seen`."""
    pk.put_person(s, pid, [(0, 18.5, (*before, 0.0)), (18.5, None, (*now, 0.0))], list(seen))
    for k in seen:
        s.parts[k, pid] = point(before if k < K else now)
    return pid


def mem(s, k: int = K):
    """`approach_mem`'s candidate, or the dead band it names, at look point `k` of `s`, the stream cut there."""
    return questions_approach.mem_candidate(pk.cut(s, k), k)


def case_of(path, s, k: int = K):
    """`s` as a case with the `approach_mem` of look point `k` written by hand from the gold function: (the case, the
    qid, the gold row)."""
    cand = questions_approach.mem_candidate(pk.world(s), k)
    case = s.write_case(path)
    qs, gs = questions.records([cand])
    ep.write_jsonl(case.questions, qs, sort_keys=True)
    ep.write_jsonl(case.gold, gs)
    return case, qs[0]["qid"], gs[0]
