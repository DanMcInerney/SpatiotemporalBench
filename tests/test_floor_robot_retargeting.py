"""The floor robots ask about the floor (FD-questions-0.10 §1.3, review S3): for a job whose relevance sets
`Relevance.floor` (the vacuum, the tote robot) `ego_now`, `ego_mem` and `object_where` ask only about a thing standing
on the floor at the sighting the gold rests on; a job that does not set it (the helper, the drone) asks of every
thing. Each scene is a floor thing and a table thing, the robot a vacuum whose relevance is set by the test, so the
job's floor rule is the only difference between the two worlds. The `ego_kit` scene: the camera at (3.0, 1.13, 0.08)
heading north at the look point 13."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from ego_kit import EGO_JOB, SMALL, K, corners, named, put, scene
from space_kit import BAG_SIZE, JOB, RELEVANT_KINDS, vacuum_run
from stbench.job_registry import Relevance
from stbench.questions_ego import ego_mem, ego_now
from stbench.questions_where import pools

TABLE_Z = 0.75   # a table top, far over the vacuum's climb of 0.02 m
ROOMS = frozenset({"kitchen", "living room"})


def job_of(base, floor: bool):
    """`base` with a relevance that sets (or not) the floor rule and nothing else."""
    return replace(base, relevance=lambda plan: Relevance(RELEVANT_KINDS, ROOMS, frozenset(), floor=floor))


def on_table(xy):
    """The corners of a small box standing on a table."""
    return [(x, y, TABLE_Z + z) for x, y, z in corners(xy, SMALL)]


def put_table(s, eid, colour, xy, seen=()):
    """A bag `eid` standing on a table at `xy`, seen at the keyframes `seen`, its pixels on the table top."""
    return put(s, eid, "bag", colour, xy, seen, where=[(0, None, (*xy, TABLE_Z))], pts=on_table(xy))


def ego_pools(s, floor: bool):
    """(the answerable entities, the real unknown entities) of `ego_mem` at the look point 13 under the job."""
    moment = ego_mem(fk.world(s, job=job_of(EGO_JOB, floor)), K)
    unknown = {c.subjects[0][2:] for c in moment.unknown if c.subjects[0].startswith("e:")}
    return list(named(moment)), unknown


def test_ego_now_asks_the_floor_things_alone_of_a_floor_robot():
    """Contract: for a floor robot's job `ego_now` names only a thing on the floor in the frame; for a job that sets no
    floor rule it names the table thing as well. Hand-worked: a blue bag F on the floor and a red bag T on a table
    0.75 m up, both recognisable in frame 13: the floor job names F alone, the other job F and T in timeline order.
    Regression caught: a mug on a table asked of a vacuum that cannot reach it, or the helper's table things lost with
    the floor ones."""
    s = scene()
    put(s, "F", "bag", "blue", (1.0, 1.13), [K])
    put_table(s, "T", "red", (5.0, 3.13), [K])
    assert list(named(ego_now(fk.world(s, job=job_of(EGO_JOB, True)), K))) == ["F"]
    assert list(named(ego_now(fk.world(s, job=job_of(EGO_JOB, False)), K))) == ["F", "T"]


def test_ego_mem_asks_the_floor_things_alone_of_a_floor_robot_in_both_pools():
    """Contract: for a floor robot's job `ego_mem` asks only of a thing that stood on the floor at the sighting its
    gold rests on, and its unknown pool names only unseen things that stand on the floor, so a table cannot predict the
    slice; a job without the rule keeps every thing in both pools. Hand-worked: F (floor) and T (table) seen at
    keyframe 2, before the recent view 6-13, and never in it; U (floor) and V (table) never seen at all. The floor job's
    answerable pool is F and the unknown pool's real things are U; the other job's are F and T, and U and V.
    Regression caught: the answerable pool filtered and the unknown one not (a table thing would be unknown more often
    than answerable, so its kind would predict the slice), or the filter applied to the helper."""
    s = scene()
    put(s, "F", "bag", "blue", (1.0, 1.13), [2])
    put_table(s, "T", "red", (5.0, 3.13), [2])
    put(s, "U", "bag", "yellow", (6.0, 2.0))
    put_table(s, "V", "green", (6.5, 1.0))
    assert ego_pools(s, True) == (["F"], {"U"})
    assert ego_pools(s, False) == (["F", "T"], {"U", "V"})


def test_object_where_asks_the_floor_things_alone_of_a_floor_robot():
    """Contract (`questions_where.critical`): where the job sets `Relevance.floor` only a thing that stood on the floor
    at its last recognisable sighting is critical and so answerable; a job that does not set it keeps the table thing.
    Hand-worked in the hand map's rooms: a red bag on the floor at (2, 1) and a blue one on a table at (3, 1), both
    recognisable at keyframe 1: the floor job asks E0001 alone, the other job E0001 and E0002. Regression caught: a
    floor robot asked where it last saw a thing on a shelf, or the table's things dropped for the helper."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (2.0, 1.0, 0.0))], size=BAG_SIZE)
    s.item("E0002", "bag", {"color": "blue"}, [(0, None, (3.0, 1.0, TABLE_Z))], size=BAG_SIZE)
    for eid in ("E0001", "E0002"):
        s.see(1, eid, 1500)
    assert [c.entities[0] for c in pools(fk.world(s, job=job_of(JOB, True)))[0]] == ["E0001"]
    assert [c.entities[0] for c in pools(fk.world(s, job=job_of(JOB, False)))[0]] == ["E0001", "E0002"]
