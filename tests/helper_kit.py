"""What the helper's job tests share: scripts of `helper/1` on `apartment_a` with the real clutter, cached per worker
process, and a hand world to put a reach pose in.

`generated(seed, n)` is the seed's `n`-task script with a copy of the world as each task was planned on it (after the
gap callback), so a test can ask `robot_obstacles` of the world a task saw. The two helper test modules share one xdist
group, so one worker builds each script once. Coordinates are the map's (`maps/apartment_a.json`): the dock at
(4.71, 5.05), 0.12 m deep facing east, so the park pose is (5.40, 5.05) facing the front door, heading -90 degrees.
"""

from __future__ import annotations

import copy
from functools import cache
from unittest import mock

import numpy as np

from stbench import job_helper
from stbench.generate import build_world, generate_script
from stbench.job_helper_reach import free_space_for
from stbench.mapspec import load_map
from stbench.robots import ROBOTS

MAP, ROBOT, JOB = "apartment_a", "helper", "helper/1"
PARK, PARK_YAW = (5.4, 5.05), -1.570796  # by hand: 4.71 + 0.06 + 0.63, the dock's axis; the door is due south
HALL, LIVING = "R1", "R2"  # the regions the qualified surfaces stand in
SURFACES = {HALL: ("F20", "F21"), LIVING: ("F10", "F11", "F12")}  # the helper's reachable surfaces, by hand


@cache
def apartment():
    return load_map(MAP)


def robot():
    return ROBOTS[ROBOT]


@cache
def generated(seed: int, n: int = 5) -> tuple[list, object]:
    """([the world as each task was planned on it], the `n`-task script of the seed), the real clutter and gaps and the
    resident. Cached per worker process; a test mutates neither."""
    worlds: list = []
    real = job_helper.SPEC

    def script(m, w, plans, seed_, between=None, **kwargs):
        def snapping(lo, hi, keep_open):
            if between is not None:
                between(lo, hi, keep_open)
            worlds.append(copy.deepcopy(w))

        return real.script(m, w, plans, seed_, between=snapping, **kwargs)

    from dataclasses import replace

    with mock.patch.object(job_helper, "SPEC", replace(real, script=script)):
        tl = generate_script(MAP, seed, ROBOT, JOB, n)
    return worlds, tl


def bare_free_space(rooms=(HALL, LIVING)):
    """The helper's free space in the map's own world (no clutter, nothing carried) for a task between `rooms`."""
    w = build_world(apartment(), 101)
    w.carried.clear()
    return w, free_space_for(apartment(), w, robot(), frozenset(rooms))


def reachable_mask(free) -> np.ndarray:
    return free.reachable_from(PARK)
