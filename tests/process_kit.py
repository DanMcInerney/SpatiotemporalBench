"""What the world-process tests share: the apartment's seeds 101-103 as `generate_script` builds one before it plans a
path (built, cluttered, the light furniture started), cheap copies of them to change, and the floor guard turned off for
a test of a law, which would otherwise build the planners' free space (25 ms a robot) for every thing it moves.

`realism_kit` caches its worlds for every test of a worker process, so a world must never be built while a test has the
guard off (the clutter would be placed without it): `fresh` and `real_guard` build under the real one."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

from realism_kit import generated_world
from stbench import dressing_robots
from stbench.changes import Carried, World

SEEDS = (101, 102, 103)
HOUR = 3600.0
REAL_HOME = dressing_robots.PLANNED_ROBOTS["home"]  # the robots planned on the home, as imported (nothing patched)


@contextmanager
def real_guard() -> Iterator[None]:
    """The planned robots of the home as the generator has them, whatever a test has patched: for building a world that
    a cache keeps."""
    patched = dressing_robots.PLANNED_ROBOTS["home"]
    dressing_robots.PLANNED_ROBOTS["home"] = REAL_HOME
    try:
        yield
    finally:
        dressing_robots.PLANNED_ROBOTS["home"] = patched


def copy_world(w: World) -> World:
    """A copy of a world that a process may change: its own ledger, dicts and carried records (the rest, which no
    process writes, is shared)."""
    return replace(w, ledger=copy.deepcopy(w.ledger), carried={k: Carried(c.entity, c.spec, c.spot)
                                                              for k, c in w.carried.items()},
                   clutter=dict(w.clutter), light_rects=dict(w.light_rects))


def fresh(seed: int) -> World:
    """A copy of `generated_world(seed)` that a test may change."""
    with real_guard():
        return copy_world(generated_world(seed))


def guard_off(monkeypatch) -> None:
    """No robot is planned on the home's maps, so `FloorGuard` allows everything and builds nothing."""
    monkeypatch.setitem(dressing_robots.PLANNED_ROBOTS, "home", ())
