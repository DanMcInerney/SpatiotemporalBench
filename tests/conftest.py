"""Shared helpers. Everything here is pure and cached per process, so tests stay parallel-safe."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from functools import cache

import pytest

from builders import keyframe, start_status
from stbench import job_registry
from stbench.generate import build_world, to_timeline
from stbench.mapspec import MapSpec, load_map
from stbench.robots import ROBOTS
from stbench.timeline import Ego, EpisodeInfo, Task, TaskPlan, Timeline

SCRIPT_START = "2026-03-02T08:00:00"
FRAME_DT_S = 1.0  # the keyframes of a hand-made script: the robots' `min_dt_s`


@cache
def apartment() -> MapSpec:
    return load_map("apartment_a")


@cache
def warehouse() -> MapSpec:
    return load_map("warehouse_a")


@cache
def _world(map_name: str, seed: int):
    """The map's initial world, cached per worker process: `script_on_map` only reads it."""
    return build_world(load_map(map_name), seed)


def script_on_map(map_name: str, seed: int, robot: str, poses, *, plan: TaskPlan | None = None) -> Timeline:
    """A one-task script on `map_name`: `generate.build_world` (the seed's items and doors) plus hand-made keyframes at
    the base poses `(x, y, z, yaw)`, 1.0 s apart, the camera at the robot's default pitch and derived from the base by
    `robots.camera_pose`, every keyframe in the robot's start set (each range at its upper bound). It is valid under
    `validate` when the poses are; `plan` defaults to the job's first task kind (`hand` for a job not written yet) with
    the room the first pose stands in (the region's name; none when it stands in no room, as in a street). The
    timeline is the caller's to edit: its entities' attributes are copies of the cached world's."""
    m, r = load_map(map_name), ROBOTS[robot]
    # a hand-built script is stamped with the vacuum's job (its tests read `vacuum/1`'s rules) or, for another robot, an
    # unregistered hand job, so `validate` applies that robot's generic rules only (a written job's checks refuse a hand
    # plan: the drone's needs its address and zone)
    job_id = f"{robot}/1" if robot == "vacuum" else f"{robot}/hand"
    if plan is None:
        room = m.region_of(poses[0][:2])
        params = {} if room is None else {"room": m.region(room).name}
        try:
            plan = TaskPlan(job_registry.get(job_id).task_kinds[0], params)
        except KeyError:
            plan = TaskPlan("hand", params)
    phase, status = r.status.start["phase"], start_status(robot)
    keyframes = [keyframe(robot, k, k * FRAME_DT_S, (x, y, z), yaw, task=0, phase=phase, status=status)
                 for k, (x, y, z, yaw) in enumerate(poses)]
    info = EpisodeInfo(m.name, seed, robot, job_id, SCRIPT_START)
    tl = to_timeline(m, _world(map_name, seed), Ego(keyframes, [Task(0, 0, len(keyframes), plan, ())], []), info)
    tl.entities = [replace(e, attributes=deepcopy(e.attributes)) for e in tl.entities]  # not the cached world's dicts
    return tl


@pytest.fixture(scope="session")
def apt() -> MapSpec:
    return apartment()


@pytest.fixture(scope="session")
def wh() -> MapSpec:
    return warehouse()


@pytest.fixture(autouse=True, scope="session")
def no_codex_home(tmp_path_factory):
    """No test, and no fixture of any scope, sees the developer's own Codex home: `Codex.prepare` refuses a home that
    holds a personal `AGENTS.md`, so a machine that has one would fail every test of the route. Session scope, because
    pytest sets up a module-scoped fixture (`test_codex_system.case` runs a whole case) before a function-scoped
    autouse one, which would leave it the real `~/.codex`. A test that needs a home sets its own."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CODEX_HOME", str(tmp_path_factory.getbasetemp() / "no-codex-home"))
        yield


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Every `docker` test joins one xdist group (`--dist loadgroup`, pyproject.toml), so under `-n` they all run in one
    worker, one after the other: they time freezes, slots and container starts on one shared Docker daemon, which
    concurrent workers (and their containers) would load unevenly. Everything else spreads over the workers. `tryfirst`:
    xdist reads the group mark when it names the node ids, which must happen after this adds it.

    The tests of a group (a module's scripts built once, in one worker) run first, the biggest group first: xdist hands
    out whole groups and single tests in collection order, so a group of a hundred seconds that comes last in the
    collection is a hundred seconds of the run in which the other workers have nothing left to do."""
    for item in items:
        if item.get_closest_marker("docker"):
            item.add_marker(pytest.mark.xdist_group("docker"))
    named = {item: _group_of(item) for item in items}
    size = Counter(name for name in named.values() if name is not None)
    items.sort(key=lambda item: (0, -size[named[item]]) if named[item] is not None else (1, 0))   # a stable sort


def _group_of(item) -> str | None:
    """The name of the xdist group the test is marked into, or None."""
    mark = item.get_closest_marker("xdist_group")
    return None if mark is None else str(mark.kwargs.get("name", mark.args[0] if mark.args else None))


def pytest_runtest_setup(item):
    """Skip a `docker` test when Docker, its daemon or the sandbox image is absent (never pull). Lazy and
    cached per process, so only a worker that is about to run a container test probes Docker."""
    if item.get_closest_marker("docker"):
        from stbench import sandbox

        if reason := sandbox.problem(sandbox.DOCKER, sandbox.IMAGE):
            pytest.skip(f"the Docker sandbox is unavailable: {reason}")
