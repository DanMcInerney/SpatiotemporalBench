"""The hooks M2's world lanes fill, with the stubs M2-0b merged: the order `generate_script` calls them in, the callback
a job's `script` gets, and the person kinds `validate` falls back to. The stubs change nothing, so these tests
spy on the hooks (monkeypatched on the module that owns them) and run no world process."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import SEED_KEY, catalog, clutter, generate, job_registry, people, validate_ego, world_processes
from stbench.catalog import MoverKind
from stbench.changes import World
from stbench.mapspec import MapSpec
from stbench.seeding import make_rng
from stbench.timeline import Timeline
from stbench.timeline_ego import Ego, TaskPlan

SEED = 101
SENTINEL_CALLBACK = object()  # what the spied `gaps` returns: a job must hand on exactly this
SENTINEL_TIMELINE = object()  # what the spied `people.add` returns: `generate_script` must return exactly this


def hooked_script(monkeypatch) -> tuple[list, object]:
    """`generate_script("apartment_a", 101, "vacuum", "vacuum/1", 1)` over spies for `clutter.place`, `light_starts`,
    the job's `plan` and `script`, `world_processes.gaps` and `people.add`: the calls in order, as (name, details), and
    what `generate_script` returned. The job is `vacuum/1`'s with its plan and script replaced by recorders (the script
    writes an empty ego), so no path is planned."""
    calls: list = []
    real = job_registry.get("vacuum/1")
    ego = Ego([], [], [])

    def plan(m, seed, n_tasks):
        calls.append(("plan", (m.name, seed, n_tasks)))
        return [TaskPlan("clean_room", {"room": "bedroom"})]

    def script(m, w, plans, seed, **kwargs):
        calls.append(("script", kwargs))
        return ego

    def place(w, m, rng):
        calls.append(("clutter", float(rng.random())))

    def light_starts(w, m, rng, anchors=frozenset()):
        calls.append(("light_start", (float(rng.random()), anchors)))

    def gaps(w, m, seed):
        calls.append(("gaps", (w, m, seed)))
        return SENTINEL_CALLBACK

    def add(tl, m, rng):
        calls.append(("people", (tl, m, float(rng.random()))))
        return SENTINEL_TIMELINE

    monkeypatch.setattr(job_registry, "get", lambda job_id: replace(real, plan=plan, script=script))
    monkeypatch.setattr(clutter, "place", place)
    monkeypatch.setattr(generate, "light_starts", light_starts)
    monkeypatch.setattr(world_processes, "gaps", gaps)
    monkeypatch.setattr(people, "add", add)
    return calls, generate.generate_script("apartment_a", SEED, "vacuum", "vacuum/1", 1)


def first_draw(*key) -> float:
    return float(make_rng("apartment_a", SEED, SEED_KEY, *key).random())


def test_generate_calls_the_hooks_in_order(monkeypatch):
    """Contract (plan §5.1, M2-0b §3): `generate_script` calls clutter, the light starts, the job's plan, the world's
    gaps (as the job's script is called), the job's script and the people overlay in that order, each from a key of its
    own (`SEED_KEY` with "clutter", "light_start", "people"), hands `people.add` the finished timeline (the job's own
    ego in it) and the map, and returns what `people.add` returned. Regression caught: a hook called before the path
    exists, or with a stream another step also draws from (which would redraw that step's world), or an overlay whose
    result is dropped."""
    calls, result = hooked_script(monkeypatch)
    assert [name for name, _ in calls] == ["clutter", "light_start", "plan", "gaps", "script", "people"]
    by_name = dict(calls)
    assert by_name["clutter"] == first_draw("clutter")
    assert by_name["light_start"] == (first_draw("light_start"), frozenset({"dock"}))
    assert by_name["plan"] == ("apartment_a", SEED, 1)
    w, m, seed = by_name["gaps"]
    assert isinstance(w, World) and isinstance(m, MapSpec) and (m.name, seed) == ("apartment_a", SEED)
    tl, m_people, draw = by_name["people"]
    assert isinstance(tl, Timeline) and tl.ego.keyframes == [] and tl.episode.job == "vacuum/1"
    assert m_people is m and draw == first_draw("people")
    assert result is SENTINEL_TIMELINE


def test_a_job_receives_the_between_callback(monkeypatch):
    """Contract (M2-0b §3): the job's `script` gets, as the keyword `between`, exactly what `world_processes.gaps(w, m,
    seed)` returned. Regression caught: a job left without the gaps' callback, so the world's changes never land in
    its gaps."""
    calls, _ = hooked_script(monkeypatch)
    assert dict(calls)["script"] == {"between": SENTINEL_CALLBACK}


def test_mover_kind_falls_back_to_people(monkeypatch):
    """Contract (M2-0b §3): when no catalog table has the kind, `validate`'s mover lookup returns `people.person_kind`'s
    record for it and propagates its KeyError, and a kind a catalog has never reaches `people`; `catalog.mover_kind`
    itself knows no person (the catalog, which neutral modules read, never imports the truth side's `people`).
    Regression caught: `validate` reporting every person as `unknown_mover_kind`, or a person kind shadowing a
    catalog's mover kind."""
    person = MoverKind(name="person", size=(0.5, 0.5, 1.7), colors=("grey",), speed=(0.8, 1.4), dwell=(3, 20),
                       same_room=0.5, camera_m=0.7)
    asked: list[str] = []

    def person_kind(name: str) -> MoverKind:
        asked.append(name)
        if name != "person":
            raise KeyError(name)
        return person

    monkeypatch.setattr(people, "person_kind", person_kind)
    assert validate_ego._kind_of("person") is person
    assert validate_ego._kind_of("robot_vacuum") is catalog.MOVER_KINDS["robot_vacuum"]
    with pytest.raises(KeyError, match="nothing"):
        validate_ego._kind_of("nothing")
    assert asked == ["person", "nothing"]
    with pytest.raises(KeyError):
        catalog.mover_kind("person")
