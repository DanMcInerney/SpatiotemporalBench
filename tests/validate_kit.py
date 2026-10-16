"""The helpers the validate test files share (`test_validate.py`, `test_validate_ego.py`, `test_validate_body.py`): the
codes a timeline's problems carry, a timeline with other keyframes, one with a mover, and a fixture that turns the
vacuum's own job checks off for a test of the generic rules."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import job_vacuum
from stbench.catalog import item_size
from stbench.geometry import quat_z
from stbench.timeline import Ego, Entity, Placement
from stbench.validate import validate


@pytest.fixture
def generic_rules_only(monkeypatch):
    """`vacuum/1`'s own checks off (`job_vacuum._checks`, tested in `test_job_vacuum.py`), for the tests that exercise
    `validate`'s generic rules, the CLI or the ledger on a hand-made vacuum script: such a script is not a whole task of
    the job (it never leaves the dock's phase and starts anywhere), which the job's checks rightly refuse."""
    monkeypatch.setattr(job_vacuum, "SPEC", replace(job_vacuum.SPEC, checks=lambda tl, m: []))


def codes(tl, m) -> set[str]:
    """The distinct problem codes `validate` finds in `tl` on map `m`."""
    return {p.code for p in validate(tl, m)}


def with_keyframes(tl, kfs, tasks=None):
    """`tl` with these keyframes (and tasks, the timeline's own when None), the decisions kept."""
    tl.ego = Ego(list(kfs), tl.ego.tasks if tasks is None else tasks, tl.ego.decisions)
    return tl


def with_mover(tl, pos, *, eid="E0006", t0=0.0, t1=None, yaw=0.4, kind="robot_vacuum"):
    """`tl` plus a mover (by default a robot vacuum, 0.34 x 0.34 m, or a kind of any catalog at its box) with one
    placement at `pos` over [t0, t1)."""
    if eid not in {e.id for e in tl.entities}:
        tl.entities.append(Entity(eid, kind, {"color": "white"}, "mover", item_size(kind)))
    tl.placements.append(Placement(eid, t0, t1, (pos[0], pos[1], 0.0), quat_z(yaw), None))
    return tl
