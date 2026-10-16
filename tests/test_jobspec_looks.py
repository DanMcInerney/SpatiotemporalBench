"""The `JobSpec` fields of the 0.8.0 slot table and clearance questions (`job_registry`, lane L0) and the 0.10.0
priority decisions of the look draw: their defaults reproduce 0.7.0 and a table or a decision list that cannot be
drawn is refused at construction, naming the job."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest

from stbench.job_registry import get

TWO = ("ego_now", "ego_mem")
BARE = dict(look_shares=None, look_unknown_keep={}, clear_phases=frozenset(), look_phases={}, fly_floor_m=None,
            one_person=False, look_decisions=())


def job(**over):
    """`vacuum/2` as the job `t/1` with these fields replaced, on a bare 0.7.0 draw: the shipped job's own table (it
    declares one since 0.8.0) is cleared, so the fields under test start at their defaults."""
    return replace(get("vacuum/2"), id="t/1", **{**BARE, **over})


def test_the_new_fields_default_to_0_7_0s_draw():
    """Contract (plan L0): a `JobSpec` built without the new fields has `look_shares` None (so `look_slots` is one slot
    a family of `looks` at 100), no unknown keep, no clear phases, no fly floor and `one_person` False. Hand-worked:
    a job asking `ego_now` and `ego_mem` has the slots ((ego_now, 100),), ((ego_mem, 100),); one asking none has none.
    Regression caught: a field added without a default (every job built by keyword breaks), or a default that draws
    differently from 0.7.0."""
    bare = job(looks=TWO)
    assert bare.look_shares is None
    assert bare.look_slots == ((("ego_now", 100),), (("ego_mem", 100),))
    assert dict(bare.look_unknown_keep) == {} and bare.clear_phases == frozenset()
    assert bare.fly_floor_m is None and bare.one_person is False
    assert job(looks=()).look_slots == ()


def test_a_table_that_sums_to_at_most_100_is_accepted_and_kept_as_declared():
    """Contract (plan L0): a slot may sum to 100 or to less (the rest asks nothing), a family may share a slot, and the
    table is read back as declared. Regression caught: a check that refuses a table that fits (a slot that sums to
    exactly 100, or a rest), or one that rewrites it."""
    table = ((("ego_now", 50), ("ego_mem", 50)), (("ego_mem", 30),))
    assert job(looks=TWO, look_shares=table).look_slots == table
    assert job(looks=("ego_now",), look_shares=((("ego_now", 1),),)).look_slots == ((("ego_now", 1),),)


def test_a_three_slot_table_is_accepted_and_kept_as_declared():
    """Contract (FD-questions-0.10 §3): a point asks one question a slot and a table may have three slots (the harm
    families, then the situation and memory families, then the rest). Hand-worked: three slots over `ego_now`,
    `ego_mem` and `clear_path` are read back as declared, and the default table of three looks has three slots at 100.
    Regression caught: a check or a property that assumes two slots, or a table rewritten."""
    three = ("ego_now", "ego_mem", "clear_path")
    table = ((("clear_path", 30),), (("ego_now", 50), ("ego_mem", 20)), (("ego_mem", 30), ("ego_now", 20)))
    assert job(looks=three, look_shares=table).look_slots == table
    assert job(looks=three).look_slots == ((("ego_now", 100),), (("ego_mem", 100),), (("clear_path", 100),))


def test_priority_decisions_default_to_none_and_must_be_decisions_of_the_job():
    """Contract (FD-questions-0.10 §3): `look_decisions` names decision kinds of the job, each once, for a job that asks
    a look family. Hand-worked on `vacuum/2` (decisions `before_room`, `before_doorway`, `room_done`,
    `cleaning_stopped`, `before_backout`): none by default; `room_done` and `before_doorway` are accepted and read back
    in order; `before_crossing` (an AMR kind), a kind twice, and any kind for a job with no look family are refused,
    naming the job. Regression caught: a priority kind that the job never reaches (silently drawing nothing), or a
    list kept for a job that asks no look."""
    assert job(looks=TWO).look_decisions == ()
    both = ("room_done", "before_doorway")
    assert job(looks=TWO, look_decisions=both).look_decisions == both
    with pytest.raises(ValueError, match="t/1.*before_crossing.*no decision of the job"):
        job(looks=TWO, look_decisions=("before_crossing",))
    with pytest.raises(ValueError, match="t/1.*twice"):
        job(looks=TWO, look_decisions=("room_done", "room_done"))
    with pytest.raises(ValueError, match="t/1.*asks no look family"):
        job(looks=(), look_decisions=("room_done",))


def test_each_shipped_job_names_its_priority_decisions():
    """Contract (FD-questions-0.10 §3 table; final review B1): the jobs that ask look families draw these decision
    frames first: the AMR `before_crossing` and `before_station`, the helper `before_grasp` and `before_doorway` and
    not `before_place`, the vacuum `before_doorway` and `room_done`, the drone `before_lower` and `before_descent`. The
    helper's `before_place` is out because a decision frame puts a structure question at the same plan pose, by the
    same furniture, in every script: with `before_place` first, 8 of the carrier gold's 10 `clear_path` rows sat there
    and the helper's evidence gate failed on them (+0.211 on decision frames alone; every gate passed without it).
    Regression caught: a job file losing or mis-spelling its list (the points would fall by hash alone), the helper's
    `before_place` back in the draw (its `clear_path` becoming a channel again), or a decision added to a list without
    the plan."""
    want = {"amr/1": ("before_crossing", "before_station"),
            "helper/1": ("before_grasp", "before_doorway"),
            "vacuum/2": ("before_doorway", "room_done"), "drone/1": ("before_lower", "before_descent")}
    assert {j: get(j).look_decisions for j in want} == want
    assert "before_place" in get("helper/1").decisions and "before_place" not in get("helper/1").look_decisions


def test_look_phases_name_looks_of_the_job_and_phases_of_its_robot():
    """Contract (0.10.0, `JobSpec.look_phases`): a mapping from a look family of the job to the status phases it is
    asked in. Hand-worked on `vacuum/2`'s robot, whose phases are docked, undocking, cleaning, transit, docking and
    paused: `tangle_ahead` in `cleaning` and `transit` is accepted and read back as declared, and the default is no
    phase rule; a family not in `looks`, no phase at all, and a phase the robot lacks (`lower`, a drone's) are refused,
    naming the job. Regression caught: a phase typed apart from the robot's vocabulary (the family would never be
    asked), a rule for a family the job does not ask, or an empty rule that asks nowhere."""
    phases = {"tangle_ahead": frozenset({"cleaning", "transit"})}
    assert dict(job(looks=TWO).look_phases) == {}
    assert dict(job(looks=("tangle_ahead",), look_phases=phases).look_phases) == phases
    with pytest.raises(ValueError, match="t/1.*look_phases for 'tangle_ahead'.*not in its looks"):
        job(looks=TWO, look_phases=phases)
    with pytest.raises(ValueError, match="t/1.*no phase for 'tangle_ahead'"):
        job(looks=("tangle_ahead",), look_phases={"tangle_ahead": frozenset()})
    with pytest.raises(ValueError, match="t/1.*\\['lower'\\].*vacuum.*does not have"):
        job(looks=("tangle_ahead",), look_phases={"tangle_ahead": frozenset({"cleaning", "lower"})})


OVER = "t/1.*more than 100"
BAD = "t/1.*outside 1 to 100"


@pytest.mark.parametrize("shares, message", [
    ((  # 110 in a slot
        (("ego_now", 60), ("ego_mem", 50)),), OVER),
    ((  # 101 in the second slot
        (("ego_now", 100),), (("ego_mem", 99), ("ego_now", 2))), OVER),
    ((  # a family of looks in no slot
        (("ego_now", 100),),), "t/1.*ego_mem.*no slot"),
    ((  # a family that is not in looks
        (("ego_now", 50), ("ghost", 50)), (("ego_mem", 100),)), "t/1.*ghost.*not in its looks"),
    ((  # a zero percent
        (("ego_now", 100), ("ego_mem", 0)),), BAD),
    ((  # a percent over 100
        (("ego_now", 101),), (("ego_mem", 100),)), BAD),
])
def test_a_slot_table_that_cannot_be_drawn_is_refused_naming_the_job(shares, message):
    """Contract (plan L0): percents over 100 in a slot, a family missing from `looks`, a family of `looks` in no slot
    and a percent outside 1 to 100 are refused with a `ValueError` naming the job, at construction. Regression caught: a
    silent mis-declared table (a slot that over-covers and never draws its last family, or a family never asked)."""
    with pytest.raises(ValueError, match=message):
        job(looks=TWO, look_shares=shares)


@pytest.mark.parametrize("keep, message", [
    ({"ego_mem": 50}, "t/1.*ego_mem.*sliced"),               # a sliced family's share is its slice
    ({"ghost": 50}, "t/1.*ghost.*not in its looks"),
    ({"ego_now": 101}, "t/1.*0 to 100"),
    ({"ego_now": -1}, "t/1.*0 to 100"),
])
def test_an_unknown_keep_that_cannot_apply_is_refused_naming_the_job(keep, message):
    """Contract (plan L0): a keep on a `SLICED` family (`ego_mem`), on a family the job does not ask, or outside 0 to
    100 is refused naming the job; `ego_now` at 0 and at 100 and an absent keep are fine. Regression caught: a keep that
    thins a sliced family on top of its slice, or a typo in a family name that thins nothing."""
    with pytest.raises(ValueError, match=message):
        job(looks=TWO, look_unknown_keep=keep)
    assert job(looks=TWO, look_unknown_keep={"ego_now": 0}).look_unknown_keep == {"ego_now": 0}
    assert job(looks=TWO, look_unknown_keep={"ego_now": 100}).look_unknown_keep == {"ego_now": 100}


def test_a_job_module_that_declares_a_keep_imports_first_in_a_fresh_process(tmp_path):
    """Contract (plan L0, the check at registration): the sliced-family check needs the family registry, which imports
    the job registry; a job module that declares a keep and is the first thing imported in a process must not hit an
    import cycle. Hand-worked: a module `job_fake_keep` replacing `vacuum/2`'s looks and keep, imported in a fresh
    interpreter, loads and keeps 50. Regression caught: a module-level import of `questions_text` in `job_registry`."""
    (tmp_path / "job_fake_keep.py").write_text("\n".join([
        "from dataclasses import replace", "from stbench.job_vacuum2 import SPEC as BASE",
        "SPEC = replace(BASE, id='fake/1', looks=('ego_now',), look_shares=None, look_phases={},",
        "               look_unknown_keep={'ego_now': 50})"]) + "\n",
        encoding="utf-8")
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import job_fake_keep; "
              "print(job_fake_keep.SPEC.look_unknown_keep)")
    out = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "{'ego_now': 50}"
