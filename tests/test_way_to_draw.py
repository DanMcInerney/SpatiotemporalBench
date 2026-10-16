"""The look draw of `way_to` beside `ego_mem` in the helper's slot 2 (`questionnaire.looks`,
`questions_looks.slot_family`): which family a point draws is the slot table and a hash of the case, the point and the
slot, never the world."""

from __future__ import annotations

from looks_kit import hand_points, look_job, world
from space_kit import vacuum_run
from stbench import job_registry, questionnaire
from stbench.seeding import stable_hash, stable_hex

SHARES = ((("ego_mem", 65), ("way_to", 35)),)   # the helper's slot 2


def test_a_point_draws_ego_mem_on_65_and_way_to_on_35_from_the_hash_alone(monkeypatch):
    """Contract (FD-0.11 §3.4.2, "the draw reads no world"): in a slot of `ego_mem` 65 and `way_to` 35 a look point asks
    `ego_mem` where `stable_hash(case id, point, "slot0") % 100` is under 65 and `way_to` where it is 65 to 99, and the
    family's gold function is called at no other point. Hand-worked over thirty case ids on the vacuum's task with both
    golds stubbed to a dead band (None): the recorded calls are exactly the points the hash assigns, so a point drawn
    by one family is never offered to the other and the draw cannot depend on what is in view. Regression caught: a
    gold function called at every point (a draw that reads the world), `way_to` taking `ego_mem`'s share, or a
    hash keyed on the family."""
    seen: dict[str, list] = {"ego_mem": [], "way_to": []}
    for family in seen:
        monkeypatch.setitem(questionnaire.GOLDS, family, lambda w, k, family=family: seen[family].append(
            (w.rec.case, k)))
    job = look_job("ego_mem", "way_to", look_shares=SHARES, look_phases={"way_to": frozenset({"cleaning"})})
    cids = [stable_hex("way-draw", i, n=16) for i in range(30)]
    for cid in cids:
        assert questionnaire.looks(world(vacuum_run(), cid, job=job), []) == []
    for family, below in (("ego_mem", True), ("way_to", False)):
        want = [(cid, k) for cid in cids for k in hand_points(cid)
                if (stable_hash(cid, k, "slot0") % 100 < 65) == below]
        assert seen[family] == want and want, family


def test_the_helper_draws_way_to_in_the_slot_ego_mem_leaves_empty():
    """Contract: `way_to` takes the 35 percent that slot 2 drew nothing on (ego_mem alone sat at 65 in 0.10.0), so
    `ego_mem` keeps its share and the slot is whole (100). By hand from the job file. Regression caught: `ego_mem`'s
    share cut to make room, or a slot over 100, which the registry refuses."""
    helper = job_registry.get("helper/1")
    assert helper.look_slots[1] == (("ego_mem", 65), ("way_to", 35))
    assert sum(percent for _, percent in helper.look_slots[1]) == 100
