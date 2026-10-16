"""What the four shipped jobs ask (0.6.0: DECISIONS, "0.6.0 declared before lanes", call 1, and "0.6.0's Phase A review
calls", 5-8; 0.8.0: "0.8.0 declared", review §4.4; 0.9.0: the final review, B1; 0.11.0: the vacuum's list): no decision
kind asks a family, each job's look families are asked at look points, and the end families and the `task_things`
predicate are each job's own.
The mappings are written out by hand, so a family silently unasked, asked twice or left unregistered fails here by job
and name."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench import abilities, brief, job_registry, questionnaire
from stbench.questions_text import FAMILIES, LOOKS
from stbench.task_predicate import TaskThings

SHIPPED = ("vacuum/2", "helper/1", "amr/1", "drone/1")
RETIRED = ("zone_clear", "out_of_view", "adjacent")
ENDS = ("object_where", "task_things")
# The ends by job; the drone asks no `task_things` (it has no predicate that scores under the empty constant: DECISIONS,
# "0.6.0's Phase A review calls", 8). The vacuum asked none in 0.9.0 and 0.10.0 (its list was a layout prior of the
# home's recurring item spots, a lead on the binding evidence verdict: DECISIONS, "the evidence verdict ships", 2) and
# asks it again in 0.11.0 as the list of what its brush could catch, limited to the kinds that do not recur at one spot
# (FD-0.11-design 3.4.3).
# `object_was` and `same_place` are asked by no job (DECISIONS, "Before the binding read", 1); their modules and
# family pages stay.
UNASKED = ("object_was", "same_place")
# The Phase A read's keeps (DECISIONS, "Empty keeps set at the Phase A read"): 100(1 - p)/(2p) of each job's
# measured empty share p after `told` and RELAX, so the expected empty share is a third. The AMR's is set again by lane
# R3 for its whole-drive path (22 empty of 41 tasks, p 0.537: 100 x 19 / 44 = 43). The vacuum's (0.11.0) is read from
# the first compile of its list: 28 empty of the 39 tasks of the 32 short and medium cases (p 0.718: 100 x 11 / 56 =
# 19.6).
EMPTY_KEEP = {"amr/1": 43, "helper/1": 26, "vacuum/2": 20}
# The kinds of the vacuum's list written out by hand: the items are cords and cloth, and the three that recur at one
# spot from script to script (`docs/families/tangle_ahead.md`: bath mat 94%, pedestal mat 82%, clothes pile 69% within
# 0.1 m) are don't-care.
CATCH = frozenset({"bath_mat", "cable_bundle", "clothes_pile", "extension_cord", "pedestal_mat", "power_strip",
                   "towel"})
SPOTS = frozenset({"bath_mat", "clothes_pile", "pedestal_mat"})
ITEM_KINDS = {"vacuum/2": CATCH}
DONT_CARE_KINDS = {"vacuum/2": SPOTS}

# 0.8.0 (review §4.4): `closest_person` takes the AMR's `object_where` slot and the drone asks no end question.
EXPECTED_ENDS = {"vacuum/2": ENDS, "helper/1": ENDS, "amr/1": ("task_things", "closest_person"), "drone/1": ()}
# The look families by job, in the order the job lists them (0.8.0, review §4.4; 0.9.0: `open_heading` joins the helper
# and the vacuum, FD-decision-questions §6).
EXPECTED_LOOKS = {"vacuum/2": ("clear_path", "tangle_ahead", "ego_mem", "open_heading"),
                  # person_mem dropped from the helper (0.10.0, DECISIONS 2026-10-14)
                  # way_to joins the helper (0.11, FD-0.11-design §3.4.2), after ego_mem in the job's order
                  # approach_mem joins the helper and the AMR, last in each job's order (0.11, FD-0.11-hard §3.2)
                  "helper/1": ("clear_path", "closest_approach", "ego_mem", "way_to", "ego_now", "open_heading",
                               "approach_mem"),
                  "amr/1": ("closest_approach", "ego_mem", "ego_now", "approach_mem"),
                  "drone/1": ("clear_path", "ego_mem")}   # drop_zone_people dropped (final review S2)
# The during sentence by job: each phrase once, in the order of `DURING_PHRASES`; `person_mem` reuses `ego_mem`'s.
# 0.10.0: the harm families add their phrases, after the older ones.
HEAD = "At some points during the job you will be asked "
NOW, MEM, MOVE, ROOM = ("where a thing you can see is from your camera",
                        "where a thing you saw before is from your camera", "how far the robot could move",
                        "which way the robot has the most room")
NEAR, FLOOR = "how close something would come to the robot", "what lies on the floor near the robot"
WAY = "which way to set off to reach a thing you saw before"   # `way_to`, 0.11
EXPECTED_DURING = {"vacuum/2": f"{HEAD}{MEM}, {MOVE}, {ROOM}, and {FLOOR}.",
                   "helper/1": f"{HEAD}{NOW}, {MEM}, {WAY}, {MOVE}, {ROOM}, and {NEAR}.",
                   "amr/1": f"{HEAD}{NOW}, {MEM}, and {NEAR}.", "drone/1": f"{HEAD}{MEM} and {MOVE}."}
# (phrase, anchor, phases, until, reach_m, band_m, floor, min_height_m) of each job's `task_things`; the light kinds are
# don't-care by their mobility in the gold, so no job lists a kind; every keep is 100 until the Phase A read sets it.
PREDICATES = {
    "vacuum/2": ("on the floor of the rooms you cleaned that your brush could catch (a cable bundle, an extension "
                 "cord, a power strip or a towel; a bath mat, a clothes pile or a pedestal mat counts neither way)",
                 "rooms", {"cleaning"}, None, 0.0, 0.0, True, 0.0),
    "helper/1": ("on the floor within 1.2 m of the way you went to fetch and put down the thing", "path",
                 {"navigating", "grasping", "carrying", "placing"}, "placing", 1.2, 0.4, True, 0.0),
    "amr/1": ("within 2 m of the way you drove", "path",
              {"transit", "approach", "at_station", "undocking", "waiting"}, None, 2.0, 0.3, False, 0.0),
    "drone/1": None,
}


def shipped():
    return {job_id: job_registry.get(job_id) for job_id in SHIPPED}


def test_no_decision_asks_a_family_and_every_job_asks_its_looks():
    """Contract (0.6.0 call 1; 0.8.0 review §4.4): each shipped job maps no family to any of its decision kinds (the
    kinds stay: the coins and responses are on them), asks its own look families at look points (0.10.0: the AMR
    `closest_approach`, `ego_mem` and `ego_now`, the drone `clear_path` and `ego_mem`, the vacuum
    `clear_path`, `tangle_ahead`, `ego_mem` and `open_heading`, the helper all six of `clear_path`, `closest_approach`,
    `ego_mem`, `person_mem`, `ego_now` and `open_heading`; 0.11: the AMR and the helper also `approach_mem`), and
    none of the retired families
    (`zone_clear`, `out_of_view`, `adjacent`) is asked anywhere. Hand-written: five, three, three and three decision
    kinds. Regression caught: a decision kind still asking the retired `zone_clear` or `out_of_view` (a question the
    0.6.0 scorer has no ability for), or a job without looks (it would ask nothing during the job)."""
    kinds = {"vacuum/2": 5, "helper/1": 3, "amr/1": 3, "drone/1": 3}
    for job_id, job in shipped().items():
        assert len(job.decisions) == kinds[job_id] and not any(job.decisions.values()), job_id
        assert job.looks == EXPECTED_LOOKS[job_id], job_id
        asked = {f for fs in job.decisions.values() for f in fs} | set(job.looks) | set(job.ends)
        assert not asked & set(RETIRED), job_id


def test_each_job_asks_its_own_end_families_and_predicate():
    """Contract (0.6.0 call 1, review calls 7 and 8, "Before the binding read" 1): the end families are `object_where`
    and `task_things` on the vacuum (0.11.0: dropped in 0.9.0, back as the list of what the brush could catch, kinds
    named by hand below) and the helper, `task_things` and `closest_person` on the AMR (0.8.0) and none on the drone;
    neither change family (`object_was`, `same_place`) is asked by any job; `task_things` is asked by exactly the jobs
    that declare a predicate, and each predicate is the table's. Regression caught: a family silently unasked (a job
    that lost `task_things`), a change family back in a job's ends (thin groups that fail gates and print BROKEN reach
    lines), the drone asking `task_things` without a predicate (an empty question for ever), a predicate for a job that
    does not ask it, a distance or phase in the text that is not the gold's, or the vacuum's list holding a kind that
    recurs at one spot as an item (the layout prior that dropped it in 0.9.0)."""
    for job_id, job in shipped().items():
        assert job.ends == EXPECTED_ENDS[job_id], job_id
        assert not set(UNASKED) & set(job.ends), job_id
        assert ("task_things" in job.ends) == (PREDICATES[job_id] is not None) == (job.task_things is not None), job_id
        if (hand := PREDICATES[job_id]) is None:
            continue
        pred = job.task_things
        phrase, anchor, phases, until, reach, band, floor, min_height = hand
        assert (pred.phrase, pred.anchor, set(pred.phases), pred.until) == (phrase, anchor, phases, until), job_id
        assert (pred.reach_m, pred.band_m, pred.floor, pred.min_height_m) == (reach, band, floor, min_height), job_id
        assert pred.dont_care_kinds == DONT_CARE_KINDS.get(job_id, frozenset()), job_id
        assert pred.item_kinds == ITEM_KINDS.get(job_id, frozenset()), job_id
        assert pred.empty_keep_percent == EMPTY_KEEP[job_id], job_id
        assert isinstance(pred, TaskThings)


def test_the_vacuum_is_told_how_a_list_is_credited_by_its_end_list_as_well_as_by_tangle_ahead():
    """Contract (0.11.0, FD-0.11-design 3.4.3 and 4.2: the vacuum gets the list paragraph): `brief.asks_lists` is true
    for the vacuum through each of its two lists alone, `task_things` after the job and `tangle_ahead` during it, so
    dropping one by the gates leaves the other's credit rules in the prompt; the list catastrophe is told for
    `tangle_ahead` only (an end list carries none). By hand: the vacuum as registered, then without `tangle_ahead`
    (still told, no hazard), then without `task_things` (still told, the cord hazard), then without both (not told).
    Regression caught: the list paragraph keyed to one family, so a vacuum whose cord list failed a gate would be asked
    for a list with no rule for crediting it, or a catastrophe told for a question that has none."""
    vacuum = job_registry.get("vacuum/2")
    no_tangle = replace(vacuum, looks=tuple(f for f in vacuum.looks if f != "tangle_ahead"), look_shares=None,
                        look_phases={})
    no_end_list = replace(vacuum, ends=("object_where",))
    neither = replace(no_tangle, ends=("object_where",))
    assert [brief.asks_lists(j) for j in (vacuum, no_tangle, no_end_list, neither)] == [True, True, True, False]
    assert [brief.list_hazards(j) for j in (vacuum, no_tangle, no_end_list, neither)] == [
        ("tangle_ahead",), (), ("tangle_ahead",), ()]


def test_every_family_a_job_asks_is_registered_and_has_an_ability():
    """Contract (plan section 3, `abilities`): each family a shipped job asks (its looks and its ends) is in the
    registry, a look family is in `LOOKS` and a during family, an end family is an end family and not a look, and each
    is listed in `abilities.FAMILY_ABILITY` (a family it does not list would be an ability of its own and weigh a
    column alone). Hand-written: distance, memory, relevance and clearance, which since 0.11.0 holds `open_heading`
    (change is asked by no job). Regression caught: a family asked but unregistered (a
    compile failure), or one missing from the ability table (the headline would weigh it alone)."""
    seen = set()
    for job_id, job in shipped().items():
        for family in (*job.looks, *job.ends):
            spec = FAMILIES[family]
            assert (family in LOOKS) == (family in job.looks) and spec.when == ("during" if family in LOOKS else "end")
            assert family in abilities.FAMILY_ABILITY, (job_id, family)
            seen.add(abilities.ability_of(family))
    assert seen == {"distance", "memory", "relevance", "clearance"}


def test_a_job_with_looks_maps_no_decision_family():
    """Contract (A2's review): no registered job that asks at look points maps any family to a decision kind (the
    shipped jobs ask during the job at look points alone), and `questionnaire.check_decisions` accepts each and refuses
    a look family at a decision kind, naming it. Hand-worked on every registered job with looks (the four shipped
    ones), and on the AMR given `ego_now` at `before_station`. Regression caught: a decision family left in a job when
    its looks were added, or a look family mapped to a decision (its gold is a moment, not a candidate)."""
    jobs = [job for job in job_registry.all_jobs() if job.looks]
    assert {job.id for job in jobs} == set(SHIPPED)
    for job in jobs:
        questionnaire.check_decisions(job)
        assert not any(job.decisions.values()), job.id
    amr = job_registry.get("amr/1")
    with pytest.raises(ValueError, match="ego_now"):
        questionnaire.check_decisions(replace(amr, decisions={**amr.decisions, "before_station": ("ego_now",)}))


def test_the_during_sentence_is_the_phrases_of_the_jobs_looks():
    """Contract: a shipped job's brief primes exactly the phrases of its look families (no decision
    family is asked), each once, in the order of `DURING_PHRASES`, and `asks_during` (the Codex prompt's fact) is true
    for it. Regression caught: a
    prompt that speaks of questions during the job for a job that asks none, or a brief that primes a retired
    family."""
    for job_id, job in shipped().items():
        assert brief.asks_during(job) and brief.during_sentence(job) == EXPECTED_DURING[job_id], job_id
