"""The four kinds of question (`abilities.FAMILY_KIND`, 0.9.0): the vision's harm, help, situation and reuse, a split of
the reports beside the abilities."""

from __future__ import annotations

from stbench.abilities import DIAGNOSTIC, FAMILY_ABILITY, FAMILY_KIND, KINDS, ability_of, kind_of
from stbench.job_registry import all_jobs


def test_the_kinds_are_the_designs_by_hand():
    """Contract (brief §5; FD-questions-0.10 §2): harm is `clear_path`, `closest_person` and `person_mem`, and since
    0.10.0 `closest_approach`, `drop_zone_people` and `tangle_ahead`, and since 0.11.0 `approach_mem`; help is
    `open_heading` and, since 0.11.0, `way_to`; situation is `ego_now` and `ego_mem`; reuse is `object_where` and
    `task_things`. Hand-written. Regression caught: a family filed under the wrong kind, so a kind's split would score
    another question."""
    assert KINDS == ("harm", "help", "situation", "reuse")
    by_kind = {kind: {f for f, k in FAMILY_KIND.items() if k == kind} for kind in KINDS}
    assert {f for f in by_kind["harm"]} == {"clear_path", "closest_person", "person_mem", "closest_approach",
                                            "drop_zone_people", "tangle_ahead", "approach_mem"}
    assert by_kind["help"] == {"open_heading", "way_to"}
    assert {"ego_now", "ego_mem"} <= by_kind["situation"] and {"object_where", "task_things"} <= by_kind["reuse"]


def test_every_family_of_the_ability_table_and_every_family_a_job_asks_has_a_kind():
    """Contract: a family that has an ability has a kind, and so does each family a shipped job asks at a look point or
    at its end, so the kind split holds every scored row. Regression caught: a family missing from the split, whose rows
    would fall out of it (or be reported as a kind of their own)."""
    assert set(FAMILY_ABILITY) <= set(FAMILY_KIND)
    asked = {f for job in all_jobs() for f in (*job.looks, *(job.ends or ()))}
    assert asked <= set(FAMILY_KIND), asked - set(FAMILY_KIND)
    assert set(FAMILY_KIND.values()) <= set(KINDS)


def test_open_heading_counts_in_clearance_and_is_the_help_kind_and_way_to_is_memory_and_help():
    """Contract (FD-0.11 §3.3, §3.4.2): `open_heading`, graded since 0.11.0, is clearance, counted in the means, and is
    still the help kind; `way_to`, the route to a remembered thing, is memory and help; `clear_path` stays clearance and
    harm. The ability and the kind are two splits of one family. Regression caught: `open_heading` back in a diagnostic
    ability of its own (the help kind would leave the headline), `way_to` filed as an ability of its own (a one-family
    cell weighing a full cell), or the help kind dropped with the ability."""
    assert ability_of("open_heading") == "clearance" and "clearance" not in DIAGNOSTIC and "heading" not in DIAGNOSTIC
    assert ability_of("way_to") == "memory" and "memory" not in DIAGNOSTIC
    assert ability_of("clear_path") == "clearance"
    assert kind_of("open_heading") == "help" and kind_of("way_to") == "help" and kind_of("clear_path") == "harm"


def test_an_unlisted_family_is_a_kind_of_its_own():
    """Contract: `kind_of` of a family the table does not list is the family itself (as `ability_of`), so the rows of an
    older run stay scorable. Regression caught: a KeyError on an older row."""
    assert [kind_of(f) for f in ("out_of_view", "zone_clear")] == ["out_of_view", "zone_clear"]
