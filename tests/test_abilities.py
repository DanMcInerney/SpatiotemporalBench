"""Which ability each family measures (`abilities.py`, 0.6.0)."""

from __future__ import annotations

from stbench.abilities import DIAGNOSTIC, FAMILY_ABILITY, ability_of
from stbench.records import BALANCE_POOLS


def test_the_table_is_the_designs_abilities_by_hand():
    """Contract (FD-0.6.0 §0, 0.8.0, 0.11.0): distance is `ego_now`; memory is `ego_mem` and `object_where`, and since
    0.11.0 `way_to`; change is `object_was` and `same_place`; relevance is `task_things`; clearance is `clear_path`,
    `closest_person` and `person_mem`, since 0.10.0 `closest_approach`, `drop_zone_people` and `tangle_ahead`
    (FD-questions-0.10 §2) and since 0.11.0 `open_heading` (FD-0.11 §3.3: counted, no longer a diagnostic) and
    `approach_mem` (FD-0.11-hard §3.2: `closest_approach` for the person last seen, a clearance decision);
    prediction (`mover_next`) is the one diagnostic, the only ability kept out of the means.
    Hand-written. Regression caught: a family filed under the wrong ability (the weighted means would then weigh it
    as another's), a heading back in a diagnostic of its own, or a diagnostic joining the headline."""
    assert FAMILY_ABILITY == {"ego_now": "distance", "ego_mem": "memory", "object_where": "memory",
                              "object_was": "change", "same_place": "change", "task_things": "relevance",
                              "mover_next": "prediction", "clear_path": "clearance", "closest_person": "clearance",
                              "person_mem": "clearance", "open_heading": "clearance", "way_to": "memory",
                              "closest_approach": "clearance", "drop_zone_people": "clearance",
                              "tangle_ahead": "clearance", "approach_mem": "clearance"}
    assert {"prediction"} == DIAGNOSTIC
    assert {ability_of(f) for f in FAMILY_ABILITY} - DIAGNOSTIC == {"distance", "memory", "change", "relevance",
                                                                    "clearance"}


def test_a_family_the_table_does_not_list_is_an_ability_of_its_own():
    """Contract: `ability_of` of an unlisted family is the family itself, so the rows of a run before 0.6.0
    (`out_of_view`, `zone_clear`, `path_ahead`) stay scorable and each counts as its own ability. Regression caught: a
    KeyError on an older row, or all unlisted families pooled as one ability."""
    assert [ability_of(f) for f in ("out_of_view", "zone_clear", "path_ahead")] == ["out_of_view", "zone_clear",
                                                                                    "path_ahead"]
    assert ability_of("object_where") == "memory"


def test_every_balanced_family_has_an_ability():
    """Contract: each family with balance pools (`records.BALANCE_POOLS`) is in the table, so the balance check and the
    ability columns speak of the same families. Regression caught: a classed family that no column reports."""
    assert set(BALANCE_POOLS) <= set(FAMILY_ABILITY)


def test_the_three_clearance_families_are_one_ability_of_their_own():
    """Contract (FD-task-questions review §5, plan L0): `clear_path`, `closest_person` and `person_mem` are all
    `clearance`, an ability counted in the means (not a diagnostic) and not folded into `memory` or `distance`; the
    ability is named by the table, not by the family. Regression caught: a new family scored as an ability of its own
    (three one-family abilities, each weighing a full cell), or clearance filed under an old ability."""
    assert {ability_of(f) for f in ("clear_path", "closest_person", "person_mem")} == {"clearance"}
    assert "clearance" not in DIAGNOSTIC
    assert ability_of("ego_mem") == "memory" and ability_of("ego_now") == "distance"
