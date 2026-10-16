"""`map_reader` in the gates (FD-task-questions review §6, item 3; VISION rule 5): the declared reference of a structure
family is no shortcut and no rival of `separation`, `other_case` is excused with it on every layout, and everything
else stays gated. Hand rows of a synthetic family whose oracle rows name `map_reader` as their reference; the layouts
and systems as in `tests/test_gates.py` (four scripts of four questions, the last unknown: `all_unknown` 0.25)."""

from __future__ import annotations

import pytest

from stbench.gates import MAP_READER, ORACLE, OTHER, PATH_LOG, SHORTCUTS, evaluate, failing
from stbench.map_files import HELD_OUT
from test_gates import N_BOOT, gates_of, rows, system_set

FAMILY = "structure_here"
NOW = {"scopes": ["now"] * 4, "reference": MAP_READER}   # four `now` questions whose reference is `map_reader`


def structure_set(reader=(1.0,) * 4, **options):
    """`system_set` of a family of `now` rows that name `map_reader`, with `map_reader`'s rows scoring `reader` (a
    layout memoriser: it answers, whatever the answer scored)."""
    options = {**NOW, **options}
    keep = ("scopes", "reference", "layout", "job")
    own = rows(FAMILY, list(reader), **{k: v for k, v in options.items() if k in keep})
    for r in own:
        r["outcome"] = "partial"
    return system_set(FAMILY, **options) | {MAP_READER: own}


@pytest.mark.parametrize("layout", ["apartment_a", sorted(HELD_OUT)[0]])
def test_the_map_reader_and_other_case_are_left_out_of_the_shortcut_gate_and_separation(layout):
    """Contract: on a group whose oracle rows all name `map_reader`, a map reader that scores the family 1.0 and an
    `other_case` that does too fail nothing, on a public layout and a held-out one alike: both are out of the shortcut
    gate and out of `separation`, whose best rival is the gated `blind` at 0.30 (a gap of 0.7). The group still
    reports the reference's score, 1.0. Regression caught: a structure family failed by its own reference, or by
    `other_case` on a public layout (where it is a gated shortcut for any other family)."""
    g = gates_of(structure_set(layout=layout, **{OTHER: [1.0] * 4}))
    assert g["shortcuts"]["pass"] is True and g["separation"]["pass"] is True
    assert g["separation"]["best"] == "blind" and g["separation"]["value"] == pytest.approx(0.7)
    assert MAP_READER not in g["shortcuts"]["scores"] and OTHER not in g["shortcuts"]["scores"]
    report = evaluate(structure_set(layout=layout, **{OTHER: [1.0] * 4}), n_boot=N_BOOT)
    assert report["rows"][0]["scores"][MAP_READER]["mean"] == pytest.approx(1.0) and list(failing(report)) == []


def test_path_log_blind_and_the_form_stay_gated_on_a_structure_family():
    """Contract: only the reference and `other_case` are excused. `path_log` (which answers a structure family from the
    robot's own log), `blind`, `questionnaire_blind` and `form` at 0.40, 0.15 over the unknown share, each fail
    `shortcuts` and are named, and are the rival `separation` measures against. Regression caught: a structure family
    that escapes the own-log gate, or one that excuses every shortcut."""
    for shortcut in (PATH_LOG, "blind", "questionnaire_blind", "form"):
        g = gates_of(structure_set(**{shortcut: [0.4] * 4}))
        assert g["shortcuts"]["pass"] is False and g["shortcuts"]["worst"] == shortcut
        assert g["separation"]["best"] == shortcut


def test_a_family_that_does_not_name_the_map_reader_keeps_other_case_gated_on_a_public_layout():
    """Contract: the excuse is read from the rows. A family with no reference, or one where only some oracle rows name
    `map_reader`, has `other_case` at 1.0 gated on a public layout (the shortcut gate fails and names it) and
    `map_reader` read as nothing: not among the shortcuts, not listed as missing. Regression caught: `map_reader`
    excusing `other_case` everywhere, or one tagged row excusing a whole group."""
    plain = structure_set(reference=None, **{OTHER: [1.0] * 4})
    (row,) = evaluate(plain, n_boot=N_BOOT)["rows"]
    assert row["gates"]["shortcuts"]["worst"] == OTHER and row["gates"]["shortcuts"]["pass"] is False
    assert row["missing"] == [] and MAP_READER not in row["gates"]["shortcuts"]["scores"]
    some = structure_set(**{OTHER: [1.0] * 4})
    some[ORACLE][0]["reference"] = None
    assert gates_of(some)["shortcuts"]["pass"] is False


def test_a_structure_family_with_no_rows_for_the_reference_lists_it_missing():
    """Contract: the group that names `map_reader` and holds no rows for it lists it in `missing` (a check that did not
    run it shows it), while the gates still pass on what they read: it is no gate. Regression caught: a reference
    silently skipped when its run is missing."""
    without = system_set(FAMILY, **NOW)
    (row,) = evaluate(without, n_boot=N_BOOT)["rows"]
    assert row["missing"] == [MAP_READER] and row["gates"]["shortcuts"]["pass"] is True
    assert MAP_READER not in SHORTCUTS


def test_memory_needed_still_reads_the_current_view_on_the_memory_rows_of_a_structure_family():
    """Contract: on `memory` rows `current_view` and `current_view_prior` (without a pose key on this family) are held
    to the memory margin as for any family. Hand rows: four `memory` questions with `current_view_prior` at 1.0 against
    the unknown share 0.25 fail `memory_needed`, 0.75 over the allowed 0.05. Regression caught: the structure family
    exempt from the check that its rows need memory."""
    g = gates_of(structure_set(scopes=["memory"] * 4, **{"current_view_prior": [1.0] * 4}))
    assert g["memory_needed"]["pass"] is False and g["memory_needed"]["worst"] == "current_view_prior"
