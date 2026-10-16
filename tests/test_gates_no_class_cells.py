"""The evidence verdict has no cell for a declared class (0.6.0 decision): a system that always gives one side's answer
on a fair coin passes every cell."""

from __future__ import annotations

from stbench import check
from stbench.gates import BLIND, FIRST, evaluate
from test_gates import N_BOOT, gates_of
from test_gates_evidence import FAMILY, shortcuts_of, world

CLASSES = ["yes", "moved", "yes", "moved"]   # a question's class by its place in its script: a fair coin


def classed(results: dict) -> dict:
    """The rows of `world` with the class of question q (`CLASSES[q]`) on each, as the compiler writes it."""
    for rs in results.values():
        for r in rs:
            r["answer_class"] = CLASSES[int(r["qid"][1:])]
    return results


def one_sided(e: int, q: int) -> int:
    """The score of a guesser that always says yes: right (+1) on a yes gold, wrong (-1) on a no gold."""
    return 1 if CLASSES[q] == "yes" else -1


def test_a_guesser_that_always_says_one_side_passes_every_cell_of_a_fair_coin():
    """Contract (coordinator's decision, valid and simple): the shortcut gates are read in the group, each length and
    each slice only, never in a class. Twelve scripts of four answerable questions, two yes and two no each; `blind`
    always says yes, +1 on the yes golds and -1 on the no ones, so it nets 0 over the group, over `all_unknown`'s 0.
    In a class cell of the yes golds alone it would score +1 and count, and the notebooks of `time_needed` (always the
    earlier state, always the later) would fail both change families on every job. The cells are all, short,
    answerable; none counts and the evidence verdict passes. Regression caught: a cell per declared class, which fails
    a fair coin because a one-sided guesser wins the side it guesses."""
    results = classed(world({BLIND: one_sided}, scripts=12, per=4))
    g = shortcuts_of(results)
    assert g["evidence"] is True and g["value"] == 0
    assert [tuple(c["cell"]) for c in g["cells"]] == [("all",), ("length", "short"), ("slice", "answerable")]
    assert not any(c["counts"] for c in g["cells"])
    assert check.outcome(evaluate(results, n_boot=N_BOOT), "evidence") == (True, [])


def test_the_time_blind_notebooks_pass_time_needed_on_a_fair_coin():
    """Contract: `time_needed` reads `first_table` (always the earlier state) like any shortcut, in the group, each
    length and each slice, and passes a fair coin whichever side it guesses. The same rows, with the guesser as
    `first_table` on a temporal family. Regression caught: `time_needed` failing a change family on every job through
    a class cell."""
    results = classed(world({FIRST: one_sided}, scripts=12, per=4))
    gate = gates_of(results, temporal=frozenset({FAMILY}))["time_needed"]
    assert gate["evidence"] is True and not any(c["counts"] for c in gate["cells"])
