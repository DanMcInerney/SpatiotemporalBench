"""`--cells-from DEVROOT` on `stbench eval` and `stbench eval compare` (FD-0.11 §3.7): a subset, or a fresh set, is
scored on the cells and weights of the release's dev root and not on its own. Hand datasets; every number is worked by
hand."""

from __future__ import annotations

import pytest

from eval_kit import cli, compiled, hand_dataset, hand_script, results
from stbench.eval_compare import compare
from stream_kit import during_q, end_q


def dev_root(root):
    """The 'dev root': one case asked a `path_ahead` question (1 in the dataset: weight 1/20) and 20 end ones
    (`object_where`, the ability `memory`: weight 1), 5 of the 20 with an unknown gold."""
    compiled(root, hand_script(1), [during_q(1, 12), *[end_q(n) for n in range(2, 22)]], unknown=(1, 2, 3, 4, 5))
    return root


def head(md: str) -> list[str]:
    line = next(x for x in md.splitlines() if x.startswith("| **headline**"))
    return [c.strip().split(" ")[0] for c in line.strip("|").split("|")]


def test_cells_from_gives_a_subset_the_dev_roots_cells_and_weights(tmp_path):
    """Contract (FD-0.11 §3.7, §4.5): with `--cells-from DEVROOT` a run's headline is weighed by the dev root's cells,
    not by those of the dataset it ran on. By hand, the subset is `eval_kit.hand_dataset`: `path_ahead` 2 answerable
    (the `now` column) and `object_where` 4 with 3 unknown golds (`memory`), so `all_unknown` scores `path_ahead` 0 and
    `memory` 3/4. On its own every cell is under 20 questions, the rule is not applied and the headline is (0 + 3/4) / 2
    = 3/8 with `thin_ignored`. On the dev root's cells (`path_ahead` weighs 1/20, `memory` 1) it is (1 * 3/4 + 1/20 * 0)
    / (1 + 1/20) = 5/7, `thin_ignored` is false and the cell is listed with its weight. Regression caught: the fast tier
    or a held-out set scored on its own handful of questions per cell, so that its headline has other cells than
    dev's."""
    dev_root(tmp_path / "dev")
    hand_dataset(tmp_path / "subset")
    assert cli("all_unknown", "--data", tmp_path / "subset", "--out", tmp_path / "own") == 0
    assert cli("all_unknown", "--data", tmp_path / "subset", "--out", tmp_path / "cells",
               "--cells-from", tmp_path / "dev") == 0
    own, cells = (results(tmp_path / name)["layouts"]["apartment_a"] for name in ("own", "cells"))
    assert own["headline"]["mean"] == pytest.approx(3 / 8) and own["thin_ignored"] is True
    assert cells["headline"]["mean"] == pytest.approx(5 / 7) and cells["thin_ignored"] is False
    assert [entry[1:] for entry in cells["thin"]] == [["path_ahead", "now", pytest.approx(1 / 20)]]


def test_compare_takes_the_dev_roots_cells_for_every_result(tmp_path, capsys):
    """Contract: `eval compare --cells-from DEVROOT` re-aggregates every result on the dev root's cells, whatever cells
    the results were made with, so the tables of a fast tier and a held-out set show headlines on one set of cells. By
    hand, two `all_unknown` results on the subset of the test above, made on its own cells: their headlines read 3/8
    (+0.375) and the difference 0; compared with the dev root's cells they read 5/7 (+0.714) and the difference
    0. Regression caught: a compare that keeps the cells the results were made with, or applies them to the first result
    alone."""
    dev_root(tmp_path / "dev")
    hand_dataset(tmp_path / "subset")
    for name in ("a", "b"):
        assert cli("all_unknown", "--data", tmp_path / "subset", "--out", tmp_path / name) == 0
    paths = [str(tmp_path / "a"), str(tmp_path / "b")]
    assert head(compare(paths))[1:3] == ["+0.375", "+0.375"]
    assert head(compare(paths, cells_from=tmp_path / "dev"))[1:3] == ["+0.714", "+0.714"]
    capsys.readouterr()
    assert cli("compare", *paths, "--cells-from", tmp_path / "dev") == 0   # the option reaches it from the command line
    assert head(capsys.readouterr().out)[1:3] == ["+0.714", "+0.714"]


def test_a_cells_from_path_with_no_compiled_case_is_refused_before_anything_runs(tmp_path, capsys):
    """Contract (final review N1): `--cells-from` names a dataset that holds compiled cases; a mistyped path or an empty
    folder is refused with exit 2 and a message naming the option, the path and what a dev root holds, on `eval`,
    `eval compare` and `eval distance`, and a refused `eval` writes nothing. Hand cases on the subset of the tests
    above: the dev root's cells run (exit 0); `dev-typo` (absent) and `empty` (a folder with nothing in it) are
    refused. Before the fix a missing index read as no cell, so every cell weighed 1 and the headline came out as the
    subset's own (3/8) without a word. Regression caught: a run scored on its own cells because the dev root's path
    was mistyped."""
    dev_root(tmp_path / "dev")
    hand_dataset(tmp_path / "subset")
    (tmp_path / "empty").mkdir()
    assert cli("all_unknown", "--data", tmp_path / "subset", "--out", tmp_path / "good", "--cells-from",
               tmp_path / "dev") == 0
    capsys.readouterr()
    for missing in (tmp_path / "dev-typo", tmp_path / "empty"):
        assert cli("all_unknown", "--data", tmp_path / "subset", "--out", tmp_path / "bad", "--cells-from",
                   missing) == 2
        error = capsys.readouterr().err
        assert "--cells-from" in error and str(missing) in error and "no compiled case" in error
        assert not (tmp_path / "bad").exists()
        assert cli("compare", tmp_path / "good", tmp_path / "good", "--cells-from", missing) == 2
        assert "no compiled case" in capsys.readouterr().err
        assert cli("distance", tmp_path / "good", "--data", tmp_path / "subset", "--cells-from", missing) == 2
        assert "no compiled case" in capsys.readouterr().err
