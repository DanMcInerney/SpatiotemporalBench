"""`stbench export`: the released set, each refusal, `release.json`, repeatability, and an export that runs from its
own files (A1 to A7). The dataset is `export_kit.build`'s; every count below is written out by hand, never read back
from the export."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest

import export_kit as kit
from eval_kit import cli as eval_cli
from recompile_kit import recompile
from stbench import export_cmd, export_files, map_files
from stbench.cli import main

GENERATED = {"README.md", "LICENSE", "release.json"}   # the files an export makes, not copies


def pin(mp: pytest.MonkeyPatch) -> None:
    """The machine's user name fixed, so an export does not depend on the user."""
    mp.setattr(export_files, "machine_names", lambda: ["zed-user"])


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    pin(monkeypatch)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> kit.Fixture:
    """The dataset, built once; a test that changes it works on `kit.clone` of it."""
    return kit.build(tmp_path_factory.mktemp("built") / "data")


@pytest.fixture(scope="module")
def rows(built, tmp_path_factory) -> Path:
    """Two reference rows over the dataset, `all_unknown` with the during questions on and off: `on/` and `off/`."""
    base = tmp_path_factory.mktemp("rows")
    assert eval_cli("all_unknown", "--data", built.data, "--out", base / "on") == 0
    assert eval_cli("all_unknown", "--data", built.data, "--during", "off", "--out", base / "off") == 0
    return base


@pytest.fixture(scope="module")
def exported(built, tmp_path_factory) -> Path:
    """One export of the dataset, made once, only read."""
    out = tmp_path_factory.mktemp("exported") / "out"
    with pytest.MonkeyPatch.context() as mp:
        pin(mp)
        assert main(["export", "--data", str(built.data), "--out", str(out)]) == 0
    return out


def run(capsys, data: Path, out: Path, *extra) -> tuple[int, str]:
    """`stbench export --data DATA --out OUT ...` through the CLI: (the exit code, what it printed to stderr)."""
    code = main(["export", "--data", str(data), "--out", str(out), *map(str, extra)])
    return code, capsys.readouterr().err


def test_an_export_holds_exactly_the_released_files(built, exported):
    """Contract (A1): an export holds the files `export_files` names and no others: every file under
    `scripts/<sid>` (the camera frames, no depth), the four private files and the `depth/` frames of each script, the
    three of each case, the index entries, the five top-level files; not `instances/`, `instance_map.json`,
    `render_tmp/`, `runs/` or a stray file of the dataset (all of which this dataset holds). Regression caught: a rule
    that copies a script's private folder or the dataset whole, which would ship the instance images and the working
    output; one that forgets the depth, which `stbench check` on the export reads for the `detections` channel; or one
    that puts it back under `scripts/`."""
    assert all((built.data / junk).is_file() for junk in kit.JUNK)
    assert all((built.script(n).private / j).is_file() for n in built.scripts for j in kit.SCRIPT_JUNK)
    assert kit.files(exported) == kit.released(built)


def test_the_copied_files_are_the_datasets_bytes(built, exported):
    """Contract (A1): every released file that is a copy (the scripts, the private files, the index, `CREDITS.md` and
    the check report) is the dataset's file byte for byte, and `LICENSE` is the package's CC BY 4.0 text. Regression
    caught: a copy that rewrites line ends or re-serialises JSON, which would change a gold value or a picture."""
    for rel in kit.released(built) - GENERATED:
        assert (exported / rel).read_bytes() == (built.data / rel).read_bytes(), rel
    licence = Path(export_cmd.__file__).parent / "release" / "CC-BY-4.0.txt"
    assert (exported / "LICENSE").read_bytes() == licence.read_bytes()


# -- the refusals (A2) -------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    """What a refusal's setup works on: the dataset (the shared one, until a setup that changes it calls `own`), a
    scratch folder, the monkeypatch, the reference rows, the `--out` folder the command is given and the extra
    arguments."""

    fx: kit.Fixture
    tmp: Path
    mp: pytest.MonkeyPatch
    rows: Path
    out: Path
    extra: list

    def own(self) -> kit.Fixture:
        """A copy of the dataset that the setup may change."""
        self.fx = kit.clone(self.fx, self.tmp / "data")
        return self.fx


def bad_results(c: Ctx, change: Callable[[dict], object]) -> None:
    path = c.tmp / "bad" / "results.json"
    path.parent.mkdir()
    shutil.copy(c.rows / "on" / "results.json", path)
    kit.edit(path, change)
    c.extra += ["--results", path]


def out_not_empty(c):
    c.out.mkdir()
    (c.out / "keep.txt").write_text("mine\n", encoding="utf-8")


def out_inside_data(c):
    c.out = c.fx.data / "release"


def out_is_data(c):
    c.out = c.fx.data


def data_inside_out(c):
    c.out = c.fx.data.parent


def heldout_split(c):
    kit.edit(c.own().data / "private" / "index" / f"{c.fx.scripts['short']}.json", lambda d: d.update(split="heldout"))


def heldout_map(c):
    c.mp.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, "apartment_a"))


def case_not_compiled(c):
    c.own().case("short").gold.unlink()


def render_is_partial(c):
    kit.edit(c.own().script("medium").render_log, lambda d: d.update(frames=[0, 3]))


def render_log_missing(c):
    c.own().script("short").render_log.unlink()


def rgb_missing(c):
    c.own().script("medium").rgb(4).unlink()


def depth_missing(c):
    c.own().script("silent").depth(2).unlink()


def agent_folder_leaks(c):
    (c.own().script("short").root / "notes.txt").write_text("notes\n", encoding="utf-8")


def released_file_holds_a_path(c):
    kit.edit(c.own().script("short").render_log, lambda d: d.update(linear_kept="C:\\Users\\someone\\linear"))


def released_file_holds_the_user_name(c):
    with (c.own().data / "CREDITS.md").open("a", encoding="utf-8") as f:
        f.write("Thanks to zed-user.\n")


def credits_missing(c):
    (c.own().data / "CREDITS.md").unlink()


def report_missing(c):
    (c.own().data / "check_report.json").unlink()


def report_over_other_cases(c):
    kit.write_report(c.own().data, asked_nothing=[{**kit.CHECK_COUNTS[0], "compiled": 3}, kit.CHECK_COUNTS[1]])


def report_of_an_earlier_compile(c):
    """The report of 0.10.0 over the 0.11 root (final review B1): the same counts of cases that ask nothing, and no gate
    row for a family the dataset has since gained (`object_where` here)."""
    kit.write_report(c.own().data, gates=kit.GATE_ROWS[:1])


def report_with_another_question_count(c):
    kit.write_report(c.own().data, gates=[kit.GATE_ROWS[0], {**kit.GATE_ROWS[1], "n": 3}])


def report_gating_a_family_the_gold_lacks(c):
    kit.write_report(c.own().data, gates=[*kit.GATE_ROWS, {**kit.GATE_ROWS[0], "family": "way_to", "n": 5}])


def report_fails_the_evidence_verdict(c):
    kit.write_report(c.own().data, passed_evidence=False)


def report_made_under_another_verdict(c):
    kit.write_report(c.own().data, verdict="point", passed_evidence=True)


def report_names_a_broken_instrument(c):
    kit.write_report(c.own().data,
                     broken=["BROKEN apartment_a vacuum/1 object_where: no level of blind finds a record"])


def results_partial(c):
    bad_results(c, lambda d: d.update(partial=True))


def results_name_a_case_not_exported(c):
    bad_results(c, lambda d: d["cases"][0].update(id="f" * 16))


def results_scored_on_another_compile(c):
    """The dataset is compiled again after the rows were scored: one case's qids are new (not a gold value edited)."""
    recompile(c.own().data, c.fx.cases["short"])
    c.extra += ["--results", c.rows / "on"]


def results_with_one_name_twice(c):
    c.extra += ["--results", c.rows / "on", c.rows / "on" / "results.json"]


REFUSALS = [(out_not_empty, "is not empty"), (out_inside_data, "overlap"), (out_is_data, "overlap"),
            (data_inside_out, "overlap"), (heldout_split, "is held out"), (heldout_map, "is held out"),
            (case_not_compiled, "is not compiled"), (render_is_partial, "a partial render"),
            (render_log_missing, "render_log.json"), (rgb_missing, "lack an rgb or a depth image"),
            (depth_missing, "lack an rgb or a depth image"), (agent_folder_leaks, "unlisted_file"),
            (released_file_holds_a_path, "absolute path"), (released_file_holds_the_user_name, "user name"),
            (credits_missing, "CREDITS.md"), (report_missing, "check_report.json is missing"),
            (report_over_other_cases, "but the export holds"),
            (report_of_an_earlier_compile, "1 group.s. of the gold it does not gate, e.g. apartment_a/vacuum/1/obj"),
            (report_with_another_question_count, "the report says 3, the gold holds 4"),
            (report_gating_a_family_the_gold_lacks, "1 it gates that the gold lacks, e.g. .*way_to"),
            (report_fails_the_evidence_verdict, "passed_evidence is not true"),
            (report_made_under_another_verdict, "not evidence"),
            (report_names_a_broken_instrument, "instrument is broken"),
            (results_partial, "is partial"), (results_name_a_case_not_exported, "does not hold"),
            (results_scored_on_another_compile, "scored on another compile for 1 case"),
            (results_with_one_name_twice, "ships under the name")]


@pytest.mark.parametrize("setup, message", REFUSALS, ids=[s.__name__ for s, _ in REFUSALS])
def test_every_refusal_exits_2_and_writes_nothing(setup, message, built, rows, tmp_path, monkeypatch, capsys):
    """Contract (A2): each condition that makes a dataset unreleasable (an `--out` that is not empty or overlaps the
    data, a held-out script by its split or its map, a case not compiled, a partial render, a keyframe without its
    pictures, a leak, an absolute path or the user's name in a released file, no `CREDITS.md`, a check report that is
    missing, over other cases, with gate rows that are not the gold's by (layout, job, family) and question count (the
    stale report of an earlier compile, found by the 0.11 review: B1), made under another verdict than the evidence
    one, failing the evidence verdict or naming a broken instrument, and a results file that is partial, names a case
    not exported, holds rows of another compile (a qid its case's gold lacks: a compile draws every qid anew) or ships
    under a name another has) is refused with exit 2 and a message that names it, before anything is written.
    Regression caught: a check that runs after the first copy, which leaves a half-written folder that looks like a
    release, or a condition that is dropped."""
    ctx = Ctx(built, tmp_path, monkeypatch, rows, tmp_path / "out", [])
    setup(ctx)

    def listing() -> list[str]:
        return sorted(f"{root.name}/{p.relative_to(root).as_posix()}" for root in {tmp_path, ctx.fx.data}
                      for p in root.rglob("*"))

    before = listing()
    code, err = run(capsys, ctx.fx.data, ctx.out, *ctx.extra)
    assert code == 2 and re.search(message, err), err
    assert listing() == before


def test_the_export_follows_the_evidence_verdict_and_not_the_point_reading(built, tmp_path, capsys):
    """Contract (DECISIONS, "Before the binding read", 2): a check report made with `--verdict evidence` that passes it
    (`passed_evidence`) with no `BROKEN` line exports, whatever the point and interval readings say, and is shipped as
    it is. At about 20 questions a family the point reading is noise, so a report failing only the point reading
    (`passed_point` and `passed_interval` false) is the evidence verdict's pass. Regression caught: an export that
    still reads `passed_point`, which refuses the dataset the binding read cleared; or one that ignores the report's
    verdict and ships a point-only pass that no evidence check cleared (the refusals above)."""
    fx = kit.clone(built, tmp_path / "data")
    kit.write_report(fx.data, passed_point=False, passed_interval=False)
    out = tmp_path / "out"
    assert run(capsys, fx.data, out)[0] == 0
    assert (out / "check_report.json").read_bytes() == (fx.data / "check_report.json").read_bytes()
    assert json.loads((out / "check_report.json").read_text(encoding="utf-8"))["passed_point"] is False


def test_a_check_report_that_still_carries_versions_is_shipped_as_it_is(built, tmp_path, capsys):
    """Contract: the export reads a check report's verdict, its counts and its `broken` list and nothing else, so a
    report written by an earlier build, with the `versions` key it then carried, exports and is shipped byte for byte.
    Regression caught: an export that compares the report's `versions` with anything (it would refuse every dataset
    checked before the key was dropped), or that rewrites the report it ships."""
    fx = kit.clone(built, tmp_path / "data")
    kit.write_report(fx.data, versions={"compiler": "old", "scorer": "old", "stream": "old"})
    out = tmp_path / "out"
    assert run(capsys, fx.data, out)[0] == 0
    assert (out / "check_report.json").read_bytes() == (fx.data / "check_report.json").read_bytes()
    assert "versions" in json.loads((out / "check_report.json").read_text(encoding="utf-8"))


def test_urls_and_ordinary_text_are_not_machine_paths():
    """Contract (A2): a machine path is a drive letter, a system folder at the start of a path, a `~/` home or the
    user's name; a URL, a relative path, a time and a word that merely contains the name are not. The fixture's
    `CREDITS.md` carries a URL and relative paths and every export test releases it. Regression caught: a pattern so
    loose that it refuses the card's links and the credits' URLs, or so tight that `C:/Users/x` or `/home/x` ships."""
    def traces(text: str) -> list[str]:
        return export_files.machine_traces([("a.md", text.encode())], ["zed-user"])

    for fine in ("https://github.com/DanMcInerney/SpatiotemporalBench/blob/v0.4.0/docs/format.md",
                 "http://localhost:8080/home/ and http://example.org/Users/", "scripts/<sid>/robot.json",
                 "private/index/<id>.json", "at 10:30/11:00", "zed-users and zed-userland"):
        assert traces(fine) == [], fine
    for bad in ("C:\\Users\\x\\data", "d:/data/x", '"/Users/someone/x"', "(/home/someone/data)", "~/data", "/tmp/x",
                "built by zed-user on a laptop", "Zed-User"):
        assert traces(bad), bad
    assert export_files.machine_traces([("a.png", b"C:/Users/x"), ("b.txt", b"C:/Users/x")], []) == []   # not text


def test_a_file_with_several_machine_traces_is_one_problem_so_the_refusal_counts_files():
    """Contract (A2): the refusal "N released files hold an absolute path or a user name" counts files, so
    `machine_traces` gives one entry a file, with its first trace and how many others it holds. Regression caught: one
    entry a trace (up to three a file), which told the dry run on ten render logs, each with a `C:\\Users\\...` path
    and the user's name in it, that twenty files held one."""
    files = [("a.md", b"see C:/x"), ("b.json", b'{"p": "C:\\\\Users\\\\x\\\\y", "q": "~/z", "r": "zed-user"}'),
             ("c.md", b"clean")]
    got = export_files.machine_traces(files, ["zed-user"])
    assert len(got) == 2
    assert got[0].startswith("a.md: ") and "more" not in got[0]
    assert got[1].startswith("b.json: ") and got[1].endswith("(+2 more)")


# -- the release document (A4) ---------------------------------------------------------------------------------
def test_release_json_holds_the_counts_by_hand(built, exported):
    """Contract (A4): `release.json` holds exactly the package version, per (layout, job) its scripts, cases by length,
    cases asking nothing, questions (and by ability) and keyframes, per case its script, layout, job, length and
    question count, and the names of the shipped results: no schema, definition version, source commit, gold hash or
    timestamp. Counts are by hand: the apartment group has 2 scripts (6 and 12 keyframes), a short and a medium case
    of 3 questions each (a `path_ahead` and two `object_where`, which measure memory: 4 memory and 2 `path_ahead`, an
    ability of its own); the warehouse group 1 script of 6 keyframes and 1 case that asks nothing. Regression caught: a
    count that includes the silent case's questions, an ability count that drops a family or reads the wrong file, a
    stamp (version, schema, commit or hash) that creeps back into the document, or a clock in it."""
    release = json.loads((exported / "release.json").read_text(encoding="utf-8"))
    assert list(release) == ["package", "groups", "cases", "results"]
    assert release["package"] == metadata.version("spatiotemporalbench") and release["results"] == []
    assert release["groups"] == [
        {"layout": "apartment_a", "job": "vacuum/1", "scripts": 2, "cases": {"short": 1, "medium": 1, "long": 0},
         "asked_nothing": 0, "questions": 6, "abilities": {"memory": 4, "path_ahead": 2}, "frames": 18},
        {"layout": "warehouse_a", "job": "amr/1", "scripts": 1, "cases": {"short": 1, "medium": 0, "long": 0},
         "asked_nothing": 1, "questions": 0, "abilities": {}, "frames": 6}]

    def case(name: str, layout: str, job: str, length: str, questions: int) -> dict:
        return {"id": built.cases[name], "script": built.scripts[name], "layout": layout, "job": job,
                "length": length, "questions": questions}

    apartment = sorted([case("short", "apartment_a", "vacuum/1", "short", 3),
                        case("medium", "apartment_a", "vacuum/1", "medium", 3)], key=lambda c: c["id"])
    assert release["cases"] == [*apartment, case("silent", "warehouse_a", "amr/1", "short", 0)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}T", json.dumps(release))


def test_the_exported_card_is_the_template_with_this_releases_counts(exported):
    """Contract (A6): the exported `README.md` is the package's template word for word outside the counts block, and the
    block holds this release's numbers, by hand: the apartment's 2 scripts, 1 short and 1 medium case, 6 questions and
    18 keyframes; the warehouse's 1 script, 1 silent case and 6 keyframes; 3 scripts, 24 keyframes in all; and the line
    "Exported by stbench <the package version>." with no commit and no definitions. Regression caught: a card made from
    another template than the package's, a block the export fills from anything but its own `release.json`, or a
    commit or definitions line that returns."""
    template = (Path(export_cmd.__file__).parent / "release" / "dataset-card.md").read_text(encoding="utf-8")
    head, rest = template.split("<!-- counts -->\n")
    tail = rest.split("<!-- /counts -->", 1)[1]
    card = (exported / "README.md").read_text(encoding="utf-8")
    assert card.startswith(head) and card.endswith(tail)
    block = card[len(head):len(card) - len(tail)]
    for row in ("| `apartment_a` | `vacuum/1` | 2 | 1 / 1 / 0 | 0 | 6 | 18 |",
                "| `warehouse_a` | `amr/1` | 1 | 1 / 0 / 0 | 1 | 0 | 6 |",
                "| **Total** |  | 3 | 2 / 1 / 0 | 1 | 6 | 24 |",
                "| Layout | Job | memory | path_ahead |", "| `apartment_a` | `vacuum/1` | 4 | 2 |",
                "| `warehouse_a` | `amr/1` | 0 | 0 |", "| **Total** |  | 4 | 2 |"):
        assert row in block, row
    assert f"\n\nExported by stbench {metadata.version('spatiotemporalbench')}.\n" in block
    assert "commit" not in block and "Definitions" not in block


# -- repeatability and an export that runs from its own files (A5, A7) ------------------------------------------
def test_two_exports_of_one_dataset_are_byte_identical(built, exported, tmp_path, capsys):
    """Contract (A5): exporting the same dataset twice gives the same files with the same bytes, `release.json`, the
    card included (no clock, no random order). Regression caught: a timestamp, a set iterated in hash
    order or a thread-pool completion order in a generated file."""
    assert run(capsys, built.data, tmp_path / "again")[0] == 0
    assert kit.tree(tmp_path / "again") == kit.tree(exported)


def test_the_export_runs_from_its_own_files(exported, tmp_path):
    """Contract (A7): `stbench eval all_unknown --data EXPORT` scores every case of the export that asks a question,
    from the export's own files. By hand: `all_unknown` scores 0 on an answerable gold and 1 on an unknown one, so the
    golds [a, a, u] and [a, u, u] give 6 rows; the `now` column is 0/2 and the `memory` column 3/4, so the headline is
    0.375; the silent case asks nothing and is not run. Regression caught: a file `eval` or `stream` reads (the pose
    rows, a picture, the index, the gold, the oracle's ids) left out of the released set."""
    assert eval_cli("all_unknown", "--data", exported, "--out", tmp_path / "run") == 0
    doc = json.loads((tmp_path / "run" / "results.json").read_text(encoding="utf-8"))
    assert [c["status"] for c in doc["cases"]] == ["complete", "complete"] and not doc["partial"]
    assert len(doc["rows"]) == 6 and doc["layouts"]["apartment_a"]["headline"]["mean"] == pytest.approx(0.375)


def test_results_ship_under_their_name_with_their_bytes(built, rows, tmp_path, capsys):
    """Contract (A1, A4): each `--results` (a `results.json` or its folder) ships as `results/<name>.json`, byte for
    byte, named from what the row is (system, channel, and `during-off` when the during questions were off), so the same
    row has the same name wherever it ran; `release.json` lists the names, and both files of two rows are kept.
    Regression caught: a results file re-serialised or renamed after its folder, which would make `eval compare` read
    another document than the one that was scored."""
    out = tmp_path / "out"
    assert run(capsys, built.data, out, "--results", rows / "on", rows / "off" / "results.json")[0] == 0
    names = ["all_unknown-rgb_pose", "all_unknown-rgb_pose-during-off"]
    assert kit.files(out) == kit.released(built) | {f"results/{n}.json" for n in names}
    assert (out / "results" / f"{names[0]}.json").read_bytes() == (rows / "on" / "results.json").read_bytes()
    assert (out / "results" / f"{names[1]}.json").read_bytes() == (rows / "off" / "results.json").read_bytes()
    assert json.loads((out / "release.json").read_text(encoding="utf-8"))["results"] == sorted(names)


def test_the_help_says_what_it_writes_and_what_it_refuses(capsys):
    """Contract (the brief's done criterion): `stbench export --help` names the files it writes and the conditions it
    refuses on. Regression caught: a help text that says only 'write the dataset', so a user learns the refusals by
    hitting them."""
    with pytest.raises(SystemExit) as stop:
        main(["export", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert stop.value.code == 0
    for word in ("release.json", "private/<cid>/", "instances/", "Refused", "held-out", "not compiled", "partial",
                 "CREDITS.md", "check_report.json", "--results", "another compile"):
        assert word in text, word
