"""`stbench build` and its commands: the stages wired in order (with no Blender, perceive or compiler: the stage runners
are injected), a render reused only when it is whole and its log says it was made as this build asks, and
`generate`/`cases`."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

import job_kit
from stbench import asset_manifest, asset_render, build, case, cli
from stbench.asset_manifest import AssetError
from stbench.build import _rendered
from stbench.case import ScriptPaths
from stbench.timeline import load


def args_for(tmp_path: Path, **over) -> argparse.Namespace:
    base = {"map": "apartment_a", "robot": "vacuum", "job": "vacuum/1", "tasks": 1, "seeds": "101-102",
            "out": tmp_path / "ds", "appearance": "placeholder", "samples": None, "fast": True,
            "keep_linear": tmp_path / "linear", "workers": 1}
    return argparse.Namespace(**(base | over))


class Stages:
    """Stand-ins for the stages that need Blender, perceive and the compiler, recording what they were asked."""

    def __init__(self):
        self.calls: list[tuple] = []

    def render(self, script: Path, flags: list[str]) -> None:
        """What `stbench render` leaves: the last keyframe's depth and instances, and a render log naming its look."""
        self.calls.append(("render", script.name, flags))
        paths = ScriptPaths.from_dir(script)
        last = load(paths.timeline).n_keyframes - 1
        for path in (paths.depth(last), paths.instances(last)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
        samples = int(flags[flags.index("--samples") + 1]) if "--samples" in flags else 32
        log = {"appearance": flags[flags.index("--appearance") + 1], "fast": "--fast" in flags,
               "isp_override": False, "samples": samples}
        paths.render_log.write_text(json.dumps(log), encoding="utf-8")

    def perceive(self, scripts: list[Path], workers: int) -> None:
        self.calls.append(("perceive", [s.name for s in scripts], workers))

    def compile(self, dataset: Path, cases: list[str], workers: int) -> None:
        self.calls.append(("compile", dataset, cases, workers))

    def run(self, args: argparse.Namespace) -> list[tuple]:
        self.calls = []
        build.build(args, render=self.render, perceive=self.perceive, compile=self.compile)
        return self.calls


def test_build_runs_without_a_render(tmp_path, capsys):
    """Contract (plan §4.3): `build` runs generate, render, perceive, the cases and compile, each stage once a script
    (compile once a dataset), in that order; the render gets the appearance, `--fast` and `--keep-linear DIR/<script
    id>`; a script whose render is finished is not rendered again, while perceive and compile run again over the same
    cases (equal entries reuse their ids). Hand case: two seeds make two scripts of one
    task, so two short windows, two cases. Regression caught: a stage left unwired, a flag dropped on the way to the
    renderer, or a resumed build that renders everything again."""
    args, stages = args_for(tmp_path), Stages()
    first = stages.run(args)
    sids = [name for kind, name, *_ in first if kind == "render"]
    assert [call[0] for call in first] == ["render", "render", "perceive", "compile"] and len(set(sids)) == 2
    assert [call[2] for call in first if call[0] == "render"] == [
        ["--appearance", "placeholder", "--fast", "--keep-linear", str(tmp_path / "linear" / sid)] for sid in sids]
    assert first[2][1:] == (sids, 1)
    dataset, cases, workers = first[3][1:]
    assert dataset == tmp_path / "ds" and len(cases) == 2 and workers == 1
    assert sorted(i for i, e in case.read_index(dataset).items() if e["kind"] == "case") == sorted(cases)
    assert "keyframes" in capsys.readouterr().out

    again = stages.run(args)  # all rendered
    assert [call[0] for call in again] == ["perceive", "compile"] and again[1][2] == cases


def test_assets_are_checked_first_credited_and_part_of_the_render_test(tmp_path):
    """Contract (plan §4.3): `--appearance assets` checks the asset library before anything is generated (no fallback to
    placeholders), writes `CREDITS.md` at the dataset root, and passes the appearance and the sample count to the
    render. Regression caught: a build with assets that falls back to placeholders, or an attribution file missing from
    the dataset."""
    info = asset_render.LibraryInfo(tmp_path / "lib.blend", asset_manifest.packaged_path())
    args = args_for(tmp_path, seeds="101", appearance="assets", samples=64)
    stages = Stages()
    with mock.patch.object(asset_render, "require_library", lambda: info):
        calls = stages.run(args)
        assert (tmp_path / "ds" / "CREDITS.md").read_text(encoding="utf-8").startswith("# Credits")
        assert calls[0][2][:4] == ["--appearance", "assets", "--samples", "64"]
        assert [c[0] for c in stages.run(args)] == ["perceive", "compile"]  # finished: nothing to render
        assert stages.run(args_for(tmp_path, seeds="101", appearance="assets", samples=128))[0][0] == "render"
    args = args_for(tmp_path, seeds="101", appearance="assets", samples=64)
    refused = mock.Mock(side_effect=AssetError("no library"))
    with mock.patch.object(asset_render, "require_library", refused), pytest.raises(
            SystemExit, match="no fallback to placeholders"):
        stages.run(args)
    assert [c[0] for c in stages.run(args_for(tmp_path, seeds="101"))] == ["render", "perceive", "compile"]


EXPECT = {"appearance": "assets", "fast": False, "samples": 128, "isp_override": False}  # what a build expects


def finished(tmp_path: Path, **log) -> ScriptPaths:
    """A script folder whose last keyframe (3) has depth and instances, and the render log `stbench render` writes,
    which is `EXPECT` with the edits."""
    paths = ScriptPaths(tmp_path, "abc")
    for path in (paths.depth(3), paths.instances(3)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    paths.render_log.parent.mkdir(parents=True, exist_ok=True)
    paths.render_log.write_text(json.dumps(EXPECT | log), encoding="utf-8")
    return paths


@pytest.mark.parametrize(("edit", "log"), [
    ("a look-dev override", {"isp_override": True}), ("another appearance", {"appearance": "placeholder"}),
    ("other samples", {"samples": 64}), ("the fast flag", {"fast": True}),
])
def test_a_render_made_another_way_is_not_finished(tmp_path, edit, log):
    """Contract (recommendations §2.6 item 3): `_rendered` is true only when the last keyframe has depth and
    instances and the log equals what the build expects (appearance, `fast`, no `--isp` override, samples when asked),
    and false when a trial `--isp` was developed or its appearance, samples or flag differ. Regression caught: a
    resumed build keeping old-look or `--fast` frames."""
    assert _rendered(finished(tmp_path), 4, EXPECT)  # the same log passes
    assert not _rendered(finished(tmp_path, **log), 4, EXPECT), edit


def test_a_render_with_a_missing_part_or_an_old_log_is_judged_by_what_is_there(tmp_path):
    """Contract: a render is not finished without depth or instances at its last keyframe, or without a readable log;
    a log with no `appearance` is a placeholder render (the first releases'), and a log without a key the build expects
    is never finished. Regression caught: a half-written render passing, or an old log taken for the current look."""
    paths = finished(tmp_path)
    paths.depth(3).unlink()
    assert not _rendered(paths, 4, EXPECT)
    paths = finished(tmp_path)
    paths.instances(3).unlink()
    assert not _rendered(paths, 4, EXPECT)
    paths = finished(tmp_path)
    paths.render_log.write_text("{", encoding="utf-8")
    assert not _rendered(paths, 4, EXPECT)
    old = {"appearance": "placeholder", "fast": False}
    paths = finished(tmp_path)
    paths.render_log.write_text(json.dumps({"fast": False}), encoding="utf-8")
    assert _rendered(paths, 4, old)
    paths.render_log.write_text(json.dumps({"samples": 32}), encoding="utf-8")
    assert not _rendered(paths, 4, old)


def test_the_commands_generate_a_valid_script_and_make_its_cases(tmp_path, capsys):
    """Contract (plan §4.3): `stbench generate` writes a script and prints its id and keyframe count, `stbench validate`
    finds nothing wrong with it (the leak scan included), and `stbench cases` writes a case of every script's window,
    the same ones again on a second run. Regression caught: a generator whose scripts `validate` refuses, or a `cases`
    command that makes new cases each time."""
    ds = str(tmp_path / "ds")
    assert cli.main(["generate", "--seed", "101", "--out", ds]) == 0
    sid, rest = capsys.readouterr().out.split(": ")
    assert rest.endswith("keyframes, 1 tasks\n") and sid in case.read_index(ds)
    assert cli.main(["validate", str(tmp_path / "ds" / "scripts" / sid)]) == 0
    assert "ok" in capsys.readouterr().out
    assert cli.main(["cases", ds]) == 0 and capsys.readouterr().out == f"1 cases in {ds}\n"
    cases = {i for i, e in case.read_index(ds).items() if e["kind"] == "case"}
    assert len(cases) == 1 and cli.main(["cases", ds]) == 0
    assert {i for i, e in case.read_index(ds).items() if e["kind"] == "case"} == cases


def test_an_invalid_script_never_reaches_the_render(tmp_path):
    """Contract (the review's M4): `build` and `stbench generate` run `validate` (the job's checks included) before a
    script is written, so a script it rejects raises naming its seed and its findings, nothing is written into the
    dataset and no stage runs. A generator that returns a one-task script with its decisions dropped (the job's own
    `job_decisions` rule) stands for gaps, drops or people that break a script found only after hours of render.
    Regression caught: an invalid script rendered before anyone looks at it."""
    good = job_kit.script(101)
    bad = replace(good, ego=replace(good.ego, decisions=[]))
    stages = Stages()
    with mock.patch.object(build, "generate_script", lambda *args: bad):
        with pytest.raises(ValueError, match=r"seed 101 .*job_decisions"):
            stages.run(args_for(tmp_path))
        with pytest.raises(ValueError, match=r"seed 101 .*job_decisions"):
            cli.main(["generate", "--seed", "101", "--out", str(tmp_path / "other")])
    assert stages.calls == [] and case.read_index(tmp_path / "ds") == {} and not (tmp_path / "ds" / "scripts").exists()
    assert not (tmp_path / "other").exists()
    with mock.patch.object(build, "generate_script", lambda *args: good):
        assert build.generate_checked("apartment_a", 101, "vacuum", "vacuum/1", 1) is good


def test_tasks_take_a_range(tmp_path, capsys):
    """Contract: `--tasks` is a count (`5`) or a range (`4-6`); each seed of a range draws its own count, uniformly,
    from a key of its own, so it is in [4, 6], all three occur, and a rebuild (the same call again, as `stbench
    generate` and `build`'s generate stage both make it) draws the same; a count is that count. A script written with
    `--tasks 4-6` holds the seed's own count. Regression caught: a range read as a string, a count redrawn per stage,
    or `generate` and `build` disagreeing about a seed's task count."""
    assert build.parse_tasks("5") == (5, 5) and build.parse_tasks("4-6") == (4, 6)
    for bad in ("0", "6-4", "a", "4-", "-6", "1-2-3", ""):
        with pytest.raises(ValueError, match="--tasks"):
            build.parse_tasks(bad)
    counts = [build.tasks_for((4, 6), "apartment_a", seed, "vacuum", "vacuum/2") for seed in range(1, 61)]
    assert set(counts) == {4, 5, 6} and counts == [
        build.tasks_for((4, 6), "apartment_a", seed, "vacuum", "vacuum/2") for seed in range(1, 61)]
    assert {build.tasks_for(5, "apartment_a", seed, "vacuum", "vacuum/2") for seed in range(1, 20)} == {5}
    assert {build.tasks_for((5, 5), "apartment_a", seed, "vacuum", "vacuum/2") for seed in range(1, 20)} == {5}
    seen = []
    script = job_kit.generated2(101)[1]
    with mock.patch.object(build, "generate_checked", lambda *args: seen.append(args) or script):
        for seed in range(1, 7):
            build._generate(("apartment_a", seed, "vacuum", "vacuum/2", (4, 6), str(tmp_path / "ds")))
        build._generate(("apartment_a", 9, "vacuum", "vacuum/2", 5, str(tmp_path / "ds")))
    assert [a[4] for a in seen] == counts[:6] + [5] and [a[1] for a in seen] == [1, 2, 3, 4, 5, 6, 9]
    ds = str(tmp_path / "range")
    assert cli.main(["generate", "--seed", "101", "--job", "vacuum/2", "--tasks", "4-6", "--out", ds]) == 0
    want = build.tasks_for((4, 6), "apartment_a", 101, "vacuum", "vacuum/2")
    assert capsys.readouterr().out.endswith(f"{want} tasks\n")
    with pytest.raises(SystemExit):
        cli.main(["generate", "--seed", "101", "--tasks", "6-4", "--out", ds])
