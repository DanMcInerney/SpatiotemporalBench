"""`stbench build`: generate, render, perceive, make the cases and compile a dataset of scripts end to end (V0 §13).

    stbench build --map apartment_a --robot vacuum --job vacuum/1 --tasks 1 --seeds 101-120 --out DATASET [--fast]
                  [--appearance placeholder|assets] [--samples N] [--keep-linear DIR] [--workers N]

`--tasks` is a count (`5`) or a range (`4-6`): each seed of a range draws its own count, uniformly, from its own key
(`tasks_for`), here and in `stbench generate`, so a rebuild draws the same.

Stages, each over every script of the seeds:
1. generate (a process pool): `generate.generate_script`, `validate.validate` (a script with any problem is refused,
   naming its seed, before anything is written or rendered) and `episode.write_script`;
2. render, one script at a time (Blender owns the GPU): `stbench render`, skipped for a script whose render is reused
   (`_rendered`: it is whole and was made with the same appearance, fast flag, samples and no look-dev override), with
   `--keep-linear DIR/<script id>` when `--keep-linear DIR` is given, so a look-dev or dev render keeps its linear
   passes and a camera-model change develops them again instead of rendering again;
3. perceive (parallel `stbench perceive`): the private visibility log and the oracle channel;
4. cases: `case.windows` and `case.write_case` for each script (`write_cases`; equal entries reuse their ids; no
   window starts at a task of a kind its job `continues`);
5. compile (`stbench compile DATASET --cases ...`): the private questions and gold.
Render, perceive and compile are injectable (`build`'s keywords), so the wiring runs in a test with no Blender and
before the compiler has merged. `--appearance assets` first checks the asset library (`asset_render.require_library`:
nothing is generated without one) and writes `CREDITS.md`, the attribution the CC BY assets require, at the dataset
root. The time per stage, seconds per frame and keyframes per script are printed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import asset_fetch, asset_render, job_registry, validate
from . import case as case_mod
from . import episode as ep
from .asset_manifest import AssetError
from .asset_manifest import load as load_manifest
from .generate import generate_script
from .mapspec import load_map
from .seeding import make_rng
from .timeline import Timeline, load

Render = Callable[[Path, list[str]], None]  # (script dir, extra `stbench render` arguments)
Perceive = Callable[[list[Path], int], None]  # (script dirs, workers)
Compile = Callable[[Path, list[str], int], None]  # (dataset, case ids, workers)


def _rendered(paths: case_mod.ScriptPaths, n_frames: int, expect: dict) -> bool:
    """Whether the script is finished: its last keyframe has depth and instances, and its render log equals `expect` on
    every key (a log without `appearance` is a placeholder render)."""
    if not (paths.depth(n_frames - 1).exists() and paths.instances(n_frames - 1).exists()):
        return False
    try:
        log = json.loads(paths.render_log.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(log.get(k, "placeholder" if k == "appearance" else None) == v for k, v in expect.items())


def _stbench(*args: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "stbench.cli", *args])


def _parallel_cli(commands: list[list[str]], workers: int) -> None:
    """Run `stbench` subprocesses, at most `workers` at once; any failure aborts after the running ones finish."""
    running: list[subprocess.Popen] = []
    failed: list[list[str]] = []
    queue = list(commands)
    while queue or running:
        while queue and len(running) < workers:
            running.append(_stbench(*queue.pop(0)))
        done = [p for p in running if p.poll() is not None]
        for p in done:
            running.remove(p)
            if p.returncode:
                failed.append(p.args[3:])
                queue.clear()
        if not done:
            time.sleep(0.2)
    if failed:
        raise SystemExit(f"stbench {' '.join(failed[0])} failed")


def _render_cli(script: Path, flags: list[str]) -> None:
    if _stbench("render", str(script), *flags).wait():
        raise SystemExit(f"stbench render {script} {' '.join(flags)} failed")


def _perceive_cli(scripts: list[Path], workers: int) -> None:
    _parallel_cli([["perceive", str(s)] for s in scripts], workers)


def _compile_cli(dataset: Path, cases: list[str], workers: int) -> None:
    _parallel_cli([["compile", str(dataset), "--cases", ",".join(cases), "--workers", str(workers)]], 1)


def write_credits(out: Path, manifest_path: Path) -> Path:
    """`CREDITS.md` at the dataset root: the attribution CC BY asks for, where no system under test can read it (the
    agent-facing files live under `scripts/<id>/` only)."""
    path = Path(out) / "CREDITS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(asset_fetch.credits(load_manifest(manifest_path)), encoding="utf-8", newline="\n")
    return path


def parse_tasks(text: str) -> tuple[int, int]:
    """`--tasks`: "5" is (5, 5) and "4-6" is (4, 6); ValueError (which argparse reports) for anything else, a count
    under 1 or a range that runs backwards."""
    found = re.fullmatch(r"(\d+)(?:-(\d+))?", text)
    if found is None:
        raise ValueError(f"--tasks is a count or a range of counts, such as 5 or 4-6, not {text!r}")
    bounds = (int(found[1]), int(found[2] or found[1]))
    if not 1 <= bounds[0] <= bounds[1]:
        raise ValueError(f"--tasks needs 1 <= low <= high, not {text!r}")
    return bounds


def tasks_for(tasks: int | tuple[int, int], map_name: str, seed: int, robot: str, job_id: str) -> int:
    """The seed's task count: `tasks` itself when it is a count, else uniform over the range from a key of the seed's
    own that no other draw shares, so `stbench generate` and a rebuild draw the same count."""
    lo, hi = (tasks, tasks) if isinstance(tasks, int) else tasks
    return lo if lo == hi else int(make_rng(map_name, seed, robot, job_id, "tasks").integers(lo, hi + 1))


def generate_checked(map_name: str, seed: int, robot: str, job_id: str, n_tasks: int) -> Timeline:
    """`generate_script`, refused with `validate`'s findings (the job's own checks included), naming the seed, when the
    script is not valid: gaps, drops or people that break one are found here, not after hours of render."""
    tl = generate_script(map_name, seed, robot, job_id, n_tasks)
    problems = validate.validate(tl, load_map(map_name))
    if problems:
        shown = "; ".join(f"{p.code}: {p.message}" for p in problems[:5]) + (" ..." if len(problems) > 5 else "")
        raise ValueError(f"the script of {map_name} seed {seed} ({job_id}, {n_tasks} tasks) is invalid and not "
                         f"written: {shown}")
    return tl


def write_cases(dataset: Path, only: set[str] | None = None) -> list[str]:
    """A case for each window `case.windows` draws over each script of the index (those in `only`, when given), the
    draw seeded from the script's own entry; equal entries reuse their ids, so this is idempotent. No window starts at
    a task whose kind its job `continues` (a `resume`: the case would show that earlier tasks exist). The case ids."""
    ids = []
    for sid, entry in case_mod.read_index(dataset).items():
        if entry["kind"] != "script" or (only is not None and sid not in only):
            continue
        rng = make_rng(entry["map"], entry["seed"], entry["robot"], entry["job"], "windows")
        continues = job_registry.get(entry["job"]).continues
        tasks = load(case_mod.ScriptPaths(dataset, sid).timeline).ego.tasks
        starts = [i for i, t in enumerate(tasks) if t.plan.kind not in continues]
        ids += [case_mod.write_case(dataset, sid, a, b, length).id
                for a, b, length in case_mod.windows(entry["tasks"], rng, starts)]
    return ids


def _generate(job: tuple[str, int, str, str, int | tuple[int, int], str]) -> tuple[str, int]:
    map_name, seed, robot, job_id, tasks, out = job
    tl = generate_checked(map_name, seed, robot, job_id, tasks_for(tasks, map_name, seed, robot, job_id))
    return ep.write_script(out, tl).id, tl.n_keyframes


def build(args: argparse.Namespace, *, render: Render = _render_cli, perceive: Perceive = _perceive_cli,
          compile: Compile = _compile_cli) -> int:
    from .cli import parse_seeds

    out, seeds = Path(args.out), parse_seeds(args.seeds)
    expect: dict = {"appearance": args.appearance, "fast": bool(args.fast), "isp_override": False}
    if args.samples:
        expect["samples"] = args.samples
    flags = ["--appearance", args.appearance] + (["--samples", str(args.samples)] if args.samples else [])
    flags += ["--fast"] if args.fast else []
    if args.appearance == "assets":
        try:
            info = asset_render.require_library()
        except AssetError as e:
            raise SystemExit(f"build: {e}\nnot building with assets: there is no fallback to placeholders") from e
    t0 = time.time()
    jobs = [(args.map, s, args.robot, args.job, args.tasks, str(out)) for s in seeds]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            made = list(pool.map(_generate, jobs))
    else:
        made = [_generate(j) for j in jobs]
    print(f"generate: {len(made)} scripts in {time.time() - t0:.1f} s")
    scripts = [case_mod.ScriptPaths(out, sid) for sid, _ in made]
    frames = {sid: n for sid, n in made}
    if args.appearance == "assets":
        write_credits(out, info.manifest_path)

    t1, rendered = time.time(), 0
    for script in scripts:
        if _rendered(script, frames[script.id], expect):
            continue
        keep = ["--keep-linear", str(Path(args.keep_linear) / script.id)] if args.keep_linear else []
        render(script.root, flags + keep)
        rendered += frames[script.id]
    if rendered:
        print(f"render: {rendered} frames in {time.time() - t1:.1f} s ({(time.time() - t1) / rendered:.2f} s/frame)")

    t2 = time.time()
    perceive([s.root for s in scripts], args.workers)
    print(f"perceive: {len(scripts)} scripts in {time.time() - t2:.1f} s")

    t3 = time.time()
    cases = write_cases(out, {s.id for s in scripts})
    print(f"cases: {len(cases)} in {time.time() - t3:.1f} s")
    t4 = time.time()
    compile(out, cases, args.workers)
    print(f"compile: {len(cases)} cases in {time.time() - t4:.1f} s")
    for (sid, n), seed in zip(made, seeds, strict=True):
        print(f"seed {seed:>3} {sid} {n:>4} keyframes")
    print(f"total {time.time() - t0:.1f} s")
    return 0
