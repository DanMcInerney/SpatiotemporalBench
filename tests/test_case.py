"""Cases: a window of a script's tasks, delivered from step 0; the dataset layout, the private index and the offsets."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from stbench import case as case_mod
from stbench import timeline as T
from stbench.case import CasePaths, ScriptPaths, case_rows, read_case, start_clock, windows, write_case
from stbench.seeding import make_rng
from stbench.visibility import load_pose
from synth import HOME, WAREHOUSE, Synth

KITCHEN = HOME.rooms[0]
SIZES = {"short": (1, 1), "medium": (2, 3), "long": (4, 6)}  # the tasks a window of each length holds (plan §2.3)


def four_tasks(**kw) -> Synth:
    """Four tasks of four keyframes each (k 0-3, 4-7, 8-11, 12-15), 7200 s apart, 1.5 s between keyframes."""
    return Synth(tasks=[[(KITCHEN, 4)] for _ in range(4)], **kw)


def force_case_id(monkeypatch, cid: str) -> None:
    """Every case written from now on claims `cid` (script ids stay random: `episode` claims its own)."""
    monkeypatch.setattr(case_mod, "claim_id", lambda dataset, entry: cid)


def test_every_window_starts_at_zero(tmp_path):
    """Contract (plan §2.3): for every window of a 4-task script, the first rebased row has k 0, status.task 0 and the
    case's own start clock, the first `gap_s` is 0, and a later task carries its real gap and its place in the window.
    Hand values: a task is 4 keyframes 1.5 s apart (its last at 4.5 s) and tasks start 7200 s apart, so the gap before
    a later task is 7200 - 4.5 = 7195.5 s. Regression caught: a window that reveals its place in the script (map
    X-H4)."""
    s = four_tasks()
    sid = s.write(tmp_path).id
    tl = T.load(ScriptPaths(tmp_path, sid).timeline)
    seen = set()
    for a, b, length in windows(4, make_rng("test_case", 1)):
        paths = write_case(tmp_path, sid, a, b, length)
        rec = read_case(paths)
        rows = case_rows(rec, tl, load_pose(paths.script.pose))
        assert [r["k"] for r in rows] == list(range(4 * (b - a + 1)))
        assert rows[0]["status"]["task"] == 0 and rows[0]["clock"] == rec.start_clock + ".0"
        assert [r["status"]["task"] for r in rows[::4]] == list(range(b - a + 1))
        assert case_mod.gap_s(tl, rec, a) == 0.0
        assert [case_mod.gap_s(tl, rec, i) for i in range(a + 1, b + 1)] == [7195.5] * (b - a)
        seen.add(length)
    assert seen == {"short", "medium", "long"}  # four tasks hold all three lengths


def test_a_window_rebases_clock_task_and_k_by_hand(tmp_path):
    """Hand case: the window of tasks 1 and 2 starts at k 4 (script time 7200 s), so its case k 7 is script k 11, and
    the case's second task begins 7200 s into the window: its first row's clock is the start clock plus exactly two
    hours, and its tasks number 0 and 1. Regression caught: an offset applied to k but not to the clock or the task."""
    s = four_tasks()
    sid = s.write(tmp_path).id
    paths = write_case(tmp_path, sid, 1, 2, "medium")
    rec = read_case(paths)
    assert (rec.a, rec.b, rec.k0, rec.k1, rec.t0, rec.length) == (1, 2, 4, 12, 7200.0, "medium")
    assert (case_mod.case_k(rec, 11), case_mod.script_k(rec, 7), case_mod.case_task(rec, 2)) == (7, 11, 1)
    assert datetime.fromisoformat(case_mod.case_clock(rec, 7201.5)) == (
        datetime.fromisoformat(rec.start_clock) + timedelta(seconds=1.5))
    rows = case_rows(rec, T.load(ScriptPaths(tmp_path, sid).timeline), load_pose(paths.script.pose))
    assert datetime.fromisoformat(rows[4]["clock"]) == datetime.fromisoformat(rec.start_clock) + timedelta(hours=2)
    assert [r["k"] for r in rows] == list(range(8)) and rows[4]["status"]["task"] == 1


def test_a_delivered_clock_carries_tenths_from_the_true_script_time(tmp_path):
    """Contract (0.10.0: accurate timestamps; `case.case_clock`): a step at script time t is delivered with the
    case's start clock plus (t minus the window's first step's time), to the tenth of a second and always as
    `YYYY-MM-DDTHH:MM:SS.s` with one decimal. Hand-built case: the window of tasks 1 and 2 of `four_tasks` starts at
    script time 7200 s, its keyframes are 1.5 s apart, so its eight rows are the start clock plus 0.0, 1.5, 3.0, 4.5
    s for the first task and, 7200 s later in the script, plus 7200.0, 7201.5, 7203.0, 7204.5 s for the second; the
    task message's clock is the first row's. By hand at the edges: a start clock of 07:59:59 plus 1.25 s is 08:00:00.3
    (half up), plus 0.04 s is 07:59:59.0, plus 0.06 s is 07:59:59.1, and plus 0.95 s is 08:00:00.0 (a carry into the
    second keeps its decimal), the script's own clock (`timeline.clock_at`) staying to the second. Regression caught: a
    clock still rounded to the second (the gold of `closest_approach` then moves by more than C on 6% of person rows),
    a clock whose decimal is dropped when it is 0, or one that reads the rounded script clock instead of the true
    time."""
    sid = four_tasks().write(tmp_path).id
    paths = write_case(tmp_path, sid, 1, 2, "medium")
    rec = read_case(paths)
    tl = T.load(ScriptPaths(tmp_path, sid).timeline)
    start = datetime.fromisoformat(rec.start_clock)
    rows = case_rows(rec, tl, load_pose(paths.script.pose))
    offsets = [0.0, 1.5, 3.0, 4.5, 7200.0, 7201.5, 7203.0, 7204.5]
    assert [datetime.fromisoformat(r["clock"]) - start for r in rows] == [timedelta(seconds=s) for s in offsets]
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d", r["clock"]) for r in rows)
    assert rows[0]["clock"] == rec.start_clock + ".0" and rows[1]["clock"].endswith(".5")
    edge = case_mod.CaseRecord("c", "s", 0, 0, 0, 1, 100.0, "2026-03-04T07:59:59", "short")
    assert [case_mod.case_clock(edge, 100.0 + s) for s in (0.0, 0.04, 0.06, 0.95, 1.25)] == [
        "2026-03-04T07:59:59.0", "2026-03-04T07:59:59.0", "2026-03-04T07:59:59.1", "2026-03-04T08:00:00.0",
        "2026-03-04T08:00:00.3"]
    assert case_mod.format_clock(datetime(2026, 3, 4, 8, 0, 0)) == "2026-03-04T08:00:00.0"
    assert T.clock_at("2026-03-04T07:59:59", 1.25) == "2026-03-04T08:00:00"   # the script's own clock: to the second


def test_case_rows_leave_their_input_alone(tmp_path):
    """Contract (review Q-4): `case_rows` returns rebased deep copies and never changes the pose rows it is given, so
    one script's rows can be rebased for several windows (a script with several cases, each at its own offset). Hand
    case: the window of tasks 1-2 rebases `status.task` by 1 and `k` by 4; run twice on one list, the list is unchanged
    and both results are equal, and editing a result changes no input row. Regression caught: rows rebased in place, so
    the second window of a script (or a rerun) starts from the first one's already shifted `k`, task and clock."""
    sid = four_tasks().write(tmp_path).id
    paths = write_case(tmp_path, sid, 1, 2, "medium")
    rec, tl = read_case(paths), T.load(ScriptPaths(tmp_path, sid).timeline)
    rows = load_pose(paths.script.pose)
    before = copy.deepcopy(rows)
    first = case_rows(rec, tl, rows)
    assert rows == before
    assert case_rows(rec, tl, rows) == first and [r["k"] for r in first] == list(range(8))
    first[0]["status"]["task"] = 99
    first[0]["camera"]["pos"][0] = 99.0
    assert rows == before


def test_the_prefix_never_reaches_the_case(tmp_path, monkeypatch):
    """Contract: two scripts that share a window's tasks but differ in what comes before it (the number of keyframes,
    the seconds the earlier tasks take, the status they ended in) give byte-equal rebased rows for that window, once the
    two cases hold the same id (so the same start clock, which is the case's own). Regression caught: an offset leaking
    through k, clock, task, battery or phase."""
    force_case_id(monkeypatch, "0123456789abcdef")
    one = Synth(tasks=[[(KITCHEN, 4)], [(KITCHEN, 5)]])
    two = Synth(tasks=[[(KITCHEN, 7)], [(KITCHEN, 5)]], gap_s=9000.0, status={0: {"battery_pct": 91}})
    rows = []
    for name, s in (("one", one), ("two", two)):
        sid = s.write(tmp_path / name).id
        paths = write_case(tmp_path / name, sid, 1, 1, "short")
        tl = T.load(ScriptPaths(tmp_path / name, sid).timeline)
        rows.append(json.dumps(case_rows(read_case(paths), tl, load_pose(paths.script.pose)), sort_keys=True))
    assert rows[0] == rows[1] and len(json.loads(rows[0])) == 5


def test_a_window_never_splits_a_task(tmp_path):
    """Contract: a window's keyframes [k0, k1) are task bounds and its sizes are the plan's (`SIZES`: 1; 2-3; 4-6
    tasks), one window for each length whose minimum fits; a window of one task is `(0, 0, "short")`. Hand values: 2 and
    3 tasks hold a short and a medium window, 4 to 9 all three. Regression caught: a case starting mid-task, or
    `LENGTHS` changed without the plan (a medium window of 2-4 tasks)."""
    assert case_mod.LENGTHS == SIZES  # the one place the table is read; every expectation below is from the hand table
    rng = make_rng("test_case", 2)
    assert windows(1, rng) == [(0, 0, "short")]
    for n, names in ((2, ["short", "medium"]), (3, ["short", "medium"]), (4, ["short", "medium", "long"]),
                     (6, ["short", "medium", "long"]), (9, ["short", "medium", "long"])):
        for _ in range(40):
            found = windows(n, rng)
            assert [w[2] for w in found] == names
            for a, b, length in found:
                lo, hi = SIZES[length]
                assert 0 <= a <= b < n and lo <= b - a + 1 <= min(hi, n)
    s = Synth(tasks=[[(KITCHEN, 3 + i)] for i in range(6)])  # tasks of different lengths
    sid = s.write(tmp_path).id
    tl = T.load(ScriptPaths(tmp_path, sid).timeline)
    ends = {t.k0: t.k1 for t in tl.ego.tasks}
    for a, b, length in windows(6, rng):
        rec = read_case(write_case(tmp_path, sid, a, b, length))
        assert rec.k0 in ends and rec.k1 in ends.values() and (rec.k0, rec.k1) == (tl.ego.tasks[a].k0,
                                                                                   tl.ego.tasks[b].k1)
    with pytest.raises(ValueError, match="short"):
        write_case(tmp_path, sid, 0, 1, "short")  # two tasks are no short window
    with pytest.raises(ValueError, match="length"):
        write_case(tmp_path, sid, 0, 0, "huge")
    with pytest.raises(ValueError, match="window"):
        write_case(tmp_path, sid, 5, 6, "medium")  # past the last task


def test_windows_draw_both_the_first_task_and_the_size():
    """Contract (plan §2.3): a window's size is uniform in [lo, min(hi, n)] and its first task uniform among those that
    leave room for it, so over many draws every (size, first task) pair of a length occurs and no other. Hand values for
    6 tasks: a short window has size 1 and its first task is any of 0-5; a medium one has size 2 (first task 0-4) or 3
    (0-3); a long one has size 4 (0-2), 5 (0-1) or 6 (0). For 2 tasks the medium window can only be both of them.
    Regression caught: a first task always 0 or a size always the largest, which a bounds check alone does not see
    (every window at the script's start, so no case would ever test a later offset)."""
    seen: dict[str, set[tuple[int, int]]] = {"short": set(), "medium": set(), "long": set()}
    for i in range(400):
        for a, b, length in windows(6, make_rng("test_case", "windows", i)):
            seen[length].add((b - a + 1, a))
    assert seen == {"short": {(1, a) for a in range(6)},
                    "medium": {(2, a) for a in range(5)} | {(3, a) for a in range(4)},
                    "long": {(4, a) for a in range(3)} | {(5, a) for a in range(2)} | {(6, 0)}}
    assert {tuple(windows(2, make_rng("test_case", "two", i))[1][:2]) for i in range(40)} == {(0, 1)}


def test_the_start_clock_comes_from_the_case_id_alone(tmp_path, monkeypatch):
    """Contract (review V-2, V-8): a case's start clock is drawn from the case's own random id and nothing else, so a
    delivered clock names neither the script's map, seed, robot or job, nor the window, nor how much precedes it. Hand
    cases: cases with one id, over a home script (seed 1, window of task 1), a script of other seed and other task
    lengths (window of tasks 2-3) and a warehouse script of another robot and job (window of tasks 0-3), all get one
    clock; 24 ids over one window draw more than 12 different clocks. Regression caught: a clock drawn from the map,
    seed, robot, job or window (a delivered clock then names the window's first task and the seed), or the same clock
    for every case."""
    cases = [
        (four_tasks(), 1, 1, "short"),
        (Synth(tasks=[[(KITCHEN, 6)], [(KITCHEN, 2)], [(KITCHEN, 5)], [(KITCHEN, 4)]], seed=7), 2, 3, "medium"),
        (Synth(tasks=[[(WAREHOUSE.rooms[0], 3)] for _ in range(6)], stage=WAREHOUSE, seed=3), 0, 3, "long"),
    ]
    force_case_id(monkeypatch, "0123456789abcdef")
    same = {read_case(write_case(tmp_path / str(i), s.write(tmp_path / str(i)).id, a, b, length)).start_clock
            for i, (s, a, b, length) in enumerate(cases)}
    assert len(same) == 1
    script = four_tasks().write(tmp_path / "ids").id
    clocks = set()
    for i in range(24):
        force_case_id(monkeypatch, f"{i:016x}")
        clocks.add(read_case(write_case(tmp_path / "ids", script, 1, 1, "short")).start_clock)
    assert len(clocks) > 12


def test_start_clock_is_a_2026_morning_drawn_from_the_generator_it_is_given():
    """Contract: `start_clock(rng)` is a 2026 date and a start time from 07:00 to 09:59 on a whole second, the same for
    the same generator, spread over the year and over the three hours. Regression caught: a clock outside the 07:00 to
    10:00 working morning every task starts in, or one that does not vary (every case at one time)."""
    clocks = [datetime.fromisoformat(start_clock(make_rng("clock", i))) for i in range(400)]
    assert all(c.year == 2026 and c.second == 0 and 7 * 60 <= c.hour * 60 + c.minute < 10 * 60 for c in clocks)
    assert {c.hour for c in clocks} == {7, 8, 9}
    assert min(c.month for c in clocks) <= 2 and max(c.month for c in clocks) >= 11  # spread over the year
    assert start_clock(make_rng("clock", 1)) == start_clock(make_rng("clock", 1)) != start_clock(make_rng("clock", 2))


def test_importing_case_pulls_in_no_script_generator():
    """Contract (review V-8): `case.py` owns the clock distribution (`start_clock`), so importing it does not
    import `generate`, whose own changes (a script's working hours) then cannot move every case's clock. Regression
    caught: `case` importing the generator for its clock draw, as it did before the review."""
    out = subprocess.run([sys.executable, "-c", "import sys, stbench.case; print('stbench.generate' in sys.modules)"],
                         capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "1"}).stdout
    assert out.strip() == "False"


def test_case_ids_reuse_equal_entries(tmp_path):
    """Contract: the same (script, a, b) twice gives one id, one index entry and one start clock (drawn from the id),
    another window a new one; a script written twice keeps its id. Regression caught: duplicate cases after a rebuild
    (map D-17), or a rebuilt case whose clock changed under its id."""
    s = four_tasks()
    first = s.write(tmp_path)
    assert s.write(tmp_path).id == first.id
    one = write_case(tmp_path, first.id, 1, 1, "short")
    again = write_case(tmp_path, first.id, 1, 1, "short")
    assert again.id == one.id and read_case(again).start_clock == read_case(one).start_clock  # an equal entry's clock
    other = write_case(tmp_path, first.id, 2, 2, "short")
    assert other.id != one.id and CasePaths.from_id(tmp_path, one.id).script.id == first.id
    index = case_mod.read_index(tmp_path)
    assert len(index) == 3 and index[one.id] == {"kind": "case", "script": first.id, "a": 1, "b": 1, "length": "short"}
    assert index[first.id] == {"kind": "script", "map": "apartment_a", "seed": 1, "robot": "vacuum",
                               "job": "vacuum/1", "tasks": 4, "split": "dev"}


def test_a_case_record_carries_no_schema_and_one_of_an_earlier_build_still_reads(tmp_path):
    """Contract: `case.json` holds the window and nothing else, and a `case.json` of an earlier build (with a `schema`)
    reads to the same record. Regression caught: a record that writes or requires a schema string, which stops every
    dataset built before the change from compiling or scoring."""
    case = write_case(tmp_path, four_tasks().write(tmp_path).id, 1, 2, "medium")
    doc = json.loads(case.record.read_text(encoding="utf-8"))
    assert set(doc) == {"case", "script", "a", "b", "k0", "k1", "t0", "start_clock", "length"}
    rec = read_case(case)
    case.record.write_text(json.dumps({**doc, "schema": "old"}), encoding="utf-8")
    assert read_case(case) == rec


def test_claim_id_finds_an_entry_written_with_the_stamps_of_an_earlier_build(tmp_path):
    """Contract: an index entry of an earlier build (a `schema` in every entry, a `generator` in a script's) reads back
    as the entry the code builds now, so `claim_id` reuses its id and writing an old dataset's script again makes no
    duplicate. Regression caught: a reader that keeps a stamp, so no entry the code builds ever equals an old one."""
    entry = {"kind": "script", "map": "apartment_a", "seed": 1, "robot": "vacuum", "job": "vacuum/1", "tasks": 4,
             "split": "dev"}
    old = {"schema": "old", **entry, "generator": "old"}
    folder = case_mod.index_dir(tmp_path)
    folder.mkdir(parents=True)
    (folder / ("a" * 16 + ".json")).write_text(json.dumps(old), encoding="utf-8")
    assert case_mod.read_entry(tmp_path, "a" * 16) == entry
    assert case_mod.read_index(tmp_path) == {"a" * 16: entry}
    assert case_mod.claim_id(tmp_path, entry) == "a" * 16
    assert case_mod.claim_id(tmp_path, {**entry, "seed": 2}) != "a" * 16


def claim_entry(args) -> str:
    """One worker's write: claim an id for a case entry and write it (run in a pool, so at module level)."""
    dataset, i = args
    entry = {"kind": "case", "script": "0" * 16, "a": i, "b": i, "length": "short"}
    iid = case_mod.claim_id(dataset, entry)
    case_mod.write_entry(dataset, iid, entry)
    return iid


def test_index_writes_are_parallel_safe(tmp_path):
    """Contract: the private index is one small file per entry, so 20 entries written from a pool of 4 processes at once
    leave 20 readable files, one per entry, no id taken twice and no temporary file left over. Regression caught: a
    shared index file that parallel `stbench build` workers overwrite with a stale copy (lost entries), or a temporary
    file left in the index folder (`read_index` would aggregate it). A torn file is the next test's."""
    with ProcessPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(claim_entry, [(tmp_path, i) for i in range(20)]))
    files = sorted((tmp_path / "private" / "index").iterdir())
    assert len(set(ids)) == 20 and [f.name for f in files] == sorted(f"{i}.json" for i in ids)
    index = case_mod.read_index(tmp_path)
    assert sorted(e["a"] for e in index.values()) == list(range(20))
    assert all("schema" not in json.loads(f.read_text(encoding="utf-8")) for f in files)


def rewrite_entries(args) -> int:
    """One writer: rewrite each of the entries `rounds` times, a new window every round (run in a pool)."""
    dataset, ids, rounds = args
    for r in range(rounds):
        for iid in ids:
            case_mod.write_entry(dataset, iid, {"kind": "case", "script": "0" * 16, "a": r, "b": r, "length": "short"})
    return rounds


def read_entries(args) -> int:
    """One reader: `rounds` times the whole index, each entry and a claim, every entry seen whole (run in a pool). An
    entry caught mid-swap may read as absent (NTFS replaces non-atomically), so a missing one is skipped, a torn one
    is not."""
    dataset, ids, rounds = args
    for _ in range(rounds):
        seen = list(case_mod.read_index(dataset).items())
        seen += [(iid, e) for iid in ids if (e := case_mod.read_entry(dataset, iid)) is not None]
        for iid, entry in seen:
            assert iid in ids and entry["kind"] == "case" and entry["a"] == entry["b"], (iid, entry)
        case_mod.claim_id(dataset, {"kind": "case", "script": "1" * 16, "a": 0, "b": 0, "length": "short"})
    return rounds


def test_index_reads_and_writes_race_without_raising(tmp_path):
    """Contract: readers and writers of the private index never raise while they run at once, and every entry a reader
    sees is whole. Four processes rewrite the same 6 entries 100 times each and four read the index, each entry and a
    claim 100 times, so readers open files while writers swap them in thousands of times; on Windows the open or the
    swap then fails with `PermissionError` unless both retry (8 of 8 runs failed with no retry on the read; one
    free-suite run under load failed the same way). Regression caught: a read that does not retry the swap
    (`read_entry` in `claim_id` died in a `stbench build` worker), a retry only on the write side, or `exists()` ahead
    of the read (on Windows it says False for a file mid-swap, so the retry never runs)."""
    ids = [f"{i:016x}" for i in range(6)]
    for iid in ids:
        case_mod.write_entry(tmp_path, iid, {"kind": "case", "script": "0" * 16, "a": 0, "b": 0, "length": "short"})
    jobs = (tmp_path, ids, 100)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(rewrite_entries, jobs) for _ in range(4)] + [pool.submit(read_entries, jobs)
                                                                            for _ in range(4)]
        assert [f.result() for f in futures] == [100] * 8  # an exception in any worker re-raises here
    assert sorted(case_mod.read_index(tmp_path)) == ids
    assert not [f for f in (tmp_path / "private" / "index").iterdir() if f.suffix == ".tmp"]


def test_an_index_file_is_swapped_in_whole(tmp_path, monkeypatch):
    """Contract: `write_entry` writes the whole entry to a temporary file and swaps it in with `os.replace`, so a reader
    sees the old entry or the new one and never half of one. Hand case: two writes to one id; at each swap the temporary
    file already holds the complete new entry and the target is absent the first time and holds the
    complete first entry the second time. Regression caught: a direct write to the entry file, which a concurrent
    reader (a parallel `stbench build` worker reading the index) can see torn."""
    swaps = []
    real_replace = os.replace

    def spy(src, dst):
        target = Path(dst)
        swaps.append((json.loads(Path(src).read_text(encoding="utf-8")),
                      json.loads(target.read_text(encoding="utf-8")) if target.exists() else None))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    one = {"kind": "case", "script": "0" * 16, "a": 0, "b": 0, "length": "short"}
    two = {**one, "a": 1, "b": 1}
    case_mod.write_entry(tmp_path, "f" * 16, one)
    case_mod.write_entry(tmp_path, "f" * 16, two)
    assert swaps == [(one, None), (two, one)]
    assert case_mod.read_entry(tmp_path, "f" * 16) == two
    assert [f.name for f in (tmp_path / "private" / "index").iterdir()] == ["f" * 16 + ".json"]


def test_paths_name_one_layout(tmp_path):
    """Contract: every agent-facing property of `ScriptPaths` lies under `scripts/<sid>`, every private one under
    `private/<sid>`, and a case's under `private/<cid>`; `from_dir` takes the agent folder or the private one and
    refuses anything else, the old `episodes/<id>` layout and a case's folder included. Regression caught: a reader
    still building the old `episodes/` path, or a path outside both folders."""
    s = four_tasks()
    script = s.write(tmp_path)
    case = write_case(tmp_path, script.id, 0, 0, "short")
    agent = {"robot", "sensors", "camera", "pose", "rgb_dir", "perception", "oracle"}
    properties = {n for n in dir(ScriptPaths) if isinstance(getattr(ScriptPaths, n), property)}
    assert agent <= properties
    for name in properties:
        path = getattr(script, name)
        assert (script.root if name in agent else script.private) in path.parents, name
    assert script.root == tmp_path / "scripts" / script.id and script.private == tmp_path / "private" / script.id
    for name in (n for n in dir(CasePaths) if isinstance(getattr(CasePaths, n), property)):
        assert case.private in getattr(case, name).parents, name
    assert case.private == tmp_path / "private" / case.id
    assert script.rgb(7) == script.root / "sensors" / "rgb" / "000007.jpg"
    assert script.depth(7) == script.private / "depth" / "000007.png"   # the scorer's, never under scripts/ (0.6.0)
    assert script.instances(7) == script.private / "instances" / "000007.png"
    assert ScriptPaths.from_dir(script.root).id == ScriptPaths.from_dir(script.private).id == script.id
    assert CasePaths.from_dir(case.private).id == case.id and CasePaths.from_dir(case.private).script.id == script.id
    for bad in (tmp_path, tmp_path / "episodes" / script.id, case.private, script.root / "sensors"):
        with pytest.raises(ValueError):
            ScriptPaths.from_dir(bad)
    with pytest.raises(ValueError, match="not a case"):
        CasePaths.from_dir(script.private)
