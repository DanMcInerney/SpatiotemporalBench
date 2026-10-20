"""`stbench view`'s reader: a run folder, and the dataset it ran on, as one plain dict (the page's data).

RUN_DIR is a run folder of `stbench eval` (`results.json`, `cases/<cid>/`), a folder of such run folders (the arms of
`--arms`, each listed by its folder's name) or the folder of one `stbench run` (`summary.json` beside the case's
files). Everything is read from the folders as they stand, so a run that is still going, was killed, or lacks a
`results.json` is shown for what it holds; a file that is missing or cut is a sentence in the case's `problems`, never
an error (`view_files`). The dict is a function of the files alone (no clock, folders in sorted order, keys in the order
the code lists them), so reading the same folders twice gives the same bytes. It holds the gold, so it belongs beside a
run, never under the dataset's `scripts/` or `private/` (`view_cmd`).

A case is the union of what `results.json` lists, what its `summary.json` says, its rows and the dataset's index; the
later of the first three wins where they differ, and `summary.json` decides the status (it is written last). Tokens
and cost are what the case reported (`eval_results.resources`, `prices`), never recomputed here.

Each case also names its stream, the steps the harness delivered on the run's channel (`view_frames`), which the page
scrubs frame by frame; the streams are the dict's `streams`, one for each case and channel however many arms ran it.
Given a folder, `read_run` draws the depth pictures of those streams into it (`view_depth`); the dict is the same
either way but for the address of that folder.
"""

from __future__ import annotations

from pathlib import Path

from . import case as case_mod
from . import eval_results as results
from .eval_systems import CASES
from .stream_protocol import DEFAULT_CHANNEL
from .view_files import json_object, jsonl
from .view_frames import Streams
from .view_log import read_log, read_transport
from .view_questions import counts, merge, read_case_files

INFO_KEYS = ("script", "layout", "robot", "job", "length")
HEADER_KEYS = ("system", "channel", "brief", "during", "sandbox", "reasoning", "text_only", "memory", "compact_at")
TOKEN_KEYS = ("tokens_in", "tokens_cached_in", "tokens_cache_write_in", "tokens_out")
KINDS = ("complete", "ended", "invalid", "refused", "unfinished")   # the colour classes of a status; else `pending`


def find_dataset(run_dir: Path) -> Path | None:
    """The dataset RUN_DIR sits in (`DATASET/runs/<name>[/<arm>]`): the nearest folder above it that holds `private/`
    and `scripts/`, None for a run folder anywhere else."""
    here = run_dir.resolve()
    return next((p for p in [here, *here.parents][:6]
                 if (p / case_mod.PRIVATE).is_dir() and (p / case_mod.SCRIPTS).is_dir()), None)


def _is_run(folder: Path) -> bool:
    return any((folder / name).exists() for name in (CASES, results.RESULTS, results.SUMMARY))


def run_folders(run_dir: Path) -> list[tuple[str, Path]]:
    """`[(arm name, folder)]`: RUN_DIR (name "") if it is a run folder, else its subfolders that are, by name."""
    if _is_run(run_dir):
        return [("", run_dir)]
    return [(p.name, p) for p in sorted(run_dir.iterdir()) if p.is_dir() and _is_run(p)]


def _case_dirs(folder: Path) -> dict[str, Path]:
    """The case folders of a run folder by case id (the folder of one `stbench run` is its own case)."""
    if (folder / CASES).is_dir():
        return {p.name: p for p in sorted((folder / CASES).iterdir()) if p.is_dir()}
    summary = json_object(folder / results.SUMMARY)
    return {} if summary is None else {str(summary.get("case") or folder.name): folder}


def _seconds(wall) -> dict[str, float]:
    """`wall_s` as one sum a key: `summary.json` lists the seconds of each message, `results.json` has the sums."""
    return {key: round(sum(v) if isinstance(v, list) else v, 3) for key, v in (wall or {}).items()
            if isinstance(v, list | int | float)}


def _status(summary: dict | None, entry: dict | None, has_files: bool) -> str:
    if summary and summary.get("status"):
        return str(summary["status"])
    if entry and entry.get("status") not in (None, "pending"):
        return str(entry["status"])
    return "unfinished (no summary.json)" if has_files else "not started"


def _info(dataset: Path | None, cid: str, record: dict | None, *sources: dict) -> dict:
    """script, layout, robot, job and length: the dataset's index first, then each of `sources` that states one."""
    index = (case_mod.read_entry(dataset, cid) or {}) if dataset is not None else {}
    sid = index.get("script") or (record or {}).get("script")
    script = (case_mod.read_entry(dataset, sid) or {}) if dataset is not None and sid else {}
    info = {"script": sid, "layout": script.get("map"), "robot": script.get("robot"), "job": script.get("job"),
            "length": index.get("length")}
    for source in sources:
        info.update({key: source[key] for key in INFO_KEYS if source.get(key) is not None})
    return info


def _first_summary(folders: dict[str, Path]) -> dict:
    """The first `summary.json` of the run's cases, the run's options being the same in each."""
    return next((s for p in folders.values() if (s := json_object(p / results.SUMMARY)) is not None), {})


def _stream(streams: Streams, cid: str, said: dict, channel: str | None, problems: list[str]) -> str | None:
    """The key of the case's stream (`view_frames`), on the channel the case states, else the run's, else the default
    one, which the case's problems then say."""
    use = said.get("channel") or channel
    if use is None:
        problems.append(f"the run does not record its channel: the stream is shown as the {DEFAULT_CHANNEL} one")
    return streams.get(cid, use or DEFAULT_CHANNEL, problems)


def _case(cid: str, folder: Path | None, entry: dict | None, dataset: Path | None, listed: bool, streams: Streams,
          channel: str | None) -> dict:
    """One case from its folder, its entry in `results.json` and the dataset; `listed` is false for a case folder that
    `results.json` (which the run has) does not list. `channel` is the run's, for a case that does not state its own."""
    problems: list[str] = []
    summary = None if folder is None else json_object(folder / results.SUMMARY, problems)
    rows = [] if folder is None else jsonl(folder / results.ROWS, problems)
    answers = [] if folder is None else jsonl(folder / "answers.jsonl", problems)
    record, questions, golds = read_case_files(dataset, cid, problems)
    info = _info(dataset, cid, record, rows[0] if rows else {}, summary or {}, entry or {})
    said = summary if summary is not None else entry or {}   # the fields both state: the summary's, being later
    usage = said.get("usage") if isinstance(said.get("usage"), dict) else None
    spent = results.resources(usage)
    status = _status(summary, entry, folder is not None and any(folder.iterdir()))
    log = None if folder is None else read_log(folder)
    wall = _seconds(said.get("wall_s"))
    stream = None if record is None else _stream(streams, cid, said, channel, problems)
    return {
        "id": cid, **info, "status": status, "kind": next((k for k in KINDS if status.startswith(k)), "pending"),
        "listed": listed, "ran": summary is not None, "error": (entry or {}).get("error"),
        "failures": said.get("failures") or {},
        "problems": [*problems, *(log["problems"] if log else [])],
        "window": None if record is None else {k: record.get(k) for k in ("a", "b", "k0", "k1", "start_clock")},
        "wall_s": round(sum(wall.values()), 3) if wall else None, "wall_by": wall,
        "state_bytes": said.get("state_bytes") or {}, "tokens": spent["tokens"],
        "cost_usd": said["cost_usd"] if "cost_usd" in said else spent["cost_usd"],
        "cost_note": said["cost_note"] if "cost_note" in said else spent["cost_note"],
        "usage_problems": spent["usage_problems"], "counts": counts(rows),
        "questions": merge(dataset, info["script"], (record or {}).get("k0"), questions, golds, rows, answers),
        "transport": None if folder is None else read_transport(folder), "log": log, "stream": stream,
    }


def _header(doc: dict | None, cases: list[dict], folders: dict[str, Path]) -> dict:
    """The run's options: `results.json`'s when it has them, else the first `summary.json`'s, with the model, the
    reasoning effort, the memory and the compaction limit from the system's own summary where it states them."""
    first = _first_summary(folders)
    head = {key: (doc if doc is not None else first).get(key) for key in HEADER_KEYS}
    own = next((t["summary"] for c in cases if (t := c["transport"])), {})
    head |= {"model": own.get("model"), "reasoning": head["reasoning"] or own.get("reasoning"),
             "memory": head["memory"] or own.get("memory"),
             "compact_at": own.get("compact_at") if head["compact_at"] is None else head["compact_at"]}
    head |= {key: (doc or {}).get(key) for key in ("verified", "partial", "max_usd")}
    # eval writes `results.json` when the run begins and when it ends, so its `partial: N of M cases finished` note is
    # that of the start (0) until then: the page counts the finished case folders itself and says so (`totals`).
    head["notes"] = [n for n in (doc or {}).get("notes") or [] if not str(n).startswith("partial:")]
    return head


def _layouts(doc: dict | None) -> dict:
    """The headline per layout (no number pools layouts) with its `now` and `memory` columns (`columns`), the secondary
    means (`abilities_mean`, `pooled`) and each ability with its count (`by.ability`: `abilities`, a diagnostic one
    flagged), exactly as `results.json` records them, None for a layout whose aggregate it holds as null. The viewer
    computes no statistic of its own: a run with no `results.json` yet has no layouts here, and the page says the
    headline comes from the file eval writes when the run ends."""
    layouts = (doc or {}).get("layouts")
    if not isinstance(layouts, dict):
        return {}
    keys = ("n", "cases", "scripts", "headline", "columns", "abilities_mean", "pooled", "thin", "counts")
    return {name: None if agg is None else {**{k: agg.get(k) for k in keys},
                                            "abilities": (agg.get("by") or {}).get("ability")}
            for name, agg in layouts.items()}


def _totals(cases: list[dict]) -> dict:
    """Cases finished of all, and tokens, cost and wall time summed over the cases that ran (a figure a case did not
    report is left out; `unpriced` counts the cases that ran without a cost). A case folder that `results.json` does
    not list (left by another selection) is shown but counted in `unlisted` only."""
    unlisted, cases = sum(1 for c in cases if not c["listed"]), [c for c in cases if c["listed"]]
    ran = [c for c in cases if c["ran"]]
    tokens = {key: [c["tokens"][key] for c in ran if c["tokens"][key] is not None] for key in TOKEN_KEYS}
    costs = [c["cost_usd"] for c in ran if c["cost_usd"] is not None]
    return {"cases": len(cases), "unlisted": unlisted, "finished": sum(1 for c in cases if c["kind"] == "complete"),
            "tokens": {key: sum(found) if found else None for key, found in tokens.items()},
            "cost_usd": round(sum(costs), 6) if costs else None, "unpriced": len(ran) - len(costs),
            "wall_s": round(sum(c["wall_s"] or 0 for c in cases), 3)}


def read_arm(name: str, folder: Path, dataset: Path | None, streams: Streams | None = None) -> dict:
    """One run folder: its header, the headline per layout (`results.json`'s), the totals, how many cases `results.json`
    lists as finished (`results`; the totals count the case folders) and every case (those `results.json` lists, in its
    order, then any other case folder). The cases' streams are added to `streams`, which draws no depth picture unless
    it was given a folder."""
    problems: list[str] = []
    doc = json_object(folder / results.RESULTS, problems)
    if doc is not None and (found := results.problems(doc)):
        problems.append(f"{results.RESULTS} is not a results document ({found[0]})")
    folders = _case_dirs(folder)
    entries = {c["id"]: c for c in (doc or {}).get("cases") or [] if isinstance(c, dict) and "id" in c}
    stated = None if doc is None else sum(1 for c in entries.values() if c.get("status") == "complete")
    channel = (doc if doc is not None else _first_summary(folders)).get("channel")
    streams = streams or Streams(dataset)
    cases = [_case(cid, folders.get(cid), entries.get(cid), dataset, doc is None or cid in entries, streams, channel)
             for cid in [*entries, *(cid for cid in folders if cid not in entries)]]
    return {"name": name, "dir": folder.resolve().as_posix(), "header": _header(doc, cases, folders),
            "layouts": _layouts(doc), "totals": _totals(cases), "problems": problems,
            "results": {"present": doc is not None, "finished": stated}, "cases": cases}


def read_run(run_dir: Path, dataset: Path | None, depth_dir: Path | None = None) -> dict:
    """The page's data: `{"run", "dataset", "depth_dir" (where the depth pictures are drawn, null if nowhere),
    "streams" (by the key a case names), "arms": [read_arm(...)]}`. With `depth_dir` the depth pictures of the streams
    are drawn there, as this reads."""
    streams = Streams(dataset, depth_dir)
    arms = [read_arm(name, folder, dataset, streams) for name, folder in run_folders(run_dir)]
    return {"run": run_dir.resolve().as_posix(),
            "dataset": None if dataset is None else dataset.resolve().as_posix(),
            "depth_dir": None if depth_dir is None else depth_dir.resolve().as_posix(),
            "streams": streams.items, "arms": arms}
