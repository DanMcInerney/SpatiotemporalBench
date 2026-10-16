"""What the `stbench view` tests share (`test_run_view.py`, `test_view_page.py`): a real `eval` run over the hand
dataset of `eval_kit` and the page's embedded data read back as a browser would see it.

`make_run` is the two-case `hand_dataset` (the first case's golds are [answerable, answerable, unknown], the second's
[answerable, unknown, unknown]; under `all_unknown` an answerable gold scores 0, abstained, and an unknown one 1,
correct) with an `all_unknown` run at `DATASET/runs/floor`, where `stbench view` finds the dataset by itself.
`window_case` and `window_run` are the stream's own: a case over the last two of three tasks, and a run of it on a
channel. `start_stamp` turns a finished run's `results.json` into the stamp eval writes at the start of a run.
`PageParts` is the page parsed with the standard library's HTML parser: the start tags it holds and the text of each
script block, by its `type` (`application/json` for the data, none for the code).
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from eval_kit import cli, compiled, hand_dataset
from stbench import case as case_mod
from stream_kit import during_q, end_q
from synth import depth_with_wall, three_tasks


def make_run(root: Path) -> tuple[list[case_mod.CasePaths], Path]:
    """(the two cases, the run folder `root/runs/floor`) of an `all_unknown` run over the hand dataset at `root`."""
    cases = hand_dataset(root)
    run = root / "runs" / "floor"
    assert cli("all_unknown", "--data", root, "--out", run) == 0
    return cases, run


def window_case(root: Path) -> case_mod.CasePaths:
    """The case the stream's tests read (`test_view_frames.py`, the page's stream test): `synth.three_tasks`, 40
    keyframes in tasks of k 0-15, 16-31 and 32-39 with decision points at 5 and 20, a red mug seen all along and a blue
    book seen from k 22, and the case over tasks 1 and 2: script keyframes 16 to 39, so its step 0 is keyframe 16, its
    task 0 has 16 steps and its task 1 has 8. It is asked a during question at keyframe 20 (step 4) and an end one.
    Keyframe 20's depth image is 2 m (2,000 mm) across its left 320 columns, nothing (0) in its top 10 rows and 6 m
    elsewhere; every other keyframe's is 6 m throughout."""
    synth = three_tasks()
    synth.decisions = [(5, "before_doorway"), (20, "before_doorway")]
    wall = depth_with_wall(2.0, 0, 320)
    wall[:10, :] = 0
    synth.depth[20] = wall
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.item("E02", "book", {"color": "blue"}, [(0, None, (2.5, 1.0, 0.9))])
    synth.see(range(40), "E01")
    synth.see(range(22, 40), "E02")
    return compiled(root, synth, [during_q(1, 20), end_q(2)], a=1, b=2, length="medium")


def window_run(root: Path, name: str = "floor", channel: str = "rgb_pose") -> Path:
    """An `all_unknown` run of the datasets at `root` on `channel`, at `root/runs/<name>`."""
    run = root / "runs" / name
    assert cli("all_unknown", "--data", root, "--out", run, "--channel", channel) == 0
    return run


def start_stamp(run: Path, *more_notes: str) -> None:
    """Rewrite `run/results.json` as `stbench eval` writes it when a run begins and leaves it until the run ends: every
    case pending, no row, no aggregate, `partial` and the note `partial: 0 of N cases finished`, then `more_notes`."""
    doc = json.loads((run / "results.json").read_text(encoding="utf-8"))
    doc |= {"partial": True, "rows": [], "cost_usd": 0, "layouts": dict.fromkeys(doc["layouts"]),
            "notes": [f"partial: 0 of {len(doc['cases'])} cases finished; not a benchmark score", *more_notes]}
    doc["cases"] = [{**c, "status": "pending", "usage": None} for c in doc["cases"]]
    (run / "results.json").write_text(json.dumps(doc), encoding="utf-8")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PageParts(HTMLParser):
    """The elements of a page: `tags` (name, attributes) in order, `scripts` the text of each script by its type."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=False)
        self.tags: list[tuple[str, dict]] = []
        self.scripts: list[tuple[str | None, str]] = []
        self._in: str | None = "-"
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self._in = dict(attrs).get("type")
            self.scripts.append((self._in, ""))

    def handle_endtag(self, tag):
        if tag == "script":
            self._in = "-"

    def handle_data(self, data):
        if self._in != "-":
            kind, text = self.scripts[-1]
            self.scripts[-1] = (kind, text + data)

    def embedded(self) -> dict:
        """The page's data: the one `application/json` block, parsed."""
        (block,) = [text for kind, text in self.scripts if kind == "application/json"]
        return json.loads(block)
