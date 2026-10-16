"""`stbench view`'s page (`view_page.py` and `src/stbench/viewer/`, docs/eval.md): one self-contained file that shows a
run's text, which is untrusted (a solver's answers, a tool's results, a question's words), only ever as text, and that
reaches no network. `runview_kit.PageParts` reads the page as a parser would; the code of the page also runs, from the
first line to the last interaction, against a stand-in DOM under node (`view_dom_stub.js`) where node is installed."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path

import pytest

from runview_kit import PageParts, jsonl, make_run, start_stamp, window_case, window_run
from stbench import view_page
from stbench.case import read_case
from stbench.view import read_run

HOSTILE = '</script><img src=x onerror=alert(1)><!-- <script>alert(2)</script> &amp;  '
STUB = Path(__file__).resolve().parent / "view_dom_stub.js"


def viewer_files() -> dict[str, str]:
    """Every file of the page's package folder by name: the template, the style and the scripts."""
    return {name: resources.files("stbench").joinpath(view_page.PACKAGE_DIR, name).read_text(encoding="utf-8")
            for name in ("page.html", "page.css", *view_page.SCRIPTS)}


def hostile_run(root: Path):
    """An `all_unknown` run whose first case has HOSTILE in a question's words, in the solver's answer and in a tool's
    result, an error, a warning and a reply of its transport log; (the cases, the page's HTML)."""
    (first, second), run = make_run(root)
    asked = jsonl(first.questions)
    asked[0]["text"] = HOSTILE
    first.questions.write_text("".join(json.dumps(q) + "\n" for q in asked), encoding="utf-8")
    folder = run / "cases" / first.id
    answers = jsonl(folder / "answers.jsonl")
    answers[0]["answer"] = {"type": "failed", "reason": HOSTILE}
    (folder / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8")
    record = {"i": 0, "purpose": "observe", "kind": "start", "messages": [HOSTILE], "stderr": HOSTILE,
              "warnings": [HOSTILE], "prompt": {"text": HOSTILE, "images": []},
              "tools": [{"tool": HOSTILE, "arguments": {"path": HOSTILE}, "result": HOSTILE}]}
    (folder / "agent.calls.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return (first, second), view_page.render(read_run(run, root))


def test_hostile_text_cannot_leave_the_data_block(tmp_path):
    """Contract (view_page): everything from the run is written into one `application/json` block with every `<`, `>`
    and `&` escaped, so a `</script>`, a `<!--` or an `<img onerror>` in an answer, a tool result or a question stays
    inside it. The page holds exactly two scripts, no element beyond its own template (no `img`), no comment, and the
    block parses back to the very strings the run held. Regression caught: data written into the page unescaped (the
    classic script-block breakout), which would run a solver's text as the viewer's own code."""
    (first, _), html = hostile_run(tmp_path)
    parts = PageParts(html)
    names = [tag for tag, _ in parts.tags]
    assert names.count("script") == 2 and "img" not in names and "iframe" not in names
    assert html.lower().count("</script") == 2 and "<!--" not in html and "<img" not in html.lower()
    data = parts.embedded()
    case = next(c for c in data["arms"][0]["cases"] if c["id"] == first.id)
    assert case["questions"][0]["text"] == HOSTILE
    assert case["questions"][0]["answer"] == {"type": "failed", "reason": HOSTILE}
    (record,) = case["log"]["records"]
    assert record["stderr"] == HOSTILE and record["tools"][0]["result"] == HOSTILE and record["warnings"] == [HOSTILE]
    assert "<" not in view_page.embed({"x": HOSTILE}) and ">" not in view_page.embed({"x": HOSTILE})
    assert not any(mark in html for mark in ("/*__CSS__*/", "/*__JS__*/", "__DATA__"))   # each mark was replaced


def test_the_code_of_the_page_makes_no_html_from_data_and_reaches_no_network():
    """Contract (docs/eval.md: no server, no network, nothing external): the page's code builds its elements with
    `createElement`, `createTextNode` and `textContent` and never from a string (no `innerHTML`, `outerHTML`,
    `insertAdjacentHTML`, `document.write`, `eval`, `new Function`, `srcdoc` or `javascript:`), names no address but
    the `file:` links it is given, loads nothing (no `fetch`, `XMLHttpRequest`, `WebSocket`, `import`, `@import`,
    `url(`, `<link>`, `<base>` or `src` or `href` in the template), and holds none of the template's own marks. The
    first list is checked against a scan that sees the page's own words (`createTextNode`, `textContent`) so the rule
    is not vacuous. Regression caught: a script that renders data as HTML, or a font, script or beacon from a
    host, which would break a page opened from disk and leak what it shows."""
    files = viewer_files()
    code = "\n".join(files[name] for name in view_page.SCRIPTS)
    assert "createTextNode" in code and "textContent" in code and "createElement" in code
    for word in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "srcdoc",
                 "javascript:"):
        assert word not in code, word
    everything = "\n".join(files.values())
    assert re.findall(r"https?:", everything) == [] and re.findall(r"(?<![\w.])//[a-z]", everything) == []
    for word in ("fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "import(", "@import", "url(", "<link", "<base"):
        assert word not in everything, word
    assert set(re.findall(r"[a-z]+:/", code)) == {"file:/"}   # the one scheme the code names: a link it is handed
    template = PageParts(files["page.html"])
    assert not [t for t in template.tags if {"src", "href"} & set(t[1])]
    assert [files["page.html"].count(m) for m in ("/*__CSS__*/", "/*__JS__*/", "__DATA__")] == [1, 1, 1]
    assert not any(m in files[name] for name in ("page.css", *view_page.SCRIPTS) for m in ("__DATA__", "/*__"))
    assert "</script" not in code.lower() and "<!--" not in code


def run_stub(tmp_path: Path, html: str) -> dict:
    """What `view_dom_stub.js` reports of the page `html` run under node: the text it came to hold and the rest."""
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    done = subprocess.run(["node", str(STUB), str(page)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_runs_on_a_run_and_data_only_becomes_text(tmp_path):
    """Contract: the code of the page runs from its first line on a run (under node, against `view_dom_stub.js`, whose
    `innerHTML`, `outerHTML` and `insertAdjacentHTML` throw): it lists the cases, opens every case, its questions and
    its whole transport log, filters (by text, by wrong answers) and sorts by every column in every direction without
    an error; the hostile text is on the page as text and no element or attribute is made of it (no `onerror`, and the
    one kind of picture is a `file:` frame of the dataset); the layouts table shows the headline's two columns with
    their counts (by hand, the run's `now` rows score 0 over 2 and its `memory` rows 3/4 over 4). Regression caught: a
    runtime error that blanks the page (no other test runs the code; one caught here was a sort that read the headers of
    the nested question tables), data turned into markup, or a headline shown without the columns it is the mean of."""
    (first, second), html = hostile_run(tmp_path)
    out = run_stub(tmp_path, html)
    assert out["cases"] == 2 and out["count"] == "showing 2 of 2" and out["title"] == "stbench view: all_unknown"
    assert out["text"].count(HOSTILE) >= 6   # the question, the answer, and the log's reply, error, warning and prompt
    assert "onerror" not in out["attributeNames"] and out["tags"].get("script", 0) == 0
    assert len(out["pictures"]) == 6 and all(src.startswith("file:///") for src in out["pictures"])
    assert out["links"] == out["pictures"]
    assert "now 0.000 over 2, memory +0.750 over 4" in out["text"]
    assert first.id in out["text"] and second.id in out["text"] and "unknown" in out["text"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_run_in_progress_says_so_the_same_way_everywhere_and_a_missing_frame_falls_back_to_its_link(tmp_path):
    """Contract (docs/eval.md, `stbench view`): the banner counts the finished case folders as the header does and says
    the run is unfinished, and why `results.json` has no headline: there is none yet, or it is the stamp eval wrote at
    the start (which lists none finished; its own `partial: 0 of N` note is not repeated). A layout with no aggregate
    says the headline comes from `results.json`, which eval writes when the run ends, and not `no case finished`; the
    page computes no statistic. A frame that cannot load (the page served over http) becomes its `file:` link as text,
    never markup, and a note says frames load when the page is opened from disk. Regression caught: a banner of 0 of N
    beside a header of 13 of N, a headline row that denies work that was done, or a thumbnail that fails silently."""
    (first, second), run = make_run(tmp_path)
    (run / "cases" / second.id / "summary.json").unlink()
    start_stamp(run)
    stale = run_stub(tmp_path, view_page.render(read_run(run, tmp_path)))
    banner = "Unfinished: 1 of 2 cases finished (counted from the case folders), so this is not a benchmark score."
    assert banner + " results.json lists 0 finished: eval writes it when the run begins" in stale["text"]
    assert "Cases finished" in stale["text"] and "1 of 2" in stale["text"]
    assert "0 of 2 cases finished" not in stale["text"] and "no case finished" not in stale["text"]
    assert "No headline yet: it is read from results.json, which eval writes when the run ends" in stale["text"]
    assert "the one here is from the start of the run and holds none" in stale["text"]
    (run / "results.json").unlink()
    absent = run_stub(tmp_path, view_page.render(read_run(run, tmp_path)))
    assert banner + " There is no results.json yet: eval writes it when the run ends." in absent["text"]
    assert "(there is none yet)" in absent["text"] and "no case finished" not in absent["text"]
    fell_back = absent["fallback"]
    assert fell_back["pictures"] == 0 and len(fell_back["notes"]) == 2   # a note a case, both cases are open
    assert all("file:// address" in note and "over http" in note for note in fell_back["notes"])
    assert len(fell_back["links"]) == 6 and all(link.startswith("k0 file:///") for link in fell_back["links"])
    assert first.id in absent["text"]


def test_wide_tables_scroll_in_their_own_box_and_long_text_wraps():
    """Contract: the page itself never scrolls sideways at a window of 800 to 900 px. The table of cases and each
    questions table sit in a `scroll` box (`overflow-x: auto`), an opened case's body is `width: 0; min-width: 100%` so
    its content cannot widen the table, and the transport log's text wraps (`pre-wrap`, `overflow-wrap: anywhere`).
    Measured in Chrome with every case and log open at 500 to 900 px (scroll width equal to client width); this keeps
    the rules from being dropped where no browser runs. Regression caught: a long JSON line or a wide table that makes
    the whole page scroll sideways."""
    files = viewer_files()
    assert re.search(r'<div class="scroll">\s*<table id="cases">', files["page.html"])
    css = files["page.css"]
    assert ".scroll { overflow-x: auto;" in css
    assert ".detail-body { width: 0; min-width: 100%; overflow-x: auto;" in css
    assert ".log, .log .rec, .rec .txt { overflow-wrap: anywhere; }" in css
    assert ".rec .txt { white-space: pre-wrap; }" in css
    assert ".rec pre, .log pre { margin: 2px 0; white-space: pre-wrap; overflow-wrap: anywhere;" in css
    assert files["page.js"].count("class: 'scroll'") >= 2   # the questions table and the layouts table


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_stream_scrubber_steps_marks_its_questions_and_shows_the_dataset_only_as_text(tmp_path):
    """Contract (frames.js, docs/eval.md): the section "The stream, frame by frame" of an opened case, run under node
    against `view_dom_stub.js` (whose `innerHTML` and friends throw), steps one frame at a time with its buttons, the
    look window buttons, the slider and the keys, stays inside the first and last frame, and shows at each frame the
    step (k from 0), the clock, the base and camera, the status, the picture file of that step (the camera image on
    `rgb_pose`; only the depth on `detections`, with the frame's detection lines), the look window and task it belongs
    to and the questions asked after it, each question a link that lights its row in the table above; a mark under the
    slider takes the reader to the frame its question was asked after. By hand over `window_case` (24 steps from
    keyframe 16; look windows end at 5, 13, 16, 24; the question at step 4): the mark of Q1 goes to "frame 5 of 24", the
    last frame lists the end question, the start frame's clock is the case's start clock and the third's 3 s later. The
    dataset's text (a status phase, a detection's attribute) is hostile here and appears as text only, and a picture
    that cannot load becomes its link as text with the note. Regression caught: a frame counter that runs past its ends,
    a question's mark at another step, a picture of another frame, or dataset text turned into markup."""
    case = window_case(tmp_path)
    camera, detections = window_run(tmp_path, "camera"), window_run(tmp_path, "det", "detections")
    poses = jsonl(case.script.pose)
    for row in poses:
        row["status"]["phase"] = HOSTILE
    write_jsonl_rows(case.script.pose, poses)
    seen = jsonl(case.script.oracle)
    for row in seen:
        for found in row["detections"]:
            found["attributes"] = {"color": HOSTILE} if found["kind"] == "mug" else found["attributes"]
    write_jsonl_rows(case.script.oracle, seen)
    start = datetime.fromisoformat(read_case(case).start_clock)
    pictures = tmp_path / "pictures"

    out = run_stub(tmp_path, view_page.render(read_run(camera, tmp_path, pictures)))
    (scrub,) = out["scrub"]
    moves = {i: f["move"] for i, f in enumerate(scrub["frames"])}
    assert [f["counter"] for f in scrub["frames"]] == [f"frame {n} of 24" for n in
                                                       (1, 2, 3, 2, 6, 1, 24, 24, 23, 1, 1, 2, 24, 24, 5)], moves
    first, third, last, asked = (scrub["frames"][i] for i in (0, 2, 12, 14))
    assert "k = 0 in the case" in first["facts"] and start.isoformat() in first["facts"]
    assert (start + timedelta(seconds=3)).isoformat() in third["facts"] and "k = 2" in third["facts"]
    assert "at (2, 1.5, 0) m, heading 0°" in first["facts"] and "at (2.13, 1.5, 0.08) m, pitch 5°" in first["facts"]
    assert f"phase {HOSTILE} · task 0" in first["facts"] and "task 0, its first step" in first["facts"]
    assert "look window 1 of 4: steps 0 to 4" in first["facts"]
    assert "Questions" not in first["facts"].replace("End questions", "")   # none is asked after step 0
    assert "k = 23" in last["facts"] and "task 1, its last step" in last["facts"] and "End questions" in last["facts"]
    assert "asked after this last step: Q2" in last["facts"]
    assert "k = 4" in asked["facts"] and "asked after this step: Q1 path_ahead" in asked["facts"]
    assert "End questions" not in asked["facts"]
    ids = [f"{16 + k:06d}" for k in (0, 1, 2, 1, 5, 0, 23, 23, 22, 0, 0, 1, 23, 23, 4)]
    for frame, number in zip(scrub["frames"], ids, strict=True):
        (rgb,) = frame["pictures"]
        assert rgb.startswith("file:///") and rgb.endswith(f"/sensors/rgb/{number}.jpg")
    assert scrub["sliderMax"] == "23" and scrub["pictureSlots"] == 1 and scrub["marks"]["windowEnds"] == 3
    assert scrub["marks"]["tasks"] == ["task 0 begins at step 0", "task 1 begins at step 16"]
    assert [(q["label"], q["click"]) for q in scrub["marks"]["questions"]] == [("Q1", 1), ("end: Q2", 1)]
    title = scrub["marks"]["questions"][0]["title"]
    assert "path_ahead" in title and "How far ahead is it clear (1)?" in title
    assert out["flash"] == {"links": True, "before": 0, "after": 1}   # Q1's link lit its row in the table above
    assert out["slot"]["img"] is True and out["slot"]["miss"] is False and out["slot"]["notes"] == [0, 1]
    assert out["slot"]["text"].startswith("camera image (rgb) file:///")
    assert HOSTILE in scrub["text"] and out["tags"].get("script", 0) == 0 and "onerror" not in out["attributeNames"]

    out = run_stub(tmp_path, view_page.render(read_run(detections, tmp_path, pictures)))
    (scrub,) = out["scrub"]
    start_frame, end_frame = scrub["frames"][0], scrub["frames"][6]
    assert len(start_frame["pictures"]) == 1 and start_frame["pictures"][0].endswith(f"/{case.script.id}/000016.png")
    assert scrub["pictureSlots"] == 1 and end_frame["counter"] == "frame 24 of 24"
    assert (start_frame["rows"], end_frame["rows"]) == (2, 3)   # the header and a detection; the header and two
    assert "mug" in start_frame["detections"] and f"color {HOSTILE}" in start_frame["detections"]
    assert "book" in end_frame["detections"] and "color blue" in end_frame["detections"]
    assert "(0, 0) to (20, 20)" in end_frame["detections"] and "0.1 x 0.1 x 0.1" in end_frame["detections"]
    assert "their keys are fresh for each frame" in end_frame["detections"]
    assert HOSTILE in scrub["text"] and "onerror" not in out["attributeNames"] and out["tags"].get("script", 0) == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_stream_scrubber_shows_the_camera_image_alone_on_rgb_pose(tmp_path):
    """Contract (frames.js; 2026-10-10: the agent is sent no depth image): the section "The stream, frame by frame"
    of a case run on `rgb_pose` has one picture slot and shows, at every frame, the camera image of that step and no
    depth picture, no note of one that is not drawn and no word of depth, though the dataset holds depth images and a
    pictures folder was given. Regression caught: a page that shows a
    depth image the agent never got (a reader would take it for what the agent saw), or a slot left empty for it."""
    window_case(tmp_path)
    drawn = tmp_path / "pictures"
    out = run_stub(tmp_path, view_page.render(read_run(window_run(tmp_path, "rgb", "rgb_pose"), tmp_path, drawn)))
    (scrub,) = out["scrub"]
    assert scrub["pictureSlots"] == 1 and "depth" not in scrub["text"].lower() and not drawn.exists()
    assert scrub["frames"] and all(len(f["pictures"]) == 1 and f["pictures"][0].startswith("file:///")
                                   and "/sensors/rgb/" in f["pictures"][0] and f["pictures"][0].endswith(".jpg")
                                   for f in scrub["frames"])
