"""The text-only option of the Codex reference agent (`llm_render`, `codex_system`, `cmd_codex`; FD-0.6.0 §2 call 7): a
zero-shot check of what the questions give away without the pictures. It is an option and not a channel: the stream is
`rgb_pose`'s, and the run withholds every image, says so and is stamped. Over the fake command; no model, no network.
(The pull's own text-only test, frame by frame, is `test_codex_pull.py`'s.)
"""

from __future__ import annotations

import argparse
import json

import pytest

from codex_kit import CREDIT, Fake, Run, job_also_asks, pulled
from stbench import cmd_codex
from stbench.case import CasePaths
from stbench.codex_log import read_log
from stbench.llm_render import frame_message, system_prompt
from test_cmd_codex import jsonl, small_case
from test_llm_render import EXPECTED_FRAMES, STEPS, public

BRIEF = "Your assignment is to clean up."


def test_a_text_only_case_sends_no_image_and_the_prompt_says_the_images_are_withheld(tmp_path):
    """Contract (plan A8): with `text_only` on `rgb_pose`, though the stream delivers a JPEG for each frame, no image
    block is sent; a frame is its header line and nothing else (no line naming an image; the end ask's questions follow
    the second frame as a text block of their own, FD-questions-0.10 §1.5); the first message tells the
    model that it is sent no image and that a frame is its header line alone, once; and the journal logs no image for
    any served item. The same case without the option sends the two JPEGs (the control that the option, and nothing
    else, withholds them). Hand lines for the kit's two frames. Regression caught: a text-only run that still sends a
    picture (the check would measure the pictures), or one that sends none but tells the model it will get some."""
    (tmp_path / "plain").mkdir()
    (tmp_path / "text").mkdir()
    with Run(tmp_path / "plain", asks_during=True, **CREDIT) as plain:
        plain.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        plain.stream([0, 1])
        plain.serve()
        assert [c["blocks"] for c in pulled(plain.fake.execs()[0])] == [["text", "image"]] * 2

    with Run(tmp_path / "text", asks_during=True, **CREDIT, text_only=True) as run:
        steps = run.batch([0])
        assert all((run.work / step["rgb"]).is_file() for step in steps)  # the stream did deliver the pictures
        run.system.task(0, "2026-03-02T08:00:00", 0.0, BRIEF)
        run.system.observe(steps)
        run.system.observe(run.batch([1]))
        run.system.ask("end", {"k": 1, "clock": "2026-03-02T08:00:01"}, [{"qid": "q" * 16, "text": "How many?",
                                                                           "answer_type": "count"}])
        (only,) = run.fake.execs()
    frames = pulled(only)[:2]
    # frame 1 comes with the end ask: its header, then the questions, both text
    assert [c["blocks"] for c in frames] == [["text"], ["text", "text"]] and [c["images"] for c in frames] == [[], []]
    assert [c["text"].split(" ")[0] for c in frames] == ["[k0", "[k1"] and "\n" not in frames[0]["text"]
    assert frames[1]["text"].splitlines()[0].startswith("[k1 ") and not any("Image" in c["text"] for c in frames)
    first = only["prompt"]
    assert first.count("In this run you are not sent the images: a frame is its header line alone.") == 1
    assert "- A frame is its header line alone, with no image. The header gives its number k" in first
    assert "attached" not in first and "Things in the image" not in first
    assert [e["images"] for e in read_log(run.log) if e.get("event") == "served"] == [[], [], []]  # task, 2 frames


def test_the_text_only_messages_are_the_camera_ones_less_the_images(tmp_path):
    """Contract: the text-only frame is the header line of the camera frame (`test_llm_render.EXPECTED_FRAMES`) with no
    image to send, whatever a step carries, and the text-only prompt differs from the camera one by the sentences about
    images alone: the frame bullet's image clauses and the added "not sent" sentence. Hand text for the kit's STEPS.
    Regression caught: a text-only run that changes anything else the model reads (another frame header, a missing
    credit sentence), which would confound the check."""
    got = [frame_message(step, "rgb_pose", text_only=True) for step in STEPS]
    assert got == [(text, None) for text in EXPECTED_FRAMES]
    # a step with no `rgb` is fine: nothing is sent
    assert frame_message({k: v for k, v in STEPS[0].items() if k != "rgb"}, "rgb_pose", text_only=True)[1] is None
    camera = system_prompt(public(tmp_path / "a"), "rgb_pose", asks_during=True, **CREDIT)
    only = system_prompt(public(tmp_path / "b"), "rgb_pose", asks_during=True, **CREDIT, text_only=True)
    assert only.replace(" In this run you are not sent the images: a frame is its header line alone.", "").replace(
        "A frame is its header line alone, with no image.", "A frame is its header line and its camera image.").replace(
        "and its status.\n- Questions", "and its status. Things in the image carry no labels or positions.\n- Questions"
    ) == camera


def test_text_only_on_detections_is_refused_everywhere_it_could_start(tmp_path):
    """Contract: the channel `detections` sends no image, so there is nothing to withhold and "text-only" there would
    only relabel the detections arm: the renderer, the system and the driver each refuse it (the driver before
    anything is written). Regression caught: a text-only stamp on a perfect-detector run, which would put the diagnostic
    in the zero-shot row."""
    with pytest.raises(ValueError, match="text_only withholds the camera images, and channel 'detections' sends none"):
        system_prompt(public(tmp_path), "detections", text_only=True)
    with pytest.raises(ValueError, match="text_only withholds the camera images, and channel 'detections' sends none"):
        frame_message(STEPS[0], "detections", text_only=True)
    (tmp_path / "work").mkdir()
    with pytest.raises(ValueError, match="text_only withholds the camera images of rgb_pose"):
        Run(tmp_path / "work", channel="detections", text_only=True)
    out = tmp_path / "out"
    with pytest.raises(cmd_codex.Refused, match="text_only withholds"):
        cmd_codex._transport("m", argparse.Namespace(channel="detections", out=out, reasoning=None, timeout=None,
                                                     text_only=True))
    assert not out.exists()


@pytest.mark.parametrize("text_only", [True, False])
def test_the_driver_stamps_a_text_only_run_in_its_journal_and_summary(tmp_path, monkeypatch, capsys, text_only):
    """Contract (plan A8: "is stamped on the run"): `cmd_codex.run` with `args.text_only` writes `"text_only": true` in
    the journal's `run` event and in `codex.summary.json`, prints that no image was sent, and sends no image block;
    without the option (an `args` that lacks the key, as `stbench run` builds today) both say false and the six frames
    of the case carry the camera images (6 image blocks). Regression caught: a text-only run that leaves no mark, so
    that its rows are read as camera ones, or a camera run marked text-only."""
    job_also_asks(monkeypatch, "distance_between")   # the case's during question is a number, which vacuum/1 asks none
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    monkeypatch.setattr(cmd_codex, "make_codex", lambda model, reasoning: fake.new(model=model, reasoning=reasoning))
    out = tmp_path / "out"
    out.mkdir()
    args = argparse.Namespace(channel="rgb_pose", brief="correct", during="on", reasoning=None, timeout=None, out=out)
    if text_only:
        args.text_only = True
    run = cmd_codex.run("fake-model", CasePaths.from_dir(case.private), args)
    assert run.status == "complete"
    assert jsonl(out / "codex.calls.jsonl")[0]["text_only"] is text_only
    assert json.loads((out / "codex.summary.json").read_text(encoding="utf-8"))["text_only"] is text_only
    assert ("text-only run: no image was sent to the model" in capsys.readouterr().out) is text_only
    (only,) = fake.execs()
    assert sum(len(c["images"]) for c in only["calls"]) == (0 if text_only else 6)
