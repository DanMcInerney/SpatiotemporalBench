"""The robot's body in the prompt (0.8.0 lane L5; FD-task-questions review §2.2: "how far could your body go" has no
answer when the prompt never states the body). `llm_render.body_sentence` reads `robot.json`'s `flies` and `body`; the
expected sentences are hand-written for made-up robots with odd sizes (no robot of the library), and the library's
robots are checked against their own fields and against `grid.band`, the scorer's own band."""

from __future__ import annotations

import re

import pytest

from stbench import job_registry, leak
from stbench.grid import band
from stbench.llm_render import body_sentence, credit_sentence, system_prompt
from stbench.robots import ROBOTS, robot_doc
from test_llm_render import BODY, CAMERA, public

ODD = {"shape": "rect", "length_m": 0.77, "width_m": 0.43, "height_m": 0.215, "climb_m": 0.035, "clearance_m": 0.1,
       "under_margin_m": 0.06}   # no robot has these sizes: a number typed into the prompt cannot match them
ROUND = {"shape": "disc", "length_m": 0.42, "width_m": 0.42, "height_m": 1.25, "climb_m": 0.01, "clearance_m": 0.1,
         "under_margin_m": 0.05}
FLIER = {"shape": "disc", "length_m": 1.6, "width_m": 1.6, "height_m": 0.45, "climb_m": 0.0, "clearance_m": 3.0,
         "under_margin_m": 0.0}


def robot(body: dict, *, flies: bool = False) -> dict:
    return {"id": "made_up", "flies": flies, "body": body, "camera": {"mount_m": [0.0, 0.0, 0.5]}}


def test_the_body_sentence_is_built_from_the_numbers_of_the_robot_file():
    """Contract (review §2.2, L5; VISION rule 8: a robot is a profile): a ground robot's sentence gives the body's
    shape and size, its height, that the base is its middle, the highest step it drives over (`climb_m`) and the height
    above the floor it passes under (`height_m` plus half `under_margin_m`); a box says "long and wide", a disc
    "across"; lengths under a metre in centimetres, else in metres. Hand sentences for three made-up bodies with sizes
    no library robot has. Regression caught: a number typed into the prompt apart from the file (the sentence would
    not follow an odd size), a disc described as a box, or the passing height read as the body's height alone."""
    assert body_sentence(robot(ODD)) == (
        "The robot's body is a box 77 cm long and 43 cm wide and 21.5 cm tall, centred on the base. It drives over "
        "anything up to 3.5 cm high and passes under anything 24.5 cm or more above the floor.")
    assert body_sentence(robot(ROUND)) == (
        "The robot's body is a disc 42 cm across and 1.25 m tall, centred on the base. It drives over anything up to "
        "1 cm high and passes under anything 1.275 m or more above the floor.")
    assert body_sentence(robot(BODY)) == (
        "The robot's body is a box 50 cm long and 40 cm wide and 30 cm tall, centred on the base. It drives over "
        "anything up to 1.5 cm high and passes under anything 32 cm or more above the floor.")


def test_a_flier_is_told_its_body_and_no_climb():
    """Contract (review §2.2): a robot that flies has its shape, size and height and that the base is its middle, and
    nothing of climbing or of passing under: its corridor is the body's own height wherever it flies. Regression
    caught: a drone told it drives over steps, or a flier's sentence that drops its size."""
    sentence = body_sentence(robot(FLIER, flies=True))
    assert sentence == "The robot's body is a disc 1.6 m across and 45 cm tall, centred on the base."
    assert not re.search(r"drives|climb|passes under|floor", sentence)


def metres(text: str) -> float:
    """A printed length ("9.5 cm", "1.3 m") read back in metres."""
    number, unit = text.split()
    return float(number) / (100 if unit == "cm" else 1)


LENGTH = r"(\d[\d.]* (?:cm|m))"
BODY_WORDS = re.compile(rf"body is a (disc|box) {LENGTH}(?: across| long and {LENGTH} wide) and {LENGTH} tall, centred "
                        rf"on the base\.(?: It drives over anything up to {LENGTH} high and passes under anything "
                        rf"{LENGTH} or more above the floor\.)?")


@pytest.mark.parametrize("robot_id", sorted(ROBOTS))
def test_every_library_robots_sentence_is_its_own_fields_and_the_scorers_band(robot_id):
    """Contract (the prompt never disagrees with the scorer): for each robot of the library the sentence's numbers,
    read back, are that robot's own `body` fields, and a ground robot's passing height is the top of `grid.band`, the
    band in which the gold and the mapper see an obstacle (a surface at exactly that height is overhead). Regression
    caught: a sentence whose passing height is not the band the gold reads (the agent would be told it clears what the
    gold says stops it), or a body number that drifts from `robot.json`."""
    r = ROBOTS[robot_id]
    shape, length, width, height, climb, top = BODY_WORDS.search(body_sentence(robot_doc(r))).groups()
    assert (shape, metres(length), metres(height)) == ("disc" if r.body.shape == "disc" else "box",
                                                       pytest.approx(r.body.length_m), pytest.approx(r.body.height_m))
    assert (width is None) == (r.body.shape == "disc") and (width is None or metres(width) == r.body.width_m)
    if r.flies:
        assert climb is None and top is None
    else:
        assert metres(climb) == pytest.approx(r.body.climb_m) and metres(top) == pytest.approx(band(r)[1])


def test_the_body_sentence_refuses_a_robot_file_with_no_valid_body():
    """Contract (as `mount_sentence` refuses a bad mount): a `robot.json` with no `flies`, no body, an unknown shape, a
    disc whose two sizes differ, or a size that is not a positive number (a climb or margin may be 0) is an error, not a
    prompt without a body. Regression caught: a prompt silently missing the body, which leaves "how far could your body
    go" with no answer."""
    assert body_sentence(robot({**ODD, "climb_m": 0, "under_margin_m": 0})).endswith(
        "up to 0 cm high and passes under anything 21.5 cm or more above the floor.")
    bad = [robot(ODD) | {"body": None}, {k: v for k, v in robot(ODD).items() if k != "flies"}, "not a robot",
           robot({**ODD, "shape": "blob"}), robot({**ROUND, "width_m": 0.5}), robot({**ODD, "height_m": 0}),
           robot({**ODD, "length_m": "77"}), robot({**ODD, "climb_m": -0.01}), robot({**ODD, "width_m": True}),
           robot({k: v for k, v in ODD.items() if k != "under_margin_m"})]
    for doc in bad:
        with pytest.raises(ValueError, match=r"robot\.json needs|body\.|disc body"):
            body_sentence(doc)


@pytest.mark.parametrize("job", job_registry.all_jobs(), ids=lambda job: job.id)
def test_each_jobs_prompt_states_its_robots_body_and_trips_no_leak_rule(tmp_path, job):
    """Contract (L5; VISION rule 6): the system prompt of every registered job, built from the job's own robot file
    and its numbers, holds that robot's body sentence once, before the camera is called the only eye, and nothing
    `leak.RULES` flags (an entity id, a region id, the word "truth"), no word of memory and no word of depth. Regression
    caught: a body sentence lost from the prompt of a robot (the lens paragraph not reading the file), or a new
    sentence that trips the leak scan of what a model is sent."""
    doc = robot_doc(ROBOTS[job.robot])
    prompt = system_prompt(public(tmp_path, CAMERA, doc), "rgb_pose", asks_during=True, match_m=job.match_m,
                           clearance_m=job.clearance_m, reacquire_m=job.reacquire_m)
    assert prompt.count(body_sentence(doc)) == 1 and prompt.index(body_sentence(doc)) < prompt.index("only eye")
    assert [rule for rule, rx in leak.RULES.items() if rx.search(prompt)] == []
    assert "memory" not in prompt.lower() and "depth" not in prompt.lower()
    assert credit_sentence(job.match_m, job.clearance_m, job.reacquire_m) in prompt
