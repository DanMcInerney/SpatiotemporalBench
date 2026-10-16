"""The moves a job's `clear_path` may ask (`JobSpec.clear_directions`, 0.9.0): declared per job and refused at
construction when the body cannot take them."""

from __future__ import annotations

from dataclasses import replace

import pytest

from stbench.job_registry import all_jobs, get
from stbench.questions_words import CLEAR_DIRECTIONS, FLY_DIRECTIONS


def test_the_default_is_ahead_and_every_registered_job_declares_a_valid_set():
    """Contract: a job that declares no directions asks ahead alone (0.8.0's corridor), and every registered job's
    directions are among its body's (the AMR asks no corridor and keeps the default). Regression caught: a default that
    moves a 0.8.0 gold, or a shipped job whose declared direction was never drawn."""
    assert get("amr/1").clear_directions == ("ahead",)
    for job in all_jobs():
        allowed = FLY_DIRECTIONS if job.robot == "drone" else CLEAR_DIRECTIONS
        assert job.clear_directions and set(job.clear_directions) <= set(allowed), job.id


def test_the_vocabularies_are_the_designs():
    """Contract (brief §4.1): a ground robot may be asked ahead, back, left and right; a flier ahead and back, since a
    sideways flight was rejected (a held-out constant leads on it). Regression caught: a vocabulary widened for a
    flier."""
    assert CLEAR_DIRECTIONS == ("ahead", "back", "left", "right") and FLY_DIRECTIONS == ("ahead", "back")


@pytest.mark.parametrize(("job", "directions"), [
    ("helper/1", ("ahead", "back", "left", "right")), ("vacuum/2", ("ahead", "back", "left")),
    ("drone/1", ("ahead", "back")), ("amr/1", ("ahead",)), ("helper/1", ("right",))])
def test_a_job_may_declare_the_moves_its_body_takes(job, directions):
    """Contract: the helper's four moves, the vacuum's three, the drone's two and a single one are all accepted.
    Regression caught: a validator that refuses the table the 0.9.0 jobs declare."""
    assert replace(get(job), clear_directions=directions).clear_directions == directions


@pytest.mark.parametrize(("job", "directions", "stray"), [
    ("vacuum/2", ("ahead", "up"), "up"), ("helper/1", ("forward",), "forward"), ("drone/1", ("ahead", "left"), "left"),
    ("drone/1", ("right",), "right")])
def test_a_misdeclared_direction_is_refused_naming_the_job(job, directions, stray):
    """Contract: a ground job that declares a direction outside ahead, back, left and right, or a flier one outside
    ahead and back, is refused at construction, and the message names the job and the direction. Regression caught: a
    direction never drawn, silently (the draw would index it and ask a corridor no gold function knows)."""
    with pytest.raises(ValueError, match=rf"{job}.*{stray}"):
        replace(get(job), clear_directions=directions)


@pytest.mark.parametrize("directions", [(), ("ahead", "ahead"), ("back", "left", "back")])
def test_an_empty_or_a_repeating_set_is_refused(directions):
    """Contract: at least one direction, each once (the draw picks each equally, so a repeat would weight one).
    Regression caught: a modulo by zero in the draw, or a direction drawn twice as often as the rest."""
    with pytest.raises(ValueError, match="vacuum/2"):
        replace(get("vacuum/2"), clear_directions=directions)
