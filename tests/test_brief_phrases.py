"""The brief's words for the 0.8.0 families (lane L5; FD-task-questions review §7 item 13). `brief.DURING_PHRASES` needs
a phrase for `clear_path`, and `person_mem` reuses `ego_mem`'s so the brief never announces people (methodology §1: the
brief never names what will be asked about). Hand-written sentences on fake jobs."""

from __future__ import annotations

import pytest

from stbench.brief import DURING_PHRASES, asks_during, compose, during_sentence
from stbench.robots import ROBOTS, robot_doc
from stbench.timeline_ego import TaskPlan
from stream_kit import fake_job

SEEN = "where a thing you can see is from your camera"
BEFORE = "where a thing you saw before is from your camera"
MOVE = "how far the robot could move"


def first_text(job) -> str:
    plans = [TaskPlan("tidy", {"room": "kitchen"})]
    return compose(plans, job=job, robot=robot_doc(ROBOTS["vacuum"]), domain="home", channel="rgb_pose")[0]


def test_a_job_asking_person_mem_has_the_same_during_sentence_as_one_asking_ego_mem():
    """Contract (review §7.13, methodology §1): `person_mem` has `ego_mem`'s phrase, so a job that asks it has the
    during sentence, and the whole first text, of a job that asks `ego_mem` in its place, and a job that asks both
    names the phrase once; no phrase of the brief speaks of a person. Regression caught: a brief that announces people
    (the agent would look for them before any question), or a family with no phrase (`during_sentence` refuses it, so
    the job would not compose)."""
    memory, person = fake_job(looks=("ego_now", "ego_mem")), fake_job(looks=("ego_now", "person_mem"))
    assert during_sentence(person) == during_sentence(memory) == (
        f"At some points during the job you will be asked {SEEN} and {BEFORE}.")
    assert first_text(person) == first_text(memory)
    both = fake_job(looks=("ego_mem", "person_mem"))
    assert during_sentence(both) == f"At some points during the job you will be asked {BEFORE}."
    assert DURING_PHRASES["person_mem"] == DURING_PHRASES["ego_mem"] == BEFORE
    assert not any(word in phrase for phrase in DURING_PHRASES.values() for word in ("person", "people", "human"))


@pytest.mark.parametrize(("looks", "kinds"), [
    (("clear_path",), MOVE),
    (("clear_path", "ego_mem"), f"{BEFORE} and {MOVE}"),
    (("ego_now", "clear_path", "person_mem", "ego_mem"), f"{SEEN}, {BEFORE}, and {MOVE}"),
])
def test_a_job_asking_clear_path_tells_the_agent_how_far_the_robot_could_move(looks, kinds):
    """Contract (plan L5, §3): a job that asks `clear_path` at its look points has "how far the robot could move" in
    its during sentence, once, in `DURING_PHRASES`' order after the ego phrases, and the first text carries the
    sentence and promises nothing else; a job that asks it is one that asks during the job. Regression caught: a
    `clear_path` job whose brief says nothing of the question (the first ask would prime beyond the brief), a phrase
    that names the corridor or a heading (it would coach), or a phrase missing from the order."""
    job = fake_job(looks=looks)
    assert asks_during(job)
    assert during_sentence(job) == f"At some points during the job you will be asked {kinds}."
    text = first_text(job)
    assert text.count(MOVE) == 1 and f"asked {kinds}." in text
    assert "corridor" not in text and "ahead" not in text


ROOM = "which way the robot has the most room"


@pytest.mark.parametrize(("looks", "kinds"), [
    (("open_heading",), ROOM),
    (("clear_path", "open_heading"), f"{MOVE} and {ROOM}"),
    (("ego_now", "open_heading", "ego_mem"), f"{SEEN}, {BEFORE}, and {ROOM}"),
])
def test_a_job_asking_open_heading_tells_the_agent_which_way_has_the_most_room_once(looks, kinds):
    """Contract (plan P2, 0.9.0): a job that asks `open_heading` at its look points has "which way the robot has the
    most room" in its during sentence, once, after `clear_path`'s phrase in `DURING_PHRASES`' order, and the first text
    carries the sentence and promises nothing else; a job that does not ask it never says so, and the phrase names
    neither a turn nor a heading (it would coach). Regression caught: a brief that promises a question never asked, a
    job that asks it with no words in the brief, or a phrase that tells how to answer."""
    job = fake_job(looks=looks)
    assert asks_during(job)
    assert during_sentence(job) == f"At some points during the job you will be asked {kinds}."
    text = first_text(job)
    assert text.count(ROOM) == 1 and f"asked {kinds}." in text
    assert "heading" not in text and "turn" not in text
    assert DURING_PHRASES["open_heading"] == ROOM
    assert ROOM not in first_text(fake_job(looks=("clear_path", "ego_mem")))
