"""What the brief and the prompt say of the harm families of 0.10.0 (`closest_approach`, `drop_zone_people`,
`tangle_ahead`): their phrases in the during sentence, and the list paragraph of the credit sentence for the two that
ask a list. Hand-written sentences on fake jobs."""

from __future__ import annotations

import pytest

from stbench.brief import DURING_PHRASES, asks_during, asks_lists, during_sentence
from stream_kit import fake_job

PHRASES = {"closest_approach": "how close something would come to the robot",
           "drop_zone_people": "what stands near the point below the robot",
           "tangle_ahead": "what lies on the floor near the robot"}


@pytest.mark.parametrize("family", sorted(PHRASES))
def test_a_job_asking_a_harm_family_has_its_phrase_in_the_during_sentence(family):
    """Contract (methodology §1: the brief never names what will be asked about): each harm family has a phrase in
    `DURING_PHRASES`, used once in the during sentence of a job that asks it at look points, and none speaks of a person
    (the existing brief test holds that for every phrase). Regression caught: a family with no phrase (`during_sentence`
    refuses the job, so it cannot compose), or a phrase that names what is asked."""
    assert DURING_PHRASES[family] == PHRASES[family]
    job = fake_job(looks=("ego_now", family))
    assert asks_during(job)
    assert during_sentence(job).count(PHRASES[family]) == 1 and during_sentence(job).startswith("At some points")
    assert PHRASES[family] not in during_sentence(fake_job(looks=("ego_now",)))


def test_the_prompt_speaks_of_lists_for_a_job_that_asks_a_list_during_the_job():
    """Contract (`brief.asks_lists`): the list paragraph of the credit sentence (how a list of places is credited) is
    shown to a job that asks `drop_zone_people` or `tangle_ahead` at a look point, as to one that ends with
    `task_things`, and not to a job that asks only `closest_approach`, a number. Regression caught: a list question
    scored per item that the prompt never explains (an agent that does not know a missed thing costs a whole thing and a
    false place half of one)."""
    assert asks_lists(fake_job(looks=("clear_path", "tangle_ahead")))
    assert asks_lists(fake_job(looks=("drop_zone_people",)))
    assert asks_lists(fake_job(ends=("object_where", "task_things")))
    assert not asks_lists(fake_job(looks=("closest_approach", "ego_mem"))) and not asks_lists(fake_job())
