"""The credit section of each registered job's prompt (`llm_credit`) against the numbers it is built from: the job's
clearance C, re-find radius R and matching radius, and the constants of the kernel and the rules. Every number is read
from there and none is typed into the words (2026-10-14: same rules and numbers in plainer words), and each
conditional part is there for the jobs that ask its families and for no other. Hand-written tables; the whole text is
pinned by hand in `test_subgoal_prompt.py`, which paragraphs a job is told in `test_credit_scope.py`, and the halves
(`HARM_FACTOR`, `FP_WEIGHT`) by `test_llm_credit_sentence.py`."""

from __future__ import annotations

import pytest

from credit_kit import JOB_FAMILIES, JOB_KINDS, text_of
from stbench import hazard_rule, job_registry, llm_credit, rules
from stbench.credit_kernel import ANGLE_DEG, CATASTROPHE_PERSON, CATASTROPHE_THING, MATCH_REACH, MIN_RHO_M
from stbench.llm_credit import credit_sentence

# job id -> (C as the prompt says it and in metres, R and in metres, the matching radius as the prompt says it and in
# metres, the margin line the prompt gives the kinds the job asks, whether it asks a list, whether it asks
# `tangle_ahead` during the job), by hand from each job's declaration (`job_*.py`)
TABLE = {
    "vacuum/1": (("10 cm", 0.1), ("1 m", 1.0), ("0.5", 0.5), "The margin is 1 m for a place.", False, False),
    "vacuum/2": (("10 cm", 0.1), ("1 m", 1.0), ("0.5", 0.5),
                 "The margin is 1 m for a place.\n- For an offset, a distance in metres or a heading, the margin is "
                 "10 cm or a tenth of the right distance (from your camera to the right point, the right number of "
                 "metres or the best heading's run), whichever is larger.", True, True),
    "helper/1": (("15 cm", 0.15), ("1 m", 1.0), ("0.5", 0.5),
                 "The margin is 1 m for a place.\n- For an offset, a distance in metres or a heading, the margin is "
                 "15 cm or a tenth of the right distance (from your camera to the right point, the right number of "
                 "metres or the best heading's run), whichever is larger.", True, False),
    "amr/1": (("45 cm", 0.45), ("1 m", 1.0), ("1", 1.0),
              "For an offset or a distance in metres, the margin is 45 cm or a tenth of the right distance (from your "
              "camera to the right point or the right number of metres), whichever is larger.", True, False),
    "drone/1": (("2.4 m", 2.4), ("2 m", 2.0), ("2", 2.0),
                "For an offset or a distance in metres, the margin is 2.4 m or a tenth of the right distance (from "
                "your camera to the right point or the right number of metres), whichever is larger.", False, False)}


@pytest.mark.parametrize("job_id", TABLE)
def test_the_numbers_in_a_jobs_credit_text_are_the_jobs_and_the_constants(job_id):
    """Contract (2026-10-14): in the credit section of each registered job the margins are that job's C (offset,
    distance, heading) and R (a single place, only for a job that asks one; a list's margin is its matching radius,
    `match_m`), the matching radius is its `match_m`, and the bearing, the least pixel, the reach of a match, the
    long-thing threshold, the catastrophe weights and the distance of a tangle are the constants of `credit_kernel`,
    `rules` and `hazard_rule`; each is printed where the job asks the kind or family it is about (the bearing for an
    offset, the long thing for a place, -2 for `clear_path`, -3 and the worked zero for a distance) and nowhere else.
    The table is checked against the job first (so the hand numbers are the job's), then each number is looked for in
    the words, the constants imported and not typed. Regression caught: a number typed into the prompt apart from the
    job or the kernel, a table that drifted from the job it describes, or a number told to a job that asks nothing it
    is about."""
    (c, c_m), (r, r_m), (match, match_m), margin, lists, tangle = TABLE[job_id]
    job = job_registry.get(job_id)
    assert (job.clearance_m, job.reacquire_m, job.match_m) == pytest.approx((c_m, r_m, match_m))
    families, kinds = set(JOB_FAMILIES[job_id]), JOB_KINDS[job_id]
    text = text_of(job_id)
    assert f"{margin}\n" in text
    assert (f"within {ANGLE_DEG} degrees of the right point" in text) is ("offset" in kinds)
    assert f"never finer than {MIN_RHO_M * 100:g} cm" in text
    place = bool({"position", "positions"} & set(kinds))
    assert (f"A thing longer than {2 * rules.WIDEN_FREE_M:g} m" in text) is place
    assert (f"beyond {rules.WIDEN_FREE_M:g} m" in text) is place
    assert (f"scores -{CATASTROPHE_THING}. That is a distance" in text) is ("clear_path" in families)
    assert (f"scores -{CATASTROPHE_PERSON}. That is a distance that says a person" in text) is bool(
        {"closest_approach", "approach_mem", "closest_person"} & families)
    assert "That is an offset that puts a person farther" not in text   # no registered job asks `person_mem`
    assert (f"a distance {c} too long scores 0" in text) is ("number" in kinds)
    assert (f"A pair within {MATCH_REACH} matching radii" in text) is lists
    assert (f"A matching radius is {match} m, grown by" in text) is lists
    assert (f"within {hazard_rule.FRONT_M * 100:g} cm of your front edge" in text) is tangle


@pytest.mark.parametrize("job_id", TABLE)
def test_the_list_catastrophe_and_the_heading_text_are_there_for_the_jobs_that_ask_them(job_id):
    """Contract (2026-10-14; final review S7): the list bullets are in the credit section of a job that asks a
    list (`task_things` after the job, or `tangle_ahead` or `drop_zone_people` during it), the clause that a list
    leaving out a critical cord scores -3 only in a job that asks `tangle_ahead` during the job (the vacuum), the
    heading's definition and regret only in a job that asks `open_heading` (the vacuum and the helper), and none of them
    in the others. Regression caught: a clause for a family the job never asks, which tells the agent of a question
    that does not come, or one left out of the job that asks it."""
    *_, lists, tangle = TABLE[job_id]
    headings = "open_heading" in JOB_FAMILIES[job_id]
    text = text_of(job_id)
    assert ("A list of places earns each thing in it" in text) is lists
    assert text.count("A list that leaves out") == int(tangle)
    assert ("A heading is a turn in place and then straight on" in text) is headings
    assert ("A heading is credited by its regret" in text) is headings
    assert ("Its right direction is room to move" in text) is headings
    assert ("A shorter run is its harmful side" in text) is headings
    assert ("For the way to a thing" in text) is ("way_to" in JOB_FAMILIES[job_id])   # 0.11: the helper alone asks it


@pytest.mark.parametrize(("name", "value", "words"), [
    ("ANGLE_DEG", 30, "within 30 degrees of the right point"),
    ("MIN_RHO_M", 0.02, "never finer than 2 cm"),
    ("MATCH_REACH", 3, "A pair within 3 matching radii"),
    ("CATASTROPHE_THING", 5, "had seen scores -5. That is"),
    ("CATASTROPHE_PERSON", 7, "Going into a person scores -7. That is")])
def test_each_number_of_the_text_moves_with_its_constant(monkeypatch, name, value, words):
    """Contract: the numbers the credit section prints are read from `credit_kernel`'s constants when it is built, so a
    changed constant is a changed text and the old number is gone (the bearing, the least pixel, the reach of a match
    and the two catastrophe weights; the others are pinned in `test_llm_credit_sentence.py`). Regression caught: a
    constant moved in the kernel while the prompt keeps telling the agent the old number."""
    before = credit_sentence(0.5, 0.1, 1.0, lists=True)
    monkeypatch.setattr(llm_credit, name, value)
    after = credit_sentence(0.5, 0.1, 1.0, lists=True)
    assert words in after and words not in before
