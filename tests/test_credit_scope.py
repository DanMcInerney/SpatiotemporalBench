"""Which paragraphs of the credit section a job is told (`llm_credit.credit_sentence`'s `kinds` and `families`;
2026-10-14: the credit text said what no question of the job needed). Hand tables: each paragraph's marker, the answer
types and families that need it, and what each registered job asks. No model, no network."""

from __future__ import annotations

import pytest

from credit_kit import JOB_FAMILIES, JOB_KINDS, section, text_of
from stbench import job_registry
from stbench.brief import answer_types, asked_families
from stbench.llm_credit import credit_sentence
from stream_kit import fake_job

# A phrase only its bullet holds, and what needs the bullet: (the answer types, the families); a bullet is needed when
# the job asks one of the types or one of the families. The bullets that are said whatever the job asks are SHARED.
NEEDS = {
    "An offset goes to the point of the thing": ({"offset"}, set()),
    "of the thing or person nearest": (set(), {"person_mem"}),
    "A place in the map, alone or in a list": ({"position", "positions"}, set()),
    "How far the body could move is how far it can go": (set(), {"clear_path"}),
    "A heading is a turn in place and then straight on": ({"heading"}, set()),
    "The right direction for an offset": ({"offset"}, set()),
    "For a distance in metres: on the same side as the right one": ({"number"}, set()),
    "(for how far the body could move, under it there is no room to move)": (set(), {"clear_path"}),
    "Where the right one is not under that room": ({"number"}, set()),
    "For a place: within the margin of the thing.": ({"position"}, set()),
    "For an offset, the harmful side is putting a thing": ({"offset"}, set()),
    "a thing or a person farther from you": (set(), {"person_mem"}),
    "For a distance in metres, it is being longer than the right one": ({"number"}, set()),
    "more room than there was": (set(), {"clear_path"}),
    "a person who came closer than you said": (set(), {"closest_approach", "approach_mem", "closest_person"}),
    "A place has no harmful side.": ({"position", "positions"}, set()),
    "Going into something the robot had seen": (set(), {"clear_path"}),
    "Going into a person scores": (set(), {"closest_approach", "approach_mem", "closest_person", "person_mem"}),
    "an offset that puts a person farther from you than they are": (set(), {"person_mem"}),
    "a distance that says a person is clear who came closer": (set(), {"closest_approach", "approach_mem",
                                                                       "closest_person"}),
    "A thing longer than": ({"position", "positions"}, set()),
    "A list of places earns each thing in it": (set(), {"task_things", "tangle_ahead", "drop_zone_people"}),
    "A list that leaves out": (set(), {"tangle_ahead", "drop_zone_people"}),
    "A heading is credited by its regret": ({"heading"}, set()),
    "Its right direction is room to move": ({"heading"}, set()),
    "A shorter run is its harmful side": ({"heading"}, set()),
    "For the way to a thing, a heading is credited": (set(), {"way_to"}),
    ("A heading with less room than the body needs, or one into floor you have not seen, scores as a wrong "
     "answer."): (set(), {"way_to"}),
    "On the harmful side that is twice the halved margin": ({"number", "offset"}, set()),
}
SHARED = ("Credit is four quarters that add up.", "Quarter 4 is whole when the answer is as exact as one pixel",
          "A margin finer than twice the pixel's exactness", "Some errors count against you twice as fast",
          "A question may name a thing your camera never showed clearly enough to tell what it is,",
          "A worse error still is one on the harmful side", "`unknown` is never one of these.",
          "An answer with credit s scores s minus (1 - s).", "An answer with the right direction alone scores -0.5.")


def check(text: str, kinds, families) -> None:
    """Each bullet of NEEDS is in `text` exactly when `kinds` or `families` ask what needs it; every SHARED one is."""
    for marker, (types, needing) in NEEDS.items():
        needed = bool(types & set(kinds)) or bool(needing & set(families))
        why = "missing, though it is needed" if needed else "told, though nothing asked needs it"
        assert (marker in text) is needed, f"{marker!r} is {why}"
    for marker in SHARED:
        assert marker in text, f"{marker!r} is missing"


@pytest.mark.parametrize("job_id", JOB_FAMILIES)
def test_each_registered_job_is_told_the_paragraph_of_each_type_and_family_it_asks_and_no_other(job_id):
    """Contract (2026-10-14): the credit section of a registered job has the paragraph of every answer type and
    family the job asks (the point an offset names for an offset, the place for a position or a list of them, the
    corridor for `clear_path`, a person's distance for `closest_approach` and `closest_person`, the list rules for the
    list families, the regret for `open_heading`, the right direction and harmful side of each kind) and no paragraph
    that none of them needs; the quarters, the halving, the opening of the catastrophes and the score are told to every
    job. The job's types and families are by hand (`credit_kit`), checked against what the code finds from the job
    first, then every paragraph's marker is looked for. Regression caught: a rule a question needs that the section
    lost, a paragraph told to a job that asks nothing it is about (the drone's place and person, the AMR's corridor),
    or a table that drifted from the job."""
    job = job_registry.get(job_id)
    assert asked_families(job) == JOB_FAMILIES[job_id] and answer_types(job) == JOB_KINDS[job_id]
    check(text_of(job_id), JOB_KINDS[job_id], JOB_FAMILIES[job_id])


def test_the_four_shipped_jobs_are_told_what_their_questions_need_and_the_drone_and_the_amr_less():
    """Contract (2026-10-14): by hand, from what the shipped jobs ask. The drone (`clear_path` and `ego_mem`: a
    corridor and an offset) is told the offset's point, the corridor and the distance's rules with the -2 for a corridor
    said clear that is not, and nothing of a place, a person, a list or a heading. The AMR (`closest_approach`,
    `closest_person`, `ego_now`, `ego_mem`, `task_things`) is told the offset's point, the place's, a person's distance
    with the -3, and the list's rules, and nothing of a corridor, a heading or a single place's margin. The helper is
    told no cord's list rule; the vacuum no person's distance. Regression caught: every paragraph told to every job in
    the same words."""
    drone, amr, helper, vacuum = (text_of(job_id) for job_id in ("drone/1", "amr/1", "helper/1", "vacuum/2"))
    for gone in ("A place in the map", "Going into a person", "A list of places", "A heading is a turn",
                 "A thing longer", "For a place: within the margin", "A place has no harmful side",
                 "a person who came closer"):
        assert gone not in drone
    for kept in ("An offset goes to the point of the thing nearest", "How far the body could move",
                 "Going into something the robot had seen scores -2", "more room than there was",
                 "For an offset or a distance in metres, the margin is 2.4 m or a tenth of the right distance "
                 "(from your camera to the right point or the right number of metres), whichever is larger."):
        assert kept in drone
    for gone in ("How far the body could move", "Going into something the robot had seen", "more room than there was",
                 "A heading is a turn", "For a place: within the margin", "(for how far the body could move",
                 "A list that leaves out"):
        assert gone not in amr
    for kept in ("An offset goes to the point of the thing nearest", "A place in the map, alone or in a list",
                 "Going into a person scores -3. That is a distance that says a person is clear who came closer.",
                 "A list of places earns each thing in it", "A thing longer than",
                 "For an offset or a distance in metres, the margin is 45 cm or a tenth of the right distance "
                 "(from your camera to the right point or the right number of metres), whichever is larger."):
        assert kept in amr
    assert "A list that leaves out" not in helper and "A list that leaves out" in vacuum
    assert "Going into a person" in helper and "Going into a person" not in vacuum
    assert "a person who came closer than you said" in helper and "a person who came closer" not in vacuum


# (a fake job's keywords, what it is for) by hand: each asks a combination no registered job does
COMBOS = {
    "an offset of a person": {"looks": ("person_mem",), "ends": ()},
    "how close a person came and nothing else": {"ends": ("closest_person",)},
    "a distance that is no corridor and no person's": {"ends": ("distance_between",)},
    "places with no list rules": {"ends": ("out_of_view",)},
    "a heading alone": {"looks": ("open_heading",), "ends": ()},
    "nothing at all": {"ends": ()},
    "both cords and people in lists": {"looks": ("drop_zone_people", "tangle_ahead"), "ends": ()},
    "everything the helper asks and a person's offset": {"looks": ("clear_path", "closest_approach", "ego_mem",
                                                                    "person_mem", "open_heading"),
                                                         "ends": ("object_where", "task_things")}}


@pytest.mark.parametrize("name", COMBOS)
def test_a_job_that_asks_a_combination_no_registered_job_asks_is_told_what_it_needs_and_no_more(name):
    """Contract (2026-10-14): the same rule for any job, built here from fake jobs through the driver's own path
    (`brief.answer_types`, `asked_families`, `asks_lists`, `list_hazards`, `asks_headings`): an offset of a person is
    told that the point may be a person's and that a person placed farther is harmful, with the -3; how close a person
    came alone, the distance's rules and the person's -3 with no corridor; a distance of no family that has words
    (`distance_between`) the distance's rules and neither a corridor nor a person; places with no list rules the place
    and not a single place's margin; a heading alone the heading's rules; no question at all the shared bullets only.
    Regression caught: a bullet that stays when the family it is about goes, or goes when another kind needs it."""
    job = fake_job(clearance=0.1, reacquire=1.0, **COMBOS[name])
    families = asked_families(job)
    check(section(job), answer_types(job), families)


def test_a_term_a_bullet_uses_is_defined_for_every_job_that_gets_the_bullet():
    """Contract (2026-10-14): the room the body needs is defined where a distance in metres is (the body's length
    plus C), and in the heading's harmful-side bullet for a job that asks a heading and no distance, so no
    job meets the term undefined; a distance in metres has no remark about a corridor or a person unless the job asks
    that family; the worked zero uses a distance when the job asks one, an offset when it asks that and no distance,
    and none when it asks neither; the margin line names the kinds the job asks. Hand cases on fake jobs. Regression
    caught: a heading bullet that uses a term nothing defines, a worked example about a kind the job never asks, or a
    margin line that names a kind of answer the job does not ask."""
    heading = section(fake_job(clearance=0.1, reacquire=1.0, looks=("open_heading",), ends=()))
    assert ("Its right direction is room to move: a run of the room the body needs (the robot body's length plus "
            "10 cm) or longer (or the best run, where no heading has that much).") in heading
    assert "A shorter run is its harmful side, where the margin is half as large." in heading
    assert ("For a heading, the margin is 10 cm or a tenth of the right distance (the best heading's run), whichever "
            "is larger.") in heading and "On the harmful side that is twice" not in heading
    both = text_of("helper/1")
    assert ("Its right direction is room to move: a run of the room the body needs or longer (or the best run, "
            "where no heading has that much).") in both
    assert " (the robot body's length plus 15 cm) or longer" not in both
    plain = section(fake_job(clearance=0.1, reacquire=1.0, ends=("distance_between",)))
    assert "For a distance in metres, it is being longer than the right one.\n" in plain
    assert "the robot body's length plus 10 cm.\n" in plain and "how far the body could move" not in plain
    assert "a distance 10 cm too long scores 0" in plain
    assert ("For a distance in metres, the margin is 10 cm or a tenth of the right distance (the right number of "
            "metres), whichever is larger.") in plain
    offset = section(fake_job(clearance=0.1, reacquire=1.0, looks=("person_mem",), ends=()))
    assert "a thing placed 10 cm too far scores 0" in offset
    assert ("For an offset, the margin is 10 cm or a tenth of the right distance (from your camera to the right "
            "point), whichever is larger.") in offset
    assert "putting a thing or a person farther from you" in offset
    assert "of the thing or person nearest the floor point below your camera" in offset
    places = section(fake_job(clearance=0.1, reacquire=1.0, ends=("out_of_view",)))
    assert "The margin is" not in places and "whichever is larger" not in places and "A place in the map" in places
    nothing = section(fake_job(clearance=0.1, reacquire=1.0, ends=()))
    assert "The margin is" not in nothing and "whichever is larger" not in nothing
    assert "On the harmful side that is twice" not in nothing
    assert "Credit is four quarters that add up." in nothing and "`unknown` scores 0" in nothing


def test_a_caller_that_knows_no_job_is_told_every_bullet_of_a_kind_and_a_family():
    """Contract: with no `kinds` and no `families` (a test or an external harness that knows no job) every bullet about
    a kind or a family is told, as before the section was scoped: with the list, heading and cord flags set the text
    holds every paragraph of `NEEDS` and `SHARED`; a `kinds` alone leaves the family bullets (the corridor, the
    person's) all told, and `families` alone the kind bullets; a name in `kinds` that is no answer type, `unknown`
    included (it is always shown), is refused. Regression caught: a default that tells a caller nothing of the
    corridor, or a stray kind that quietly drops a paragraph."""
    full = credit_sentence(0.5, 0.1, 1.0, lists=True, headings=True, hazards=("drop_zone_people", "tangle_ahead"))
    check(full, {"offset", "position", "positions", "number", "heading"}, {
        "person_mem", "clear_path", "closest_approach", "task_things", "tangle_ahead", "open_heading", "way_to"})
    kinds_only = credit_sentence(0.5, 0.1, 1.0, kinds=("offset",))
    for marker in ("How far the body could move", "Going into a person scores", "a thing or a person farther"):
        assert marker in kinds_only
    assert "A place in the map" not in kinds_only and "The right direction for an offset" in kinds_only
    families_only = credit_sentence(0.5, 0.1, 1.0, families=("clear_path",))
    assert "A place in the map" in families_only and "Going into a person" not in families_only
    assert "How far the body could move" in families_only
    for stray in (("blob",), ("position", "unknown")):
        with pytest.raises(ValueError, match="is no answer type"):
            credit_sentence(0.5, 0.1, 1.0, kinds=stray)


def test_the_families_a_job_asks_are_its_decision_look_and_end_families_sorted():
    """Contract: `brief.asked_families` is the families of a job's decision kinds and looks (during the job) and of its
    end families after it (the questionnaire's default `object_where` for a job that names none, none for `ends=()`),
    each once and sorted, whatever the order they were written in; a family the registry lacks is refused naming the
    job. Hand cases on fake jobs. Regression caught: an end family left out (the AMR's `closest_person`), the default
    end forgotten, a family counted twice, or one the registry lacks passing silently into a section."""
    assert asked_families(fake_job(looks=("ego_mem",))) == ("ego_mem", "object_where")
    assert asked_families(fake_job(looks=("ego_mem", "clear_path"), ends=())) == ("clear_path", "ego_mem")
    assert asked_families(fake_job(decisions={"before_doorway": ("path_ahead", "out_of_view"),
                                              "room_done": ("path_ahead",)}, ends=("order", "when_seen"))) == (
        "order", "out_of_view", "path_ahead", "when_seen")
    assert asked_families(fake_job(ends=())) == ()
    with pytest.raises(ValueError, match="fake/1 asks no_such_family, which the family registry lacks"):
        asked_families(fake_job(looks=("no_such_family",)))


@pytest.mark.parametrize("job_id, turns", [("drone/1", False), ("helper/1", True), ("vacuum/2", True),
                                          ("amr/1", False)])
def test_the_turned_move_is_told_only_to_a_job_whose_corridor_turns(job_id, turns):
    """Contract: the corridor's turn-in-place bullet (`llm_credit.CORRIDOR_TURN`) is in a job's credit section exactly
    when its `clear_path` may ask a turned move (`brief.asks_turns`: left or right in `JobSpec.clear_directions`); hand
    table: the drone flies ahead and back only, the AMR asks no corridor, the helper and the vacuum turn. Regression
    caught: a flier told how a turn in place is measured, a move it is never asked about."""
    from stbench import brief, job_registry, llm_credit
    job = job_registry.get(job_id)
    assert brief.asks_turns(job) is turns
    text = llm_credit.credit_sentence(job.match_m, job.clearance_m, job.reacquire_m, lists=brief.asks_lists(job),
                                      headings=brief.asks_headings(job), hazards=brief.list_hazards(job),
                                      kinds=brief.answer_types(job), families=brief.asked_families(job),
                                      turns=brief.asks_turns(job))
    assert ("turn in place to its left or right first" in text) is turns
