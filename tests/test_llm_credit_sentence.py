"""The credit section of the reference agent's prompt (`llm_credit`) against the credit's own numbers
(`credit_kernel`). The whole section is pinned by hand in `test_subgoal_prompt.py`; this file ties what it says to what
the kernel computes, so a prompt cannot disagree with the scorer."""

from __future__ import annotations

import math
import re

import pytest

import family_kit as fk
import task_kit as tk
from credit_kit import JOB_FAMILIES, text_of
from space_kit import vacuum_run
from stbench import credit_kernel, job_registry, leak, llm_credit, llm_scan
from stbench.brief import asks_headings, asks_lists, list_hazards
from stbench.credit_kernel import ANGLE_DEG, MATCH_REACH, MIN_RHO_M, graded
from stbench.llm_credit import PLACE_TARGET, credit_sentence
from stbench.questions_where import pools
from stbench.score_set import assignment
from stream_kit import fake_job

FLAGS = [(False, False), (True, False), (False, True), (True, True)]
MARGINS = re.compile(r"The margin is (.+?) for a place\.\n- For an offset, a distance in metres or a heading, the "
                     r"margin is (.+?) or a tenth of the right distance \(")
PRINTED = {0.1: "10 cm", 0.15: "15 cm", 0.45: "45 cm", 2.4: "2.4 m"}   # a job's C as the prompt says it, by hand


def metres(text: str) -> float:
    number, unit = text.split()
    return float(number) / (100 if unit == "cm" else 1)


@pytest.mark.parametrize("c", [0.1, 0.15, 0.45, 2.4])
def test_the_margins_are_the_jobs_c_and_r_and_the_quarters_are_the_kernels(c):
    """Contract (plan P2; brief §3.1): the margins the section prints are the job's C and R (the numbers the gold rows
    carry as `need_m`), and what it says each distance earns is what `credit_kernel.graded` computes at them, worked
    by hand: an answer within one pixel scores 1 (+1 penalised), at the margin 3/4 (+0.5), at twice the margin 1/2 (0,
    as good as `unknown`), at four times it with the right direction 1/4 (-0.5), and a wrong direction beyond that 0
    (-1); on the harmful side the margin is halved, so half the margin scores 3/4, and a distance too long by the whole
    margin (twice the halved margin) scores 1/2 (0), which is the example the section works at the job's own C.
    Regression caught: a prompt whose margin or whose zero point disagrees with the kernel, a margin typed apart from
    the job, an example that does not follow from the halving, or a halving that the scorer does not apply. (Told every
    kind of answer, as for a caller that knows no job; which kinds a job's margin line names is
    `test_credit_scope.py`'s.)"""
    sentence = credit_sentence(0.5, c, 1.0, headings=True)
    printed_r, printed_c = MARGINS.search(sentence).groups()
    assert metres(printed_c) == pytest.approx(c) and metres(printed_r) == pytest.approx(1.0)
    rho = MIN_RHO_M
    worked = [(rho, True, 1.0), (c, True, 0.75), (2 * c, True, 0.5), (4 * c, True, 0.25), (5 * c, True, 0.25),
              (5 * c, False, 0.0)]
    for error, direction, credit in worked:
        assert graded(error, rho, c, direction, False) == pytest.approx(credit)
    assert graded(c / 2, rho, c, True, True) == pytest.approx(0.75)
    assert graded(c, rho, c, True, True) == pytest.approx(0.5)
    for words in ("An exact answer scores 1. An answer off by the margin scores 0.5.",
                  "An answer off by twice the margin scores 0, as good as `unknown`.",
                  f"On the harmful side that is twice the halved margin: with a margin of {PRINTED[c]}, a distance "
                  f"{PRINTED[c]} too long scores 0.",
                  "An answer with the right direction alone scores -0.5. A wrong answer scores -1."):
        assert words in sentence


@pytest.mark.parametrize(("lists", "headings"), FLAGS)
def test_the_sentence_names_the_kernels_constants_and_each_kinds_harmful_side(lists, headings):
    """Contract (plan P2): the bearing test is the kernel's `ANGLE_DEG`, a list pairs within `MATCH_REACH` matching
    radii; the harmful side is named for an offset (a thing or a person farther than it is), a distance (longer than
    the right one) and, when the job asks a heading, a run shorter than the room the body needs, and a place has none;
    the corridor covers a reverse (the rear edge) and a turn in place (the front edge after it). Regression caught: a
    bearing or a reach typed apart from the kernel, a harmful side the scorer does not apply to that kind, or a
    corridor that defines only the move straight ahead."""
    sentence = credit_sentence(0.5, 0.1, 1.0, lists=lists, headings=headings)
    assert f"within {ANGLE_DEG:g} degrees" in sentence
    assert (f"A pair within {MATCH_REACH} matching radii" in sentence) is lists
    assert "the harmful side is putting a thing or a person farther from you than it is" in sentence
    assert "it is being longer than the right one" in sentence
    assert "A place has no harmful side." in sentence
    assert ("A shorter run is its harmful side, where the margin is half as large" in sentence) is headings
    assert "the room the body needs, which is the robot body's length plus 10 cm (for how far" in sentence
    assert "Straight back, count the distance its rear edge moves. The body does not turn." in sentence
    assert "turn in place to its left or right first" in sentence
    assert "from where that edge stands after the turn" in sentence


# the sections these tests look at: every combination of the list and heading bullets told every kind and family (as to
# a caller that knows no job), and the section of each registered job as the driver builds it
CASES = [(lists, headings, None) for lists, headings in FLAGS] + [(False, False, job_id) for job_id in JOB_FAMILIES]


def built(lists: bool, headings: bool, job_id: str | None) -> str:
    return text_of(job_id) if job_id else credit_sentence(0.5, 0.1, 1.0, lists=lists, headings=headings)


@pytest.mark.parametrize(("lists", "headings", "job_id"), CASES)
def test_the_sentence_trips_no_leak_rule_and_coaches_on_nothing(lists, headings, job_id):
    """Contract (plan P2; the leak scan's rules): the section holds nothing `leak.RULES` flags (an entity id, a region
    id, the word "truth"), no word of memory or depth, no share of a range, and none of the phrases the prompt's tests
    keep out ("at least", "% of r"), in every combination of the list and heading bullets and in the section of each
    registered job (whose kinds and families leave bullets out and put others in their place). Regression caught: a new
    bullet that trips the scan of what a model is sent or coaches on what to keep."""
    sentence = built(lists, headings, job_id)
    assert [rule for rule, rx in leak.RULES.items() if rx.search(sentence)] == []
    for word in ("memory", "depth", "at least", "% of r", "straight-line distance", "of the right answer"):
        assert word not in sentence.lower()


@pytest.mark.parametrize(("lists", "headings", "job_id"), CASES)
def test_the_section_is_a_titled_list_of_plain_bullets_in_short_sentences(lists, headings, job_id):
    """Contract (2026-10-14: the credit text was one paragraph of about 700 words): the section is its title
    "How an answer is credited:" and then bullets only, one line each (so no line of it can start a frame header, which
    `llm_scan.HEADER` reads), and no sentence of it runs past 50 words (the longest is 38, and the old paragraph's
    were over 100), in every combination of the list and heading bullets and in the section of each registered job.
    Regression caught: the section written again as one paragraph, or a bullet that packs a whole paragraph into one
    sentence."""
    sentence = built(lists, headings, job_id)
    title, *bullets = sentence.splitlines()
    assert title == "How an answer is credited:" and len(bullets) > 15
    assert all(line.startswith("- ") and not line.startswith("- -") for line in bullets)
    assert not any(llm_scan.HEADER.findall(line) for line in bullets)
    sentences = [s for line in bullets for s in re.split(r"(?<=\.)\s+(?=[A-Z`])", line[2:])]
    assert max(len(s.split()) for s in sentences) <= 50


def test_the_prompt_names_the_point_a_place_measures_and_it_is_the_point_the_golds_use():
    """Contract (0.9.0 review B3; FD-decision-questions: full credit is always reachable): the credit sentence says that
    a place in the map, alone or in a list, is the middle of the thing where it stood, as x and y in the map frame, not
    the point nearest the camera that an offset names; and that is the point the golds of `object_where` and of a
    `task_things` item hold. By hand:
    - `object_where`: a 1.0 x 0.4 x 0.5 m box standing at (2.0, 1.0), seen from a camera east of it at x = 5.9. Its gold
      is (2.0, 1.0), the middle of its footprint; the nearest point of its outline to the camera is (2.5, 1.2), 0.54 m
      from it, where the full quarter is one pixel wide;
    - `task_things`: a 1.0 x 1.0 x 0.5 m box standing at (14, 1.8) beside a way along y = 1. Its item is (14.0, 1.8);
      the nearest point of its outline to the way is (14, 1.3), 0.5 m off.
    Each gold is also `World.middle`'s x and y of the thing at its sighting, the function the widening and the range
    rest on. The place clause says only "within the margin of the thing", and the thing's room is not told. Regression
    caught: a prompt that tells the reader an offset's convention only (so an ideal reader answers the nearest visible
    point and loses a quarter on every place), a target typed apart from the golds, or the room quarter still told."""
    s = vacuum_run()
    s.item("E0001", "bag", {"color": "red"}, [(0, None, (2.0, 1.0, 0.0))], size=(1.0, 0.4, 0.5))
    s.see([10], "E0001", 1500)
    w = fk.world(s)
    (where,) = pools(w)[0]
    assert where.gold == {"type": "position", "xy": [2.0, 1.0]}
    thing = next(e for e in s.entities if e.id == "E0001")
    assert where.gold["xy"] == pytest.approx(w.middle(thing, where.evidence_k[0])[:2])
    assert math.dist(where.gold["xy"], (2.5, 1.2)) == pytest.approx(0.5385, abs=1e-4)
    t = tk.script()
    tk.put(t, "E", (14, tk.y(0.3, 1.0)), tk.BIG, at=(5,))
    item = tk.places(tk.cand(t))[0]
    tw = tk.world(t)
    assert item == [[14.0, 1.8]]
    assert item[0] == pytest.approx(tw.middle(next(e for e in t.entities if e.id == "E"), 5)[:2])
    for lists in (False, True):
        sentence = credit_sentence(0.5, 0.1, 1.0, lists=lists)
        assert PLACE_TARGET in sentence
        assert ("A place in the map, alone or in a list, is the middle of the thing where it stood: its x and y in "
                "the map frame") in sentence
        assert "It is not the point nearest your camera." in sentence
        assert "- For a place: within the margin of the thing." in sentence
        assert "same room" not in sentence


def test_the_pairing_and_the_threshold_clause_say_what_the_scorer_does():
    """Contract (0.9.0 review N4, N6): a list's pairing is "the pairing whose summed credit is largest", which is
    `score_set.assignment`'s (by hand: credits [[0.9, 0.3], [0.5, 0]] pair the first answer with the first thing, 0.9
    against 0.3 + 0.5 = 0.8, though the other pairing would score the list higher); and "there is no room to move" is
    said only of the corridor, since the other distance a job is asked, how close a person came, has no such meaning.
    Regression caught: a pairing worded as the one that earns the most (which reads as the list's score, not the sum),
    or a corridor remark said of every distance."""
    assert assignment([[0.9, 0.3], [0.5, 0.0]]) == [(0, 0), (1, 1)]
    sentence = credit_sentence(0.5, 0.1, 1.0, lists=True)
    assert "in the pairing whose summed credit is largest" in sentence and "earns the most" not in sentence
    assert sentence.count("no room to move") == 1
    assert "(for how far the body could move, under it there is no room to move)" in sentence


def test_the_list_bullets_say_what_the_scorer_does_with_a_place_and_with_an_empty_list():
    """Contract (0.11.0): the list bullets state `score_set.graded_set`. A pair's right direction is being within a
    matching radius of the thing, as a place's is within R (a list item is a place), not being paired; and where there
    were none the empty list is the one thing, credited 1 divided by 1 plus half the number of false places (by hand,
    1 / 1.5 = 2/3 for one false place, with `FP_WEIGHT` one half), where before it was right only when there were none
    (one false place scored 0, as ten did). Regression caught: a prompt that tells the agent any paired place has the
    right direction, or that an empty list is right or wrong, while the scorer grades both."""
    sentence = credit_sentence(0.5, 0.1, 1.0, lists=True)
    assert (f"A pair within {MATCH_REACH} matching radii earns the four quarters above: its right direction is being "
            "within a matching radius of the thing, and the matching radius is its margin.") in sentence
    assert ("When there were none, the empty list is the one thing: the credit is 1 divided by 1 plus half the number "
            "of false places.") in sentence
    assert "being paired is its right direction" not in sentence
    assert "An empty list is right only when there were none" not in sentence
    assert "The list's credit is the sum of its pairs' credits divided by the number of things plus half" in sentence


def test_approach_mem_is_told_the_person_distance_paragraphs_as_closest_approach_is():
    """Contract (0.11.0): `approach_mem`, the out-of-view twin of `closest_approach`, is a person's distance in metres,
    so a job that asks it is told the person who came closer than the answer said and the -3 for going into them,
    word for word as for `closest_approach`; a job that asks neither is told neither. Regression caught: a job whose
    only person-distance family is the new one scored for a catastrophe the prompt never states."""
    kinds = ("number",)
    twin = credit_sentence(0.45, 0.45, 1.0, kinds=kinds, families=("approach_mem",))
    assert twin == credit_sentence(0.45, 0.45, 1.0, kinds=kinds, families=("closest_approach",))
    assert "a distance that says a person is clear who came closer" in twin and "Going into a person scores -3." in twin
    plain = credit_sentence(0.45, 0.45, 1.0, kinds=kinds, families=("ego_now",))
    assert "who came closer" not in plain and "Going into a person" not in plain


def test_a_kernel_whose_harm_or_false_place_weight_is_not_a_half_cannot_leave_the_prompt_saying_half(monkeypatch):
    """Contract (plan P2): the sentence says the harmful margin and a false place are each half, from
    `credit_kernel.HARM_FACTOR` and `FP_WEIGHT`; if either changes the sentence refuses to be built until its words are
    changed. Regression caught: a kernel constant moved while the prompt still tells the agent the old halving."""
    credit_sentence(0.5, 0.1, 1.0, lists=True)
    monkeypatch.setattr(llm_credit, "HARM_FACTOR", 0.4)
    with pytest.raises(ValueError, match="HARM_FACTOR"):
        credit_sentence(0.5, 0.1, 1.0)
    monkeypatch.setattr(llm_credit, "HARM_FACTOR", credit_kernel.HARM_FACTOR)
    monkeypatch.setattr(llm_credit, "FP_WEIGHT", 0.25)
    with pytest.raises(ValueError, match="FP_WEIGHT"):
        credit_sentence(0.5, 0.1, 1.0, lists=True)


def test_a_job_that_asks_a_list_or_a_heading_is_found_from_its_ends_and_its_during_families():
    """Contract (plan P2; 0.10.0): `brief.asks_lists` is true for a job whose `ends` hold `task_things` or that asks
    `tangle_ahead` or `drop_zone_people` at a look point, and `asks_headings` for a job that asks `open_heading` at a
    decision point or a look point; a job with the default ends or none asks neither. Hand cases on fake jobs, then the
    registered jobs: the helper and the AMR end with `task_things` and the vacuum asks the cable list, so those three
    are told how a list is credited and the drone, which asks none, is not. Regression caught: a paragraph shown to a
    job that never asks that question (or hidden from one that does), which the prompt tests would not see because they
    are given the flags."""
    assert not asks_lists(fake_job()) and not asks_lists(fake_job(ends=("object_where",))) and not asks_headings(
        fake_job())
    assert asks_lists(fake_job(ends=("object_where", "task_things"))) and not asks_lists(fake_job(ends=()))
    assert asks_headings(fake_job(looks=("clear_path", "open_heading")))
    assert asks_headings(fake_job(decisions={"before_doorway": ("open_heading",)}))
    assert not asks_headings(fake_job(looks=("clear_path", "ego_mem")))
    assert asks_lists(fake_job(looks=("tangle_ahead",))) and asks_lists(fake_job(looks=("drop_zone_people",), ends=()))
    assert not asks_lists(fake_job(looks=("closest_approach", "ego_mem"), ends=()))
    # 0.10.0: the helper and the AMR end with `task_things` and the vacuum asks `tangle_ahead`, so they are told how a
    # list is credited; the drone asks `drop_zone_people` no more (final review S2) and no list at all
    listing = {job.id for job in job_registry.all_jobs() if asks_lists(job)}
    assert listing == {"helper/1", "amr/1", "vacuum/2"}


@pytest.mark.parametrize("bad", [(0, 0.1, 0.5), (0.5, -1.0, 0.5), (0.5, 0.1, 0)])
def test_a_number_that_is_no_distance_is_refused(bad):
    """Contract: the sentence is built from the job's three numbers, each above 0. Regression caught: a sentence that
    prints a zero or negative margin."""
    with pytest.raises(ValueError, match="must be positive"):
        credit_sentence(*bad)


@pytest.mark.parametrize(("lists", "headings"), FLAGS)
def test_the_catastrophe_sentence_follows_the_harmful_side_and_states_the_kernels_weights(lists, headings):
    """Contract (0.10.0, FD-questions-0.10 §4.7; final review S5): once, in every combination of the bullets, straight
    after the harmful side and before the long thing's place, the section says that an error on the harmful side by more
    than the whole margin that would put the robot into harm scores a flat negative, whatever the credit; `-CATASTROPHE_
    THING` (2) for going into something it had seen and `-CATASTROPHE_PERSON` (3) for going into a person (a list that
    leaves out a cord is the list bullet's, below); and that `unknown` is never one of these. A distance that is the
    first of these is longer than the right one, when the right one is under the room the body needs (the body's length
    plus the margin) and the answer is not (`rules.CATASTROPHES`' threshold, the row's `threshold_m`). Told every kind
    and family, as for a caller that knows no job. Regression caught: a prompt that omits the harm term the reported
    score charges, one that states it in the wrong place, or one that still says any overstatement scores -2 after the
    scorer asks for a corridor shorter than the room the body needs."""
    sentence = credit_sentence(0.5, 0.1, 1.0, lists=lists, headings=headings)
    assert sentence.count("A worse error still is one on the harmful side") == 1
    assert "by more than the whole margin that would put the robot into harm. It scores a flat negative, whatever" in (
        sentence)
    assert ("Going into something the robot had seen scores -2. That is a distance in metres longer than the right "
            "one, when the right one is under the room the body needs and yours is not.") in sentence
    assert ("Going into a person scores -3. That is an offset that puts a person farther from you than they are, or a "
            "distance that says a person is clear who came closer.") in sentence
    assert "whatever credit it earned." in sentence and "- `unknown` is never one of these." in sentence
    harm, worse, widening = (sentence.index(s) for s in ("A place has no harmful side.", "A worse error still",
                                                         "A thing longer than"))
    assert harm < worse < widening


def test_the_catastrophe_weights_in_the_prompt_come_from_the_kernel(monkeypatch):
    """Contract: the two numbers the sentence prints are `credit_kernel.CATASTROPHE_THING` and `CATASTROPHE_PERSON`,
    read when it is built, not typed in the prompt: with the kernel's 2 and 3 it says -2 and -3 (above); were they 5 and
    7 it would say "scores -5, and" and "scores -7, whatever" and no longer -2 or -3 there. Regression caught: a weight
    typed apart from the kernel, so that the prompt and the scorer disagree."""
    monkeypatch.setattr(llm_credit, "CATASTROPHE_THING", 5)
    monkeypatch.setattr(llm_credit, "CATASTROPHE_PERSON", 7)
    sentence = credit_sentence(0.5, 0.1, 1.0)
    assert "had seen scores -5. That is" in sentence and "Going into a person scores -7. That is" in sentence
    assert "scores -2" not in sentence and "scores -3" not in sentence


TANGLE = ("A list that leaves out a thing your brush could catch within 30 cm of your front edge, with no place within "
          "a matching radius of it, scores -3 when it is asked during the job, whatever credit it earned.")
ZONE = ("A list that leaves out a person within 2.4 m of the point below you, with no place within a matching radius "
        "of it, scores -3 when it is asked during the job, whatever credit it earned.")


def test_the_list_catastrophe_is_told_only_for_the_list_families_a_job_asks_and_from_the_codes_constants(monkeypatch):
    """Contract (final review S7): a job that asks `tangle_ahead` during the job is told that a list leaving out a thing
    its brush could catch within `hazard_rule.FRONT_M` (0.3 m: "30 cm") of its front edge, that is, with no place within
    a matching radius of it, scores `-CATASTROPHE_PERSON` (3); a job that asks `drop_zone_people` the same of a person
    within its clearance C of the point below (here 2.4 m, the drone's C, from the argument and not a constant); both
    together, one sentence naming both; none otherwise, so the helper and the AMR (a list of `task_things` after the
    job) are told nothing of it. It follows the list paragraph and comes before the heading paragraph. The weight is
    read from the kernel when the sentence is built. `brief.list_hazards` finds the families from the job's during
    families (fake jobs; the vacuum asks `tangle_ahead` and no registered job `drop_zone_people`). A clause for a job
    with no list paragraph, or for a family it has no words for, is refused. Regression caught: an omission the
    scorer charges 3 that the prompt never states, a clause shown to a job that asks no such list, a distance typed
    apart from the code, or a weight typed apart from the kernel."""
    assert TANGLE in credit_sentence(0.5, 0.1, 1.0, lists=True, hazards=("tangle_ahead",))
    assert ZONE in credit_sentence(2.0, 2.4, 2.0, lists=True, hazards=("drop_zone_people",))
    both = credit_sentence(0.5, 2.4, 1.0, lists=True, headings=True, hazards=("drop_zone_people", "tangle_ahead"))
    assert both.count("A list that leaves out") == 1
    assert both.count("scores -3 when it is asked during the job, whatever credit it earned.") == 1
    assert "a person within 2.4 m of the point below you or a thing your brush could catch within 30 cm" in both
    empty = both.index("When there were none, the empty list is the one thing")
    assert empty < both.index("A list that leaves out") < (
        both.index("A heading is credited by its regret"))
    plain = credit_sentence(0.5, 0.1, 1.0, lists=True)
    assert "A list that leaves out" not in plain and "within a matching radius of it" not in plain
    assert "A list that leaves out" not in credit_sentence(0.5, 0.1, 1.0)
    monkeypatch.setattr(llm_credit, "CATASTROPHE_PERSON", 7)
    assert "of it, scores -7 when it is asked during the job, whatever credit it earned." in credit_sentence(
        0.5, 0.1, 1.0, lists=True, hazards=("tangle_ahead",))
    monkeypatch.setattr(llm_credit, "FRONT_M", 0.45)
    assert "within 45 cm of your front edge" in credit_sentence(0.5, 0.1, 1.0, lists=True, hazards=("tangle_ahead",))
    with pytest.raises(ValueError, match="needs `lists`"):
        credit_sentence(0.5, 0.1, 1.0, hazards=("tangle_ahead",))
    with pytest.raises(ValueError, match="no words for the list family 'task_things'"):
        credit_sentence(0.5, 0.1, 1.0, lists=True, hazards=("task_things",))
    assert list_hazards(fake_job(looks=("tangle_ahead", "ego_mem"))) == ("tangle_ahead",)
    assert list_hazards(fake_job(looks=("drop_zone_people", "tangle_ahead"))) == ("drop_zone_people", "tangle_ahead")
    assert list_hazards(fake_job(looks=("closest_approach",), ends=("task_things",))) == ()
    assert {job.id: list_hazards(job) for job in job_registry.all_jobs()} == {
        "vacuum/1": (), "vacuum/2": ("tangle_ahead",), "helper/1": (), "amr/1": (), "drone/1": ()}


def test_the_margin_says_a_tenth_of_the_right_distance_as_the_kernel_does_and_a_changed_constant_is_refused(
        monkeypatch):
    """Contract (FD-0.11 §3.1, §3.7): the margin bullet tells the agent that for an offset, a distance and a heading it
    is C or a tenth of the right distance, whichever is larger, and that tenth is `credit_kernel.KAPPA`: by hand, the
    helper's door 5 m away (C = 0.15, a tenth 0.50) answered 4 m has an error of 1.0 m, twice the margin, so the
    credit is the direction and the amount quarter, (1 + 1 + 0 + 0) / 4 = 0.5 and the score 0, which is what the section
    says an answer off by twice the margin earns; a kernel with another share refuses to build the sentence. Regression
    caught: a prompt that stays on C alone while the scorer widens the need with the distance, or a share changed in
    the kernel with the words left behind."""
    sentence = credit_sentence(0.5, 0.15, 1.0)
    assert "the margin is 15 cm or a tenth of the right distance (" in sentence and "whichever is larger." in sentence
    assert graded(1.0, 0.1365, 0.15, True, False, g=5.0) == pytest.approx(0.5)
    assert "An answer off by twice the margin scores 0, as good as `unknown`." in sentence
    monkeypatch.setattr(llm_credit, "KAPPA", 0.2)
    with pytest.raises(ValueError, match="KAPPA"):
        credit_sentence(0.5, 0.15, 1.0)


def test_the_unknown_sentence_is_told_to_every_job_and_the_way_to_bullets_only_to_a_job_that_asks_it():
    """Contract (FD-0.11 §3.8, §3.4.2): every job's credit section ends with the sentence that a question may name a
    thing the camera never showed clearly or one that is not in the place, and that `unknown` is then right (true by
    construction of every named-thing family); the way-to bullets (a heading credited by how much nearer to the thing
    R of driving brings the robot, with its own margin, and a heading without room scoring as a wrong answer) are in
    the section of a job that asks `way_to` and of a caller that knows no job, and in no other. By hand, the helper
    (C = 15 cm, R = 1 m): 'nearer to it 1 m of driving along it' and a margin of 20 cm, twice the field's 10 cm cell
    (the numbers are tied to the scorer in `test_llm_credit_route`). Regression caught: the sentence missing from a
    job, the way-to rule told to a job that never asks the question, or its numbers typed apart from the job's."""
    said = ("A question may name a thing your camera never showed clearly enough to tell what it is, or one that is "
            "not in this place at all; then `unknown` is right.")
    assert all(text_of(job_id).endswith(said) for job_id in JOB_FAMILIES)
    asks = credit_sentence(0.5, 0.15, 1.0, families=("way_to", "ego_mem"), kinds=("heading",))
    assert ("For the way to a thing, a heading is credited by how much nearer to it 1 m of driving along it brings "
            "you, against the best heading's, in the quarters above.\n- Its margin is 20 cm: ") in asks
    assert ("A heading with less room than the body needs, or one into floor you have not seen, scores as a wrong "
            "answer.") in asks
    assert "For the way to a thing" not in credit_sentence(0.5, 0.15, 1.0, families=("ego_mem", "open_heading"),
                                                          headings=True)
    assert "For the way to a thing" in credit_sentence(0.5, 0.15, 1.0)   # a caller that knows no job is told all
