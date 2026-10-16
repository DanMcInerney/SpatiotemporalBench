"""The brief (`brief.py`, plan §4.4): the texts a system is told about its job, composed at run time.
Hand-written texts for the vacuum, the variants on a fake job with a swap pair, the during sentence a job's own
`decisions` and `looks` give (none for a job that asks no during family), the first text of every job (the job and the
protocol, no coaching and no kinds), the refusals, and the scans that no brief names an object, a layout or the exam."""

from __future__ import annotations

import re
import string
from dataclasses import replace

import pytest

from stbench import job_registry, leak, mapspec
from stbench.brief import DOMAINS, DURING_PHRASES, NEUTRAL_ASSIGNMENT, VARIANTS, asks_during, asks_end, compose
from stbench.robots import ROBOTS, robot_doc
from stbench.stream_protocol import CHANNELS
from stbench.timeline_ego import TaskPlan
from stream_kit import fake_job
from test_architecture import SRC, string_constants

VACUUM = robot_doc(ROBOTS["vacuum"])
JOB = job_registry.get("vacuum/1")
CLEAN_KITCHEN = TaskPlan("clean_room", {"room": "kitchen"})
CLEAN_LIVING = TaskPlan("clean_room", {"room": "living room"})

# The design's paragraph, written out by hand once for each channel: only the
# sensor words differ. `vacuum/1` asks no question during the job, so its brief has no during sentence.
SENSORS = {"rgb_pose": "camera images and the robot's own pose and status",
           "detections": "depth, detections of the things in view, and the robot's own pose and status"}
PROTOCOL = ("Use only evidence available so far. Answer unknown when it is insufficient: a wrong answer costs more "
            "than unknown.")
END = "After the job you will answer structured questions about the environment and its history. " + PROTOCOL
PARAGRAPH = (
    "You are the onboard assistant of a robot vacuum working on one floor of a furnished home. Its current assignment "
    "is to clean the kitchen. You will receive timestamped {sensors} as it works.\n\n" + END)
RGB_POSE = PARAGRAPH.format(sensors=SENSORS["rgb_pose"])
DETECTIONS = PARAGRAPH.format(sensors=SENSORS["detections"])
OUT_OF_VIEW = "At some points during the job you will be asked what is out of view."
MANOEUVRE_CLEAR = "At some points during the job you will be asked whether a manoeuvre is clear."
MEM = "where a thing you saw before is from your camera"
NEAR = "how close something would come to the robot"   # `closest_approach`'s (0.10.0)
WAY = "which way to set off to reach a thing you saw before"   # `way_to`'s (0.11)
# the AMR's: `ego_now`, `ego_mem` and `closest_approach`, asked at look points alone
LOOKS = ("At some points during the job you will be asked where a thing you can see is from your camera, "
         f"{MEM}, and {NEAR}.")
# The during sentence of each job that asks during the job (0.8.0, review §4.4: the corridor adds "how far the robot
# could move", and `person_mem` reuses `ego_mem`'s phrase; 0.9.0: the helper and the vacuum add `open_heading`'s "which
# way the robot has the most room"; 0.10.0: the harm families add "how close something would come to the robot" for the
# AMR and the helper, "what stands near the point below the robot" for no job (the drone asked it until the final
# review's S2) and "what lies on the floor near the robot" for the vacuum), hand-written: each phrase once, in
# `DURING_PHRASES` order.
DURING = {
    "amr/1": LOOKS,
    "drone/1": f"At some points during the job you will be asked {MEM} and how far the robot could move.",
    "helper/1": ("At some points during the job you will be asked where a thing you can see is from your camera, "
                 f"{MEM}, {WAY}, how far the robot could move, which way the robot has the most room, and {NEAR}."),
    "vacuum/2": ("At some points during the job you will be asked "
                 f"{MEM}, how far the robot could move, which way the robot has the most room, and what lies on the "
                 "floor near the robot."),
}
ASKS_DURING = set(DURING)   # the jobs that ask a family during the job


def vacuum(plans, channel="rgb_pose", **kw):
    return compose(plans, job=JOB, robot=VACUUM, domain="home", channel=channel, **kw)


def test_the_vacuums_brief_by_hand():
    """Contract (design §7, plan §4.4): `compose` for the vacuum is the design's paragraph with the first assignment and
    the sensors of the channel in words (pictures and the robot's own readings, or depth, detections and the
    readings), with no during sentence (`vacuum/1` asks nothing during the job); each later task's text is its
    assignment alone. Hand-written texts, both channels. Regression caught: a brief that
    drifts from the design's words, or whose sensors are not the channel's."""
    assert vacuum([CLEAN_KITCHEN]) == [RGB_POSE]
    assert vacuum([CLEAN_KITCHEN], "detections") == [DETECTIONS]
    assert vacuum([CLEAN_KITCHEN, CLEAN_LIVING, CLEAN_KITCHEN]) == [
        RGB_POSE, "Its current assignment is to clean the living room.",
        "Its current assignment is to clean the kitchen."]


def test_variants_change_the_assignment_and_nothing_else():
    """Contract (plan §4.4): `correct` is each task's own assignment, `swapped` the assignment of the kind its job
    pairs with it (the task's own parameters filled in), `neutral` no assignment (the same sentence for every task);
    the rest of the first text, the during sentence included, is byte-identical in every variant. Hand-written texts on
    a fake job of two paired kinds that asks `out_of_view` during the job. Regression caught: a swapped brief that
    names the wrong kind or leaves a placeholder, a neutral brief that still names the task, or a variant that
    changes more than the assignment (the brief-swap check would then measure the wording)."""
    job = fake_job(decisions={"before_doorway": ("out_of_view",)})
    plans = [TaskPlan("tidy", {"room": "kitchen"}), TaskPlan("check", {"room": "hall"})]
    got = {variant: compose(plans, job=job, robot=VACUUM, domain="home", channel="rgb_pose", variant=variant)
           for variant in VARIANTS}
    tidy, check = "Its current assignment is to tidy the {}.", "Its current assignment is to check the {}."
    texts = {"correct": (tidy.format("kitchen"), check.format("hall")),
             "swapped": (check.format("kitchen"), tidy.format("hall")),
             "neutral": (NEUTRAL_ASSIGNMENT, NEUTRAL_ASSIGNMENT)}
    for variant, (first, second) in texts.items():
        assert got[variant][1] == second
        assert first in got[variant][0] and got[variant][0].replace(first, "{a}") == got["correct"][0].replace(
            texts["correct"][0], "{a}")
    assert NEUTRAL_ASSIGNMENT == "It is carrying on with its usual work."
    assert all(OUT_OF_VIEW in got[variant][0] for variant in VARIANTS)


def test_compose_refuses_what_it_cannot_write():
    """Contract (plan §4.4): a variant, domain or channel that is not one, no tasks, a task kind the job has no text
    for, a plan that lacks a field its text names, and `swapped` for a job that pairs the kind with none (the vacuum,
    until M3) are refused with a `ValueError` that says why. Regression caught: a brief written for a typo, or a
    swapped brief that quietly falls back to the correct one (the swap check would then measure nothing)."""
    job, plan = fake_job(), TaskPlan("tidy", {"room": "kitchen"})

    def write(plans=(plan,), job=job, **kw):
        return compose(list(plans), job=job, robot=VACUUM, **{"domain": "home", "channel": "rgb_pose", **kw})

    assert write()[0].count("Its current assignment is to tidy the kitchen.") == 1  # the control: this one writes
    for kwargs, why in [({"variant": "sarcastic"}, "variants are"), ({"domain": "garage"}, "domains are"),
                        ({"channel": "rgb"}, "unknown channel"), ({"plans": ()}, "at least one task"),
                        ({"plans": (TaskPlan("sweep", {"room": "x"}),)}, "no assignment text"),
                        ({"plans": (TaskPlan("tidy", {}),)}, "does not give")]:
        with pytest.raises(ValueError, match=why):
            write(**kwargs)
    with pytest.raises(ValueError, match="no swapped brief"):
        vacuum([CLEAN_KITCHEN], variant="swapped")
    unpaired = fake_job(task_kinds=("tidy", "check", "mop"),
                        assignments={**job.assignments, "mop": "It is to mop the {room}."})
    with pytest.raises(ValueError, match="no swapped brief"):
        write((TaskPlan("mop", {"room": "x"}),), unpaired, variant="swapped")


def first_text(job, channel="rgb_pose", **kw) -> str:
    """The first text a job's brief writes, for its first task kind with every field of its template a word."""
    plans = [TaskPlan(job.task_kinds[0], dict.fromkeys(template_fields(job), "x"))]
    return compose(plans, job=job, robot=robot_doc(ROBOTS[job.robot]), domain=job.domain, channel=channel, **kw)[0]


def asked_during(job) -> set[str]:
    return {family for families in job.decisions.values() for family in families}


def test_the_during_sentence_names_only_the_jobs_own_phrases():
    """Contract (FD-0.5.0-impl §2, FD-0.6.0-impl §4 A2): the first text names, in one sentence, the phrase of
    each during family the job asks, at a decision point (`decisions`) or a look point (`looks`), and no other, each
    phrase once, in `DURING_PHRASES`' order, in every variant. Hand-written on fake jobs: one family; four, two of which
    share a phrase ("whether a manoeuvre is clear", named once); the two look families alone and a look family beside
    a decision family; and the AMR, which asks the two look families at its look points and nothing at a decision
    (0.6.0). Regression caught: a brief that primes a question the job never asks (the old sentence named all five
    kinds for every job), or one that drops a kind it does ask (the first ask would prime beyond the brief)."""
    one = fake_job(decisions={"d": ("out_of_view",)})
    many = fake_job(decisions={"a": ("rear_clear", "path_ahead"), "b": ("zone_clear", "out_of_view")})
    amr = job_registry.get("amr/1")
    looking = fake_job(looks=("ego_mem", "ego_now"))
    mixed = fake_job(decisions={"d": ("out_of_view",)}, looks=("ego_now",))
    sentence = "At some points during the job you will be asked {}."
    seen = "where a thing you can see is from your camera"
    before = "where a thing you saw before is from your camera"
    around = "what is in view around the robot, what is out of view, and whether a manoeuvre is clear"
    for job, phrases in ((one, "what is out of view"), (many, around), (looking, f"{seen} and {before}"),
                         (mixed, f"what is out of view and {seen}"),
                         (amr, f"{seen}, {before}, and {NEAR}")):
        for variant in ("correct", "neutral"):
            text = first_text(job, variant=variant)
            assert sentence.format(phrases) in text and text.count("At some points") == 1, (job.id, variant)
    assert DURING_PHRASES["rear_clear"] == DURING_PHRASES["zone_clear"]   # the shared phrase `many` names once
    for job in (one, amr, looking):
        assert not any(DURING_PHRASES[f] in first_text(job) for f in ("where_was_i", "uncovered", "path_ahead"))
    assert seen not in first_text(one) and before not in first_text(mixed)


def test_a_job_with_no_during_family_has_no_during_sentence():
    """Contract (FD-0.5.0-impl §2; DECISIONS "0.5.0 declared": `where_was_i` is dropped): the sentence is
    in the first text of a registered job exactly when its `decisions` list a family or it has `looks`, and no phrase of
    any during family is in the text of a job that asks none. Hand-written for 0.6.0: amr/1, drone/1, helper/1 and
    vacuum/2 ask at look points and at no decision, so their sentence is the job's own in `DURING` (0.8.0) and
    `asks_during` (the Codex prompt's fact, `cmd_codex` reads it) is true for them; every other job asks none.
    Regression caught: a brief that promises
    questions at decision points that never come, a job whose look questions the brief never primes, or a sentence
    that outlives the drop of a family from the jobs."""
    jobs = job_registry.all_jobs()
    assert {j.id for j in jobs if j.looks} == ASKS_DURING and not [j.id for j in jobs if asked_during(j)]
    assert {j.id for j in jobs if asks_during(j)} == ASKS_DURING
    for job in jobs:
        text = first_text(job)
        assert ("At some points" in text) == bool(asked_during(job) or job.looks) == asks_during(job), job.id
        assert (DURING[job.id] in text) if job.id in ASKS_DURING else ("At some points" not in text), job.id
        if not asks_during(job):
            assert not any(phrase in text for phrase in DURING_PHRASES.values()), job.id


def test_only_a_job_with_an_end_family_is_told_of_questions_after_the_job():
    """Contract (final review, nit): the brief and the Codex prompt promise questions after the job only to a job that
    asks any (`JobSpec.ends`; None is the questionnaire's default, one family). Hand-written: of the registered jobs
    only `drone/1` declares no end family (it asks at look points alone), so it alone is not told of them, in the brief
    or in the prompt's `asks_end`; a job made to ask none is not told either. Regression caught: a prompt that promises
    the drone a question after the job that never comes (the agent waits for it, or answers the last look's as one)."""
    jobs = job_registry.all_jobs()
    assert {j.id for j in jobs if not asks_end(j)} == {"drone/1"}
    for job in jobs:
        for channel in SENSORS:
            assert ("After the job" in first_text(job, channel)) == (job.id != "drone/1"), (job.id, channel)
    assert not asks_end(replace(job_registry.get("vacuum/2"), ends=()))
    assert asks_end(replace(job_registry.get("vacuum/2"), ends=("object_where",)))


def test_a_during_family_without_words_is_refused():
    """Contract: a job that asks a during family `DURING_PHRASES` has no words for is refused, naming the family.
    Regression caught: a family added to a job without a phrase, whose first ask would prime the agent."""
    with pytest.raises(ValueError, match="asks zone_blocked"):
        first_text(fake_job(decisions={"d": ("out_of_view", "zone_blocked")}))


PLACES = {"home": "one floor of a furnished home", "warehouse": "an aisle-and-rack warehouse",
          "delivery": "a few streets of a town"}
NAMED = ("mug", "glass", "bowl", "plate", "book", "remote", "bag", "shoes", "laptop", "plant", "bottle", "box", "towel",
         "toy", "chair", "stool", "laundry basket", "pallet", "tote", "drum", "sack", "bin", "pallet jack", "roll cage",
         "car", "van", "pickup", "box truck", "skip", "dumpster", "portable toilet", "food cart", "red", "blue",
         "green", "yellow", "white", "black", "orange", "purple", "grey", "brown", "pink", "beige")


def names_a_kind_or_colour(text: str) -> list[str]:
    """The kinds and colours of the catalogs (`NAMED`, written by hand) that `text` holds as whole words."""
    return [word for word in NAMED if re.search(rf"\b{word}\b", text)]


def test_the_first_brief_is_the_job_and_the_protocol_and_no_help():
    """Contract (the agent is given no help with its task): the first text of every
    registered job on both channels is the robot, the place, the job's first assignment, the channel's sensors, the
    job's own during sentence (only the jobs that ask during the job have one) and the protocol facts (questions after
    the job, only for a job that asks any: the drone asks none; evidence so far, the unknown rule), and nothing else:
    no "Keep whatever ..." line and no word on what to keep, no list of kinds or colours. Hand-written around each
    job's own robot name and assignment, so the jobs of a domain differ only by their robot, assignment and during
    sentence. The scan for kinds and colours reads the text less the robot's name and the assignment (a tote robot's
    name holds a kind) and is run on planted text first.
    Regression caught: a brief that coaches what to keep, or names the kinds and colours a question may use (naming
    what it sees is perception, and the questions' own words are what the agent gets)."""
    planted = "Movable things are named by kind (mug, drinking glass, pair of shoes) and by colour (red, blue)."
    assert names_a_kind_or_colour(planted) == ["mug", "glass", "shoes", "red", "blue"]
    for sentence in (OUT_OF_VIEW, MANOEUVRE_CLEAR, *DURING.values()):
        assert "keep" not in sentence and names_a_kind_or_colour(sentence) == []
    during = {job_id: sentence + " " for job_id, sentence in DURING.items()}
    jobs = job_registry.all_jobs()
    assert {job.domain for job in jobs} == set(PLACES)
    for job in jobs:
        name = robot_doc(ROBOTS[job.robot])["name"]
        for channel, sensors in SENSORS.items():
            text = first_text(job, channel)
            assignment = job.assignments[job.task_kinds[0]].format(**dict.fromkeys(template_fields(job), "x"))
            head = (f"You are the onboard assistant of a {name} working on {PLACES[job.domain]}. {assignment} You will "
                    f"receive timestamped {sensors} as it works.\n\n")
            tail = PROTOCOL if job.id == "drone/1" else END   # the one job with no end family
            assert text == head + during.get(job.id, "") + tail, (job.id, channel)
            assert "Keep whatever" not in text and "keep" not in text.lower(), (job.id, channel)
            assert names_a_kind_or_colour(text.replace(assignment, "").replace(name, "")) == [], (job.id, channel)


def leaks_in(text: str, names: list[str]) -> list[str]:
    """Every `leak.RULES` hit in `text`, and every map name in it (any case)."""
    found = [rule for rule, rx in leak.RULES.items() if rx.search(text)]
    return found + [f"map name {name}" for name in names if name.lower() in text.lower()]


def template_fields(job) -> set[str]:
    return {field for text in job.assignments.values() for _, field, _, _ in string.Formatter().parse(text) if field}


def test_every_brief_is_clean():
    """Contract (VISION rule 6, design §7): no text a registered job can produce, for any domain, channel and variant
    the job allows, with each region kind and name of every bundled map as every field of its templates, and no event
    wording of any job, has a `leak.RULES` hit (an entity id, a region id, the word "truth") or names a bundled map. A
    job whose texts take no field is scanned as its fixed text, and its swapped brief only for the kinds it pairs; the
    scan stays non-vacuous through the vacuum's templates. Regression caught: a brief or an event naming an object, a
    layout or the exam, hidden by placeholder parameters. The scan is run on planted text first, so it is not passing
    vacuously."""
    maps = list(mapspec.bundled_maps())
    assert len(maps) >= 3 and leaks_in("the apartment_a robot is near E0042 in R3, the truth", maps) == [
        "entity_id", "region_id", "truth", "map name apartment_a"]
    words = sorted({word for name in maps for r in mapspec.load_map(name).regions for word in (r.kind, r.name)})
    jobs = job_registry.all_jobs()
    assert jobs and any(template_fields(job) for job in jobs)
    scanned = 0
    for job in jobs:
        assert all(leaks_in(wording, maps) == [] for wording in job.events)
        variants = VARIANTS if job.swaps else ("correct", "neutral")
        for word in words:
            plans = [TaskPlan(kind, dict.fromkeys(template_fields(job), word)) for kind in job.task_kinds]
            for domain, channel, variant in ((d, c, v) for d in DOMAINS for c in CHANNELS for v in variants):
                window = [p for p in plans if variant != "swapped" or any(p.kind in pair for pair in job.swaps)]
                for text in compose(window, job=job, robot=robot_doc(ROBOTS[job.robot]), domain=domain, channel=channel,
                                    variant=variant):
                    assert leaks_in(text, maps) == [], (job.id, word, domain, channel, variant)
                    scanned += 1
    assert scanned >= len(words) * len(DOMAINS) * len(CHANNELS) * 2


def test_the_harness_names_no_robot_or_task_kind():
    """Contract (AGENTS "Never": no special cases): `brief.py` and `stream.py` hold no string constant (docstrings
    aside) that names a robot, a registered task kind or the dock as a whole word: a job's texts and kinds are the
    job's. Regression caught: a brief or a stream that special-cases the vacuum, so adding a robot would edit
    the brief or the harness. The scan is run on a planted constant first."""
    words = {kind for job in job_registry.all_jobs() for kind in job.task_kinds} | {"dock", *ROBOTS}
    pattern = re.compile(r"(?<![A-Za-z_])(" + "|".join(sorted(words)) + r")(?![A-Za-z_])")
    assert [c for c in string_constants("X = 'then clean_room at the dock'") if pattern.search(c)]
    for module in ("brief", "stream"):
        text = (SRC / f"{module}.py").read_text(encoding="utf-8")
        assert [c for c in string_constants(text) if pattern.search(c)] == [], module
