"""The decoys (`questions_where.never_seen`, `questionnaire`): the absent descriptions an `unknown` gold may name, and
how many of them a case asks. A decoy that the world gives away (its kind never in the frames, its colour on a
lookalike) lets an agent answer `unknown` for the wrong reason (0.5.0), so it is offered only
when it gives nothing away, and one a case at most; the world's own unseen things come before it."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from family_kit import TASK_FRAMES, multi_run
from space_kit import record, vacuum_run
from stbench import questionnaire
from stbench.descriptions import desc_key
from stbench.questions_where import never_seen
from stbench.questions_world import World
from stbench.seeding import stable_hex
from test_questionnaire import end_job

HOME_COLOURS = {  # the home catalog's colours of the four relevance kinds of the kit's job
    "bag": {"black", "red", "blue", "green", "brown", "grey", "orange"},
    "box": {"brown", "white", "blue", "red", "green"},
    "shoes": {"black", "white", "red", "blue", "brown"},
    "toy": {"red", "blue", "yellow", "green", "orange", "pink"}}


def colours(keys) -> dict[str, set[str]]:
    """The colours of the descriptions, by kind."""
    out: dict[str, set[str]] = {}
    for kind, attrs in keys:
        out.setdefault(kind, set()).add(dict(attrs)["color"])
    return out


def sight(s, eid, kind, colour, x, *, seen=True, **attrs):
    """A thing standing at (x, 1) that the camera shows (1,500 px) at keyframe 1, or never."""
    s.item(eid, kind, {"color": colour, **attrs}, [(0, None, (x, 1.0, 0.0))])
    if seen:
        s.see(1, eid, 1500)


def test_a_decoy_gives_nothing_away():
    """Contract: an absent description is offered only when a thing of its kind is sighted in the window and no sighted
    thing of its own kind or of a kind it can be mistaken for (`descriptions.CONFUSABLE`) has its colour; the colours
    of an unsighted thing say nothing. Hand-worked on a task with a red bag, a large brown bag, a blue box, a red tote
    (a lookalike of a box), a green laundry basket (of a bag) and black shoes that no frame shows, over the home
    catalog's colours: shoes and toys have no sighted thing of their kind, so no decoy; the bag's are the catalog's
    colours but red (in the world and on the tote), brown (the large brown bag shares it), green (the laundry basket)
    and blue (the box, a bag's look-alike since 0.5.0's fix round), so black, grey and orange; the box's are the
    catalog's but blue (in the world), red (the tote and the red bag) and brown (the large brown bag), so white and
    green. With the tote and the basket gone, the bag keeps black, green, grey and orange (red, brown and the box's
    blue are taken) and the box white and green (blue, and the bags' red and brown, are taken). Regression caught: a
    red-box decoy beside a red tote (the agent sees a thing that looks like it and answers where it is), a blue-bag
    decoy beside a blue box, a decoy of a kind that was never in the frames (`unknown` for the kind's absence), or the
    colours of a thing no frame showed counted."""
    s = vacuum_run()
    sight(s, "E0001", "bag", "red", 1.0)
    sight(s, "E0002", "bag", "brown", 1.5, size="large")
    sight(s, "E0003", "box", "blue", 2.0)
    sight(s, "E0004", "tote", "red", 2.5)
    sight(s, "E0005", "laundry_basket", "green", 3.0)
    sight(s, "E0006", "shoes", "black", 3.5, seen=False)
    assert colours(never_seen(fk.world(s))) == {"bag": {"black", "grey", "orange"}, "box": {"white", "green"}}
    plain = vacuum_run()
    sight(plain, "E0001", "bag", "red", 1.0)
    sight(plain, "E0002", "bag", "brown", 1.5, size="large")
    sight(plain, "E0003", "box", "blue", 2.0)
    assert colours(never_seen(fk.world(plain))) == {"bag": HOME_COLOURS["bag"] - {"red", "brown", "blue"},
                                                    "box": HOME_COLOURS["box"] - {"blue", "red", "brown"}}


def test_a_decoy_is_a_description_the_world_lacks():
    """Contract: an admissible decoy still matches nothing in the world (a description in the world is no decoy
    whatever its colour is shared with). Hand-worked: with a red bag sighted, the description red bag is never offered;
    nor is it needed, as the bag's other colours are. Regression caught: a real thing offered as an `unknown` gold."""
    s = vacuum_run()
    sight(s, "E0001", "bag", "red", 1.0)
    w = fk.world(s)
    assert desc_key("bag", {"color": "red"}) in w.groups and desc_key("bag", {"color": "red"}) not in never_seen(w)


def medium_world(unseen: bool):
    """Three tasks of the kit's vacuum, each with four static things recognisable at its first steps: task t has the
    bag, box, shoes and toy of the t-th colour of its kind's palette below (so every kind is sighted and the decoys that
    remain are the catalog's other colours), and with `unseen` a pink toy no frame shows: the world's one unseen thing.
    Twelve moments, twelve answerable things."""
    s = multi_run(3)
    table = {"bag": ("black", "red", "blue"), "box": ("brown", "white", "blue"), "shoes": ("black", "white", "red"),
             "toy": ("red", "blue", "yellow")}
    for t in range(3):
        for j, (kind, palette) in enumerate(table.items()):
            sight(s, f"{kind[0].upper()}{t}", kind, palette[t], 0.5 + 0.8 * j, seen=False)
            s.see(TASK_FRAMES * t + 1, f"{kind[0].upper()}{t}", 1500)
    if unseen:
        sight(s, "U0", "toy", "pink", 4.0, seen=False)
    return s


def worlds(s, tag: str, n: int):
    """The medium window of the script under `n` case ids that differ only in their hash."""
    inp, rec = fk.inputs(s), record(s, a=0, b=2, length="medium")
    return [World(inp, replace(rec, case=stable_hex(tag, i, n=16))) for i in range(n)]


def test_a_case_asks_at_most_one_absent_description():
    """Contract: a case asks at most one question that names an absent description (a `d:` subject), however many
    moments draw the unknown slice, and in a world with no unseen thing of its own the unknown slice is that one
    question at most, so the unknown share can not be earned by counting. Hand-worked on three tasks with twelve
    answerable things, every kind sighted and no unseen thing (an all-decoy world), over 60 case ids: no case asks two
    `d:` subjects or two `unknown` questions, some case asks one and some case asks more than one question (a case
    stops at the first moment that draws the unknown slice, whose one decoy leaves the later moments without an
    unknown candidate). Regression caught: a case of several decoys, which an agent that answers `unknown` for whatever
    it cannot find scores on."""
    asked_one = longest = 0
    for w in worlds(medium_world(unseen=False), "one absent", 60):
        taken = questionnaire.end(replace_job(w))
        absent = [x for c in taken for x in c.subjects if x.startswith("d:")]
        unknown = [c for c in taken if c.slice == "unknown"]
        assert len(absent) <= 1 and len(unknown) == len(absent) and taken
        asked_one += len(unknown)
        longest = max(longest, len(taken))
    assert asked_one > 0 and longest > 1


def replace_job(w: World) -> World:
    """The world asked `object_where` alone."""
    return World(replace(w.inp, job=end_job("object_where")), w.rec)


def test_real_unseen_things_come_before_absent_descriptions():
    """Contract: a thing of the world that no frame shows is offered before an absent description, so a case asks a
    decoy only beside the unseen thing and never in its place: the unknown questions of a case are the unseen thing,
    then at most one decoy. Hand-worked on the same three tasks with a pink toy no frame shows, over 60 case ids: no
    case asks a decoy without asking the pink toy, none asks the pink toy twice (one question to a thing), at most two
    unknown questions are asked and some case asks both. Regression caught: the hash order picking a decoy while the
    world's own unseen thing was free, which spends the one decoy a case may ask on the first moment that draws the
    unknown slice."""
    both = 0
    for w in worlds(medium_world(unseen=True), "real first", 60):
        taken = questionnaire.end(replace_job(w))
        unknown = [c for c in taken if c.slice == "unknown"]
        subjects = [x for c in unknown for x in c.subjects if x.startswith(("e:", "d:"))]
        assert subjects.count("e:U0") <= 1 and len(subjects) <= 2 and len(unknown) == len(subjects)
        assert "e:U0" in subjects or not subjects
        both += len(subjects) == 2
    assert both > 0
