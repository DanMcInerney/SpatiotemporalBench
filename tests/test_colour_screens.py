"""A question names a thing only by a colour its render shows (0.5.0-control, review defect 1; `World.rendered`,
`may_be_taken_for`, `colour_holds`, and `questions_where.never_seen`, `lookalike_sighted`, `nameable`, `askable`).

The kit's pictures are faithful (`synth.RENDER_RGB`: a thing renders in its own colour word) unless a sighting says
otherwise (`Synth.see(..., rgb=)`) or the script is an old root's (`rgb_med = False`: no render, the label stands,
a neutral label for all three neutrals). Black, grey and white are one group when two things are compared. Hand-built
worlds; every expected set is written out by hand."""

from __future__ import annotations

from dataclasses import replace

import family_kit as fk
from space_kit import BAG_SIZE, JOB, RELEVANT_KINDS, vacuum_run
from stbench.job_registry import Relevance
from stbench.questions_select import Moment
from stbench.questions_things import distance_between
from stbench.questions_where import never_seen, pools
from synth import RENDER_RGB
from things_kit import RECOG_PX, flat, put, world

WHITE, BLACK = RENDER_RGB["white"], RENDER_RGB["black"]
RED_ORANGE_EDGE = (210, 60, 30)  # Lab hue 41 degrees: a red thing and an orange one at once
ORANGE_YELLOW_EDGE = (255, 165, 0)  # CSS orange, Lab hue 73 degrees: the orange / yellow edge is at 72
LAPTOP_JOB = replace(JOB, relevance=lambda plan: Relevance(
    RELEVANT_KINDS | {"laptop"}, frozenset({"kitchen", "living room"}), frozenset({"cleaning", "transit", "docking"})))


def sight(s, eid, kind, colour, x, ks=(1,), *, rgb=None, mobility="carried", lum=None, px=1500, y=1.0):
    """A thing labelled `colour` standing at (x, y), sighted (recognisably, unless `lum` or `px` say not) at the
    keyframes `ks`, in the sRGB median `rgb` (the faithful one of its label when None)."""
    s.item(eid, kind, {"color": colour}, [(0, None, (x, y, 0.0))], mobility=mobility,
           size=BAG_SIZE if kind == "bag" else (0.1, 0.1, 0.1))
    s.see(list(ks), eid, px, lum=lum, rgb=rgb)


def colours(keys, kind) -> set[str]:
    """The colours of the decoy descriptions of `kind`."""
    return {dict(attrs)["color"] for k, attrs in keys if k == kind}


def entities(cands) -> set[str]:
    """The ids of the world's things the candidates name."""
    return {x[2:] for c in cands for x in c.subjects if x.startswith("e:")}


def asked_pools(w) -> Moment:
    """`object_where`'s two pools for the world's window, as a `Moment` (answerable, unknown)."""
    return Moment("window", *pools(w))


def test_a_decoy_is_dropped_by_a_sighted_look_alike_that_renders_its_colour_and_the_neutrals_are_one_group():
    """Contract: an absent description is offered only when no sighted thing of its kind (or a kind it can be mistaken
    for) could be taken for its colour by the picture, not the label: it renders that word or sits next to a boundary of
    it; and black, grey and white are one group, so a thing that may be taken for any neutral hides every neutral decoy.
    Hand-worked on the home catalog's toys (red, blue, yellow, green, orange, pink) with one sighted toy labelled
    yellow: in yellow the decoys are red, blue, green, orange, pink (yellow is the world's); rendering green, green goes
    and yellow is still no decoy; rendering CSS orange, on the orange / yellow edge, orange goes. On the shoes (black,
    white, red, blue, brown): a sighted red pair leaves black and white offered, red rendering grey takes both and
    leaves blue and brown, and a black pair, rendering black or white (silver), takes white and leaves red, blue, brown
    (the neutrals are one group, whichever it renders). Regression caught: the 0.5.0 log review's 'Where was the white
    laptop?' asked with a laptop in plain view that a viewer could not tell from a white one, a decoy of a colour a
    thing renders in left because the label differs, a boundary render counted as one colour only, or a chromatic
    thing hiding the neutrals."""
    def toy_decoys(**sighting):
        s = vacuum_run()
        sight(s, "E0001", "toy", "yellow", 1.0, **sighting)
        return colours(never_seen(fk.world(s)), "toy")

    def shoe_decoys(label, rgb=None):
        s = vacuum_run()
        sight(s, "E0001", "shoes", label, 1.0, rgb=rgb)
        return colours(never_seen(fk.world(s)), "shoes")

    assert toy_decoys() == {"red", "blue", "green", "orange", "pink"}
    assert toy_decoys(rgb=RENDER_RGB["green"]) == {"red", "blue", "orange", "pink"}
    assert toy_decoys(rgb=ORANGE_YELLOW_EDGE) == {"red", "blue", "green", "pink"}
    assert shoe_decoys("red") == {"black", "white", "blue", "brown"}
    assert shoe_decoys("red", rgb=RENDER_RGB["grey"]) == {"blue", "brown"}
    assert shoe_decoys("black") == shoe_decoys("black", rgb=WHITE) == {"red", "blue", "brown"}


def test_the_review_laptop_a_sighted_neutral_thing_hides_an_unknown_question_about_a_neutral_one():
    """Contract: an `unknown` question about a real unseen thing of a neutral colour is not asked while any sighted
    thing of its kind could be taken for a neutral (black, grey and white are one group; `World.may_be_taken_for`),
    whatever it is labelled, and is asked while the sighted thing could only be taken for a colour. Hand-worked on the
    log review's case (helper case bc5f): a white laptop that no frame shows is not asked
    beside a sighted laptop labelled grey rendering (94, 93, 97) (L 40: grey, ten lightness units under the white edge),
    nor beside one labelled black rendering black, nor beside a red-labelled one that renders white; it is asked beside
    a red laptop, and beside one labelled red that renders orange. Regression caught: 'Where was the white laptop?'
    asked as `unknown` with the grey laptop in plain view (the review's defect 1, left by a render screen that kept grey
    and white apart)."""
    def unknown_things(label, rgb=None):
        s = vacuum_run()
        sight(s, "E0001", "laptop", label, 1.0, rgb=rgb)
        s.item("E0002", "laptop", {"color": "white"}, [(0, None, (2.0, 1.0, 0.0))])
        return entities(asked_pools(fk.world(s, job=LAPTOP_JOB)).unknown)
    assert unknown_things("grey", rgb=(94, 93, 97)) == set()
    assert unknown_things("black") == set()
    assert unknown_things("red", rgb=WHITE) == set()
    assert unknown_things("red") == {"E0002"}
    assert unknown_things("red", rgb=RENDER_RGB["orange"]) == {"E0002"}


def test_a_neutral_thing_beside_another_neutral_look_alike_is_not_asked_and_a_clean_chromatic_pair_is():
    """Contract: `nameable` hides an answerable thing of a neutral colour (its picture shows its word) while another
    sighted thing of a confusable kind may be taken for any neutral, and keeps one of a chromatic colour beside a
    look-alike that renders cleanly in another. Hand-worked on the window: a black bag beside a grey bag
    (each rendering its own word) is not asked, nor is either beside a white box (a bag's look-alike); a black bag
    beside a red bag is asked, and so is the red bag; a red bag beside an orange box, each rendering cleanly in its own
    colour (red (200, 30, 30), orange (240, 130, 20)), are both asked; a red bag beside a box labelled red whose render
    sits on the red / orange edge asks neither (the box shows no word of its own, and it may be taken for red, which
    hides the bag). Regression caught: a black bag asked beside a grey-rendering one (the review's bag labelled black,
    lighter than the grey one), or the neutral group spread to colours that render apart."""
    def asked(*things):
        s = vacuum_run()
        for i, (kind, colour, rgb) in enumerate(things, 1):
            sight(s, f"E000{i}", kind, colour, float(i), rgb=rgb)
        return entities(asked_pools(fk.world(s)).answerable)
    assert asked(("bag", "black", None), ("bag", "grey", None)) == set()
    assert asked(("bag", "black", None), ("bag", "grey", None), ("box", "white", None)) == set()
    assert asked(("bag", "black", None), ("bag", "red", None)) == {"E0001", "E0002"}
    assert asked(("bag", "red", None), ("box", "orange", None)) == {"E0001", "E0002"}
    assert asked(("bag", "red", None), ("box", "red", RED_ORANGE_EDGE)) == set()


def test_a_thing_is_asked_only_if_its_picture_shows_its_colour_word():
    """Contract: `askable` (so `object_where`'s answerable and unknown pools, and `object_was`, `distance_between`)
    leaves out a thing whose recognisable frames do not show its label: the confident colour that the most of them show
    it in must be its label, and a render on a boundary is no vote. Frames that are not recognisable (here a dark one)
    do not vote. Hand-worked in the window, on frames 1, 3 and 5: a red bag in red is asked;
    a 'black' bag that renders white is not; a green bag in green, green, white is asked (two of three); an orange bag
    in orange, white, white is not (white has the most); red shoes rendering on the red / orange edge in all three are
    not (no confident frame); blue shoes in blue plus a dark frame (luminance 0.005) in white are asked (the dark frame
    is no sighting a person could name); a pink toy in pink, then on the red / orange edge twice, is asked (the
    boundary frames cannot veto the one confident colour). The same world as an old root's log (no `rgb_med`, no render)
    asks all seven. Regression caught: the review's remote labelled black rendering silver asked as 'black', a single
    frame deciding a thing's colour against the rest, a boundary frame counted against a thing, or a colour read from a
    frame the recognisability floor refuses."""
    s = vacuum_run()
    sight(s, "E0001", "bag", "red", 1.0, ks=(1, 3, 5))
    sight(s, "E0002", "bag", "black", 2.0, ks=(1, 3, 5), rgb=WHITE)
    sight(s, "E0003", "bag", "green", 3.0, ks=(1, 3))
    s.see(5, "E0003", 1500, rgb=WHITE)
    sight(s, "E0004", "bag", "orange", 4.0, ks=(1,))
    s.see([3, 5], "E0004", 1500, rgb=WHITE)
    sight(s, "E0005", "shoes", "red", 1.0, ks=(1, 3, 5), rgb=RED_ORANGE_EDGE, y=2.0)
    sight(s, "E0006", "shoes", "blue", 2.0, ks=(1,), y=2.0)
    s.see(3, "E0006", 1500, lum=0.005, rgb=WHITE)
    sight(s, "E0007", "toy", "pink", 3.0, ks=(1,), y=2.0)
    s.see([3, 5], "E0007", 1500, rgb=RED_ORANGE_EDGE)
    assert entities(asked_pools(fk.world(s)).answerable) == {"E0001", "E0003", "E0006", "E0007"}
    s.rgb_med = False
    assert entities(asked_pools(fk.world(s)).answerable) == {f"E000{i}" for i in range(1, 8)}


def test_on_an_old_roots_log_the_three_neutral_labels_are_one_group():
    """Contract: with no `rgb_med` (an old log, or a thing no frame shows recognisably) the label stands for the render,
    and a neutral label stands for all three neutrals (black, grey and white are one group, black and white included),
    so the screens never trust a neutral label. Hand-worked on the home catalog's bag (black, red, blue, green, brown,
    grey, orange) in an old-log world: a grey bag sighted hides the decoy 'black bag', and the other decoys are the
    catalog's colours (red, blue, green, brown, orange); in the window a grey bag, a black bag and a white box (a bag's
    look-alike) hide one another's questions and a red bag and black shoes are asked; a black bag beside a white box
    alone are not asked either (black and white are strangers if only the next neutral over counts). On a log with
    `rgb_med`, a grey bag that no frame shows recognisably (a 100 px glimpse) keeps its label's group: it still hides
    the black decoy. Regression caught: the review's 22% of helper questions with a neutral twin in view, asked on an
    old root; black and white made strangers; or a thing with nothing measured taken for its own colour alone."""
    old = vacuum_run()
    old.rgb_med = False
    sight(old, "E0001", "bag", "grey", 1.0)
    assert colours(never_seen(fk.world(old)), "bag") == {"red", "blue", "green", "brown", "orange"}
    sight(old, "E0002", "bag", "black", 2.0)
    sight(old, "E0003", "box", "white", 3.0)
    sight(old, "E0004", "bag", "red", 4.0)
    sight(old, "E0005", "shoes", "black", 1.0, y=2.0)
    assert entities(asked_pools(fk.world(old)).answerable) == {"E0004", "E0005"}
    pair = vacuum_run()
    pair.rgb_med = False
    sight(pair, "E0001", "bag", "black", 1.0)
    sight(pair, "E0002", "box", "white", 2.0)
    assert entities(asked_pools(fk.world(pair)).answerable) == set()
    glimpse = vacuum_run()
    sight(glimpse, "E0001", "bag", "grey", 1.0, px=100)
    assert "black" not in colours(never_seen(fk.world(glimpse)), "bag")


def test_a_clutter_look_alike_and_a_toys_blocks_hide_a_thing_of_their_colour():
    """Contract (`descriptions.CONFUSABLE`, `World.sighted_looks`): a sighted thing of a kind that can be mistaken for
    the asked one hides it when it renders in its colour, and that includes the clutter kinds the review found (a
    clothes pile for a toy) and the toy's stacked blocks for a box; clutter of a kind in no group (a toaster) hides
    nothing. Hand-worked on the window: a pink toy beside a sighted pink clothes pile is not asked, but
    is when the pile is never sighted; a grey bag beside a sighted grey toaster (clutter, kind in no group) is asked; a
    green toy and a green box, both sighted, hide one another, while a blue toy and a red box beside them are asked.
    Regression caught: 'a clothes pile for a pink toy' and 'green toy blocks for a green box' (the log review), or
    every clutter thing made a look-alike of everything."""
    seen, unseen, toaster, blocks = vacuum_run(), vacuum_run(), vacuum_run(), vacuum_run()
    sight(seen, "E0001", "toy", "pink", 1.0)
    sight(seen, "E0100", "clothes_pile", "pink", 2.0, mobility="clutter")
    sight(unseen, "E0001", "toy", "pink", 1.0)
    unseen.item("E0100", "clothes_pile", {"color": "pink"}, [(0, None, (2.0, 1.0, 0.0))], mobility="clutter")
    sight(toaster, "E0001", "bag", "grey", 1.0)
    sight(toaster, "E0100", "toaster", "grey", 2.0, mobility="clutter")
    sight(blocks, "E0001", "toy", "green", 1.0)
    sight(blocks, "E0002", "box", "green", 2.0)
    sight(blocks, "E0003", "toy", "blue", 3.0)
    sight(blocks, "E0004", "box", "red", 4.0)
    asked = {name: entities(asked_pools(fk.world(s)).answerable)
             for name, s in (("seen", seen), ("unseen", unseen), ("toaster", toaster), ("blocks", blocks))}
    assert asked == {"seen": set(), "unseen": {"E0001"}, "toaster": {"E0001"}, "blocks": {"E0003", "E0004"}}


def test_distance_between_names_no_thing_whose_picture_does_not_show_its_word():
    """Contract: `distance_between` screens both things of a row with the same rule as `object_where`
    (`questions_where.nameable`): a thing whose recognisable frames do not show its colour word is in no row, a neutral
    thing beside another neutral look-alike is in none, and the others are still paired. Hand-worked on one task, things
    3 m apart: a red bag (1, 1), a blue box (4, 1) and a 'black' bag at (1, 3) that renders white are sighted
    recognisably: the only answerable row is the red bag with the blue box; with the black bag in its own colour it is
    paired with both as well; with a grey bag beside it (each in its own colour, the neutrals one group) neither
    neutral bag is in a row. Regression caught: a pair naming a bag the picture shows white (the question 'how far is
    the black bag from the box?' has no referent), a pair of two neutral look-alikes, or the screen applied to one
    family and forgotten in the family that shares its things."""
    def pairs(rgb, grey=False):
        s = flat()
        put(s, "E0001", "bag", "red", (1.0, 1.0), recognised=[10])
        put(s, "E0002", "box", "blue", (4.0, 1.0), recognised=[12])
        put(s, "E0003", "bag", "black", (1.0, 3.0), recognised=[10])
        s.see([10], "E0003", RECOG_PX, rgb=rgb)
        if grey:
            put(s, "E0004", "bag", "grey", (4.0, 3.0), recognised=[14])
        found = distance_between(world(s))
        return {frozenset(c.entities) for m in found for c in m.answerable}
    assert pairs(WHITE) == {frozenset({"E0001", "E0002"})}
    assert pairs(None) == {frozenset({"E0001", "E0002"}), frozenset({"E0001", "E0003"}), frozenset({"E0002", "E0003"})}
    assert pairs(None, grey=True) == {frozenset({"E0001", "E0002"})}
