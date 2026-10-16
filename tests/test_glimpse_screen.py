"""The S4 glimpse screen (0.11.0, FD-0.11-design 3.8 and 4.5; `questions_where.glimpse_hides`, used by `object_where`'s
and `ego_mem`'s `unknown` pools): an `unknown` candidate is dropped when a look-alike kind was glimpsed (sighted, never
recognisably) in a frame whose render shows the asked colour. The kit's pictures are faithful (`synth.RENDER_RGB`)
unless a sighting says otherwise; a glimpse is 100 px, under the 1,024 px recognisability floor. Hand-built worlds."""

from __future__ import annotations

import family_kit as fk
from ego_kit import K, put, scene
from ego_kit import world as ego_world
from space_kit import BAG_SIZE, vacuum_run
from stbench.questions_ego import ego_mem
from stbench.questions_where import glimpse_hides, pools
from synth import RENDER_RGB

BLUE = RENDER_RGB["blue"]
GLIMPSE_PX = 100


def glimpsed_box(s, eid="G", label="red", rgb=None, ks=(1,), px=GLIMPSE_PX):
    """A box labelled `label` standing at (3, 1), sighted in `ks` with `px` pixels, rendered in `rgb` (its label's
    faithful colour when None)."""
    s.item(eid, "box", {"color": label}, [(0, None, (3.0, 1.0, 0.0))], size=BAG_SIZE)
    s.see(list(ks), eid, px, rgb=rgb)


def test_a_glimpsed_look_alike_that_renders_the_asked_colour_hides_it_and_nothing_else_does():
    """Contract (S4): `glimpse_hides(w, kind, attributes, entity, upto)` is true exactly when a thing of a kind the name
    can be mistaken for (a box for a bag) other than `entity` was sighted at or before `upto` and in none of those
    frames recognisably, in a frame whose `rgb_med` may be taken for the asked colour (black, grey and white are one
    group). By hand, asking for a blue bag in a window with a box labelled red: glimpsed (100 px) and rendered blue,
    hidden; rendered red, not (a different colour); sighted at 1,500 px and rendered blue, not (it is recognisable,
    which the older screens read, not this one); glimpsed in frame 20 and rendered blue, hidden for the whole window
    but not as of frame 13 (nothing after the ask counts); glimpsed and rendered white, a black bag hidden and a blue
    one not (the neutral group); the box's own entity, a toy (not a look-alike of a bag) and a name with no colour,
    not hidden. Regression caught: a glimpse read by its label only (the older screens' reading, which misses a skip
    the catalog calls grey that a glimpse shows blue), a recognisable look-alike dropped twice, a frame after the ask
    deciding a question, or a chromatic colour hidden by a neutral glimpse."""
    def hides(asked="blue", kind="bag", entity=None, upto=None, **box):
        s = vacuum_run()
        glimpsed_box(s, **box)
        return glimpse_hides(fk.world(s), kind, {"color": asked}, entity, upto)

    assert hides(rgb=BLUE)
    assert not hides()                                       # faithful: it renders red, and a red box is no blue bag
    assert not hides(rgb=BLUE, px=1500)                      # recognisable: the other screens' business
    assert hides(rgb=BLUE, ks=(20,))
    assert not hides(rgb=BLUE, ks=(20,), upto=13)
    assert hides(rgb=BLUE, ks=(1, 20), upto=20) and not hides(rgb=BLUE, ks=(1, 20), upto=0)
    assert hides(asked="black", rgb=RENDER_RGB["white"]) and not hides(rgb=RENDER_RGB["white"])
    assert not hides(rgb=BLUE, entity="G")
    assert not hides(rgb=BLUE, kind="mug")
    s = vacuum_run()
    glimpsed_box(s, rgb=BLUE)
    assert not glimpse_hides(fk.world(s), "bag", {}, None)


def test_object_where_drops_an_unknown_thing_whose_colour_a_glimpse_showed_and_keeps_the_answerable_one():
    """Contract (S4 in `questions_where.pools`): the `unknown` pool leaves out a real unseen thing (here a blue bag with
    no pixel) while a look-alike of it was glimpsed in the colour (a box labelled red whose 100 px render blue), and
    keeps it when the glimpse renders red; the answerable pool is not screened (a green bag in plain view stays in it
    either way). Regression caught: `object_where` asking 'where did you last see the blue bag' of a window that
    glimpsed a bluish box, which a person could not tell from the bag; or the screen reaching the answerable pool."""
    def asked(rgb):
        s = vacuum_run()
        glimpsed_box(s, rgb=rgb)
        s.item("U", "bag", {"color": "blue"}, [(0, None, (2.0, 1.0, 0.0))], size=BAG_SIZE)
        s.item("A", "bag", {"color": "green"}, [(0, None, (1.0, 1.0, 0.0))], size=BAG_SIZE)
        s.see([1], "A", 1500)
        answerable, unknown = pools(fk.world(s))
        return ({x[2:] for c in answerable for x in c.subjects}, {x[2:] for c in unknown for x in c.subjects
                                                                  if x.startswith("e:")})
    assert asked(BLUE) == ({"A"}, set())
    assert asked(None) == ({"A"}, {"U"})


def test_ego_mem_drops_an_unknown_thing_and_an_absent_description_whose_colour_a_glimpse_showed():
    """Contract (S4 in `questions_ego._unknown_pool`): at the look point, the `unknown` pool leaves out a real unseen
    blue bag and the absent description of a blue box while a box labelled green was glimpsed (100 px) at keyframe 3
    rendering blue, and keeps both when it renders green; the answerable red bag seen at keyframe 2 is asked either way.
    A glimpse in a frame after the look point (keyframe 20) changes nothing (the box decoys are then of a kind not yet
    sighted, so there are none). Regression caught: the dev root's `ego_mem`
    asking for a thing a bluish glimpse had just shown, or a frame after the ask deciding a question (the cut of
    `upto`)."""
    def pool(rgb, glimpse_at=3):
        s = scene()
        put(s, "A", "bag", "red", (5.0, 3.13), [2])
        put(s, "U", "bag", "blue", (6.0, 2.0))
        put(s, "G", "box", "green", (7.0, 2.0), [glimpse_at], px=GLIMPSE_PX, rgb=rgb)
        moment = ego_mem(ego_world(s), K)
        real = {c.subjects[0] for c in moment.unknown if c.subjects[0].startswith("e:")}
        decoys = {c.slots["attributes"]["color"] for c in moment.unknown if c.slots["kind"] == "box"
                  and c.subjects[0].startswith("d:")}
        return {c.entities[0] for c in moment.answerable}, real, "blue" in decoys
    assert pool(BLUE) == ({"A"}, set(), False)
    assert pool(None) == ({"A"}, {"e:U"}, True)
    assert pool(BLUE, glimpse_at=20)[:2] == ({"A"}, {"e:U"})
