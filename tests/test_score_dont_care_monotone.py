"""A don't-care circle never costs a list credit (`score_set.graded_set`, `score_set._f1`): "a place inside a circle
counts neither way" means an answered item inside one can add a match and cannot take a gold item from an item outside,
so adding it never lowers the credit. Expected values are worked by hand from the kernel's formulas."""

from __future__ import annotations

import math
import random

import pytest

from score_kit import places, rec
from stbench.credit_kernel import graded
from stbench.score import credit

M = 0.5   # the match radius of the rows below; the reach is 4 M = 2 m


def circled(gold_points, circles, *, resolution=0.01, name="graded_set"):
    """A list gold with don't-care circles `(x, y, radius)`, scored by `graded_set` or, as `f1`, by item counts."""
    gold = places(*gold_points)
    gold["dont_care"] = [{"xy": [x, y], "radius": r} for x, y, r in circles]
    tolerances = ({"match_m": M, "resolutions_m": [resolution] * len(gold_points), "fp_weight": 0.5}
                  if name == "graded_set" else {"match_m": M})
    return rec(gold, credit=name, family="task_things", tolerances=tolerances)


def test_an_item_inside_a_circle_does_not_take_a_gold_item_from_one_outside():
    """Contract: items outside every circle are paired first, so one inside a circle cannot displace one outside and
    make it a false place. Reproduction, M = 0.5, rho 0.01, gold (0, 0), a circle of radius 0.2 at (0.3, 0):
    - (-0.3, 0) alone is outside and 0.3 m off: t = a = 1 and f = 1 - ln 30 / ln 50 = 0.1306, so
      (1 + 1 + 1 + 0.1306) / 4 = 0.7826;
    - the same plus (0.12, 0), which is inside the circle and 0.12 m from the gold: still 0.7826. The closer
      item used to take the gold and leave the outside one a false place, 0.7826 -> 0.5608.
    Regression caught: in-circle items in the one assignment with the others."""
    row = circled([(0, 0)], [(0.3, 0, 0.2)])
    expected = (3 + (1 - math.log(30) / math.log(50))) / 4
    assert credit(places((-0.3, 0)), row) == pytest.approx(expected)
    assert credit(places((-0.3, 0), (0.12, 0)), row) == pytest.approx(expected)


def test_an_item_inside_a_circle_still_takes_a_gold_item_left_over():
    """Contract: an item inside a circle that matches a gold item no outside item took earns its credit, whatever its
    place. By hand, M = 0.5, rho 0.01, gold (0, 0) and (1, 0), a circle of radius 0.15 at (0.6, 0): the item (0.3, 0) is
    outside and pairs with (0, 0) at 0.3 m; (0.45, 0) is inside the circle and takes the gold (1, 0) left over, 0.55 m
    off, with no direction as it is past M. The list is the two credits over two golds, no item false; the first item
    alone leaves the second gold missed, half the first credit. Regression caught: an item in a circle dropped before it
    could match, which would cost a correct answer its match."""
    row = circled([(0, 0), (1, 0)], [(0.6, 0, 0.15)])
    expected = (graded(0.3, 0.01, M, True, False) + graded(0.55, 0.01, M, False, False)) / 2
    assert credit(places((0.3, 0), (0.45, 0)), row) == pytest.approx(expected)
    assert credit(places((0.3, 0)), row) == pytest.approx(graded(0.3, 0.01, M, True, False) / 2)


def _random_case(rng: random.Random):
    """A gold of one to three places on a 0.1 m grid, one or two circles, an answer of up to four places (anywhere in
    the 3 m square, or near a gold place) and one more place inside a circle."""
    def point():
        return round(rng.uniform(0, 3), 1), round(rng.uniform(0, 3), 1)

    def answered(gold):
        x, y = rng.choice(gold)
        return point() if rng.random() < 0.5 else (x + round(rng.uniform(-0.5, 0.5), 1), y)

    gold = [point() for _ in range(rng.randint(1, 3))]
    circles = [(*point(), round(rng.uniform(0.2, 0.8), 1)) for _ in range(rng.randint(1, 2))]
    x, y, radius = rng.choice(circles)
    angle, reach = rng.uniform(0, math.tau), rng.uniform(0, radius)
    return gold, circles, [answered(gold) for _ in range(rng.randint(0, 4))], (x + reach * math.cos(angle),
                                                                                  y + reach * math.sin(angle))


@pytest.mark.parametrize("name", ["graded_set", "f1"])
def test_adding_an_item_inside_a_circle_never_lowers_a_list_s_credit(name):
    """Contract: for any gold, circles and answer, the credit of the answer plus one item that lies inside a circle is
    at least the credit of the answer, for the graded list and for the F1 list alike. Checked on 400 seeded random
    small cases (golds of up to three places, answers of up to four, near and far from the golds). Regression caught:
    an in-circle item pairing ahead of, or in competition with, the items outside (the graded list's 0.7826 falling to
    0.5608), or an F1 whose matching lets a circle's item displace one outside."""
    rng = random.Random(11)
    for _ in range(400):
        gold, circles, answer, extra = _random_case(rng)
        row = circled(gold, circles, name=name)
        before = credit(places(*answer), row)
        after = credit(places(*answer, extra), row)
        assert after >= before - 1e-9, (gold, circles, answer, extra, before, after)
