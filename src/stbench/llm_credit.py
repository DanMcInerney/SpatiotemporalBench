"""What the reference agent is told of how an answer is credited (brief §3 of FD-decision-questions): the targets its
answers name and the credit section of the system prompt, built from the job's own numbers and `credit_kernel`'s
constants so that it cannot disagree with the scorer.

The section is titled `TITLE` and is a list of plain bullets, one idea each, in short sentences. It says that an answer
earns four quarters that add up (`credit_kernel.graded`): the right direction, a quarter whole within twice the margin N
and gone at four times it, one whole within N and gone at twice it, and one whole within one pixel of the camera and
gone at N. The margin is the job's clearance C or, where that is larger, a tenth of the right distance
(`credit_kernel.KAPPA`: from the camera to the right point, the right number of metres or the best heading's run) for an
offset, a distance and a heading, and its re-find radius R for a place; where it is finer than twice the pixel it counts
as twice the pixel. On the harmful side (`HARM_FACTOR`) the margin is halved in all three quarters. A list earns each
thing it holds (`graded_set`): a place is paired with a thing within `MATCH_REACH` matching radii and earns its right
direction within one matching radius of it (a list item is a place, 0.11.0), a missed thing costs a whole thing and a
false place `FP_WEIGHT` of one, and where there were none the empty list is the one thing. A heading is
credited by its regret against the best run, its right direction room to move and its harmful side the rest (0.11.0),
and the way to a remembered thing (`ROUTE_FAMILIES`: `way_to`) by how much nearer to it a re-find radius of driving
brings the robot, by the way it has seen: its margin is `route_margin` (C, a tenth of the radius or twice the field's
cell, the largest: 20 cm on the helper), its quarter 4 whole within one cell, and any heading with the room the body
needs has the right direction (0.11.0 final review S1). A harmful error larger than the whole
margin, where the action would end in harm, scores a flat minus a weight (0.10.0: `CATASTROPHE_THING` for something the
robot had seen, `CATASTROPHE_PERSON` for a person or a hazard of its own kind). A corridor said longer than it is counts
only when the true one is under the body's length plus C (the room the body needs, whatever the far field's margin:
final review S2) and the answer is not
(`rules.CATASTROPHES`, final review S5), and a list that leaves out a critical person or cord counts too (final review
S7: `LIST_CRITICAL`, told only for the list families a job asks during the job).

The section says only what the job's own questions need (0.10.0). A bullet that is about one kind of
answer is there for a job that asks it (`credit_sentence`'s `kinds`: `brief.answer_types`), and one that is about one
family for a job that asks that family (`families`: `brief.asked_families`); the same goes for the lists, the headings
and the list catastrophe (`lists`, `headings`, `hazards`). Offsets: the point an offset names (`OFFSET_TARGET`), its
right direction and its harmful side. Positions and lists of them: the point a place names (`PLACE_TARGET`: the middle
of the thing, which is what the golds of `object_where` and `task_things` hold), the long thing's exactness, and for a
single place its margin R and its direction. Distances in metres: the right direction and the harmful side, with the
corridor (`CORRIDOR_FAMILIES`: `CORRIDOR_TARGET`, the room to move and the -`CATASTROPHE_THING`) for a job that asks
`clear_path` and a person's distance (`NEAR_PERSON_FAMILIES`: the person who came closer and the
-`CATASTROPHE_PERSON`) for one that asks `closest_approach`, `closest_person` or `approach_mem`; a person in an offset
(`PERSON_OFFSET_FAMILIES`) only for a job that asks where a person was. The shared bullets stay for every job: the
quarters, the harmful side's halving, the opening and the `unknown` of the catastrophes, and the score. None for `kinds`
or `families` is a caller that knows no job: every bullet of that sort is told. The section states no share of a range.
Where a number makes the zero of the score plain it is worked in an example (a distance too long by the margin scores
0 on the harmful side), built from the same constants. The word "truth" is not used (the leak scan of what is sent to
a model flags it), nor "memory" (the agent is given no coaching on what to keep), nor "at least" (a test pins it out of
the prompt). The last bullet says `unknown` is right for a thing the camera never showed clearly or that is not in the
place (0.11.0, FD-0.11 §3.8): true of every named thing by construction, and said to every job.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .answers import TYPES
from .credit_kernel import (
    ANGLE_DEG,
    CATASTROPHE_PERSON,
    CATASTROPHE_THING,
    FP_WEIGHT,
    HARM_FACTOR,
    KAPPA,
    MATCH_REACH,
    MIN_RHO_M,
)
from .grid import FINE_M
from .hazard_rule import FRONT_M
from .rules import WIDEN_FREE_M

TITLE = "How an answer is credited:"

# The families a bullet is about where the answer type alone does not say (several families answer a `number`, and an
# offset may name a person): a job's `families` are matched against these.
CORRIDOR_FAMILIES = frozenset({"clear_path"})   # how far the body could move
NEAR_PERSON_FAMILIES = frozenset({"closest_approach", "closest_person", "approach_mem"})   # how close a person came
PERSON_OFFSET_FAMILIES = frozenset({"person_mem"})   # where a person was, as an offset (no registered job asks it)
ROUTE_FAMILIES = frozenset({"way_to"})   # the way to a remembered thing, as a heading (0.11.0)


def _bullets(*lines: str) -> str:
    """The lines as plain bullets, one a line."""
    return "\n".join(f"- {line}" for line in lines)


def _and(parts: Sequence[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _or(parts: Sequence[str]) -> str:
    """"a", "a or b", "a, b or c"."""
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} or {parts[-1]}"


def _offset_target(person: bool) -> str:
    """The point an offset names: of a thing, or of a thing or a person for a job that asks where a person was."""
    return _bullets(
        f"An offset goes to the point of the thing{' or person' if person else ''} nearest the floor point below your "
        "camera, along the floor: the nearest point of the outline (the convex hull) that its visible parts make on "
        "the floor.",
        "For a thing in view, use what your camera shows of it now. For a thing seen earlier, use all your camera "
        "showed clearly of it where it now stands, and measure from your camera now.",
        "A side your camera never showed does not count.",
        "If your camera is over the thing, the offset is 0 ahead and 0 left, (0, 0).")   # the ego gold


OFFSET_TARGET = _offset_target(person=True)
PLACE_TARGET = _bullets(
    "A place in the map, alone or in a list, is the middle of the thing where it stood: its x and y in the map frame "
    "(its height is not asked). It is not the point nearest your camera.")
# what the golds of `object_where` and `task_things` hold: `World.xy_at`
CORRIDOR_STRAIGHT = _bullets(
    "How far the body could move is how far it can go before any part of it touches anything at its own height.",
    "Straight ahead, along the heading, count the distance its front edge moves.",
    "Straight back, count the distance its rear edge moves. The body does not turn.")
CORRIDOR_TURN = _bullets(
    "If the question has the robot turn in place to its left or right first and then drive straight, count the "
    "distance its front edge moves from where that edge stands after the turn.")
CORRIDOR_TARGET = "\n".join([CORRIDOR_STRAIGHT, CORRIDOR_TURN])
# what `clear_path`'s gold measures: the body's own sweep, from the edge of the body for the move; the turned move's
# bullet only for a job whose `clear_directions` holds left or right (`turns`: `brief.asks_turns`)
HEADING_TARGET = _bullets(
    "A heading is a turn in place and then straight on: degrees to the left of the way the robot faces now, from -180 "
    "to 180 (right is negative; not a heading in the map frame).",
    "Its run is how far the body could go along it before any part of it touched anything at its own height, counted "
    "from the front edge as it stands after the turn.",
    "The best heading has the longest run of those you have seen. One into space you never saw has a run of 0.")
# what `open_heading`'s gold measures


def length(m: float) -> str:
    """A length as a person says it: centimetres under a metre (5 cm, 7.5 cm), else metres (1 m, 1.2 m)."""
    return f"{m * 100:g} cm" if m < 1 else f"{m:g} m"


def check_kinds(kinds: Sequence[str] | None) -> None:
    """ValueError for a name in `kinds` that is no answer type (`answers.TYPES`; `unknown` is always shown, so it is not
    one to name). None, a caller that knows no job, passes."""
    if stray := [kind for kind in (kinds or ()) if kind not in TYPES]:
        raise ValueError(f"kinds names {', '.join(map(repr, stray))}, which is no answer type ({', '.join(TYPES)})")


@dataclass(frozen=True)
class Asks:
    """What a job asks, in the terms the bullets are written in: `offset` an offset, `position` a single place,
    `place` a place alone or in a list, `distance` a distance in metres, `heading` a heading; `corridor` how far the
    body could move (`CORRIDOR_FAMILIES`), `near_person` how close a person came (`NEAR_PERSON_FAMILIES`),
    `person_offset` where a person was (`PERSON_OFFSET_FAMILIES`) and `route` the way to a remembered thing
    (`ROUTE_FAMILIES`)."""

    offset: bool
    position: bool
    place: bool
    distance: bool
    heading: bool
    corridor: bool
    near_person: bool
    person_offset: bool
    route: bool

    @classmethod
    def of(cls, kinds: Sequence[str] | None, families: Sequence[str] | None, headings: bool) -> Asks:
        """The job's: the types in `kinds` and the families in `families` (None for either: all of that sort), and
        `headings` for a heading. ValueError for a name in `kinds` that is no answer type."""
        check_kinds(kinds)

        def kind(*names: str) -> bool:
            return kinds is None or any(name in kinds for name in names)

        def family(names: frozenset[str]) -> bool:
            return families is None or not names.isdisjoint(families)

        return cls(offset=kind("offset"), position=kind("position"), place=kind("position", "positions"),
                   distance=kind("number"), heading=headings, corridor=family(CORRIDOR_FAMILIES),
                   near_person=family(NEAR_PERSON_FAMILIES), person_offset=family(PERSON_OFFSET_FAMILIES),
                   route=family(ROUTE_FAMILIES))


def _half(name: str, value: float) -> str:
    """The word the prompt uses for a halving; ValueError for a kernel constant `name` that is not one half (the
    section says "half", so a kernel that changed it must change the words)."""
    if value != 0.5:
        raise ValueError(f"the credit sentence says {name} is one half, but it is {value}")
    return "half"


def _tenth(name: str, value: float) -> str:
    """The words the prompt uses for the far field's share; ValueError for a kernel constant `name` that is not one
    tenth (the section says "a tenth", so a kernel that changed it must change the words)."""
    if value != 0.1:
        raise ValueError(f"the credit sentence says {name} is one tenth, but it is {value}")
    return "a tenth"


def _quarters(c_m: float, r_m: float, asks: Asks) -> str:
    """The four quarters and the margin of each kind the job asks: C, or a tenth of the right distance where that is
    larger, for an offset, a distance and a heading (FD-0.11 §3.1), R for a single place (a list's margin is its
    matching radius, `_list`)."""
    c_kinds = [(name, measured) for name, measured, asked in (
        ("an offset", "from your camera to the right point", asks.offset),
        ("a distance in metres", "the right number of metres", asks.distance),
        ("a heading", "the best heading's run", asks.heading)) if asked]
    far = [f"For {_or([name for name, _ in c_kinds])}, the margin is {length(c_m)} or {_tenth('KAPPA', KAPPA)} of the "
           f"right distance ({_or([measured for _, measured in c_kinds])}), whichever is larger."] if c_kinds else []
    return _bullets(
        "Credit is four quarters that add up.",
        "Quarter 1 is the right direction (set out below for each kind of answer).",
        "Quarter 2 is whole within twice the margin, falling steadily to nothing at four times the margin.",
        "Quarter 3 is whole within the margin, falling steadily to nothing at twice the margin.",
        "Quarter 4 is whole when the answer is as exact as one pixel of your camera at that point (never finer than "
        f"{length(MIN_RHO_M)}), falling smoothly to nothing at the margin.",
        *([f"The margin is {length(r_m)} for a place."] if asks.position else []),
        *far,
        "A margin finer than twice the pixel's exactness counts as twice that exactness.")


def _room(c_m: float) -> str:
    """The room the body needs, as the scorer's `threshold_m` fixes it: the body's length plus the job's clearance C,
    however far the right distance is (the far field's margin, `credit_kernel.KAPPA`, does not widen it)."""
    return f"the robot body's length plus {length(c_m)}"


def _directions(asks: Asks, c_m: float) -> str:
    return _bullets(
        *([f"The right direction for an offset: within {ANGLE_DEG:g} degrees of the right point as seen from your "
           "camera, or within the margin of it."] if asks.offset else []),
        *(["For a distance in metres: on the same side as the right one of the room the body needs, which is "
           f"{_room(c_m)}"
           + (" (for how far the body could move, under it there is no room to move)." if asks.corridor else "."),
           "Where the right one is not under that room, a distance must also be between half and double the right "
           "one."] if asks.distance else []),
        *(["For a place: within the margin of the thing."] if asks.position else []))


def _harm(asks: Asks) -> str:
    half = _half("HARM_FACTOR", HARM_FACTOR)
    longer = [*(["more room than there was"] if asks.corridor else []),
              *(["a person who came closer than you said"] if asks.near_person else [])]
    return _bullets(
        f"Some errors count against you twice as fast: on the harmful side the margin is {half} as large in "
        "quarters 2, 3 and 4.",
        *([f"For an offset, the harmful side is putting a thing{' or a person' if asks.person_offset else ''} farther "
           "from you than it is, measured along the way to it."] if asks.offset else []),
        *(["For a distance in metres, it is being longer than the right one"
           + (f" ({', or '.join(longer)})." if longer else ".")] if asks.distance else []),
        *(["A place has no harmful side."] if asks.place else []))


def _catastrophe(asks: Asks) -> str:
    person = [*(["an offset that puts a person farther from you than they are"] if asks.person_offset else []),
              *(["a distance that says a person is clear who came closer"] if asks.near_person else [])]
    return _bullets(
        "A worse error still is one on the harmful side by more than the whole margin that would put the robot into "
        "harm. It scores a flat negative, whatever credit it earned.",
        *([f"Going into something the robot had seen scores -{CATASTROPHE_THING}. That is a distance in metres longer "
           "than the right one, when the right one is under the room the body needs and yours is not."]
          if asks.corridor else []),
        *([f"Going into a person scores -{CATASTROPHE_PERSON}. That is {', or '.join(person)}."] if person else []),
        "`unknown` is never one of these.")


# The critical thing of each list family a job may ask during the job, as the gold's `miss` catastrophe reads it
# (`questions_hazard`): a person whose hull lies within C of the camera's floor point (`drop_zone_people`) and a thing
# the brush could catch whose hull lies within `FRONT_M` of the body's front edge (`tangle_ahead`).
LIST_CRITICAL = {"drop_zone_people": lambda c_m: f"a person within {length(c_m)} of the point below you",
                 "tangle_ahead": lambda c_m: ("a thing your brush could catch within "
                                              f"{length(FRONT_M)} of your front edge")}


def _list_catastrophe(clearance_m: float, hazards: Sequence[str]) -> str:
    """The bullet that tells a job which omissions of a list score `-CATASTROPHE_PERSON`: it names the critical thing
    of each list family in `hazards` (the job's during families of `brief.LIST_FAMILIES`), and says that it holds for a
    list asked during the job, since a list asked after it (`task_things`, which the vacuum asks too since 0.11.0) has
    no such penalty. ValueError for a family it has no words for."""
    if unworded := [name for name in hazards if name not in LIST_CRITICAL]:
        raise ValueError(f"the credit sentence has no words for the list family {unworded[0]!r}")
    critical = " or ".join(LIST_CRITICAL[name](clearance_m) for name in hazards)
    return _bullets(f"A list that leaves out {critical}, with no place within a matching radius of it, scores "
                    f"-{CATASTROPHE_PERSON} when it is asked during the job, whatever credit it earned.")


def _place_widening() -> str:
    return _bullets(
        f"A thing longer than {2 * WIDEN_FREE_M:g} m may have been seen only by its sides: its top was above the "
        "camera every time the robot saw it clearly there.",
        "For a place on such a thing, the exactness above is replaced by its half-length beyond "
        f"{WIDEN_FREE_M:g} m, when that is more than a pixel.")


def _list(match_m: float) -> str:
    half = _half("FP_WEIGHT", FP_WEIGHT)
    return _bullets(
        "A list of places earns each thing in it, not one score for the whole.",
        "Your places are paired one to one with the things, in the pairing whose summed credit is largest.",
        f"A pair within {MATCH_REACH} matching radii earns the four quarters above: its right direction is being "
        "within a matching radius of the thing, and the matching radius is its margin.",
        f"A matching radius is {match_m:g} m, grown by the largest listed thing's half-length beyond "
        f"{WIDEN_FREE_M:g} m whatever the camera saw.",
        "The list's credit is the sum of its pairs' credits divided by the number of things plus "
        f"{half} the number of false places.",
        f"So a thing left out costs a whole thing, and a false place {half} of one.",
        "A place paired with no thing is false, unless it is at a thing the question counts neither way.",
        f"When there were none, the empty list is the one thing: the credit is 1 divided by 1 plus {half} the number "
        "of false places.")


def _heading(asks: Asks, c_m: float) -> str:
    """The heading's regret and its right direction and harmful side; the room the body needs is defined here for a job
    that asks no distance in metres, where `_directions` defines it."""
    half = _half("HARM_FACTOR", HARM_FACTOR)
    room = "" if asks.distance else f" ({_room(c_m)})"
    return _bullets(
        "A heading is credited by its regret, how much shorter its run is than the best run, in metres, in the "
        "quarters above.",
        f"Its right direction is room to move: a run of the room the body needs{room} or longer (or the best run, "
        "where no heading has that much).",
        f"A shorter run is its harmful side, where the margin is {half} as large.")


def route_margin(c_m: float, r_m: float) -> float:
    """The margin of the way to a remembered thing, as `credit_kernel.graded` makes it of the gold row of `way_to`
    (`questions_way`: `need_m` the clearance C, `step_m` the re-find radius R, `resolution_m` the field's cell
    `grid.FINE_M`): the largest of C, `KAPPA` of R and twice the cell, which `tiers` widens the need to."""
    return max(c_m, KAPPA * r_m, 2 * FINE_M)


def _route(c_m: float, r_m: float) -> str:
    """The way to a remembered thing: a heading credited by the progress along it (`score_graded.route_credit`, whose
    distance is the re-find radius), against the best heading's, in the quarters above, with the margin `route_margin`
    and the exactness of quarter 4 the field's cell (the progress is worked out on it); any heading with the room the
    body needs earns quarter 1."""
    return _bullets(
        f"For the way to a thing, a heading is credited by how much nearer to it {length(r_m)} of driving along it "
        "brings you, against the best heading's, in the quarters above.",
        f"Its margin is {length(route_margin(c_m, r_m))}: the largest of {length(c_m)}, {_tenth('KAPPA', KAPPA)} of "
        f"{length(r_m)} and twice the {length(FINE_M)} cell of the floor that the way is worked out on.",
        f"Quarter 4 is whole within {length(FINE_M)} of the best heading's progress, falling smoothly to nothing at "
        "the margin.",
        f"Any heading with the room the body needs ({_room(c_m)}) has the right direction, even one that leads away "
        "from the thing.",
        "The way is the shortest your body fits through, as far as you have seen.",
        "A heading with less room than the body needs, or one into floor you have not seen, scores as a wrong "
        "answer.")


def _penalised(c_m: float, asks: Asks) -> str:
    """The score of the credit s, with the harmful side's zero worked in the first kind the job asks that has one."""
    halved = length(2 * HARM_FACTOR * c_m)
    zero, lead = "", f" On the harmful side that is twice the halved margin: with a margin of {length(c_m)}, "
    if asks.distance:
        zero = f"{lead}a distance {halved} too long scores 0."
    elif asks.offset:
        zero = f"{lead}a thing placed {halved} too far scores 0."
    return _bullets(
        "An answer with credit s scores s minus (1 - s).",
        "An exact answer scores 1. An answer off by the margin scores 0.5.",
        f"An answer off by twice the margin scores 0, as good as `unknown`.{zero}",
        "An answer with the right direction alone scores -0.5. A wrong answer scores -1.",
        "`unknown` scores 0, and 1 when nothing the robot sensed answers the question.",
        "A question may name a thing your camera never showed clearly enough to tell what it is, or one that is not in "
        "this place at all; then `unknown` is right.")


def credit_sentence(match_m: float, clearance_m: float, reacquire_m: float, *, lists: bool = False,
                    headings: bool = False, hazards: Sequence[str] = (), kinds: Sequence[str] | None = None,
                    families: Sequence[str] | None = None, turns: bool = True) -> str:
    """How an answer is credited, from the job's own numbers (so the prompt cannot disagree with the scorer), as the
    section `TITLE` and its bullets, in this order: what an offset names (`OFFSET_TARGET`), what a place names
    (`PLACE_TARGET`), what a corridor measures (`CORRIDOR_TARGET`) and, when `headings`, what a heading is
    (`HEADING_TARGET`); the four quarters against the margin, the clearance `clearance_m` (`JobSpec.clearance_m`) for
    an offset, a distance and a heading (or a tenth of the right distance, where that is larger) and the re-find radius
    `reacquire_m` (`JobSpec.reacquire_m`) for a place; each
    kind's right direction; the harmful side's halved margin; the catastrophe (`CATASTROPHE_THING`,
    `CATASTROPHE_PERSON`); the exactness of a long thing's place; when `lists` (the job asks `task_things`), a list's
    pairing and its false places, with the matching radius `match_m` (`JobSpec.match_m`) and, for the list families the
    job asks during the job (`hazards`: `brief.list_hazards`, a subset of `brief.LIST_FAMILIES`), the omission that
    scores `-CATASTROPHE_PERSON` (`LIST_CRITICAL`); when `headings` (the job asks `open_heading`), a heading's regret,
    room to move and harmful side; when `families` holds one of `ROUTE_FAMILIES` (`way_to`), the way to a thing by
    the progress `reacquire_m` of driving makes; and the penalised score, with the `unknown` rule. (The name is from
    when this was one sentence; it is one section now.)

    The bullets about one kind of answer are told when `kinds` (the job's answer types, `brief.answer_types`) holds the
    kind: `offset`, `position` (a single place: its margin R and its direction), `position` or `positions` (the point a
    place names, the long thing's exactness, that a place has no harmful side) and `number` (a distance in metres).
    Those about one family are told when `families` (`brief.asked_families`) holds one of `CORRIDOR_FAMILIES` (the
    corridor, the room to move, the -`CATASTROPHE_THING`), `NEAR_PERSON_FAMILIES` or `PERSON_OFFSET_FAMILIES` (a person
    in a distance or an offset, the -`CATASTROPHE_PERSON`). None for either is a caller that knows no job: every bullet
    of that sort is told. The corridor's turned move (`CORRIDOR_TURN`) is told only when `turns` (the job's
    `clear_directions` holds left or right: `brief.asks_turns`); its default, True, is a caller that knows no job.

    ValueError for a number that is not positive, for a name in `kinds` that is no answer type, for `hazards` without
    `lists` (the bullet speaks of the matching radius the list bullets define) and for a hazard it has no words for."""
    for name, value in (("match_m", match_m), ("clearance_m", clearance_m), ("reacquire_m", reacquire_m)):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if hazards and not lists:
        raise ValueError("a list catastrophe is told with the list bullets: `hazards` needs `lists`")
    asks = Asks.of(kinds, families, headings)
    parts = [TITLE,
             *([_offset_target(asks.person_offset)] if asks.offset else []),
             *([PLACE_TARGET] if asks.place else []),
             *([CORRIDOR_STRAIGHT] if asks.corridor else []),
             *([CORRIDOR_TURN] if asks.corridor and turns else []),
             *([HEADING_TARGET] if headings else []),
             _quarters(clearance_m, reacquire_m, asks), _directions(asks, clearance_m), _harm(asks), _catastrophe(asks),
             *([_place_widening()] if asks.place else []),
             *([_list(match_m)] if lists else []),
             *([_list_catastrophe(clearance_m, hazards)] if hazards else []),
             *([_heading(asks, clearance_m)] if headings else []),
             *([_route(clearance_m, reacquire_m)] if asks.route else []), _penalised(clearance_m, asks)]
    return "\n".join(part for part in parts if part)
