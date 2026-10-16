"""`adjacent` (end, `memory`; held-out layouts only): a point in the room that connects two named rooms.

"Give a point in the room that connects the kitchen to the bedroom." The layout's topology is the same in every case of
a layout, so on a public layout `blind` and `other_case` learn it from the other cases: the family is asked only on a
layout of `map_files.HELD_OUT`, where the unseen layout is the ability measured (design §8.3, D1). It is answerable
only from the robot's own delivered path and what it observed, never from the end look window alone.

A moment is a pair of regions A and B, each of a kind unique in the layout, with one region R joined to both and A, B
not joined to each other (`adjacency.connector`: a door, an arch, or an open edge of 0.8 m or more joins). It is asked
only when
- the window's delivered base positions cross A-R and R-B (`adjacency.crossed`, in either direction);
- R holds neither the window's first delivered position nor any position of the end look window (the last frames the
  system was sent), so the connector is never where the robot started or is: both jobs start and end a task at their
  dock or charger, and "where I started / where I am" would score without a sensor. This also keeps the end look
  window from showing the connection itself (rule 4): crossing both boundaries puts a position in R;
- A and B were each **toured** (`adjacency.toured`: at least half their inside fine cells observed, in any state, the
  share `changed` asks), over the window's keyframes, so the camera showed both rooms and a perceiver can tell which
  they are;
- neither A nor B is placed without the camera: not a kind a window brief names (`answer_keys.rooms_named`, over the
  text `brief.compose` writes, as `path_log` and `suite_priors` read it), not a kind the job's relevance lists for a
  task of the window (where the job takes the robot), and not the region holding the window's first delivered
  position. Else the briefs, the pose log or a prior from the public layouts ("the dock stands in the hall") place the
  room, and R is the room the path entered on leaving it: an answer that needs no sensor.

The gold is a point of R at least `adjacency.ERODE_M` inside it, scored by credit `region` (any point in R's polygon
eroded by that much); never `unknown`. Evidence: the keyframes of the two first crossings. Subjects: the three
regions. A robot that flies is never asked: the table's `Mapper` keeps no robot band for a flier (`table.Mapper.band`
is None), so a table system observes no cell of any room, no room is toured and the gold asks what none could answer.

A family module (`questions_text.FAMILY_MODULES`): `SPECS`, `GOLDS` (`moments`, an end family's) and `SLICED` (none).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from . import adjacency, map_files
from .answer_keys import rooms_named
from .brief import compose
from .questions_select import Cand, Moment
from .questions_world import World
from .records import FamilySpec
from .robots import robot_doc
from .stream_protocol import GATE_CHANNEL


def _named(w: World) -> set[str]:
    """The region kinds that a brief of the window names, over the texts `brief.compose` writes for its tasks (the
    paragraph of the first one included), as `path_log` and `suite_priors` read them on the gate channel."""
    plans = [task.plan for task in w.tl.ego.tasks[w.rec.a:w.rec.b + 1]]
    briefs = compose(plans, job=w.job, robot=robot_doc(w.robot), domain=w.job.domain, channel=GATE_CHANNEL)
    return set(rooms_named(briefs, (r.kind for r in w.map.regions)))


def _relevant(w: World) -> set[str]:
    """The region kinds that the job's relevance lists for a task of the window."""
    return {kind for i, _, _ in w.tasks for kind in w.relevance_of(i).regions}


def moments(w: World) -> list[Moment]:
    """One moment for each askable pair of rooms, keyed "<kind a>|<kind b>" (the kinds in sorted order); none on a
    layout outside `map_files.HELD_OUT`, nor for a robot that flies."""
    if w.tl.episode.map not in map_files.HELD_OUT or w.robot.flies:
        return []
    m, ks = w.map, sorted(w.rows)
    first = adjacency.first_crossings(m, [w.base(k) for k in ks])
    relevant = _relevant(w)
    placed, start = _named(w) | relevant, m.region_of(w.base(ks[0]))
    held = {start, *(m.region_of(w.base(k)) for k in w.end_window)}  # R is none of these (None: a doorway)
    kinds = Counter(r.kind for r in m.regions)
    pending = []
    for a, b in combinations(sorted((r for r in m.regions if kinds[r.kind] == 1), key=lambda r: r.kind), 2):
        via = adjacency.connector(m, a.id, b.id)
        if via is None or {a.kind, b.kind} & placed or start in (a.id, b.id) or via in held:
            continue
        legs = (frozenset((a.id, via)), frozenset((via, b.id)))
        if all(leg in first for leg in legs):
            pending.append((a, m.region(via), b, legs))
    if not pending:
        return []
    cells, space = w.latest_cells(ks), "room" if w.job.domain == "home" else "area"
    out = []
    for a, via, b, legs in pending:
        point = adjacency.answer_point(via)
        if point is None or not (adjacency.toured(a, cells) and adjacency.toured(b, cells)):
            continue
        cand = Cand("adjacent", {"space": space, "a": a.kind, "b": b.kind}, {"type": "position", "xy": list(point)},
                    "answerable", tuple(sorted({ks[first[leg]] for leg in legs})), (), {"erode_m": adjacency.ERODE_M},
                    subjects=(f"r:{a.id}", f"r:{via.id}", f"r:{b.id}"),
                    critical=bool({a.kind, via.kind, b.kind} & relevant), region=[list(p) for p in via.polygon])
        out.append(Moment(f"{a.kind}|{b.kind}", [cand], []))
    return out


SPECS = (FamilySpec("adjacent", "end", "position", "region", ("see", "pose"), "memory", "other_case", True,
                    ("space", "a", "b"),
                    lambda s: f"Give a point in the {s['space']} that connects the {s['a']} to the {s['b']}."),)
GOLDS = {"adjacent": moments}
SLICED: frozenset[str] = frozenset()
