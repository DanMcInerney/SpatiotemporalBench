"""The questionnaire: which questions a case asks, and in what order.

**During** (`during`): a seeded draw over the window's decision points, independent of the world and of every gold
(design §8.5), so which points are asked reveals nothing about the world. Candidates are the decision points whose kind
asks a family (`JobSpec.decisions`), less those too early in their task (`min_steps`): a question that may reach into
the past needs history behind it, so a point whose kind asks any family that is not `now` is kept from `GAP_STEPS` steps
into its task; a `now` question is answered from the latest frame, so a point whose kind asks only `now` families
is kept from `NOW_MIN_STEPS`, half a look window (the robot has moved and the view is its own). They are taken in
`stable_hash(case id, k, "ask")` order, a point being kept when it is at least `GAP_STEPS` steps from every point kept,
until the case's quota (3 short, 5 medium, 8 long); the family is the kind's `stable_hash(case id, k, "family")`-th. The
draw reads kinds and family specs, never the world. A drawn point whose evidence rule fails is asked anyway, with gold
`unknown`; a family's gold function may return None (a dead band at that point), and then the point is not asked, no
other family drawn for it: the draw stays independent of the gold.

**End** (`end`): the families the job asks (`JobSpec.ends`; None is `DEFAULT_ENDS`, M1's `object_where`), each giving
its moments (`GOLDS[family](w)`: a named time, a task and a time, a kind). A moment of a family that draws an unknown
slice (`SLICED`) or a coin (`COINS`) is kept only when both its pools are non-empty, one of another family when its
answerable pool is.
Families are visited in `stable_hash(case id, family, "family")` order, one question a turn, round robin, to
`MAX_END`: on its turn a family tries its next moments, in `stable_hash(case id, family, moment key, "moment")` order,
until one gives a question: the first candidate of a pool in `stable_hash(case id, candidate key)` order that the
**subject rule** allows (no two end questions share an entity, a description or a named time: equal or overlapping `w:`
subjects; during questions' subjects are recorded and exempt; and at most one asks about an absent description, a `d:`
subject: the decoys give nothing away only while they are few). The candidates of a pool are taken in **selection
order**: the task-critical ones first, then the world's own things before an absent description, then by
`stable_hash(case id, candidate key)`; and a round visits the temporal families (`FamilySpec.temporal`) first. The
selection order decides only which of the questions a case could ask it does ask, never the ask order below. A moment
of a sliced family gives a question only when
the rule allows a candidate in both its pools, so a pool the earlier questions have used up skips the moment whatever
its slice; the pool taken is `unknown` for `UNKNOWN_PERCENT` of a sliced family's moments, drawn per moment
(`stable_hash(case id, family, moment key, "slice")`), else `answerable`, so neither the moment nor whether it is
asked predicts the slice, and the asked unknown share is the declared one. A `COINS` family (a yes/no whose yes
candidates are the moment's `answerable` pool and no candidates its `unknown`) is drawn the same way with
`COIN_PERCENT` for the `unknown` (no) pool, so whether a moment is asked never depends on its answer. A family with no
moment left drops out of the round.

**Looks** (`looks`): the families of `JobSpec.looks`, asked at the look points (`questions_looks`: any frame at least
`LOOK_MIN_STEPS` into its task, the frames of the job's `look_decisions` first and then the rest in `stable_hash(case
id, k, "ask")` order as the decision points are, up to `ceil(frames / LOOK_PER_FRAMES)`). A point asks one question a
slot of the job's slot table (`JobSpec.look_slots`, any number of slots), in slot order: the slot's family is drawn by
`questions_looks.slot_family`, from the case id, the point and the table alone, and asks through the same `_turn` an end
moment uses (a sliced family needs a candidate allowed in both its pools, and draws its unknown slice per point as an
end moment does). A drawn family with no candidate there, or none the subject rule allows, asks nothing: no other family
is tried, and nothing replaces it. A point of a family that is not `SLICED` offers the candidates of both its pools,
since it draws no slice between them (a corridor or a person is either seen or not). Last, a look question whose class
is `unknown` is dropped unless `stable_hash(case id, family, str(k), "unknown") % 100` is under the job's
`look_unknown_keep[family]` (absent: all are kept); the draw reads the class and nothing else.

**The order** (`ask`; DECISIONS, "0.6.0's direction"): the end questions are drawn first, by the rules above, and their
subjects are reserved; the look draw sees those and the looks so far, so no look names a reserved subject and no entity
is named twice. A thing a look or an end question names is then don't-care in the families that relax it (`RELAX`:
`task_things` carries no subject, so it reserves nothing and is relaxed instead), which recomputes their
`answer_class`; last, `CAPPED` thins the empty golds of an end family by the job's `task_things.empty_keep_percent`,
a question being kept when `stable_hash(case id, family, key, "empty") % 100` is under it, so the cap reads the class
after `RELAX`.

The ask order is the during rows (decision points and look points) in keyframe order, a decision point's before a look
point's at one keyframe and the looks of a point in slot order, then the end rows in
`stable_hash(case id, candidate key)` order.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable

from . import questions_looks
from .job_registry import JobSpec
from .questions_select import Cand, Moment
from .questions_text import CAPPED, COINS, FAMILIES, GOLDS, LOOKS, RELAX, SLICED
from .questions_world import World
from .records import BALANCE_POOLS, pool_of
from .seeding import stable_hash
from .timeline_ego import DecisionPoint

GAP_STEPS = 16  # asked points are this many steps apart, and a point asking a memory family this far into its task
NOW_MIN_STEPS = 4  # a point whose kind asks only `now` families is asked from this far into its task: half a window
MAX_DURING = {"short": 3, "medium": 5, "long": 8}
MAX_END = {"short": 12, "medium": 18, "long": 20}
UNKNOWN_PERCENT = 20  # of a sliced family's moments whose question is the unknown slice: the middle of the 15-25% band
COIN_PERCENT = 50  # of a `COINS` family's moments whose question is the second (no) pool: a fair coin
DEFAULT_ENDS = ("object_where",)  # the end families of a job that names none (M1's)

# `GOLDS` (family -> its gold function) is the registry's, `questions_text.GOLDS`: a during family's takes (world,
# decision keyframe) and gives a `Cand` or None, an end family's takes the world and gives its `Moment`s.

_TIME = re.compile(r"w:(\d+)(?:-(\d+))?")


def _times(subject: str) -> tuple[int, int] | None:
    named = _TIME.fullmatch(subject)
    return None if named is None else (int(named[1]), int(named[2] or named[1]))


def _absent(subjects: Iterable[str]) -> bool:
    """Whether any subject is an absent description (`d:<kind>/<attributes>`): a description that matches nothing in the
    world."""
    return any(x.startswith("d:") for x in subjects)


def conflicts(a: Iterable[str], b: Iterable[str]) -> bool:
    """Whether two questions' subjects share a thing, a description or a named time: equal subjects, or `w:` times
    whose keyframe intervals overlap; or each names an absent description (a case asks about at most one)."""
    if _absent(a) and _absent(b):
        return True
    for x in a:
        for y in b:
            tx, ty = _times(x), _times(y)
            if x == y or (tx and ty and tx[0] <= ty[1] and ty[0] <= tx[1]):
                return True
    return False


def min_steps(job: JobSpec, kind: str) -> int:
    """How far into its task a decision point of this kind must lie to be asked: `NOW_MIN_STEPS` when every family the
    kind asks has scope `now` (a `now` question is answered from the latest frame and needs no history), else
    `GAP_STEPS` (an `"evidence"` family counts as `memory`). The kind must ask a family."""
    return NOW_MIN_STEPS if all(FAMILIES[f].scope == "now" for f in job.decisions[kind]) else GAP_STEPS


def check_decisions(job: JobSpec) -> None:
    """ValueError naming a look family (`questions_text.LOOKS`) that the job's `decisions` map to a decision kind: a
    look family is asked at look points (`JobSpec.looks`) and its gold gives a `Moment`, not a `Cand`."""
    for kind, families in job.decisions.items():
        if stray := sorted(set(families) & LOOKS):
            raise ValueError(f"the job {job.id} maps the look famil{'y' if len(stray) == 1 else 'ies'} "
                             f"{', '.join(stray)} to the decision kind {kind!r}; a look family is named in "
                             f"`JobSpec.looks` and asked at look points")


def decision_points(w: World) -> list[DecisionPoint]:
    """The window's decision points that may be asked: their kind asks a family, and they lie at least `min_steps` of
    that kind into their task. ValueError for a job that maps a look family to a decision kind."""
    check_decisions(w.job)
    return [d for d in w.decisions
            if w.job.decisions.get(d.kind) and d.k - w.tl.task_of(d.k).k0 >= min_steps(w.job, d.kind)]


def drawn(w: World) -> list[DecisionPoint]:
    """The decision points asked, in decision order."""
    cid, kept = w.rec.case, []
    for d in sorted(decision_points(w), key=lambda d: stable_hash(cid, d.k, "ask")):
        if len(kept) < MAX_DURING[w.rec.length] and all(abs(d.k - o.k) >= GAP_STEPS for o in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.k)


def during(w: World) -> list[Cand]:
    """The during questions, in decision order: a drawn point whose family finds a dead band there is not asked."""
    out = []
    for d in drawn(w):
        families = w.job.decisions[d.kind]
        if (c := GOLDS[families[stable_hash(w.rec.case, d.k, "family") % len(families)]](w, d.k)) is not None:
            out.append(c)
    return out


def asked_ends(job: JobSpec) -> tuple[str, ...]:
    """The end families `job` asks: its `ends`, or `DEFAULT_ENDS` for a job that names none (the one owner of that
    default; `brief.answer_types` reads it too)."""
    return DEFAULT_ENDS if job.ends is None else job.ends


def end_families(w: World) -> list[str]:
    """The end families the job asks (`asked_ends`), in registry order. ValueError names one the registry lacks or that
    is not an end family."""
    asked = asked_ends(w.job)
    for family in asked:
        if family not in FAMILIES or FAMILIES[family].when != "end":
            ends = sorted(f for f, spec in FAMILIES.items() if spec.when == "end")
            raise ValueError(f"the job {w.job.id} asks {family!r} at the end, which is not an end family of the "
                             f"registry ({ends})")
    return [f for f in FAMILIES if f in asked]


def two_pools(family: str) -> bool:
    """The family's moments hold two pools and draw between them: a sliced family's (answerable, unknown) or a coin's
    (yes, no)."""
    return family in SLICED or family in COINS


def unknown_slice(case_id: str, family: str, key: str) -> bool:
    """Whether the end question about this moment of a two-pool family is taken from its `unknown` pool: the unknown
    slice of a sliced family (`UNKNOWN_PERCENT` of moments), the no of a `COINS` family (`COIN_PERCENT`)."""
    percent = COIN_PERCENT if family in COINS else UNKNOWN_PERCENT
    return stable_hash(case_id, family, key, "slice") % 100 < percent


def _by_hash(case_id: str, cands: Iterable[Cand]) -> list[Cand]:
    return sorted(cands, key=lambda c: (stable_hash(case_id, c.key), c.key))


def _selection_order(case_id: str, cands: Iterable[Cand]) -> list[Cand]:
    """The candidates in the order a pool offers them: critical ones first, a world's thing before an absent
    description, then by hash."""
    return sorted(cands, key=lambda c: (not c.critical, _absent(c.subjects), stable_hash(case_id, c.key), c.key))


def _moments(w: World, family: str) -> deque[Moment]:
    """The family's moments that can be asked, in this case's order for the family."""
    cid, paired = w.rec.case, two_pools(family)
    kept = [m for m in GOLDS[family](w) if m.answerable and (m.unknown or not paired)]
    return deque(sorted(kept, key=lambda m: (stable_hash(cid, family, m.key, "moment"), m.key)))


def _allowed(cid: str, pool: Iterable[Cand], taken: list[Cand]) -> Cand | None:
    """The pool's first candidate, in selection order, that the subject rule allows."""
    return next((c for c in _selection_order(cid, pool) if not any(conflicts(c.subjects, t.subjects) for t in taken)),
                None)


def _check_classes(family: str, m: Moment) -> None:
    """ValueError for a candidate of a two-pool family whose `answer_class` does not fit the pool of the moment it is
    in: by `records.pool_of`, a candidate of `Moment.answerable` is of the family's first pool (`BALANCE_POOLS`), one of
    `Moment.unknown` of its second. A coin's sides swapped would balance the wrong classes. A candidate with no class
    is refused too: every family with pools sets one (the balance is read between its classes), and an unclassed
    candidate would pass the check unseen."""
    pools = list(BALANCE_POOLS.get(family, ()))
    if len(pools) < 2 or not two_pools(family):
        return
    for pool, cands in zip(pools[:2], (m.answerable, m.unknown), strict=True):
        for c in cands:
            if c.answer_class is None:
                raise ValueError(f"the {family} candidate {c.key} has no answer_class; the family declares balance "
                                 f"pools ({', '.join(pools)}), so each candidate carries its class")
            if pool_of(family, c.answer_class) != pool:
                raise ValueError(f"the {family} candidate {c.key} is of class {c.answer_class!r}, which belongs to the "
                                 f"{pool_of(family, c.answer_class)!r} pool, but it is in the {pool!r} pool of moment "
                                 f"{m.key!r}")


def _turn(cid: str, family: str, queue: deque[Moment], taken: list[Cand]) -> Cand | None:
    """The family's question for its turn: its next moments are tried until one gives a candidate the subject rule
    allows; None when none is left. A moment of a sliced or coin family needs a candidate allowed in both its pools
    before its slice is looked at (else the answerable pool, which earlier questions use up, would skip the moments that
    draw it and the unknown pool, which does not run out, would not); it then gives the candidate of the pool it
    draws."""
    paired = two_pools(family)
    while queue:
        m = queue.popleft()
        _check_classes(family, m)
        answerable = _allowed(cid, m.answerable, taken)
        unknown = _allowed(cid, m.unknown, taken) if paired else None
        if answerable is not None and (unknown is not None or not paired):
            return unknown if paired and unknown_slice(cid, family, m.key) else answerable
    return None


def end(w: World) -> list[Cand]:
    """The end questions, in ask order."""
    cid, size = w.rec.case, MAX_END[w.rec.length]
    queues = {f: _moments(w, f) for f in end_families(w)}
    round_ = sorted(queues, key=lambda f: (not FAMILIES[f].temporal, stable_hash(cid, f, "family"), f))
    taken: list[Cand] = []
    while round_ and len(taken) < size:
        for family in list(round_):
            if len(taken) == size:
                break
            if (c := _turn(cid, family, queues[family], taken)) is None:
                round_.remove(family)
            else:
                taken.append(c)
    return _by_hash(cid, taken)


def look_families(w: World) -> list[str]:
    """The look families the job asks (`JobSpec.looks`), in its order. ValueError names one that is not a look family of
    the registry (`questions_text.LOOKS`)."""
    for family in w.job.looks:
        if family not in LOOKS:
            raise ValueError(f"the job {w.job.id} asks {family!r} at look points, which is not a look family of the "
                             f"registry ({sorted(LOOKS)})")
    return list(w.job.looks)


def _dropped(cid: str, family: str, k: int, keep: int) -> bool:
    """Whether the unknown look question of `family` at point `k` is thinned out: `stable_hash(case id, family, str(k),
    "unknown") % 100` is `keep` or more."""
    return stable_hash(cid, family, str(k), "unknown") % 100 >= keep


def looks(w: World, reserved: list[Cand]) -> list[Cand]:
    """The look questions, by point and then by slot (the module docstring's "Looks"): at each point of
    `questions_looks.points` each slot of `JobSpec.look_slots` draws its family (`questions_looks.slot_family`), which
    asks at most one candidate through `_turn` over `reserved` (the end questions drawn before) and the looks so far, so
    it names no reserved subject and no entity twice. A slot that draws no family, a family with no moment at the point
    (a dead band) or no candidate the subject rule allows asks nothing there, and no other family of the slot is tried.
    A point of an unsliced family offers its candidates of both pools (a corridor or a person is either seen or not,
    never both). A candidate of class `unknown` is then kept by the job's `look_unknown_keep` of its family. ValueError
    for a candidate not asked at its point."""
    cid, out = w.rec.case, []
    look_families(w)
    keep = w.job.look_unknown_keep
    for k in questions_looks.points(w):
        for i, shares in enumerate(w.job.look_slots):
            family = questions_looks.slot_family(cid, k, i, shares)
            if family is None or (m := GOLDS[family](w, k)) is None:
                continue
            if family not in SLICED:   # no slice between the pools: the point's candidates, answerable or unknown
                m = Moment(m.key, [*m.answerable, *m.unknown], [])
            if (c := _turn(cid, family, deque([m]), [*reserved, *out])) is None:
                continue
            if c.decision_k != k:
                raise ValueError(f"the {family} candidate {c.key} is asked at look point {k} but carries decision_k "
                                 f"{c.decision_k}")
            if c.answer_class == "unknown" and _dropped(cid, family, k, keep.get(family, 100)):
                continue
            out.append(c)
    return out


def _named(asked: Iterable[Cand]) -> frozenset[str]:
    """The ids of the entities the questions name (`e:<id>` subjects)."""
    return frozenset(s[2:] for c in asked for s in c.subjects if s.startswith("e:"))


def _relaxed(ends: list[Cand], named: frozenset[str]) -> list[Cand]:
    """The end questions with the entities `named` (by a look or an end question) made don't-care by their family's
    `RELAX`."""
    return [RELAX[c.family](c, named) if c.family in RELAX else c for c in ends]


def _thinned(w: World, ends: list[Cand]) -> list[Cand]:
    """The end questions less the empty golds of a `CAPPED` family the draw drops: an empty question (`answer_class`
    "empty", read after `RELAX`) is kept when `stable_hash(case id, family, key, "empty") % 100` is under the job's
    `task_things.empty_keep_percent`; a job with no `task_things` thins nothing. ValueError for a `CAPPED` candidate
    with no class, thinned or not: the thinning reads it, so a family that is capped sets one."""
    keep, out = 100 if w.job.task_things is None else w.job.task_things.empty_keep_percent, []
    for c in ends:
        if c.family in CAPPED:
            if c.answer_class is None:
                raise ValueError(f"the {c.family} candidate {c.key} has no answer_class; the family is capped, and "
                                 f"the thinning reads the class of each candidate")
            if c.answer_class == "empty" and stable_hash(w.rec.case, c.family, c.key, "empty") % 100 >= keep:
                continue
        out.append(c)
    return out


def ask(w: World) -> list[Cand]:
    """Every question the case asks, in ask order (the module docstring's "order"): the end draw first, the looks over
    its subjects, `RELAX` over what the looks and the end questions name, `CAPPED`; then the during rows by keyframe and
    the end rows."""
    ends = end(w)
    seen = looks(w, ends)
    ends = _thinned(w, _relaxed(ends, _named(seen) | _named(ends)))
    return sorted([*during(w), *seen], key=lambda c: c.decision_k or 0) + ends


def candidates(w: World) -> list[Cand]:
    """Every question the world could ask, before the draw, the slice and the subject rule: each during family at every
    decision point that asks it (a dead band skipped), each look family of the job at every candidate look point, and
    every candidate of every moment of every end family the job asks. The hidden-change checks compare these."""
    check_decisions(w.job)
    out: list[Cand] = []
    for spec in FAMILIES.values():
        if spec.when == "during" and spec.name not in LOOKS:
            found = (GOLDS[spec.name](w, d.k) for d in w.decisions if spec.name in w.job.decisions.get(d.kind, ()))
            out += [c for c in found if c is not None]
    for family in look_families(w):
        moments = (GOLDS[family](w, k) for k in questions_looks.candidate_points(w))
        out += [c for m in moments if m is not None for c in m.answerable + m.unknown]
    for family in end_families(w):
        out += [c for m in GOLDS[family](w) for c in m.answerable + m.unknown]
    return out
