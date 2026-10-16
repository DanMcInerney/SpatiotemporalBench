"""The question and gold rows the compiler writes, and the spec of a question family: key sets and validators.

A question row is what is asked (`private/<cid>/questions.jsonl`); a gold row is what answers it and how it is scored
(`gold.jsonl`). The compiler writes both, and the harness, the scorer and the gates call `check_question` and
`check_gold` on what they read. The validators judge one row on its own: that `text` is its family's rendering of
`slots` (render purity) and that the row fits its family's spec is the compiler's check, since this module cannot import
the family registry. `FamilySpec` is the shape of a registry entry (the registry itself is the compiler's) and
`check_family` judges one entry on its own. It is neutral: it imports `answers` and the channel vocabulary of
`stream_protocol`, nothing else.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from .answers import TYPES, Number, Position, Positions, Unknown, answer_type_of, is_polygon, parse_gold
from .stream_protocol import CAPABILITIES

QUESTION_KEYS = ("qid", "when", "decision_k", "text", "template", "slots", "answer_type", "needs")
GOLD_KEYS = ("qid", "family", "gold", "slice", "answer_class", "range_m", "credit", "region", "tolerances", "metric",
             "scope", "reference", "critical", "subjects", "evidence_k", "entities")
# Keys a gold row may hold beyond `GOLD_KEYS` (0.10.0, FD-questions-0.10 §4.5): the evidence age the compile writes on
# every row, absent on the rows of earlier builds, which stay readable. `age_s` is null or the clock seconds from the
# gold's oldest evidence keyframe to the ask (0 on a row whose scope is now); `age_band` null or one of `AGE_BANDS`
# (`now` exactly when scope is now).
GOLD_OPTIONAL = ("age_s", "age_band")
AGE_BANDS = ("now", "recent", "task", "earlier")
WHEN = ("during", "end")
SCOPES = ("now", "memory")
FAMILY_SCOPES = SCOPES + ("evidence",)   # FamilySpec.scope: "evidence" = now exactly when the evidence is frame k
SLICES = ("answerable", "unknown")
# The declared class a draw balanced (a gold row's `answer_class`; null where the family declares none): a sliced
# family's slice, a coin's two sides, `change`'s state and `task_things`' empty or non-empty gold.
ANSWER_CLASSES = ("yes", "moved", "vacated", "earlier", "later", "answerable", "unknown", "empty", "items", "person",
                  "no_one")
# The pools a family's balance is read between (`suite_balance`, `gates_evidence`): pool name -> the `answer_class`
# values that make it. A family absent here declares no balance (`ego_now`, the families not drawn by a class). A class
# of a family is in exactly one of its pools.
BALANCE_POOLS = {
    "object_where": {"answerable": frozenset({"answerable"}), "unknown": frozenset({"unknown"})},
    "ego_mem": {"answerable": frozenset({"answerable"}), "unknown": frozenset({"unknown"})},
    "object_was": {"earlier": frozenset({"earlier"}), "later": frozenset({"later"})},
    "same_place": {"yes": frozenset({"yes"}), "no": frozenset({"moved", "vacated"})},
    "task_things": {"empty": frozenset({"empty"}), "items": frozenset({"items"})},
    "clear_path": {"answerable": frozenset({"answerable"}), "unknown": frozenset({"unknown"})},
    "person_mem": {"answerable": frozenset({"answerable"}), "unknown": frozenset({"unknown"})},
    "closest_person": {"person": frozenset({"person"}), "no_one": frozenset({"no_one"})},
    "closest_approach": {"person": frozenset({"person"}), "no_one": frozenset({"no_one"})},   # 0.10.0
    "approach_mem": {"person": frozenset({"person"}), "no_one": frozenset({"no_one"})},       # 0.11.0, as its twin
    "drop_zone_people": {"empty": frozenset({"empty"}), "items": frozenset({"items"})},       # 0.10.0
    "tangle_ahead": {"empty": frozenset({"empty"}), "items": frozenset({"items"})},           # 0.10.0
    "open_heading": {"answerable": frozenset({"answerable"})},   # one pool: it asks no `unknown` (0.9.0)
    "way_to": {"answerable": frozenset({"answerable"}), "unknown": frozenset({"unknown"})},   # 0.11.0, as `ego_mem`
}
REFERENCES = (None, "path_log", "other_case", "kind_range", "map_reader")
# Each credit (plan §2.6) with the gold types it scores and, per type, the exact `tolerances` keys the gold row holds
# for it (an empty tuple: the row holds none). The ladder, the match radius, the erosion or the graded credit's
# resolution and need is written on the row, so no scorer default decides a credit unseen. The graded credits (0.9.0,
# `credit_kernel`) hold several: `graded` the resolution `rho`, the need and which side is harmful (and for a number
# the threshold of room to move); `graded_set` the match radius, one resolution per gold item and the weight of a false
# item; `regret` the run of every sampled heading and the resolution, need and threshold of the best run; `route`
# (0.11.0), beside the route of points, scores a heading to a remembered thing by the outcome of every sampled heading
# (how much nearer `step_m` of driving along it brings the robot), the resolution and the need.
_GRADED = ("resolution_m", "need_m", "harm")
CREDIT_KEYS = {
    "ladder": {"position": ("ladder",), "offset": ("ladder",), "number": ("ladder",), "time": ("ladder",)},
    "region": {"position": ("erode_m",)},
    "relative": {"number": ()},
    "mra": {"count": ()},
    "exact": {"yesno": ()},
    "f1": {"set": ("match_m",), "positions": ("match_m",), "cells": ()},
    "route": {"route": (), "heading": ("profile_m", "step_deg", "step_m", "resolution_m", "need_m")},
    "graded": {"offset": _GRADED, "number": (*_GRADED, "threshold_m"), "position": _GRADED},
    "graded_set": {"positions": ("match_m", "resolutions_m", "fp_weight")},
    "regret": {"heading": ("profile_m", "step_deg", "resolution_m", "need_m", "threshold_m")},
}
CREDIT_FITS = CREDIT_KEYS   # the name the scorer and the validators read: which credit scores which gold type
CREDITS = tuple(CREDIT_KEYS)
METRICS = (None, "xy", "xyz")
TOLERANCES = ("ladder", "match_m", "erode_m", "resolution_m", "need_m", "harm", "threshold_m", "resolutions_m",
              "fp_weight", "profile_m", "step_deg", "step_m", "catastrophe")
# `catastrophe` (0.10.0, FD-questions-0.10 §4.4) is optional beside any credit's keys: a harm row's rule for when an
# answer is a catastrophe, which the reported score (never a gate's) charges at its weight. Its shape:
# {"weight": 2 or 3, "rule": "over" | "radial" | "miss", "margin_m": > 0, "threshold_m": > 0 (over only, optional),
#  "critical": [[x, y], ...] (miss only)}.
OPTIONAL_TOLERANCES = frozenset({"catastrophe"})
CATASTROPHE_RULES = ("over", "radial", "miss")
# Which `harm` a graded gold's type may hold: a number answered longer than it is, an offset placed farther from the
# camera than it is; a place is never harmful.
HARM_OF = {"number": ("over", None), "offset": ("radial", None), "position": (None,)}
FULL_TURN_DEG = 360

_QID = re.compile(r"[0-9a-f]{16}")
# an entity, a description absent from the world (kind, then attributes), a region, a keyframe or a keyframe interval,
# or (0.11.0, `approach_mem`) a sighting: the keyframe of the last sighting of a person, who is named by no entity
_SUBJECT = re.compile(r"e:\S+|d:[^/]+/.*|r:\S+|a:\d+|w:(?P<k0>\d+)(?:-(?P<k1>\d+))?")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def pool_of(family: str, answer_class: str | None) -> str | None:
    """The balance pool of a gold row's class (`BALANCE_POOLS`); None for a family with no pools or an unclassed row."""
    for pool, classes in BALANCE_POOLS.get(family, {}).items():
        if answer_class in classes:
            return pool
    return None


def _is_text(v: object) -> bool:
    return isinstance(v, str) and bool(v)


def _key_problems(row: object, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(row, dict):
        return [f"a row must be an object, got {type(row).__name__}"]
    missing, extra = [k for k in keys if k not in row], [k for k in row if k not in keys]
    if missing or extra:
        return [f"keys must be exactly {list(keys)}; missing {missing}, unexpected {extra}"]
    return []


def _needs_problems(needs: object) -> list[str]:
    if not isinstance(needs, list) or not all(isinstance(n, str) for n in needs):
        return [f"needs must be a list of strings, got {needs!r}"]
    problems = []
    for n in needs:
        if n in SCOPES:
            problems.append(f"needs holds {n!r}, a scope: now and memory are the gold row's `scope`, never a need")
        elif n not in CAPABILITIES:
            problems.append(f"needs holds {n!r}, which is not one of {list(CAPABILITIES)}")
    if len(set(needs)) != len(needs):
        problems.append(f"needs must not repeat a capability, got {needs!r}")
    return problems


def _qid_problems(qid: object) -> list[str]:
    """A gold row's `qid` is its question's, so both rows hold the same form."""
    if isinstance(qid, str) and _QID.fullmatch(qid):
        return []
    return [f"qid must be 16 lowercase hex characters, got {qid!r}"]


def check_question(row: dict) -> list[str]:
    """The problems of a question row; empty when it is valid."""
    problems = _key_problems(row, QUESTION_KEYS)
    if problems:
        return problems
    problems += _qid_problems(row["qid"])
    when, k = row["when"], row["decision_k"]
    if when not in WHEN:
        problems.append(f"when must be one of {list(WHEN)}, got {when!r}")
    elif when == "during" and not _is_int(k):
        problems.append(f"decision_k must be an integer on a during row, got {k!r}")
    elif when == "end" and k is not None:
        problems.append(f"decision_k must be null on an end row, got {k!r}")
    for key in ("text", "template"):
        if not _is_text(row[key]):
            problems.append(f"{key} must be a non-empty string, got {row[key]!r}")
    slots = row["slots"]
    if not isinstance(slots, dict) or not all(isinstance(name, str) for name in slots):
        problems.append(f"slots must be an object with string keys, got {slots!r}")
    if row["answer_type"] not in TYPES:
        problems.append(f"answer_type must be one of {list(TYPES)}, got {row['answer_type']!r}")
    return problems + _needs_problems(row["needs"])


@dataclass(frozen=True)
class FamilySpec:
    """One question family's registry entry. The registry (`questions_text.FAMILIES`) holds them; this module only
    defines the shape and `check_family`."""

    name: str
    when: str                    # "during" or "end"
    answer_type: str
    credit: str                  # one of CREDITS
    # The capabilities a system needs to answer, checked against its channel's (`stream_protocol.provides`); the gold
    # may read more (the depth of the look window), which a system that sees through the camera does not need.
    needs: tuple[str, ...]
    scope: str                   # "now", "memory", or "evidence": now when every evidence keyframe is the asked frame k
    reference: str | None        # "path_log" for own-path families
    mapper_gold: bool            # the gold and the table systems call one rule (free space, `adjacency`): no `knowable`
    slots: tuple[str, ...]       # the slot names
    render: Callable[[dict], str]
    temporal: bool = False       # asks about change over time: the `time_needed` gate reads only these families
    # a heading family's fallback (0.11, `heading_screen`): a point is a dead band where the robot's own log already
    # points the best way. Off unless a gate asks for it.
    own_log_screen: bool = False


def check_family(spec: FamilySpec) -> list[str]:
    """The problems of a family spec; empty when it is valid. That `render` is pure and that the family's rows fit the
    spec are the compiler's checks."""
    problems = []
    if not _is_text(spec.name):
        problems.append(f"name must be a non-empty string, got {spec.name!r}")
    if spec.when not in WHEN:
        problems.append(f"when must be one of {list(WHEN)}, got {spec.when!r}")
    if spec.answer_type not in TYPES:
        problems.append(f"answer_type must be one of {list(TYPES)}, got {spec.answer_type!r}")
    if spec.credit not in CREDITS:
        problems.append(f"credit must be one of {list(CREDITS)}, got {spec.credit!r}")
    elif spec.answer_type in TYPES and spec.answer_type not in CREDIT_FITS[spec.credit]:
        problems.append(f"credit {spec.credit!r} scores {list(CREDIT_FITS[spec.credit])} answers, not "
                        f"{spec.answer_type!r}")
    if not isinstance(spec.needs, tuple):
        problems.append(f"needs must be a tuple of capabilities, got {spec.needs!r}")
    else:
        problems += _needs_problems(list(spec.needs))
    if spec.scope not in FAMILY_SCOPES:
        problems.append(f"scope must be one of {list(FAMILY_SCOPES)}, got {spec.scope!r}")
    elif spec.when == "end" and spec.scope != "memory":
        problems.append(f"scope must be 'memory' on an end family, got {spec.scope!r}: every end question is memory")
    if spec.reference not in REFERENCES:
        problems.append(f"reference must be one of {list(REFERENCES)}, got {spec.reference!r}")
    if not isinstance(spec.mapper_gold, bool):
        problems.append(f"mapper_gold must be a boolean, got {spec.mapper_gold!r}")
    if not isinstance(spec.slots, tuple) or not all(_is_text(name) for name in spec.slots):
        problems.append(f"slots must be a tuple of non-empty slot names, got {spec.slots!r}")
    elif len(set(spec.slots)) != len(spec.slots):
        problems.append(f"slots must not repeat a name, got {spec.slots!r}")
    if not callable(spec.render):
        problems.append(f"render must be callable, got {spec.render!r}")
    if not isinstance(spec.temporal, bool):
        problems.append(f"temporal must be a boolean, got {spec.temporal!r}")
    if not isinstance(spec.own_log_screen, bool):
        problems.append(f"own_log_screen must be a boolean, got {spec.own_log_screen!r}")
    elif spec.own_log_screen and spec.answer_type != "heading":
        problems.append(f"own_log_screen is a heading family's fallback, not {spec.answer_type!r}'s")
    return problems


def _positive(v: object) -> bool:
    return _is_number(v) and v > 0


def _ladder_ok(v: object) -> bool:
    return isinstance(v, list) and bool(v) and all(_positive(s) for s in v) and all(a < b for a, b in pairwise(v))


def _catastrophe_ok(v: object) -> bool:
    """A `catastrophe` tolerance (`TOLERANCES`' comment): its weight 2 or 3, a known rule, a positive margin, and the
    rule's own key: `threshold_m` only for `over` (optional there) and the `critical` points for `miss`."""
    if not isinstance(v, dict) or v.get("rule") not in CATASTROPHE_RULES or v.get("weight") not in (2, 3):
        return False
    if not _positive(v.get("margin_m")):
        return False
    if set(v) - {"weight", "rule", "margin_m", "threshold_m", "critical"}:
        return False
    if "threshold_m" in v and (v["rule"] != "over" or not _positive(v["threshold_m"])):
        return False
    critical = v.get("critical")
    if v["rule"] == "miss":
        return isinstance(critical, list) and all(
            isinstance(p, list) and len(p) == 2 and all(_is_number(c) for c in p) for p in critical)
    return critical is None


# tolerance key -> (its validator, what a valid value is called in the message)
_TOLERANCE_CHECKS: dict[str, tuple[Callable[[object], bool], str]] = {
    "ladder": (_ladder_ok, "a non-empty increasing list of positive numbers"),
    "match_m": (lambda v: _is_number(v) and v >= 0, "a number of at least 0"),
    "erode_m": (lambda v: _is_number(v) and v >= 0, "a number of at least 0"),
    "resolution_m": (_positive, "a number above 0"), "need_m": (_positive, "a number above 0"),
    "threshold_m": (_positive, "a number above 0"),
    "harm": (lambda v: v is None or v in ("over", "radial"), "null, 'over' or 'radial'"),
    "resolutions_m": (lambda v: isinstance(v, list) and all(_positive(r) for r in v),
                      "a list of numbers above 0, one per gold item"),
    "fp_weight": (lambda v: _positive(v) and v <= 1, "a number above 0 and at most 1"),
    "profile_m": (lambda v: isinstance(v, list) and bool(v) and all(r is None or _is_number(r) for r in v),
                  "a non-empty list of numbers or nulls"),   # `regret`'s runs are also at least 0 (`_graded_problems`)
    "catastrophe": (lambda v: _catastrophe_ok(v), "an object {weight, rule, margin_m[, threshold_m | critical]}"),
    "step_deg": (lambda v: _is_int(v) and v > 0 and FULL_TURN_DEG % v == 0,
                 "a whole number of degrees that divides 360"),
    "step_m": (_positive, "a number above 0"),
}


def _tolerance_problems(tolerances: object) -> list[str]:
    """The problems of each key a row's `tolerances` holds, whatever the credit: a key outside `TOLERANCES`, or a value
    that is not valid for its key. Which keys a credit needs is `_credit_problems`'."""
    if not isinstance(tolerances, dict):
        return [f"tolerances must be an object, got {tolerances!r}"]
    problems = []
    for name, value in tolerances.items():
        if name not in _TOLERANCE_CHECKS:
            problems.append(f"tolerances holds {name!r}, which is none of {list(TOLERANCES)}")
        elif not _TOLERANCE_CHECKS[name][0](value):
            problems.append(f"tolerances.{name} must be {_TOLERANCE_CHECKS[name][1]}, got {value!r}")
    return problems


def _graded_problems(credit: str, kind: str, gold: object, tolerances: dict) -> list[str]:
    """What the keys of a graded credit say of each other and of the gold: the harmful side fits the gold's type,
    `regret`'s and `route`'s profile is `360 / step_deg` long and `regret`'s runs are at least 0, `graded_set` has a
    resolution for each gold item."""
    harm = tolerances.get("harm")
    if credit == "graded" and harm not in HARM_OF[kind]:
        return [f"tolerances.harm must be one of {list(HARM_OF[kind])} on a {kind} gold, got {harm!r}"]
    step, profile = tolerances.get("step_deg"), tolerances.get("profile_m")
    if credit in ("regret", "route") and _is_int(step) and step > 0 and isinstance(profile, list) \
            and len(profile) != FULL_TURN_DEG // step:
        return [f"tolerances.profile_m must hold {FULL_TURN_DEG // step} entries (360 / step_deg), got {len(profile)}"]
    if credit == "regret" and isinstance(profile, list) and any(isinstance(r, int | float) and r < 0 for r in profile):
        return ["tolerances.profile_m must hold runs of at least 0 or nulls for credit 'regret'"]
    resolutions = tolerances.get("resolutions_m")
    if credit == "graded_set" and isinstance(gold, Positions) and isinstance(resolutions, list) \
            and len(resolutions) != len(gold.items):
        return [f"tolerances.resolutions_m must hold one resolution a gold item ({len(gold.items)}), got "
                f"{len(resolutions)}"]
    return []


def _credit_problems(credit: str, gold: object, tolerances: object) -> list[str]:
    """`credit` scores the gold's type and `tolerances` holds exactly the keys that credit reads for it (`CREDIT_KEYS`),
    each valid; a `relative` gold is above 0, since the credit divides by it. An unknown gold has no answer to score,
    so none of this applies to it."""
    if gold is None or isinstance(gold, Unknown) or not isinstance(tolerances, dict):
        return []
    kind, fits = answer_type_of(gold), CREDIT_KEYS[credit]
    if kind not in fits:
        return [f"credit {credit!r} scores {list(fits)} golds, not {kind!r}"]
    problems = []
    if credit == "relative" and isinstance(gold, Number) and not gold.value > 0:
        problems.append(f"credit relative divides by the gold, so a number gold must be above 0, got {gold.value}")
    keys, held = fits[kind], {k: v for k, v in tolerances.items() if k not in OPTIONAL_TOLERANCES}
    if set(held) != set(keys):
        missing, extra = [k for k in keys if k not in held], [k for k in held if k not in keys]
        held = f"exactly {list(keys)}" if keys else "nothing"
        return problems + [f"tolerances must hold {held} for credit {credit!r} on a {kind} gold; missing {missing}, "
                           f"unexpected {extra}"]
    return problems + _graded_problems(credit, kind, gold, tolerances)


def _keyframe_problems(row: dict) -> list[str]:
    problems = []
    evidence, entities, subjects = row["evidence_k"], row["entities"], row["subjects"]
    if not isinstance(evidence, list) or not all(_is_int(k) for k in evidence) or evidence != sorted(evidence):
        problems.append(f"evidence_k must be a sorted list of integers, got {evidence!r}")
    elif row["slice"] == "answerable" and not evidence:
        problems.append("evidence_k must name the evidence of an answerable gold, got none")
    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
        problems.append(f"entities must be a list of strings, got {entities!r}")
    elif isinstance(evidence, list) and entities and len(entities) != len(evidence):
        problems.append(f"entities must be empty or as long as evidence_k ({len(evidence)}), got {len(entities)}")
    if not isinstance(subjects, list) or not all(isinstance(s, str) for s in subjects):
        return problems + [f"subjects must be a list of strings, got {subjects!r}"]
    for s in subjects:
        named = _SUBJECT.fullmatch(s)
        if not named:
            problems.append(f"subject {s!r} must be e:<id>, d:<kind>/<attributes>, r:<region id>, a:<k>, w:<k> or "
                            f"w:<k0>-<k1>")
        elif named["k1"] is not None and int(named["k0"]) > int(named["k1"]):
            problems.append(f"subject {s!r} is an interval that ends before it starts")
    if isinstance(entities, list):
        problems += [f"entity {e!r} is not among the subjects (as e:{e})" for e in entities
                     if isinstance(e, str) and f"e:{e}" not in subjects]
    return problems


def check_gold(row: dict) -> list[str]:
    """The problems of a gold row; empty when it is valid. A `GOLD_OPTIONAL` key may be absent (earlier builds)."""
    problems = _key_problems({k: v for k, v in row.items() if k not in GOLD_OPTIONAL}
                             if isinstance(row, dict) else row, GOLD_KEYS)
    if problems:
        return problems
    if row.get("age_s") is not None and not (_is_number(row["age_s"]) and row["age_s"] >= 0):
        problems.append(f"age_s must be null or a number of at least 0, got {row['age_s']!r}")
    if row.get("age_band") is not None and row["age_band"] not in AGE_BANDS:
        problems.append(f"age_band must be null or one of {list(AGE_BANDS)}, got {row['age_band']!r}")
    problems += _qid_problems(row["qid"])
    if not _is_text(row["family"]):
        problems.append(f"family must be a non-empty string, got {row['family']!r}")
    try:
        gold = parse_gold(row["gold"])
    except ValueError as e:
        gold = None
        problems.append(str(e))
    if row["answer_class"] is not None and row["answer_class"] not in ANSWER_CLASSES:
        problems.append(f"answer_class must be null or one of {list(ANSWER_CLASSES)}, got {row['answer_class']!r}")
    elif (row["answer_class"] is not None and _is_text(row["family"])
          and pool_of(row["family"], row["answer_class"]) is None):
        pools = sorted(c for classes in BALANCE_POOLS.get(row["family"], {}).values() for c in classes)
        problems.append(f"answer_class {row['answer_class']!r} is in no balance pool of family {row['family']!r}, "
                        f"whose classes are {pools}: a family declares only the classes its draw balances")
    if row["range_m"] is not None and not (_is_number(row["range_m"]) and row["range_m"] >= 0):
        problems.append(f"range_m must be null or a number of at least 0, got {row['range_m']!r}")
    if row["answer_class"] in SLICES and row["answer_class"] != row["slice"]:
        problems.append(f"answer_class is {row['answer_class']!r} but slice is {row['slice']!r}: the two slice classes "
                        "are the slice")
    if row["slice"] not in SLICES:
        problems.append(f"slice must be one of {list(SLICES)}, got {row['slice']!r}")
    elif gold is not None and isinstance(gold, Unknown) != (row["slice"] == "unknown"):
        problems.append(f"slice is {row['slice']!r} but gold is {row['gold']!r}: the gold is unknown exactly when the "
                        "slice is unknown")
    credit, region = row["credit"], row["region"]
    if credit not in CREDITS:
        problems.append(f"credit must be one of {list(CREDITS)}, got {credit!r}")
    if credit == "region" and not is_polygon(region):
        problems.append(f"credit region needs region to be a polygon of three or more [x, y] points, got {region!r}")
    if credit != "region" and region is not None:
        problems.append(f"region must be null unless credit is region, got a region with credit {credit!r}")
    problems += _tolerance_problems(row["tolerances"])
    if credit in CREDITS:
        problems += _credit_problems(credit, gold, row["tolerances"])
    if row["metric"] not in METRICS:
        problems.append(f"metric must be null, 'xy' or 'xyz', got {row['metric']!r}")
    elif row["metric"] is None and isinstance(gold, Position):
        problems.append("metric must be 'xy' or 'xyz' on a position gold")
    if row["scope"] not in SCOPES:
        problems.append(f"scope must be one of {list(SCOPES)}, got {row['scope']!r}")
    if row["reference"] not in REFERENCES:
        problems.append(f"reference must be one of {list(REFERENCES)}, got {row['reference']!r}")
    if not isinstance(row["critical"], bool):
        problems.append(f"critical must be a boolean, got {row['critical']!r}")
    return problems + _keyframe_problems(row)
