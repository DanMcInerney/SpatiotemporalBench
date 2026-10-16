"""The question families' registry: `FAMILIES`, `GOLDS`, `SLICED`, `COINS`, `LOOKS`, `CAPPED` and `RELAX`, built from
the family modules that exist.

A family is a `records.FamilySpec`: when it is asked, its answer type and credit, the sensors it needs, its scope, and
`render`, the one function from its slots to the question's `text`. Render purity (plan §2.5): `text ==
render(slots)` for every compiled question, and every wording choice is a slot, so a baseline that reads the structured
view learns nothing the wording does not say. No gold-side resolution (the decision's keyframe, an event's resolved
step, the task's room) is a slot.

A family module `questions_<x>` (named in `FAMILY_MODULES`; the truth side by the fence's `questions\\w*`) defines
`SPECS` (its families' specs, each passing `records.check_family`), `GOLDS` (family -> gold function) and, when it has
an end family that draws an unknown slice per moment, `SLICED`, or one whose moments are drawn by a 50% coin (a yes/no
question with a yes candidate in `Moment.answerable` and a no candidate in `Moment.unknown`), `COINS`; a family is in at
most one of the two. A during gold is `(w: World, k: int) -> Cand | None`
(None: a dead band at that point, which is then not asked) and an end gold `(w: World) -> list[Moment]`.

Four more names, each optional (0.6.0):
- `LOOKS`: during families asked at look points (`JobSpec.looks`) and not at a decision. Their gold is `(w, k) ->
  Moment | None`: at the look point `k` (any frame of the window), the moment's two pools, its `key` `str(k)`; None is
  a dead band.
  A look family may be in `SLICED` (its pools are then answerable and unknown, drawn as an end moment's are).
- `COINS`, as above, for end families.
- `CAPPED`: end families whose empty golds the draw may thin, by the job's own keep rule (`questionnaire._thinned`:
  `TaskThings.empty_keep_percent`, one per job). Their candidates all carry an `answer_class`.
- `RELAX`: end family -> `(Cand, named ids) -> Cand`, the function that makes the entities a look question names
  don't-care in that family's candidate.
Every name in `LOOKS` must be a during family of its module, and in `CAPPED` and `RELAX` an end family of it; `SLICED`
names an end family or a look family. A module
imports `questions_world`, `questions_select`, `questions_words`, `records` and neutral modules, never this module or
`questionnaire`. The names of M3's and M4's modules sit in the list already, as `job_registry.JOB_MODULES` holds jobs
not written yet: a module that does not exist is skipped, so a family lane adds files and edits no registry.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import NamedTuple

from . import questions_words
from .records import FamilySpec

FAMILY_MODULES = ("questions_space", "questions_where", "questions_self", "questions_cover", "questions_view",
                  "questions_changed", "questions_things", "questions_route", "questions_zone", "questions_adjacent",
                  "questions_change", "questions_ego", "questions_task", "questions_clear", "questions_people",
                  "questions_heading", "questions_approach", "questions_hazard", "questions_way")
DISTANCE_LADDER = questions_words.DISTANCE_LADDER  # kept for its importers


def _import(name: str):
    """The module `name` of this package, or None when it does not exist yet (ModuleNotFoundError naming that module);
    any other import error is raised."""
    module = f"{__package__}.{name}"
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return None


class Registry(NamedTuple):
    """What `_load` finds: the specs and gold functions by family, and the names of each optional hook."""

    families: dict[str, FamilySpec]
    golds: dict[str, Callable]
    sliced: frozenset[str]
    coins: frozenset[str]
    looks: frozenset[str]
    capped: frozenset[str]
    relax: dict[str, Callable]


def _names(name: str, module, attr: str, specs: dict[str, FamilySpec], allowed: dict[str, str]) -> frozenset[str]:
    """The family names the module's `attr` lists (none when it has none), each of a family of the module whose `when`
    is among `allowed` (`when` -> what that is called in the error). ValueError names the stray one."""
    names = getattr(module, attr, frozenset())
    for family in names:
        if family not in specs or specs[family].when not in allowed:
            raise ValueError(f"{name} names {family!r} in {attr}, which is not {' or '.join(allowed.values())} of the "
                             f"module")
    return frozenset(names)


def _load(modules: tuple[str, ...] = FAMILY_MODULES) -> Registry:
    """The `Registry` over the modules that exist, in the order given. ValueError names a family two specs name, a
    `GOLDS` key with no spec, a spec with no gold, a `SLICED` name that is neither an end family nor a `LOOKS` family
    of its module, a `COINS`, `CAPPED` or `RELAX` name that is not an end family of its module, a `LOOKS` name that is
    not a during family of it, and a family in both `SLICED` and `COINS`."""
    families: dict[str, FamilySpec] = {}
    golds: dict[str, Callable] = {}
    sliced: set[str] = set()
    coins: set[str] = set()
    looks: set[str] = set()
    capped: set[str] = set()
    relax: dict[str, Callable] = {}
    end, during = {"end": "an end family"}, {"during": "a during family"}
    for name in modules:
        if (module := _import(name)) is None:
            continue
        specs: dict[str, FamilySpec] = {}
        for spec in module.SPECS:
            if spec.name in families or spec.name in specs:
                raise ValueError(f"the family {spec.name!r} is named twice: {name} names it again")
            specs[spec.name] = spec
        for family in module.GOLDS:
            if family not in specs:
                raise ValueError(f"{name} has a gold function for {family!r}, which is not among its SPECS")
        for family in specs:
            if family not in module.GOLDS:
                raise ValueError(f"{name} has the spec {family!r} and no gold function")
        look_names = _names(name, module, "LOOKS", specs, during)
        slice_names = getattr(module, "SLICED", frozenset())
        for family in slice_names:
            if family not in specs or (specs[family].when != "end" and family not in look_names):
                raise ValueError(f"{name} names {family!r} in SLICED, which is not an end family or a look family of "
                                 f"the module")
        coin_names = _names(name, module, "COINS", specs, end)
        capped_names = _names(name, module, "CAPPED", specs, end)
        relax_fns = getattr(module, "RELAX", {})
        _names(name, SimpleNamespace(RELAX=relax_fns), "RELAX", specs, end)
        families.update(specs)
        golds.update(module.GOLDS)
        sliced |= set(slice_names)
        coins |= set(coin_names)
        looks |= look_names
        capped |= capped_names
        relax.update(relax_fns)
    if both := sorted(sliced & coins):
        raise ValueError(f"the families {both} are in both SLICED and COINS: a family's moments are drawn by one rule")
    return Registry(families, golds, frozenset(sliced), frozenset(coins), frozenset(looks), frozenset(capped), relax)


FAMILIES, GOLDS, SLICED, COINS, LOOKS, CAPPED, RELAX = _load()
