"""The one loader of the answer registries: families answered by modules found through a pre-seeded list.

`table_answers.ANSWER_MODULES` and `baselines_path.PATH_MODULES` name the modules that answer the families, one module
a family group and paired one to one with `questions_text.FAMILY_MODULES`, as `job_registry.JOB_MODULES` names the jobs:
a family lane adds a module and edits no shared registry. A module that does not exist yet (the later milestones' names
cost nothing until then) is skipped, the ModuleNotFoundError naming that module; any other import error is raised. A
family answered twice raises, naming it and both claimants.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from types import ModuleType


def _import(name: str) -> ModuleType | None:
    """The package's module `name`, or None when it does not exist yet."""
    full = f"{__package__}.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name != full:
            raise
        return None


def collect(modules: Sequence[str], attr: str, taken: Iterable[str] = ()) -> dict[str, Callable]:
    """The `attr` dict (family -> answerer) of each module of `modules` that exists, merged in order. `taken` names the
    families answered elsewhere (a built-in): a module that claims one raises ValueError, as one that claims a family an
    earlier module did, naming the family and both. A module with no `attr` raises ValueError naming it."""
    claimed = dict.fromkeys(taken, "a built-in answerer")
    out: dict[str, Callable] = {}
    for name in modules:
        if (module := _import(name)) is None:
            continue
        if (found := getattr(module, attr, None)) is None:
            raise ValueError(f"{name} defines no {attr}")
        for family, answerer in found.items():
            if family in claimed:
                raise ValueError(f"the family {family!r} is answered twice: by {claimed[family]} and by {name}")
            claimed[family] = name
            out[family] = answerer
    return out
