"""Which compiled cases a command works on: `--cases` and `--select`, one owner for `stbench eval` and `stbench check`.

A command starts from the dataset's compiled cases that ask a question (`check.compiled_cases`). `--cases ID,...` keeps
those ids; `--select KEY=VALUE` keeps the cases whose `suite.case_info` KEY (`layout`, `robot`, `job` or `length`) is
VALUE: a key given twice is either value, different keys must all hold. Both together are the intersection. `Bad` (a
ValueError) names what cannot be selected; each command turns it into its own refusal, exit 2, before anything runs.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import suite
from .case import CasePaths

SELECT_KEYS = ("layout", "robot", "job", "length")


class Bad(ValueError):
    """A selection that cannot be made."""


def parse(items: Sequence[str]) -> dict[str, set[str]]:
    """`--select KEY=VALUE` (repeated): the values wanted for each key. Bad for anything else."""
    picks: dict[str, set[str]] = {}
    for item in items:
        key, equals, value = item.partition("=")
        if not equals or key not in SELECT_KEYS or not value:
            raise Bad(f"--select takes KEY=VALUE with KEY one of {', '.join(SELECT_KEYS)}, not {item!r}")
        picks.setdefault(key, set()).add(value)
    return picks


def keep(cases: list[CasePaths], ids: list[str] | None, picks: dict[str, set[str]]) -> list[CasePaths]:
    """The cases among `cases` that `ids` and `picks` keep, in their order."""
    found = cases if ids is None else [case for case in cases if case.id in set(ids)]
    return [case for case in found if all(suite.case_info(case)[key] in values for key, values in picks.items())]


def narrow(asking: list[CasePaths], ids: list[str] | None, picks: dict[str, set[str]], data) -> list[CasePaths]:
    """The cases of `asking` (the compiled cases of `data` that ask a question) that `ids` and `picks` keep. Bad for an
    id that is none of them, and when none is left."""
    if ids is not None:
        known = {case.id for case in asking}
        if unknown := [i for i in ids if i not in known]:
            raise Bad(f"--cases names {', '.join(unknown[:3])}, which {data} holds no compiled case that asks a "
                      "question for")
    if not (found := keep(asking, ids, picks)):
        raise Bad(f"no case of the {len(asking)} compiled ones matches --cases and --select")
    return found


def describe(ids: list[str] | None, picks: dict[str, set[str]]) -> str:
    """The selection in words, `job=helper/1, layout=a|b; cases 3a,9f`; empty when it selects everything."""
    parts = [f"{key}={'|'.join(sorted(values))}" for key, values in picks.items()]
    if ids is not None:
        parts.append(f"cases {','.join(ids)}")
    return ", ".join(parts)
