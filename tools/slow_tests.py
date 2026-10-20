# SpatiotemporalBench: writes the list of slow test modules that CI's fast set leaves out (a development tool).
# Copyright (c) 2026 the SpatiotemporalBench authors. Released under the MIT licence (see LICENSE).
"""The slow test modules, read from a timed run of the whole suite.

    uv run pytest -q -n auto -m "not docker and not blender" --junitxml=junit.xml
    uv run python tools/slow_tests.py junit.xml [--threshold-s 3] [--write]

A module is slow when its tests' times sum to `--threshold-s` seconds or more. A module is the unit, not a test: a
module-scoped fixture's cost is charged to the module's first test, so leaving that one test out would leave the cost.
For the same reason the modules that share an xdist group (the `@group` that `--dist loadgroup` adds to a test's name)
are one unit, timed together and slow together: one worker builds their shared cache once and its cost lands on
whichever module runs first, so leaving that one out moves the cost to the next.

The list is `tests/slow_modules.txt`, one module name a line (`test_check_scoped`), sorted; `tests/conftest.py` marks
every test of a listed module `slow`. CI's pull requests and pushes to main run `-m "not slow"`; its nightly run and a
manual run with `full` run everything. Without `--write` the list is printed and nothing is written.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

LIST = Path(__file__).resolve().parent.parent / "tests" / "slow_modules.txt"


def module_of(classname: str) -> str:
    """The test module of a JUnit `classname`: `tests.test_x` or `tests.test_x.TestY` gives `test_x`."""
    parts = classname.split(".")
    return parts[1] if parts[0] == "tests" and len(parts) > 1 else parts[0]


def module_times(junit: Path) -> Counter:
    """Seconds of test time per module, summed over the JUnit file's test cases."""
    times: Counter = Counter()
    for case in ET.parse(junit).iter("testcase"):
        times[module_of(case.get("classname", ""))] += float(case.get("time") or 0.0)
    return times


def module_groups(junit: Path) -> dict[str, set[str]]:
    """The modules of each xdist group, read from the `@group` at the end of a test case's name."""
    groups: dict[str, set[str]] = {}
    for case in ET.parse(junit).iter("testcase"):
        name = case.get("name", "")
        if "@" in name:
            groups.setdefault(name.rsplit("@", 1)[1], set()).add(module_of(case.get("classname", "")))
    return groups


def slow_modules(times: Counter, threshold_s: float, groups: dict[str, set[str]] | None = None) -> list[str]:
    """The modules whose summed time is at least `threshold_s`, sorted by name; the modules that share a group of
    `groups` are summed as one unit and are slow together."""
    unit = {m: {m} for m in times}
    for members in (groups or {}).values():
        merged = set().union(*(unit.get(m, {m}) for m in members))
        for m in merged:
            unit[m] = merged
    return sorted(m for m in times if sum(times[n] for n in unit[m]) >= threshold_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("junit", type=Path, help="the JUnit XML of a timed run of the whole suite")
    ap.add_argument("--threshold-s", type=float, default=3.0, help="a module this slow or slower is slow (default 3)")
    ap.add_argument("--write", action="store_true", help=f"write {LIST.name} (else print)")
    args = ap.parse_args(argv)
    times = module_times(args.junit)
    slow = slow_modules(times, args.threshold_s, module_groups(args.junit))
    kept = sum(t for m, t in times.items() if m not in slow)
    print(f"{len(slow)} of {len(times)} modules are slow ({sum(times[m] for m in slow):.0f} s); the fast set is "
          f"{kept:.0f} s of test time", file=sys.stderr)
    text = "".join(f"{m}\n" for m in slow)
    if args.write:
        LIST.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
