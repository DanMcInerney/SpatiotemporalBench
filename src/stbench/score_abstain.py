"""The abstention report (0.11.0, FD-0.11-design section 3.8): when a system said `unknown`, and whether it should have.

Counted from result rows alone, the way `score_stats` counts its one abstention line, but by job and family and with
Wilson intervals: the shares are counts of a few dozen questions, where the bootstrap over scripts has too few scripts
to say anything. Three shares, each a count over a count:
- **recall**: of the questions whose gold is `unknown`, the share the system answered `unknown`;
- **precision**: of the `unknown` answers, the share on a gold that is `unknown`;
- **over-abstention**: of the answerable questions, the share the system answered `unknown`.
A row was answered `unknown` when its outcome is `abstained` (an answerable gold) or `correct` on an `unknown` gold
(every other answer to an `unknown` gold is `wrong`), the test `eval_report.credit_band` applies too. The `unknown`
golds are every family's: a thing never seen, a description the world lacks and a corridor never looked at.

The design also splits the `unknown`-gold rows by whether a look-alike kind was glimpsed (its S3). That is a fact of
the visibility log, which a result row does not carry, so it is not counted here: the S4 screen
(`questions_where.glimpse_hides`) removes the rows whose glimpse shows the asked colour at the compile, and the wider
split stays a read of the visibility log by a script outside the package.

`python -m stbench.score_abstain RUN [RUN ...]` prints the models side by side (RUN: a results folder or its
`results.json`).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

Row = dict
Z = 1.959964   # the normal quantile of a 95% interval


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float] | None:
    """The Wilson score interval of `k` successes in `n` trials, None when there are no trials."""
    if n <= 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


class Share(NamedTuple):
    """`k` of `n`: a rate and its Wilson interval."""

    k: int
    n: int

    @property
    def rate(self) -> float | None:
        return None if self.n == 0 else self.k / self.n

    @property
    def ci(self) -> tuple[float, float] | None:
        return wilson(self.k, self.n)

    def text(self) -> str:
        """`0.844 [0.68, 0.93] (27/32)`; `n/a (0/0)` with no trials."""
        if self.n == 0:
            return "n/a (0/0)"
        lo, hi = self.ci  # type: ignore[misc]
        return f"{self.k / self.n:.3f} [{lo:.2f}, {hi:.2f}] ({self.k}/{self.n})"


class Summary(NamedTuple):
    recall: Share      # `unknown` answered / questions whose gold is `unknown`
    precision: Share   # `unknown` answered on an `unknown` gold / `unknown` answered
    over: Share        # `unknown` answered on an answerable gold / answerable questions


def said_unknown(row: Row) -> bool:
    """Whether the row's answer was `unknown`: an abstention on an answerable gold, or the right answer to an `unknown`
    one."""
    return row["outcome"] == "abstained" or (row["slice"] == "unknown" and row["outcome"] == "correct")


def summary(rows: Iterable[Row]) -> Summary:
    """The three shares of `rows`."""
    rows = list(rows)
    gold = [r for r in rows if r["slice"] == "unknown"]
    answerable = [r for r in rows if r["slice"] == "answerable"]
    hit, over = sum(said_unknown(r) for r in gold), sum(said_unknown(r) for r in answerable)
    return Summary(Share(hit, len(gold)), Share(hit, hit + over), Share(over, len(answerable)))


def groups(rows: Sequence[Row]) -> list[tuple[str, Summary]]:
    """The summary of all `rows`, then of each job, then of each (job, family): `(label, Summary)` in that order, jobs
    and families sorted. A label is `all`, the job (`helper/1`) or `job family`."""
    out = [("all", summary(rows))]
    for job in sorted({r["job"] for r in rows}):
        mine = [r for r in rows if r["job"] == job]
        out.append((job, summary(mine)))
        out += [(f"{job} {family}", summary(r for r in mine if r["family"] == family))
                for family in sorted({r["family"] for r in mine})]
    return out


def by_model(runs: Mapping[str, Sequence[Row]]) -> dict[str, Summary]:
    """Each system's summary over all its rows, in the order given."""
    return {name: summary(rows) for name, rows in runs.items()}


def model_table(runs: Mapping[str, Sequence[Row]]) -> list[str]:
    """The systems side by side as a markdown table: recall, precision and the `unknown` rate on answerable ones."""
    lines = ["| system | recall | precision | `unknown` on answerable |", "| --- | --- | --- | --- |"]
    lines += [f"| {name} | {s.recall.text()} | {s.precision.text()} | {s.over.text()} |"
              for name, s in by_model(runs).items()]
    return lines


def main(argv: Sequence[str]) -> int:
    from .eval_results import load

    if not argv:
        print("usage: python -m stbench.score_abstain RUN [RUN ...]   (a results folder or its results.json)",
              file=sys.stderr)
        return 2
    runs = {}
    for arg in argv:
        path = Path(arg)
        doc = load(path / "results.json" if path.is_dir() else path)
        runs[doc["system"] if doc["system"] not in runs else str(path)] = doc["rows"]
    print("\n".join(model_table(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
