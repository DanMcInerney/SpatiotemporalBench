"""What the tests of the subgoal share: a job that states one, made from a registered job.

`with_subgoals` replaces `job_registry.get` so that the jobs it is asked for state `subgoals` (default `stages`) and,
when given, a `ladder`; every other field is the registered job's. A timeline made by `synth` or
`conftest.script_on_map` for the vacuum (job `vacuum/1`) then writes pose rows that carry a subgoal, and everything
that reads the registry (the stream, the prior records, the Codex driver) sees the same job.
"""

from __future__ import annotations

from dataclasses import replace

from stbench import job_registry
from stbench.job_registry import JobSpec


def stages(tl, m) -> list[str]:
    """A subgoal for every keyframe that names only what the robot is doing: four texts in turn, by the keyframe."""
    return [f"cleaning the kitchen, stage {k % 4}" for k in range(tl.n_keyframes)]


def with_subgoals(monkeypatch, subgoals=stages, **over) -> JobSpec:
    """Every job the registry is asked for states `subgoals`, with the fields in `over` changed; the spec of
    `vacuum/1` so changed."""
    real = job_registry.get
    monkeypatch.setattr(job_registry, "get", lambda job_id: replace(real(job_id), subgoals=subgoals, **over))
    return job_registry.get("vacuum/1")
