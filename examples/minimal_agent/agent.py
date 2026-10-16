"""The smallest agent that runs: it logs what the robot reports and says "unknown" to every question.

Each step carries the robot's clock, its base pose and its status. This agent appends them to a file in `state_dir`, the
one place the harness measures (its size is reported at the end of each task and at `finish`), and answers `unknown`
to every question: `{"type": "unknown"}` is always an allowed answer, and a wrong answer scores below it. A real agent
would look at the pictures (`step["rgb"]`: the robot has a camera and no depth sensor) and keep what it learns;
`template_agent` shows where.

`system_shim.System` is the optional base class the shim provides (a constructor that sets `state_dir`, `workdir` and
`opts`, and defaults for `start`, `task`, `ask`, `finish` and `close`); `ask` calls `answer` for each question, so
this agent defines only `observe` and `answer`. Standard library only, so it runs unchanged with `--local` and in the
Docker sandbox. Run it on one case (`DATASET/private/<case id>`):

    stbench run examples/minimal_agent/agent.py:MinimalAgent DATASET/private/<case id> --local
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from system_shim import System  # `stbench run` starts this file through the shim, which provides it
except ImportError:  # run on its own (a test of yours): the same constructor, the one thing this file needs of it

    class System:
        def __init__(self, state_dir, workdir, **opts):
            self.state_dir, self.workdir, self.opts = Path(state_dir), Path(workdir), opts


class MinimalAgent(System):
    def __init__(self, state_dir, workdir, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.steps = self.state_dir / "steps.jsonl"  # the harness gives each case an empty state_dir and a new process

    def observe(self, steps):
        """One frame, a list of one step: keep its clock, base position and status. Returning asks for the next."""
        with self.steps.open("a", encoding="utf-8") as f:
            for step in steps:
                row = {"clock": step["clock"], "base": step["pose"]["pos"], "status": step["status"]}
                f.write(json.dumps(row) + "\n")

    def answer(self, question):
        """One question, `{qid, text, answer_type}`: this agent knows nothing it could answer from."""
        return {"type": "unknown"}
