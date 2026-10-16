"""Shared by test_local_system.py and test_template_agent.py: the example agent's path and small agents written as
source text (standard library only, as the shim runs them)."""

from __future__ import annotations

from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "examples" / "template_agent" / "agent.py"

# Says which process it is and where it runs; the answer is any JSON (the shim does not look at it).
WHERE = r'''
import os
import sys


class Where:
    def __init__(self, state_dir, workdir, **opts):
        self.opts, self.paths, self.started = opts, [str(state_dir), str(workdir)], None

    def start(self, session, robot, channel):
        self.started = [session, robot, channel]

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "unknown", "pid": os.getpid(), "cwd": os.getcwd(), "opts": self.opts, "paths": self.paths,
                "started": self.started, "flags": [sys.flags.isolated, sys.flags.dont_write_bytecode],
                "shim": sys.modules["system_shim"].__file__}
'''

# Dies in observe, saying so on stderr.
CRASH = r'''
import os
import sys


class Crash:
    def __init__(self, state_dir, workdir, **opts):
        pass

    def observe(self, steps):
        sys.stderr.write("goodbye from the agent\n")
        sys.stderr.flush()
        os._exit(3)

    def answer(self, question):
        return {"type": "unknown"}
'''

# Ticks into state/beat.txt from a thread once observed, and never answers: if the harness leaves it running after a
# timeout, the file keeps growing.
BEAT = r'''
import threading
import time
from pathlib import Path


class Beat:
    def __init__(self, state_dir, workdir, **opts):
        self.beat = Path(state_dir) / "beat.txt"

    def tick(self):
        while True:
            with open(self.beat, "a") as f:
                f.write("x")
            time.sleep(0.05)

    def observe(self, steps):
        threading.Thread(target=self.tick, daemon=True).start()

    def answer(self, question):
        time.sleep(3600)
'''
