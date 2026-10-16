"""The stream's and the brief's output in a second process (`determinism_kit.py`: nothing is recorded, the two are
compared).

The stream's covers its constants (the ops, the channels, the timeouts, the cap, the usage keys and the
keys of one model's tokens) and what four runs of one fixed synthetic case send a system: the agents' channel, the gate
channel, a privileged system on it and a structured one (the messages with the robot and the brief's words left out,
which the robot's and the brief's own tests cover). The brief's covers its constants and every text `compose` writes on
a fixed fake job, for every domain, channel and variant. Both are built on the frozen vacuum of
`test_determinism_records.py` and a fixed case id, so a robot tune moves neither.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from determinism_kit import digest
from stbench import brief, robots
from stbench import case as case_mod
from stbench import stream_protocol as sp
from stbench.look_windows import LOOK_WINDOW
from stbench.timeline_ego import TaskPlan
from stream_kit import Recorder, case_on, during_q, end_q, fake_job, play
from synth import HOME, Synth
from test_determinism_records import FROZEN

CASE_ID = "0123456789abcdef"  # the id the fixture's case claims, so its start clock (drawn from the id) is fixed
K, L = HOME.rooms


class Privileged(Recorder):
    privileged = True


class Structured(Recorder):
    structured = True


def told(log: list) -> list:
    """What a system was sent, without what other tests cover: a start's channel (not the robot or the session) and a
    task's index, clock and gap (not its brief)."""
    out = []
    for op, *args, _held in log:
        out.append([op, args[2]] if op == "start" else [op, *args[:3]] if op == "task" else [op, *args])
    return out


def stream_output(dataset: Path) -> str:
    """The digest of the stream: its constants and the messages of four runs of a fixed two-task case."""
    synth = Synth(tasks=[[(K, 10)], [(L, 4)]], decisions=[(3, "before_doorway"), (11, "before_doorway")])
    synth.item("E01", "mug", {"color": "red"}, [(0, None, (2.0, 1.0, 0.9))])
    synth.item("E02", "book", {"color": "blue"}, [(0, None, (6.0, 1.0, 0.9))])
    synth.see(range(0, 14), "E01")
    synth.see(range(8, 14), "E02")
    questions = [during_q(1, 3), during_q(2, 11), end_q(3), end_q(4)]
    runs = {}
    with mock.patch.dict(robots.ROBOTS, {"vacuum": FROZEN}), mock.patch.object(
            case_mod, "claim_id", lambda dataset, entry: CASE_ID):
        case = case_on(dataset, synth, questions, b=1, length="medium")
        for name, system, channel in (("agents", Recorder, "rgb_pose"), ("gate", Recorder, "detections"),
                                      ("privileged", Privileged, "detections"),
                                      ("structured", Structured, "rgb_pose")):
            runs[name] = told(play(case, system, channel=channel)[1])
    constants = {"ops": sp.OPS, "look_window": LOOK_WINDOW, "channels": {name: sorted(sends) for name, sends in
                                                                  sp.CHANNELS.items()},
                 "default": sp.DEFAULT_CHANNEL, "gate": sp.GATE_CHANNEL, "capabilities": sp.CAPABILITIES,
                 "timeouts": sp.TIMEOUTS, "cap_s": sp.CASE_CAP_S, "usage_keys": sp.USAGE_KEYS,
                 "model_token_keys": sp.MODEL_TOKEN_KEYS}
    return digest({"constants": constants, "runs": runs})


def brief_output() -> str:
    """The digest of the brief: its constants and every text `compose` writes for a fixed fake job's three tasks (the
    job asks `out_of_view` and `rear_clear` at decision points and the two look families at look points, so the during
    sentence is in the texts)."""
    plans = [TaskPlan("tidy", {"room": "kitchen"}), TaskPlan("check", {"room": "hall"}),
             TaskPlan("tidy", {"room": "living room"})]
    job = fake_job(decisions={"before_doorway": ("out_of_view", "rear_clear")}, looks=("ego_now", "ego_mem"))
    texts = {f"{domain}/{channel}/{variant}": brief.compose(plans, job=job, robot={"name": "robot vacuum"},
                                                            domain=domain, channel=channel, variant=variant)
             for domain in brief.DOMAINS for channel in sp.CHANNELS for variant in brief.VARIANTS}
    constants = {"domains": brief.DOMAINS, "phrases": brief.DURING_PHRASES, "sentence": brief.DURING_SENTENCE,
                 "neutral": brief.NEUTRAL_ASSIGNMENT, "variants": brief.VARIANTS, "sensors": brief.SENSOR_WORDS,
                 "own": brief.OWN_WORDS}
    return digest({"constants": constants, "texts": texts})


def outputs(dataset: Path) -> dict[str, str]:
    return {"stream": stream_output(dataset), "brief": brief_output()}


def test_stream_and_brief_outputs_are_deterministic(tmp_path):
    """Contract: the stream's output (its constants and the messages of a fixed synthetic case on both channels, for a
    privileged and a structured system) and the brief's (its constants and every text on a fixed fake job) are the same
    in a second process (another `PYTHONHASHSEED`) and differ from each other. Regression caught: an output that
    depends on dict or set order or on the process."""
    here = outputs(tmp_path / "here")
    out = subprocess.run([sys.executable, __file__, str(tmp_path / "there")], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": "12345"}).stdout
    assert json.loads(out) == here
    assert len(here) == 2 and len(set(here.values())) == 2


if __name__ == "__main__":  # the second process of the test above
    print(json.dumps(outputs(Path(sys.argv[1]))))
