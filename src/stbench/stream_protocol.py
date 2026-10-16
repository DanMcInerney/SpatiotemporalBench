"""The stream protocol: the protocol between the harness and a system under test, and the in-process `System` API.

One neutral module holds the channels, the messages' reply checks, the timeouts, the failure marker and the base class
every system subclasses. The harness, the scripted baselines and the Codex agent all import it, so none of them invents
its own. It imports nothing from the package, so every one of them may. The adapters (`system.CommandSystem`,
`local_system.LocalCommandSystem`, `shim.serve`) speak the messages below. No agent channel carries depth, and the depth
images of a window are served from the dataset's `private/` folder, to the gate channel `detections` alone.

Messages. JSON lines over `JsonPipe` and the shim, or method calls in process. One reply per request, no request ids.

    start    {"op": "start", "session": str, "robot": dict, "channel": str, "sends": [str],
              "workdir": str, "state_dir": str, "opts": dict}
                 -> {"ok": true}
    task     {"op": "task", "index": int, "clock": str, "gap_s": float, "brief": str}
                 -> {"ok": true}
    observe  {"op": "observe", "steps": [step]}, exactly one step: one frame
                 -> {"ok": true}                                          (the reply is the request for the next frame)
    ask      {"op": "ask", "when": "during" or "end", "now": {"k": int, "clock": str},
              "questions": [{"qid": str, "text": str, "answer_type": str}, ...]}
                 -> {"answers": [{"qid": str, "answer": dict}, ...]}
    finish   {"op": "finish"}
                 -> {"ok": true, "usage": {"models": {"<model id>": {"tokens_in": int, "tokens_cached_in": int,
                                                                     "tokens_cache_write_in": int, "tokens_out": int}},
                                           "peak_context": int, "remote_store_bytes": int}}    each value may be null
    (any, on failure)
                 -> {"error": str}

`session` is a fresh 16-hex token per run (the same value as `STBENCH_CASE_NONCE`), never the case id: a rerun of a
case gets a new namespace for remote memory, and nothing the agent sees names a dataset file. `robot` is `robot.json`'s
dict (`public/robot.json` and `public/camera.json` are also in the working folder). `sends` lists what the channel
delivers, so an external system need not infer its channel from its steps. `opts` carries constructor options for an
in-process system and `--opt` values for an external one; nothing about the world travels in it.

A step has `k` and `clock` always, the rest by channel; keys are case-relative. Every clock a system is sent (a step's,
a task's, an ask's `now.clock`) is the local ISO time `YYYY-MM-DDTHH:MM:SS.s`, with one decimal of a second always: the
case's own start clock plus the true time since its first frame, to the tenth (`case.case_clock`).

    {"k": int, "clock": str,
     "pose":   {"pos": [x, y, z], "rot": [w, x, y, z], "yaw_deg": float},      # the base
     "camera": {"pos": [x, y, z], "rot": [w, x, y, z], "pitch_deg": float},    # the camera
     "velocity": [vx, vy],            # the base's own velocity, m/s in the map frame (the controller's odometry)
     "status": {"phase": str, "task": int, <the robot's other keys>},
     "subgoal": str,                  # what the robot is doing now, in words: with `status`, on every channel
     "rgb": "inbox/rgb/<k:06d>.jpg", "depth": "inbox/depth/<k:06d>.png",
     "detections": [...],               # the perfect-detections channel's rows for this k
     "identity": {d: entity id}}        # a privileged in-process baseline only (the oracle)

`velocity` is the robot's own motion, which its controller knows (vision: what the robot knows about itself): the
displacement of the delivered base position from the previous delivered frame of the same task to this one, over the
true time between them (not the tenth-rounded clocks), to the millimetre a second. It is `[0.0, 0.0]` at a task's first
frame, which has no frame before it in its task (`case.base_velocity`, the one owner; the gold of `closest_approach`
reads the same numbers). A system need not difference two poses to know how it is moving.

Channels (`CHANNELS` below): `pose` in `sends` delivers `pose`, `camera` and `velocity`; `see` is computed (rgb or
detections); `task` messages and `robot.json` reach every channel. `rgb_pose`, the default and the only agent channel,
sends rgb, pose and status and no depth image: the agent judges distance from the picture (VISION rule 9; a camera-only
robot, 0.6.0). `detections`, the gate channel, sends depth, pose, status and detections; it is the instruments', and the
reference agent is never sent its depth. `provides(channel)` is what a question's `needs` is checked against,
so a question that needs depth is not asked on `rgb_pose`. A run whose channel answers no question of its case is
refused before `start`, and a name that is not a channel (a stale `rgb`) is refused rather than mapped to the weakest
one.

Frames: every system gets one frame per `observe`, in time order (`steps` is a list of one step), and the reply is the
request for the next: the harness sends nothing more until it has come. A `task` message comes first when the frame
begins a task. The inbox holds only the current frame's files, and still holds them during the ask that follows. No
frame and no task message marks a decision point or a task's end, so the during-on and during-off arms of the leak
check stream identical frames. A question's presence can: asks come first at the robot's decision frames
(`JobSpec.look_decisions`: before a crossing, a grasp, a doorway, a lowering), so a frame that carries one may be one
of the robot's decision moments. Its own status and subgoal already show that, and the plan, never the world, decides
where those moments fall, so a question being asked says nothing of what the world holds. (The question compiler
groups the frames into look windows, `look_windows`; a system never sees one.)

Asks: a during ask follows the frame it is about (a look point: any frame of the case, the robot's decision frames
first in the draw) and carries one question, `now` being that step; two questions at one frame are two asks in turn.
The end ask follows the last frame and carries every end question in the compiled order. No feedback is ever given;
between messages the system may do anything in its own process. A question reaches a system as `ask_view`, or as
`structured_view` for a class that sets `structured`.

Timeouts (`TIMEOUTS`, seconds): `start` 120, `task`, `observe`, a during ask and `finish` 600, the end ask 3,600, and
`CASE_CAP_S` (12 h) for the whole case, checked between messages. In-process scripted baselines are not timed.

Failure semantics. JsonPipe breaks the pipe for good after any timeout or bad line, and a late reply must never pair
with a later request, so a transport failure always ends the case. `check_reply` says whether a reply is good; the
harness acts on a bad one as follows:

    start fails (error reply, timeout, exit)                   the case ends; every question is failed
    task or observe: error, timeout, broken pipe or bad reply  the case ends; every question not yet answered is failed
    ask: timeout or broken pipe                                the case ends; this ask's questions and every later
                                                               one are failed
    ask: an {"error"} reply                                    this ask's questions are failed; the case goes on
    ask: a qid missing, answered twice, or an answer that is   that question is failed (`failed`); the others
      not an object                                            stand; the case goes on
    finish fails                                               the answers stand; usage is null; the status says so
    the 12 h cap                                               after the current message, every question not yet
                                                               answered is failed

A failed question's answer is `failed(reason)`, which parses as malformed (credit 0). A case's status is `complete` or
`ended: <reason>`.

Usage. `finish` reports what the case cost, as case totals (never per message), by the model id the system called
each model under. `tokens_in` counts every input token, the cached ones and the ones written to a cache included;
`tokens_cached_in` the ones read from a cache; `tokens_cache_write_in` the ones written to one; `tokens_out` every
output token, reasoning included. `null` says the system does not know; `{}` for `models` says it called no model.
`peak_context` is the most tokens one model request held, null when unknown. `remote_store_bytes` is the size of a
remote store at the end of the case, null when the system keeps none; any number, 0 included, says the run used one,
and `remote_memory` stamps it (such a run is scored as unverified). The harness marks the report `self_reported` and
never scores it. A cost is these tokens times the price table (`prices.py`), never a figure of the system's;
`usage_problems` says whether a report is well formed and `usage_totals` sums it over the models.
"""

from __future__ import annotations

import copy
from pathlib import Path

CAPABILITIES = ("rgb", "depth", "pose", "status", "detections", "see")   # what a question's `needs` may name
CHANNELS: dict[str, frozenset[str]] = {          # channel -> what it sends; `see` is computed (rgb or detections)
    "rgb_pose": frozenset({"rgb", "pose", "status"}),
    "detections": frozenset({"depth", "pose", "status", "detections"}),
}
DEFAULT_CHANNEL = "rgb_pose"   # the camera alone, the one channel an agent has
GATE_CHANNEL = "detections"

OPS = ("start", "task", "observe", "ask", "finish")
TIMEOUTS = {"start": 120.0, "task": 600.0, "observe": 600.0, "ask_during": 600.0, "ask_end": 3600.0,
            "finish": 600.0}   # seconds a system may take over one message
CASE_CAP_S = 12 * 3600.0   # the whole case, checked between messages; in-process scripted baselines are not timed
USAGE_KEYS = ("models", "peak_context", "remote_store_bytes")   # what `finish` reports; each may be null
MODEL_TOKEN_KEYS = ("tokens_in", "tokens_cached_in", "tokens_cache_write_in", "tokens_out")   # one model's tokens
_ASK_KEYS = ("qid", "text", "answer_type")
_STRUCTURED_KEYS = (*_ASK_KEYS, "template", "slots")


def provides(channel: str) -> frozenset[str]:
    """What a channel lets a question need: what it sends, plus `see` when it sends rgb or detections. ValueError for
    a name that is not a channel."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; the channels are {', '.join(CHANNELS)}")
    sends = CHANNELS[channel]
    return sends | {"see"} if sends & {"rgb", "detections"} else sends


def failed(reason: str) -> dict:
    """The answer of a question the harness could not get one for: scored as malformed (credit 0)."""
    return {"type": "failed", "reason": reason}


def check_reply(op: str, reply: object) -> str | None:
    """None for a good reply to `op`, else why not. `{"error": ...}` fails every op; start, task and observe need `ok`
    true; ask needs an `answers` list (each answer is checked per question by the harness); finish needs `ok` true and
    a `usage` dict. ValueError for an op not in `OPS`."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; the ops are {', '.join(OPS)}")
    if not isinstance(reply, dict):
        return f"the reply to {op!r} is not an object: {str(reply)[:200]}"
    if "error" in reply:
        return f"{op!r} failed: {reply['error']}"
    if op == "ask":
        ok = isinstance(reply.get("answers"), list)
    else:
        ok = reply.get("ok") is True and (op != "finish" or isinstance(reply.get("usage"), dict))
    return None if ok else f"an unexpected reply to {op!r}: {str(reply)[:200]}"


def ask_view(q: dict) -> dict:
    """What a system is asked: a deep copy of exactly `qid`, `text` and `answer_type`. Never a gold-side key."""
    return {key: copy.deepcopy(q[key]) for key in _ASK_KEYS}


def structured_view(q: dict) -> dict:
    """`ask_view` plus deep copies of exactly `template` and `slots`, for a class that sets `structured`."""
    return {key: copy.deepcopy(q[key]) for key in _STRUCTURED_KEYS}


class System:
    """What a memory system implements; every system, in process or external, subclasses it (or `shim.System`, its
    standard-library copy for the sandbox). The harness builds it with `state_dir` (its memory; nothing else survives)
    and `workdir` (`public/` holds what never changes, `inbox/` only the current frame's files), then calls `start`
    once, `task` before each task's first step, `observe` with each frame (a list of one step), `ask` with each
    question set, `finish` once, and `close` when it drops the instance. The messages, the step and the failure rules
    are in the module's docstring.

    The contract the harness relies on: `answer` and `ask` never change what a later `observe` or `ask` sees (there
    are no forks, so `ask` runs on the live instance); a system never receives a gold-side key, only `ask_view`, or
    `structured_view` when the class sets `structured`; and `when` is `"during"` or `"end"`. `channels` lists the
    channels it can run on, `privileged` says its steps carry `identity` (the oracle only), `reads_files` that it
    opens the picture files its steps name, and the harness reads all four from the class.
    """

    channels: tuple[str, ...] = tuple(CHANNELS)   # the channels it can run on
    privileged: bool = False                      # steps carry `identity` (the oracle only)
    structured: bool = False                      # questions carry `template` and `slots` (scripted baselines)
    reads_files: bool = True                      # it opens the pictures its steps name (False: its inbox stays empty)
    session: str | None = None                    # set by `start`
    robot: dict | None = None
    channel: str | None = None

    def __init__(self, state_dir: Path, workdir: Path, **opts):
        self.state_dir, self.workdir, self.opts = Path(state_dir), Path(workdir), opts

    @property
    def public(self) -> Path:
        return self.workdir / "public"

    def start(self, session: str, robot: dict, channel: str) -> None:
        """Called once, before anything else: the run's session token, the robot's `robot.json` and the channel."""
        self.session, self.robot, self.channel = session, robot, channel

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        """A task begins (its brief, the clock it starts at, the gap since the last task ended). Default: nothing."""

    def observe(self, steps: list[dict]) -> None:
        """One frame: a list of one step. Returning is the request for the next frame."""
        raise NotImplementedError

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        """Answers by qid for `questions`, asked at step `now["k"]` (`when` "during") or after the last (`"end"`).
        Default: `answer` for each, so a system that answers one question at a time implements only `answer`."""
        return {q["qid"]: self.answer(q) for q in questions}

    def answer(self, question: dict) -> dict:
        """One question's answer object (`answers.py`); the per-question default that `ask` calls."""
        raise NotImplementedError

    def finish(self) -> dict:
        """The run is over: the usage the system reports (keys in `USAGE_KEYS`, each may be null; the module's "Usage"
        paragraph says what each holds). Default: none."""
        return {}

    def close(self) -> None:
        """Release what the instance holds (a process, a file); the harness calls it when it drops an instance."""


class TransportError(RuntimeError):
    """The pipe to an external system broke (a timeout, an exit, a line that is not a JSON object): the case ends. An
    in-process system raises it for a failure that must end the case too (an ask that cannot be retried)."""


def timeout_key(op: str, when: str | None = None) -> str:
    """The key of `TIMEOUTS` (and of a case's per-message wall times) for a message: the op, and for `ask` also its
    `when`. ("ask", "during") is "ask_during", ("observe",) is "observe"; ValueError for anything else."""
    key = f"ask_{when}" if op == "ask" else op
    if op not in OPS or (op == "ask") != (when is not None) or key not in TIMEOUTS:
        raise ValueError(f"no timeout for op {op!r} with when {when!r}; the keys are {', '.join(TIMEOUTS)}")
    return key


def start_message(session: str, robot: dict, channel: str, workdir: str, state_dir: str, opts: dict) -> dict:
    """The `start` message of the module's table, `sends` being the channel's. ValueError for a name that is no
    channel."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; the channels are {', '.join(CHANNELS)}")
    return {"op": "start", "session": session, "robot": robot, "channel": channel,
            "sends": sorted(CHANNELS[channel]), "workdir": workdir, "state_dir": state_dir, "opts": opts}


def read_answers(questions: list[dict], reply: list | dict) -> tuple[dict[str, dict], dict[str, str]]:
    """(answers by qid, failure reason by qid) for the asked `questions`, from an `answers` list (the wire's) or a
    `{qid: answer}` dict (an in-process system's). A qid with no answer is "missing", one with two is "answered twice"
    (a list only), an answer that is not an object is "not an object"; the others stand. An answer for a qid that was
    not asked is ignored. ValueError when `reply` is neither a list nor a dict."""
    given: dict[str, list] = {}
    if isinstance(reply, dict):
        given = {qid: [answer] for qid, answer in reply.items()}
    elif isinstance(reply, list):
        for item in reply:
            if isinstance(item, dict) and isinstance(item.get("qid"), str):
                given.setdefault(item["qid"], []).append(item.get("answer"))
    else:
        raise ValueError(f"an ask reply is a list of answers or an object by qid, not {type(reply).__name__}")
    answers: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for q in questions:
        got = given.get(q["qid"], [])
        if not got:
            failures[q["qid"]] = "missing"
        elif len(got) > 1:
            failures[q["qid"]] = "answered twice"
        elif not isinstance(got[0], dict):
            failures[q["qid"]] = "not an object"
        else:
            answers[q["qid"]] = got[0]
    return answers, failures


def _count_problem(label: str, value: object) -> str | None:
    """Why `value` is not null or a non-negative integer (a bool is not one), else None."""
    ok = value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
    return None if ok else f"{label} is not null or a non-negative integer: {value!r}"


def usage_problems(usage: dict | None) -> list[str]:
    """Why `usage` is not a usage report, or [] (None, a missing report, is no problem). Keys only from
    USAGE_KEYS and the harness's "self_reported"; `models` null or an object from a non-empty model id to an object of
    exactly MODEL_TOKEN_KEYS, each a non-negative int or null (a bool is not an int), with tokens_in at least
    tokens_cached_in + tokens_cache_write_in when all three are known; `peak_context` and `remote_store_bytes` null or
    a non-negative int. One short sentence a problem, naming the key and the model."""
    if usage is None:
        return []
    if not isinstance(usage, dict):
        return [f"usage is not an object: {usage!r}"]
    problems = [f"usage has an unknown key {key!r}" for key in usage if key not in (*USAGE_KEYS, "self_reported")]
    models = usage.get("models")
    if models is not None and not isinstance(models, dict):
        problems.append(f"models is not null or an object by model id: {models!r}")
    for model, tokens in (models.items() if isinstance(models, dict) else ()):
        if not isinstance(model, str) or not model:
            problems.append(f"a model id is not a non-empty string: {model!r}")
        elif not isinstance(tokens, dict):
            problems.append(f"the tokens of model {model!r} are not an object: {tokens!r}")
        else:
            problems += [f"model {model!r} lacks {key}" for key in MODEL_TOKEN_KEYS if key not in tokens]
            problems += [f"model {model!r} has an unknown key {key!r}" for key in tokens if key not in MODEL_TOKEN_KEYS]
            counts = [_count_problem(f"{key} of model {model!r}", tokens[key]) for key in MODEL_TOKEN_KEYS
                      if key in tokens]
            problems += [problem for problem in counts if problem]
            held = [tokens.get(key) for key in ("tokens_in", "tokens_cached_in", "tokens_cache_write_in")]
            if not any(counts) and None not in held and held[0] < held[1] + held[2]:
                problems.append(f"tokens_in of model {model!r} is below tokens_cached_in + tokens_cache_write_in")
    for key in ("peak_context", "remote_store_bytes"):
        problem = _count_problem(key, usage.get(key))
        problems += [problem] if problem else []
    return problems


def usage_totals(usage: dict | None) -> dict[str, int | None]:
    """MODEL_TOKEN_KEYS summed over `models`: a key is None when the report is missing, `usage_problems` finds any
    problem with it (a report that is not well formed is never summed), `models` is null, or any model's value for it
    is null; `{}` gives zeros. `cost_usd` refuses the same reports, so the two readers agree."""
    models = usage.get("models") if usage and not usage_problems(usage) else None
    if models is None:
        return dict.fromkeys(MODEL_TOKEN_KEYS)
    counts = {key: [tokens[key] for tokens in models.values()] for key in MODEL_TOKEN_KEYS}
    return {key: None if None in found else sum(found) for key, found in counts.items()}


def remote_memory(usage: dict | None) -> bool:
    """True when the report names a remote store: `remote_store_bytes` is not null (0 included)."""
    return isinstance(usage, dict) and usage.get("remote_store_bytes") is not None
