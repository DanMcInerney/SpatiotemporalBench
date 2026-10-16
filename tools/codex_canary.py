"""The Codex canary (M5-d): does the reference agent carry anything from one case's thread into another's?

`tests/test_codex_isolation.py` shows, on a fake Codex, that the harness starts every case on a fresh thread that holds
nothing of another case. What no flag can show is whether Codex itself (its session files, its `memories` feature, its
daemon) lets one thread's content reach another's. This script asks the real one, with a token planted in one case and
a probe that every case ends with. M6's leak check needs the answer: anything Codex carried from the on arm to the off
arm would bias the leak estimate toward zero. A release tool, not package code (the sdist and the wheel leave
`tools/` out); run it on every new Codex version:

    uv run python tools/codex_canary.py --data DATASET --cases CID_A,CID_B --model MODEL --reasoning EFFORT \\
        --out OUT [--dry-run]

The transport is the reference agent's (`cmd_codex.make_codex`, `prepare`: a personal `AGENTS.md` is refused, as in a
real run); the argv, the tools and the prompt are the current transport's. A case is one long exec in which the model
pulls the frames (`codex_system`). Three runs, one after the other:

- A, a case of `--data` (`stream.run_case` on a `CodexSystem` whose first task message ends with the plant, "Remember
  this code word; you may be asked for it later in this job: <token>"), then the probe as one more exec on A's thread
  (a resume of it, once the job is over);
- B, another case (a different script) after A, with no plant, and the same probe at its end;
- C, one start exec outside any case whose whole prompt is the probe, on a fresh thread.

The token is `CANARY-` and eight hex digits, drawn fresh for each attempt and written to `canary.json` (it is not a
secret). The during asks are off unless `--during on`: nothing here depends on them.

The verdict is fixed here, before any exec (`verdict`). A text holds the token when its hex digits stand alone in it
(any case, with or without `CANARY-`). `fail` when B's or C's reply holds the token (or the token of an attempt
before), or any prompt, tool reply or model message of B's journal does. Else `pass` when A's reply holds it (the
positive control: the probe can detect recall), and `inconclusive` when it does not, whereupon the canary runs once
more with a new token, the plant repeated before A's last frame, and a second `inconclusive` is reported as it is and
never passed. A case or probe that cannot finish (a quota, a timeout) is `error`: no verdict, nothing passed.

`--out` gets `attempt-N/A`, `B` and `C` (each a `codex.calls.jsonl`; A's and B's hold the planted text and the cases'
questions, so they stay outside the dataset) and `canary.json`: the model, the effort, Codex's version, the token, the
case ids, each case's status, each probe's reply, the verdict, the wall time, the attempts before the last and
`features`, the lines of `codex features list` run under an empty temp `CODEX_HOME` (free: no model call), which put on
record the defaults of this Codex version (`memories`, which `codex_exec.DISABLED` does not name, among them). A Codex
run cannot be replayed: a folder that holds a `canary.json` is refused. `--dry-run` runs nothing on Codex (it checks
the model and the effort with `codex debug models`, which calls no model): it prints the token, the planted first task
message, the probe and the cases, over a system that sends nothing and fails every question.

Exit codes: 0 `pass`, 1 `fail`, 2 `inconclusive`, `error` or a refusal before anything ran.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from stbench import case as case_mod
from stbench import cmd_codex, llm_render
from stbench.agent_tools import ContextTooLong
from stbench.case import CasePaths
from stbench.codex_exec import Codex, CodexError, CodexTimeout
from stbench.codex_feed import Feed, Item
from stbench.codex_live import Live
from stbench.codex_log import Journal
from stbench.codex_system import CodexSystem
from stbench.safecopy import remove_tree
from stbench.stream import run_case
from stbench.stream_protocol import DEFAULT_CHANNEL, TIMEOUTS, System, failed, timeout_key

PREFIX = "CANARY-"
PLANT = "Remember this code word; you may be asked for it later in this job: {token}."
PROBE = ("Before this job began, or earlier in it, were you given a code word? If so, reply with the code word only; "
         "if not, reply NONE.")
JOURNAL = "codex.calls.jsonl"


def holds(text: str, token: str) -> bool:
    """True when the token's hex digits stand alone in `text` (any case): not inside a longer run of hex digits, so a
    hash in a journal never matches by chance."""
    digits = token.removeprefix(PREFIX).lower()
    return re.search(rf"(?<![0-9a-f]){digits}(?![0-9a-f])", text.lower()) is not None


def verdict(token: str, a_reply: str, b_reply: str, c_reply: str, b_journal: str, earlier: tuple[str, ...] = ()) -> str:
    """`fail`, `pass` or `inconclusive`, by the rules in the module docstring; `earlier` are the tokens of the attempts
    before this one, which B and C must not hold either."""
    if any(holds(text, t) for t in (token, *earlier) for text in (b_reply, c_reply, b_journal)):
        return "fail"
    return "pass" if holds(a_reply, token) else "inconclusive"


class Probed(CodexSystem):
    """A `CodexSystem` that plants `token` in its first task message (when it has one), repeats it as one more message
    before the last frame (`repeat`, with `last_k` the case's last step), and ends with one probe exec on its own
    thread whose reply it keeps in `reply`. `planted` is the first task message as sent."""

    def __init__(self, state_dir, workdir, *, token: str | None = None, repeat: bool = False, last_k: int = 0, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.token, self.repeat, self.last_k = token, repeat, last_k
        self.reply: str | None = None
        self.planted: str | None = None

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        if self.token and self.execs == 0:
            brief = f"{brief}\n{PLANT.format(token=self.token)}"
            self.planted = llm_render.task_message(index, clock, gap_s, brief)
        super().task(index, clock, gap_s, brief)

    def observe(self, steps: list[dict]) -> None:
        if self.token and self.repeat and steps[-1]["k"] == self.last_k:
            self._deliver(Item("task", PLANT.format(token=self.token)), "observe", timeout_key("observe"))
        super().observe(steps)

    def finish(self) -> dict:
        self._end()
        self.reply = "\n".join(self._more(PROBE, TIMEOUTS["task"]))
        return self._usage()


class Dry(System):
    """The system of `--dry-run`: it sends nothing to Codex, keeps the first task message as the canary would plant it
    (`planted`), counts the messages it is sent (`sent`) and fails every question."""

    def __init__(self, state_dir, workdir, *, token: str | None = None, **opts):
        super().__init__(state_dir, workdir, **opts)
        self.token, self.planted, self.sent = token, None, 0

    def task(self, index: int, clock: str, gap_s: float, brief: str) -> None:
        if self.token and self.planted is None:
            self.planted = llm_render.task_message(index, clock, gap_s, f"{brief}\n{PLANT.format(token=self.token)}")
        self.sent += 1

    def observe(self, steps: list[dict]) -> None:
        self.sent += 1

    def ask(self, when: str, now: dict, questions: list[dict]) -> dict[str, dict]:
        self.sent += 1
        return {q["qid"]: failed("dry run") for q in questions}


def play(codex, case: CasePaths, folder: Path, *, token: str | None, repeat: bool, during: bool):
    """One case on a `Probed` system: (the `CaseRun`, the system). Its journal is `folder/codex.calls.jsonl`."""
    journal = Journal(folder / JOURNAL)
    rec = case_mod.read_case(case)
    made: list[Probed] = []

    def make(state: Path, work: Path) -> Probed:
        made.append(Probed(state, work, codex=codex, journal=journal, token=token, repeat=repeat,
                           last_k=rec.k1 - rec.k0 - 1))
        return made[0]

    return run_case(make, case, channel=DEFAULT_CHANNEL, during=during), made[0]


def bare(codex: Codex, folder: Path) -> str:
    """C: one start exec whose whole prompt is the probe, on a feed that has nothing to give; its reply."""
    journal = Journal(folder / JOURNAL)
    feed = Feed(journal.note)
    feed.supply(Item("end", llm_render.END_MESSAGE))  # for a model that pulls: the probe is all it has been asked
    journal.note({"event": "exec_started", "kind": "start", "base": None, "prompt": PROBE, "attempt": 1})
    live = Live(codex, kind="start", thread=None, prompt=PROBE, server=feed.server_config(), wake=feed.wake)
    try:
        if not live.wait(TIMEOUTS["task"]):
            live.kill()
            raise CodexTimeout(f"the bare exec ran past {TIMEOUTS['task']:g} s and was killed")
        result = live.result()
    finally:
        live.close()
        feed.close()
    journal.exec(session="canary-c", kind="start", base=None, result=result)
    return "\n".join(result.messages)


def features(codex: Codex) -> list[str] | str:
    """The lines of `codex features list` run under an empty temp `CODEX_HOME` (free: no model call, no login), so the
    defaults of this Codex version are on record beside the verdict, whatever the user's own home holds. The reason, a
    string, when the command fails: the canary goes on without it."""
    home = Path(tempfile.mkdtemp(prefix="stbench-canary-home-"))
    try:
        done = subprocess.run([*codex.command, "features", "list"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=60,
                              env={**os.environ, **codex.env, "CODEX_HOME": str(home)})
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"cannot run `codex features list`: {error}"
    finally:
        remove_tree(home)
    if done.returncode:
        return f"`codex features list` exited {done.returncode}: {done.stderr.strip()[-300:]}"
    return done.stdout.splitlines()


def attempt(codex, a: CasePaths, b: CasePaths, out: Path, *, repeat: bool, during: bool,
            earlier: tuple[str, ...] = ()) -> dict:
    """A, then B, then C, on a new token, under `out` (`earlier`: the tokens of the attempts before); the attempt's
    record (its `verdict` is `error`, with `error` saying why, when a case or the probe could not finish)."""
    token = PREFIX + secrets.token_hex(4).upper()
    found: dict = {"token": token, "repeated_plant": repeat, "status": {}, "replies": {}}
    for name, case, planted in (("A", a, token), ("B", b, None)):
        run, system = play(codex, case, out / name, token=planted, repeat=repeat and planted is not None,
                           during=during)
        found["status"][name], found["replies"][name] = run.status, system.reply
        if run.status != "complete" or system.reply is None:
            return {**found, "verdict": "error", "error": f"case {name} did not finish: {run.status}"}
    try:
        found["replies"]["C"] = bare(codex, out / "C")
    except (CodexError, ContextTooLong) as error:
        return {**found, "verdict": "error", "error": f"the bare exec C failed: {error}"}
    replies = found["replies"]
    journal = (out / "B" / JOURNAL).read_text(encoding="utf-8")
    return {**found, "verdict": verdict(token, replies["A"], replies["B"], replies["C"], journal, earlier)}


def describe(case: CasePaths) -> str:
    rec = case_mod.read_case(case)
    return f"{case.id} (script {case.script.id}, tasks {rec.a}-{rec.b}, {rec.k1 - rec.k0} steps)"


def dry_run(codex: Codex, a: CasePaths, b: CasePaths, during: bool) -> None:
    token = PREFIX + secrets.token_hex(4).upper()
    print(f"model {codex.model} at {codex.reasoning}, {codex.version}")
    print(f"token: {token}")
    print(f"probe: {PROBE}")
    for name, case, planted in (("A", a, token), ("B", b, None)):
        made: list[Dry] = []

        def make(state: Path, work: Path, planted=planted, made=made) -> Dry:
            made.append(Dry(state, work, token=planted))
            return made[0]

        run = run_case(make, case, channel=DEFAULT_CHANNEL, during=during)
        print(f"case {name}: {describe(case)}: {run.status}, {made[0].sent} messages sent (a live ask may pull again)")
        print(f"planted first task message of {name}:" if planted else f"{name} has no plant")
        if made[0].planted:
            print("  " + made[0].planted.replace("\n", "\n  "))
    print("C: one start exec of the probe alone, on a fresh thread")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="plant a token in one Codex case and probe another case and a bare thread")
    p.add_argument("--data", type=Path, required=True, help="the dataset: DATA/private/<case id> holds each case")
    p.add_argument("--cases", required=True, help="two case ids on two different scripts: A (planted), then B")
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning", default=cmd_codex.DEFAULT_REASONING)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--during", choices=("on", "off"), default="off")
    p.add_argument("--dry-run", action="store_true", help="print what would be sent and run no exec")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    try:
        ids = args.cases.split(",")
        if len(ids) != 2 or ids[0] == ids[1]:
            raise ValueError("--cases takes two different case ids, A (planted) and B, as A,B")
        a, b = (CasePaths.from_id(args.data, cid) for cid in ids)
        if a.script.id == b.script.id:
            raise ValueError(f"cases {a.id} and {b.id} are on one script ({a.script.id}): choose two scripts")
        if not args.dry_run and (args.out / "canary.json").exists():
            raise ValueError(f"{args.out / 'canary.json'} exists: a Codex run cannot be replayed, so choose another "
                             "--out")
        codex = cmd_codex.make_codex(args.model, args.reasoning, None)
        codex.prepare()
    except (CodexError, ValueError, KeyError, OSError) as error:
        print(f"codex_canary: {error}", file=sys.stderr)
        return 2
    try:
        if args.dry_run:
            dry_run(codex, a, b, args.during == "on")
            return 0
        return live(codex, a, b, args)
    except ValueError as error:  # `run_case` refused a case before anything was sent: nothing ran, no record is kept
        print(f"codex_canary: {error}", file=sys.stderr)
        return 2
    finally:
        codex.close()


def live(codex: Codex, a: CasePaths, b: CasePaths, args: argparse.Namespace) -> int:
    """The attempts (a second only after an `inconclusive` one), then `canary.json` and the exit code."""
    began = time.monotonic()
    done: list[dict] = []
    for n in (1, 2):
        try:
            found = attempt(codex, a, b, args.out / f"attempt-{n}", repeat=n == 2, during=args.during == "on",
                            earlier=tuple(d["token"] for d in done if d["token"]))
        except OSError as error:  # a file of the case missing, or a folder that cannot be written
            found = {"token": None, "status": {}, "replies": {}, "verdict": "error", "error": f"OSError: {error}"}
        done.append(found)
        print(f"attempt {n}: {found['verdict']}" + (f" ({found['error']})" if "error" in found else ""))
        if found["verdict"] != "inconclusive":
            break
    last = done[-1]
    record = {"model": codex.model, "reasoning": codex.reasoning, "codex": codex.version,
              "cases": {"A": a.id, "B": b.id}, "probe": PROBE, **last,
              "earlier_attempts": done[:-1], "features": features(codex),
              "wall_s": round(time.monotonic() - began, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "canary.json").write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    for name, reply in last["replies"].items():
        print(f"probe reply {name}: {reply!r}")
    print(f"canary: {last['verdict']} (token {last['token']}, {args.out / 'canary.json'})")
    return {"pass": 0, "fail": 1}.get(last["verdict"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
