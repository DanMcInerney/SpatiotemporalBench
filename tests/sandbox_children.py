"""Test children for the container tests (test_command_system.py), and the helpers around them.

A child is a tiny system written as source text, copied into a container with `sandbox.program` and run by the shim
there. It uses the standard library only (the container has nothing else) and, where it reports what it can observe,
reports it as plain JSON through its answers or its state directory. Lane E's CLI test reuses `ECHO`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from stbench import sandbox

# /proc/self/mountinfo writes a space (and a few other characters) in a path as an octal escape such as \040.
UNESCAPE = r'''
import re


def unescape(text):
    return re.sub(r"\\(\d{3})", lambda m: chr(int(m.group(1), 8)), text)
'''

# What a child can observe of its sandbox, as a dict: the `report()` of PROBE. The truth file names are spelled out
# here, not read from `stbench.sandbox`, so the test does not take its expectation from the code it checks.
PROBE = UNESCAPE + r'''
import errno, importlib.util, json, os, socket, sys
from pathlib import Path

NAMES = {"gold.jsonl", "questions.jsonl", "timeline.json", "case.json", "visibility.jsonl", "perception_ids.jsonl",
         "instance_map.json", "protocol.json"}
WRITES = ["/sandbox/work/public/x", "/sandbox/work/public/robot.json", "/sandbox/work/inbox/x", "/sandbox/work/x",
          "/sandbox/x", "/sandbox/shim/x", "/sandbox/code/x", "/sandbox/state/../x", "/x", "/usr/x", "/etc/x",
          "/sandbox/state/probe-ok", "/tmp/probe-ok"]


def files(top):
    return sorted(p.relative_to(top).as_posix() for p in Path(top).rglob("*") if p.is_file())


def writes():
    out = {}
    for path in WRITES:
        try:
            with open(path, "a") as f:
                f.write("x")
            out[path] = "ok"
        except OSError as e:
            out[path] = errno.errorcode.get(e.errno, str(e.errno))
    for path in ("/sandbox/state/probe-ok", "/tmp/probe-ok"):
        if out[path] == "ok":
            os.unlink(path)
    return out


def report(steps=()):
    r = {"spec": importlib.util.find_spec("stbench") is not None, "sys_path": list(sys.path), "uid": os.getuid(),
         "gid": os.getgid(), "env": dict(os.environ), "cwd": os.getcwd()}
    try:
        import stbench  # noqa: F401
        r["import_error"] = None
    except Exception as e:
        r["import_error"] = type(e).__name__
    r["cmdline"] = open("/proc/1/cmdline", "rb").read().decode().split("\0")[:-1]
    r["pids"] = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
    status = {}
    for line in open("/proc/self/status"):
        key, _, value = line.partition(":")
        if key in ("CapEff", "CapBnd", "NoNewPrivs", "Seccomp"):
            status[key] = value.strip()
    r["status"] = status
    r["cgroup"] = {n: open("/sys/fs/cgroup/" + n).read().strip() for n in ("memory.max", "pids.max")}
    r["ifaces"] = [line.split(":")[0].strip() for line in open("/proc/net/dev").read().splitlines()[2:]]
    try:
        socket.getaddrinfo("example.com", 443)
        r["dns"] = "resolved"
    except OSError as e:
        r["dns"] = type(e).__name__
    try:
        socket.create_connection(("1.1.1.1", 53), 2).close()
        r["connect"] = "connected"
    except OSError as e:
        r["connect"] = errno.errorcode.get(e.errno, str(e))
    r["docker_sock"] = os.path.exists("/var/run/docker.sock")
    r["mountinfo"] = open("/proc/self/mountinfo").read()
    r["binds"] = []
    for line in r["mountinfo"].splitlines():
        fields = line.split()
        if unescape(fields[4]).startswith("/sandbox"):
            r["binds"].append([unescape(fields[4]), unescape(fields[3]), fields[5]])
    r["hits"] = []
    for top, dirs, names in os.walk("/"):
        if top == "/":
            dirs[:] = [d for d in dirs if d not in ("proc", "sys", "dev")]
        r["hits"] += [os.path.join(top, n) for n in dirs + names if n in NAMES or n.startswith("stbench")]
    r["work_files"] = files("/sandbox/work")
    r["shim_files"] = files("/sandbox/shim")
    r["code_files"] = files("/sandbox/code")
    r["step_files"] = [s[kind] for s in steps for kind in ("depth", "rgb") if kind in s]
    r["writes"] = writes()
    return r


class Probe:
    def __init__(self, state_dir, workdir, **opts):
        self.state = Path(state_dir)

    def observe(self, steps):  # it reports into its state at each observe, and again, as `fork`, when it is asked
        (self.state / f"live-{steps[0]['k']:04d}.json").write_text(json.dumps(report(steps)))

    def answer(self, question):
        live = [json.loads(p.read_text()) for p in sorted(self.state.glob("live-*.json"))]
        return {"type": "unknown", "fork": report(), "live": live}
'''

HANG = r'''
import time


class Hang:
    def __init__(self, state_dir, workdir, **opts):
        pass

    def observe(self, steps):
        time.sleep(3600)

    def answer(self, question):
        return {"type": "unknown"}
'''

BOOM = r'''
import sys

sys.stderr.write("boom\n")
sys.exit(3)
'''

ECHO = r'''
class Echo:
    def __init__(self, state_dir, workdir, **opts):
        pass

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "unknown"}
'''


def write_child(folder: Path, name: str, source: str) -> Path:
    """The child `source` as `folder/name` (the folder is made); returns the file."""
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / name
    file.write_text(source, encoding="utf-8")
    return file


def host_form(path) -> str:
    """A host path as a container shows it in /proc/self/mountinfo (resolved, POSIX slashes, no drive letter), in lower
    case, for looking for it in text whatever case it was written in."""
    text = Path(path).resolve().as_posix()
    return (text[2:] if text[1:2] == ":" else text).lower()


def gone(name: str, within: float = 10.0) -> bool:
    """The container is no longer known to Docker, within `within` seconds."""
    deadline = time.monotonic() + within
    while sandbox.exists(name) and time.monotonic() < deadline:
        time.sleep(0.2)
    return not sandbox.exists(name)


def call_within(seconds: float, fn: Callable):
    """`fn()`'s result or exception, or a test failure (not a hang) if it has not returned after `seconds`."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001  (handed back to the test thread)
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(f"no return within {seconds} s: the deadline did not fire")
    if "error" in box:
        raise box["error"]
    return box["value"]
