"""The Docker sandbox an external system runs in (VISION rule 6).

One container per system instance, from a local image that is never pulled. It sees exactly four mounts: its
state directory (read-write), the working folder with `public/` and `inbox/` (read-only), the shim (read-only) and a
copy of the system's own folder (read-only), all from temporary folders outside the repository and the dataset (the
container can read its mount sources' host paths in /proc/self/mountinfo). Root file system read-only, no network,
user 65534, no capabilities, no new privileges, a private /tmp, memory and process limits. Docker is the verified
mode: without Docker and the image, `require` raises and says so. `--local` (`local_system.py`) runs an agent without
this sandbox, unsandboxed and stamped, and never goes through here.

This module builds the command line and prepares the mount sources; `system.CommandSystem` starts the container. It
removes a container only by the name of one this harness started (`NAME_PREFIX`), never by pattern or label.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .safecopy import copy_tree, long_path, remove_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "python:3.12-slim"
DOCKER = "docker"
STATE, WORK, SHIM, CODE = "/sandbox/state", "/sandbox/work", "/sandbox/shim", "/sandbox/code"
SHIM_FILE = "system_shim.py"
SHIM_SOURCE = Path(__file__).with_name("shim.py")
MEMORY, PIDS, TMP_SIZE = "4g", 512, "1g"
NAME_PREFIX = "stbench-"
CODE_LIMIT = 2**30  # bytes of a system's folder the sandbox will copy
PROBE_TIMEOUT = 30.0  # seconds `docker version` and `docker image inspect` get
TIMED_OUT = 124  # the return code `docker()` reports for a command that ran out of time
WIN_PATHEXT = ".COM;.EXE;.BAT;.CMD"  # what `locate` tries on Windows when PATHEXT is unset
# Everything the dataset's private/ folder holds, and its own name (case.ScriptPaths, case.CasePaths and case.index_dir
# are the layout; spelled out here because the baselines import this module and may not import the layout's modules,
# and a test checks it covers them, except `depth/`: the `detections` inbox calls its own folder of depth images by that
# name, so it cannot be a truth name). A system's folder holding any of these is refused, as a copy of the private
# folder would be. That is a guard on a mistake, not the guarantee: a verified container mounts only temporary copies
# of the allowlisted files, and the dataset is a denied mount source, so `private/` (depth included) is never reachable.
TRUTH_FILES = frozenset({"gold.jsonl", "questions.jsonl", "timeline.json", "case.json", "visibility.jsonl",
                         "perception_ids.jsonl", "instance_map.json", "render_log.json", "instances", "index",
                         "private"})
_OWN_NAME = re.compile(re.escape(NAME_PREFIX) + r"[0-9A-Za-z][0-9A-Za-z_.-]*")
_RESOLVED: dict[tuple, str] = {}  # `require`'s answers that were a path, by what they were looked up under


class SandboxUnavailable(RuntimeError):
    """Docker, its daemon or the image is missing: an external system cannot run here."""


@dataclass(frozen=True)
class Program:
    """A system ready to mount: `root/shim/system_shim.py` and `root/code/` (a copy of the system's folder)."""

    root: Path
    entry: str  # the system's file name inside code/
    cls: str  # its class name
    deny: tuple[str, ...] = ()  # resolved paths no mount source may overlap: the repository and the dataset
    shim_dir = property(lambda self: self.root / "shim")
    code_dir = property(lambda self: self.root / "code")


# -- paths ----------------------------------------------------------------------------------------------------------
def _real(path: str | os.PathLike) -> str:
    """`path` with links resolved, absolute and case-folded where the file system is, so strings compare."""
    return os.path.normcase(os.path.realpath(path))


def _inside(inner: str, outer: str) -> bool:
    """`inner` is `outer` or below it (both from `_real`; different drives never overlap)."""
    return inner == outer or inner.startswith(outer.rstrip(os.sep) + os.sep)


def _overlap(a: str, b: str) -> bool:
    return _inside(a, b) or _inside(b, a)


# -- whether the sandbox can run here -------------------------------------------------------------------------------
def locate(command: str) -> str | None:
    """The absolute path of the executable `command`, or None. It is looked for in the folders of PATH that are
    absolute paths and are not the working folder, in order (with PATHEXT on Windows), and nowhere else. Not
    `shutil.which`: that looks in the working folder first on Windows (unless NoDefaultCurrentDirectoryInExePath is
    set) and reads an empty or relative PATH entry as the working folder elsewhere, so a `docker.exe` planted where
    the harness runs (a system's folder, say) would run in place of Docker, outside the sandbox. A `command` with a
    directory part is looked for in that folder alone, and only if it is absolute."""
    dirname, name = os.path.split(command)
    folders = [dirname] if dirname else os.environ.get("PATH", "").split(os.pathsep)
    names = [name]
    if os.name == "nt":
        exts = [e for e in (os.environ.get("PATHEXT") or WIN_PATHEXT).split(os.pathsep) if e]
        names = [name + e for e in exts]
        if name.upper().endswith(tuple(e.upper() for e in exts)):
            names.insert(0, name)
    here = _real(os.getcwd())
    for folder in folders:
        if not os.path.isabs(folder) or _real(folder) == here:
            continue
        for candidate in (os.path.join(folder, n) for n in names):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _run(exe: str, args: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess:
    argv = [exe, *args]
    try:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, TIMED_OUT, "", f"timed out after {timeout:g} s")


def _absent(docker: str) -> str:
    return f"the `{docker}` command is not on PATH"


@functools.cache
def problem(docker: str, image: str) -> str | None:
    """Why the sandbox cannot run here, or None. Checked once per (docker, image) per process."""
    exe = locate(docker)
    if exe is None:
        return _absent(docker)
    done = _run(exe, ("version", "--format", "{{.Server.Os}}"), PROBE_TIMEOUT)
    if done.returncode != 0:
        return f"the Docker daemon is not reachable: {done.stderr.strip()[-300:]}"
    if done.stdout.strip() != "linux":
        return f"Docker runs {done.stdout.strip() or 'unknown'} containers; switch it to Linux containers"
    if _run(exe, ("image", "inspect", "--format", "{{.Id}}", image), PROBE_TIMEOUT).returncode != 0:
        return f"the image {image} is not on this machine, and the harness never pulls (docker pull {image})"
    return None


def require() -> str:
    """The docker executable's absolute path (`locate`), or `SandboxUnavailable`. Reads `DOCKER` and `IMAGE` at call
    time. Never a bare name: the platform's own search of a bare name includes the working folder. A path that was
    found, and the daemon and image behind it (`problem`), are resolved once per process and per (`DOCKER`, `IMAGE`,
    PATH, PATHEXT, working folder): every docker command of a checkpoint window then costs no search of PATH and no
    probe, only the command (a failure is never kept: it is looked for again)."""
    key = (DOCKER, IMAGE, os.environ.get("PATH"), os.environ.get("PATHEXT"), os.getcwd())
    if (known := _RESOLVED.get(key)) is not None:
        return known
    exe = locate(DOCKER)
    reason = problem(DOCKER, IMAGE) if exe else _absent(DOCKER)
    if reason:
        raise SandboxUnavailable(f"an external system runs in the Docker sandbox, the verified mode, and {reason}. "
                                 "--local runs it without the sandbox: unsandboxed, not a verified score.")
    _RESOLVED[key] = exe
    return exe


# -- the container --------------------------------------------------------------------------------------------------
def container_name() -> str:
    return NAME_PREFIX + uuid.uuid4().hex


def container_args(docker: str, name: str, state: Path, work: Path, program: Program, *,
                   memory: str = MEMORY) -> list[str]:
    """The `docker run` command line of one instance. Pure: it builds the list and starts nothing. `memory` is the
    instance's memory limit (and its swap limit: none); it changes nothing the instance can reach: no mount, network or
    privilege depends on it."""

    def bind(source: Path, target: str, readonly: bool) -> list[str]:
        return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]

    return [docker, "run", "--rm", "-i", "--name", name, "--pull", "never",
            "--network", "none", "--read-only", "--user", "65534:65534",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", memory, "--memory-swap", memory, "--pids-limit", str(PIDS),
            "--tmpfs", f"/tmp:rw,nosuid,nodev,size={TMP_SIZE}",
            "--log-driver", "none", "--workdir", "/tmp", "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
            *bind(state, STATE, False), *bind(work, WORK, True),
            *bind(program.shim_dir, SHIM, True), *bind(program.code_dir, CODE, True),
            IMAGE, "python", "-I", "-B", f"{SHIM}/{SHIM_FILE}", f"{CODE}/{program.entry}", program.cls]


def check_sources(program: Program, *sources: Path) -> None:
    """ValueError unless every mount source (the given ones, the shim and the code folder) is an absolute path free
    of `,` `"` and newlines (the --mount syntax) that neither contains nor sits inside any path in `program.deny`."""
    for source in (*sources, program.shim_dir, program.code_dir):
        if not Path(source).is_absolute() or any(c in str(source) for c in ',"\r\n'):
            raise ValueError(f"mount source {source} is not an absolute path free of commas, quotes and newlines")
        real = _real(source)
        if clash := next((d for d in program.deny if _overlap(real, d)), None):
            raise ValueError(f"mount source {source} overlaps {clash}, which the sandbox must not reveal")


@contextmanager
def program(file: Path, cls: str, deny: Iterable[Path] = ()) -> Iterator[Program]:
    """Copy the shim and FILE's folder into a new `tempfile.mkdtemp(prefix="stbench-program-")`, yield the `Program`,
    and remove the copy afterwards. ValueError, before copying: FILE not an existing `.py`; `cls` not an identifier;
    a folder that is, contains or sits inside the repository or a `deny` path; a folder that contains the temp root
    (where the run folders live). After copying (the copy is removed): a link or special file, or more than
    `CODE_LIMIT` bytes (`UnsafeTree`), or an entry named in `TRUTH_FILES` (ValueError). Needs no Docker."""
    path = Path(file)
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"{file} is not an existing Python file (FILE.py)")
    if not cls.isidentifier():
        raise ValueError(f"{cls!r} is not a class name")
    folder = _real(path.parent)
    denied = tuple(_real(d) for d in (REPO_ROOT, *deny))
    if clash := next((d for d in denied if _overlap(folder, d)), None):
        raise ValueError(f"the system's folder {path.parent} must not be, contain or sit inside {clash}: "
                         "the repository and the dataset stay out of the sandbox")
    temp = _real(tempfile.gettempdir())
    if _inside(temp, folder):
        raise ValueError(f"the system's folder {path.parent} contains the temporary folder {temp}, where the run "
                         "folders live; move the system into a folder of its own")
    root = Path(tempfile.mkdtemp(prefix="stbench-program-"))
    try:
        prog = Program(root, path.name, cls, denied)
        prog.shim_dir.mkdir()
        shutil.copyfile(SHIM_SOURCE, prog.shim_dir / SHIM_FILE)
        copy_tree(path.resolve().parent, prog.code_dir, max_bytes=CODE_LIMIT)
        top = long_path(prog.code_dir)
        truth = sorted(Path(here, n).relative_to(top).as_posix()
                       for here, dirs, files in os.walk(top) for n in (*dirs, *files) if n.lower() in TRUTH_FILES)
        if truth:
            raise ValueError(f"the system's folder holds files named like the benchmark's private ones: {truth}")
        yield prog
    finally:
        remove_tree(root)


def docker(*args: str, check: bool = True, timeout: float = 60) -> subprocess.CompletedProcess:
    """`docker <args>` with captured text output, run by the path `require()` resolved, never by a bare name left to
    the platform's own search (`SandboxUnavailable` where there is no Docker). A command that runs out of time reports
    `TIMED_OUT`. RuntimeError with the command's stderr when `check` and it failed."""
    done = _run(require(), args, timeout)
    if check and done.returncode != 0:
        raise RuntimeError(f"`docker {' '.join(args)}` failed ({done.returncode}): {done.stderr.strip()[-500:]}")
    return done


def remove(name: str) -> None:
    """`docker rm -f <name>` for one of this harness's own containers (ValueError if the name is not one it makes);
    quiet if the container is already gone or being removed."""
    if not _OWN_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a container this harness started (its names begin with {NAME_PREFIX!r})")
    done = docker("rm", "-f", name, check=False)
    gone = ("No such container", "already in progress")
    if done.returncode != 0 and not any(text in done.stderr for text in gone):
        raise RuntimeError(f"`docker rm -f {name}` failed ({done.returncode}): {done.stderr.strip()[-500:]}")


def exists(name: str) -> bool:
    """`docker container inspect <name>` succeeds."""
    return docker("container", "inspect", name, check=False, timeout=PROBE_TIMEOUT).returncode == 0
