"""The sandbox module: what it will and will not put in front of a container, and when it says the sandbox cannot run.
Everything here but the last test needs no Docker; the container tests that observe the isolation live with
`CommandSystem` (test_command_system.py)."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from links import dir_link
from stbench import sandbox
from stbench.case import PRIVATE, CasePaths, ScriptPaths, index_dir
from stbench.safecopy import UnsafeTree

REPO = Path(__file__).resolve().parents[1]  # worked out here, not read from the code under test
SYS_SOURCE = b"class Sys:\n    pass\n"


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """A temporary folder of this test's own for `program()` to copy into, so its leftovers can be counted."""
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def system(folder: Path, **extra: bytes) -> Path:
    """A system folder holding sys.py and the given files; returns sys.py."""
    folder.mkdir(parents=True)
    (folder / "sys.py").write_bytes(SYS_SOURCE)
    for name, data in extra.items():
        (folder / name).write_bytes(data)
    return folder / "sys.py"


def leftovers(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("stbench-program-*"))


def test_the_container_never_pulls_and_runs_the_image(tmp_path):
    """Contract (VISION rule 6: asset downloads need `--yes`): a container is started from the local
    image, with `--pull never` and `--rm`. Regression caught: a run that silently pulls (a download) or runs an image
    other than the one the isolation tests ran on. (What the other flags do is observed from inside a container by the
    container tests.)"""
    prog = sandbox.Program(tmp_path / "program", "sys.py", "Sys")
    args = sandbox.container_args("docker", "stbench-x", tmp_path / "state", tmp_path / "work", prog)
    assert args[args.index("--pull") + 1] == "never"
    assert args[args.index("python") - 1] == sandbox.IMAGE
    assert "--rm" in args


def test_a_system_folder_near_the_repo_the_dataset_or_the_temp_root_is_refused(tmp_path, temp_root, monkeypatch):
    """Contract (VISION rule 6, V0 §9): the system's folder is copied into the sandbox, so it must be nowhere near
    the repository, the dataset or the folder the run folders live in, and must not hold a file or folder named like
    one of the dataset's private ones (a copy of `private/` or a part of it); a refused folder leaves no copy behind.
    Regression caught: copying the repository, the gold or other runs' state into the sandbox, or a refusal that
    leaves a half-made mount source in the temporary folder."""
    # Were the temp-root check missing, copying `top` into the temporary folder inside it would recurse without end;
    # a small size limit makes that fail the test (UnsafeTree, not ValueError) instead of hanging it.
    monkeypatch.setattr(sandbox, "CODE_LIMIT", 10_000)
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    inside_data = system(data / "sub" / "system")
    deny_inside = system(tmp_path / "deny_me")
    outer = system(tmp_path / "outer")
    (tmp_path / "outer" / "data").mkdir()
    top = system(tmp_path / "top")
    (tmp_path / "top" / "tmp").mkdir()
    truthy = system(tmp_path / "truthy", **{"gold.jsonl": b"{}"})
    nested = system(tmp_path / "nested")
    (tmp_path / "nested" / "sub").mkdir()
    (tmp_path / "nested" / "sub" / "Timeline.JSON").write_bytes(b"{}")
    folders = {}
    for name in ("private", "index", "sub/Instances"):  # folders of the dataset's private area, in any depth or case
        folders[name] = system(tmp_path / f"holds-{name.replace('/', '-')}")
        (folders[name].parent / name).mkdir(parents=True)

    refused = [  # (what is refused, file, class, deny, temp root, a word of the message)
        ("a file of the repository", REPO / "src" / "stbench" / "shim.py", "System", [], temp_root, "repository"),
        ("a folder inside the dataset", inside_data, "Sys", [data], temp_root, "dataset"),
        ("the dataset's own folder", deny_inside, "Sys", [tmp_path / "deny_me"], temp_root, "dataset"),
        ("a folder that contains the dataset", outer, "Sys", [tmp_path / "outer" / "data"], temp_root, "dataset"),
        ("a folder that contains the temporary folder", top, "Sys", [], tmp_path / "top" / "tmp", "temporary"),
        ("a folder holding gold.jsonl", truthy, "Sys", [], temp_root, "gold.jsonl"),
        ("a nested private name in another case", nested, "Sys", [], temp_root, "Timeline.JSON"),
        ("a folder named private", folders["private"], "Sys", [], temp_root, "private"),
        ("a folder named index", folders["index"], "Sys", [], temp_root, "index"),
        ("a nested folder named Instances", folders["sub/Instances"], "Sys", [], temp_root, "Instances"),
    ]
    for what, file, cls, deny, root, word in refused:
        monkeypatch.setattr(tempfile, "tempdir", str(root))
        with pytest.raises(ValueError, match=re.escape(word)), sandbox.program(file, cls, deny=deny):
            pytest.fail(f"{what} was accepted")
        assert leftovers(root) == [], what


def test_truth_files_cover_the_private_layout():
    """Contract (VISION rule 6, V0 §9): a system's folder holding a name from the dataset's private area is refused, so
    the list of names must cover that area as the dataset layout defines it (`ScriptPaths` and `CasePaths`, `index_dir`
    and the folder's own name), which it cannot import (the baselines import `sandbox`). Regression caught: a private
    file or folder added to either class of the layout and not to `TRUTH_FILES`, which a copy of the dataset inside a
    system's folder would then carry into the sandbox unnoticed (`render_log.json`, which holds the render seed, was
    one; `case.json` is the case's)."""
    script = ScriptPaths("dataset", "s")
    case = CasePaths(script, "c")
    private = {(type(paths).__name__, name): getattr(paths, name) for paths in (script, case)
               for name, prop in vars(type(paths)).items()
               if isinstance(prop, property) and paths.private in getattr(paths, name).parents}
    # the scan sees them, so the rule below is not vacuous
    assert {name for cls, name in private if cls == "ScriptPaths"} == {
        "timeline", "depth_dir", "instances_dir", "instance_map", "visibility", "perception_ids", "render_log"}
    assert {name for cls, name in private if cls == "CasePaths"} == {"record", "questions", "gold"}
    # `depth` is the one private name that is no truth name: the sensor's frames are what the `detections` gate channel
    # is sent (`inbox/depth/`), and a copy of `private/` is caught by the truth files beside it
    names = {path.name for path in private.values()} | {PRIVATE, index_dir("dataset").name}
    assert names - {"depth"} <= sandbox.TRUTH_FILES and "depth" not in sandbox.TRUTH_FILES


def test_program_runs_only_an_existing_python_file_and_an_identifier_as_the_class(tmp_path, temp_root):
    """Contract (VISION rule 6): `program` copies and mounts only an existing `.py` file and runs a class named by
    an identifier (the class name reaches the container's command line). Regression caught: a missing file, a
    non-Python file or a class name with a shell-looking tail being carried into a container command."""
    file = system(tmp_path / "ok")
    other = tmp_path / "ok" / "notes.txt"
    other.write_text("x", encoding="utf-8")
    for bad, cls in [(tmp_path / "ok" / "missing.py", "Sys"), (other, "Sys"), (file, "Sys; rm -rf /"), (file, "")]:
        with pytest.raises(ValueError), sandbox.program(bad, cls):
            pytest.fail(f"{bad} {cls!r} was accepted")
    assert leftovers(temp_root) == []


def test_program_copies_the_systems_folder_and_the_shim_and_removes_them_after(tmp_path, temp_root):
    """Contract (V0 §9): the sandbox mounts a copy of the system's own folder (so it can import its sibling modules)
    and the shim, both outside the repository, and the copy is gone when the run ends. Regression caught: a system
    that cannot import its own helper, a shim that differs from `shim.py`, a mount source inside the repository or
    one left behind in the temporary folder."""
    file = system(tmp_path / "system", **{"helper.py": b"X = 1\n"})
    (tmp_path / "system" / "pkg").mkdir()
    (tmp_path / "system" / "pkg" / "__init__.py").write_bytes(b"")
    with sandbox.program(file, "Sys", deny=[tmp_path / "data"]) as prog:
        assert (prog.entry, prog.cls) == ("sys.py", "Sys")
        assert prog.root.parent == temp_root and prog.root.name.startswith("stbench-program-")
        assert not prog.root.is_relative_to(REPO)
        assert sorted(p.name for p in prog.code_dir.iterdir()) == ["helper.py", "pkg", "sys.py"]
        assert (prog.code_dir / "sys.py").read_bytes() == SYS_SOURCE
        assert (prog.code_dir / "helper.py").read_bytes() == b"X = 1\n"
        assert [p.name for p in prog.shim_dir.iterdir()] == ["system_shim.py"]
        assert (prog.shim_dir / "system_shim.py").read_bytes() == (REPO / "src" / "stbench" / "shim.py").read_bytes()
        assert (tmp_path / "system" / "sys.py").read_bytes() == SYS_SOURCE  # the original is untouched
    assert leftovers(temp_root) == [] and (tmp_path / "system" / "helper.py").exists()


def test_program_refuses_links_and_oversized_folders_and_leaves_no_copy(tmp_path, temp_root, monkeypatch):
    """Contract (VISION rule 6): the system's folder is copied following nothing and within a size limit, as a
    snapshot is (`safecopy.copy_tree`). Regression caught: a link in the folder (to a private folder, say) being
    followed into the copy the container reads, or a huge folder filling the disk."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "gold.jsonl").write_text("secret", encoding="utf-8")
    linked = system(tmp_path / "linked")
    dir_link(tmp_path / "linked" / "peek", outside)
    with pytest.raises(UnsafeTree) as refused, sandbox.program(linked, "Sys"):
        pytest.fail("a link was accepted")
    assert refused.value.offenders == ["peek"]
    assert (outside / "gold.jsonl").read_text(encoding="utf-8") == "secret"

    monkeypatch.setattr(sandbox, "CODE_LIMIT", 100)
    big = system(tmp_path / "big", **{"blob.bin": bytes(200)})
    with pytest.raises(UnsafeTree, match="more than 100 bytes"), sandbox.program(big, "Sys"):
        pytest.fail("an oversized folder was accepted")
    assert leftovers(temp_root) == []


def test_mount_sources_must_stay_out_of_the_repo_and_the_dataset_and_out_of_the_mount_syntax(
        tmp_path, temp_root):
    """Contract (VISION rule 6): a container's mount sources are checked just before it starts, links resolved: none
    may be, contain or sit inside the repository or the dataset (the container reads their host paths), and none may
    carry a character that would change the `--mount` option. Regression caught: a state or work folder reached
    through a link into the dataset, a snapshot made inside the repository, or a comma in a path adding a mount
    option."""
    data = tmp_path / "data"
    data.mkdir()
    file = system(tmp_path / "system")
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    (run / "work").mkdir()
    sneaky = tmp_path / "sneaky"
    dir_link(sneaky, data)
    with sandbox.program(file, "Sys", deny=[data]) as prog:
        sandbox.check_sources(prog, run / "state", run / "work")  # the normal case passes
        for source, why in [(data, "the dataset"), (data / "scripts" / "x", "inside the dataset"),
                            (tmp_path, "contains the dataset"), (sneaky / "state", "a link into the dataset"),
                            (REPO, "the repository"), (REPO / "src", "inside the repository"),
                            (tmp_path / "a,readonly", "a comma"), (tmp_path / 'a"b', "a quote"),
                            (tmp_path / "a\nb", "a newline"), (Path("relative") / "state", "a relative path")]:
            try:
                sandbox.check_sources(prog, run / "state", source)
            except ValueError:
                continue
            pytest.fail(f"{why} was accepted as a mount source")


def test_docker_commands_run_by_the_resolved_path_and_only_where_there_is_docker(monkeypatch):
    """Contract (VISION rule 6): every `docker` command the harness runs (pause, unpause, remove, inspect) goes by the
    path `require()` resolved, never a bare name left to the platform's search, and raises `SandboxUnavailable` where
    there is no Docker. Regression caught: a bare `docker` in a command, which on Windows is looked up in the working
    folder first, so a planted `docker.exe` there would run in place of Docker. (Where the path comes from is the
    next test.)"""
    _docker_says(monkeypatch)
    monkeypatch.setattr(sandbox, "DOCKER", "docker-d")
    monkeypatch.setattr(sandbox, "IMAGE", "image-d")
    ran = []
    inner = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: (ran.append(argv), inner(argv, **kw))[1])
    sandbox.docker("ps")
    assert ran[-1] == ["/usr/bin/docker", "ps"]
    _docker_says(monkeypatch, which=None)
    monkeypatch.setattr(sandbox, "DOCKER", "docker-e")
    with pytest.raises(sandbox.SandboxUnavailable, match="--local"):
        sandbox.docker("ps")


def test_docker_is_looked_for_only_in_the_absolute_folders_of_path_never_in_the_working_folder(tmp_path, monkeypatch):
    """Contract (VISION rule 6): the `docker` that runs is found in a folder of PATH that is an absolute path and is not
    the working folder, in PATHEXT order on Windows, and nowhere else; with no such folder holding it, `require` raises
    `SandboxUnavailable` and never falls back to the bare name. Regression caught: `shutil.which`, which on Windows
    searches the working folder first (unless NoDefaultCurrentDirectoryInExePath is set) and elsewhere reads an empty or
    relative PATH entry as the working folder, so a `docker.exe` planted where the harness runs (a system's folder, the
    folder of a dataset) ran in place of Docker, outside the sandbox. The planted files are named as PATHEXT lists them;
    nothing is executed (the probe is stubbed)."""
    work, tools = tmp_path / "work", tmp_path / "tools"
    (work / "bin").mkdir(parents=True)
    tools.mkdir()
    exts = [".com", ".exe", ".bat", ".cmd"] if os.name == "nt" else []
    for folder in (work, work / "bin"):  # planted in the working folder and in a folder below it
        for name in ("docker", *(f"docker{e}" for e in exts)):
            (folder / name).write_bytes(b"planted")
            os.chmod(folder / name, 0o755)
    real = tools / ("docker.exe" if exts else "docker")
    real.write_bytes(b"docker")
    os.chmod(real, 0o755)
    monkeypatch.chdir(work)
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setattr(sandbox, "problem", lambda docker, image: None)  # nothing is run: the path is the question
    monkeypatch.setattr(sandbox, "DOCKER", "docker")
    unsafe = ["", ".", "bin", str(work)]  # empty; relative, so the working folder and one below it; spelled out
    monkeypatch.setenv("PATH", os.pathsep.join([*unsafe, str(tools)]))
    assert os.path.normcase(sandbox.require()) == os.path.normcase(str(real))
    monkeypatch.setenv("PATH", os.pathsep.join(unsafe))
    with pytest.raises(sandbox.SandboxUnavailable, match=r"(?s)`docker` command is not on PATH.*--local"):
        sandbox.require()


@pytest.mark.parametrize("name", ["my-container", "stbench", "stbench-", "stbench-a b", "stbench--rm", "x-stbench-a",
                                  "stbench-a*", "stbench-a\nb", ""])
def test_remove_refuses_a_container_this_harness_did_not_start(name, monkeypatch):
    """Contract (HANDOFF §4: only stop what you started): `remove` acts on a name this harness makes
    (`stbench-` and a plain tail) and on nothing else, and never by pattern. Regression caught: a pattern, an empty
    name or another project's container name reaching `docker rm -f` on a machine that runs other work."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("docker was run for a foreign name"))
    with pytest.raises(ValueError):
        sandbox.remove(name)
    assert re.fullmatch(r"stbench-[0-9a-f]{32}", sandbox.container_name())  # the names it does make are accepted


def _docker_says(monkeypatch, *, which="/usr/bin/docker", version=(0, "linux\n", ""), image=0, hang=False):
    """Replace the machine's Docker with scripted answers."""
    monkeypatch.setattr(sandbox, "locate", lambda command: which)

    def run(argv, **kw):
        if argv[1] == "version":
            if hang:
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            return subprocess.CompletedProcess(argv, version[0], version[1], version[2])
        return subprocess.CompletedProcess(argv, image, "sha256:abc" if image == 0 else "", "")

    monkeypatch.setattr(subprocess, "run", run)


@pytest.mark.parametrize("case, script, words", [
    ("no docker command", {"which": None}, ["`docker no docker command`", "not on PATH"]),
    ("daemon down", {"version": (1, "", "Cannot connect to the Docker daemon at npipe")}, ["daemon", "Cannot connect"]),
    ("daemon hangs", {"hang": True}, ["daemon", "timed out"]),
    ("windows containers", {"version": (0, "windows\n", "")}, ["Linux containers", "windows"]),
    ("image absent", {"image": 1}, ["image image absent", "never pulls", "docker pull"]),
])
def test_problem_says_why_the_sandbox_cannot_run(case, script, words, monkeypatch):
    """Contract (VISION rule 6; Docker is the verified mode): `problem` names what is missing (the command, a
    reachable Linux daemon, the image) so the author can fix it, and it answers None only when all three are
    there. Regression caught: a missing Docker reported as a traceback, a Windows-containers daemon passing as
    usable, or an absent image left to be pulled by the first `docker run`."""
    _docker_says(monkeypatch, **script)
    reason = sandbox.problem(f"docker {case}", f"image {case}")  # a pair of its own: the answer is cached per pair
    assert reason is not None and all(word in reason for word in words), case


def test_problem_is_none_when_docker_and_the_image_are_there_and_require_gives_the_command(monkeypatch):
    """Contract (VISION rule 6): with a reachable Linux Docker and the image, `require` returns the command to run;
    without, it raises `SandboxUnavailable` pointing at `--local`, reading the module's `DOCKER` and
    `IMAGE` at call time. Regression caught: `require` caching a stale answer, or falling back to running anywhere."""
    _docker_says(monkeypatch)
    monkeypatch.setattr(sandbox, "DOCKER", "docker-b")
    monkeypatch.setattr(sandbox, "IMAGE", "image-b")
    assert sandbox.problem("docker-b", "image-b") is None
    assert sandbox.require() == "/usr/bin/docker"
    _docker_says(monkeypatch, which=None)
    monkeypatch.setattr(sandbox, "DOCKER", "docker-c")
    with pytest.raises(sandbox.SandboxUnavailable, match=r"(?s)`docker-c`.*--local"):
        sandbox.require()


@pytest.mark.docker
def test_remove_takes_a_running_or_paused_container_by_name_and_is_quiet_about_one_that_is_gone():
    """Contract (HANDOFF §4, V0 §9): the harness removes a hung or frozen container by its own name (a checkpoint
    pauses the live container, and a timeout can fall while it is paused) and removing one twice is harmless.
    Regression caught: a timeout that leaves a paused container running, or a teardown that raises because the
    container is already gone."""
    name = sandbox.container_name()
    try:
        sandbox.docker("run", "-d", "--rm", "--pull", "never", "--network", "none", "--read-only", "--name", name,
                       sandbox.IMAGE, "sleep", "120")
        assert sandbox.exists(name)
        sandbox.docker("pause", name)
        sandbox.remove(name)
        deadline = time.monotonic() + 15
        while sandbox.exists(name) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not sandbox.exists(name)
        sandbox.remove(name)  # already gone: quiet
    finally:
        sandbox.remove(name)
