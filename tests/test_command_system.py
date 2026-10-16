"""`CommandSystem` in the Docker sandbox (VISION rule 6, V0 §9): what a system under test can and cannot reach, and how
its timeouts and failures behave. The container tests (marker `docker`) run small children inside real containers and
read what the children observed; they skip when Docker or the sandbox image is absent, and never pull. The protocol
replies are tested without Docker in test_shim.py, the deadline mechanics in test_jsonpipe.py and the copy in
test_safecopy.py."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from sandbox_children import BOOM, ECHO, HANG, PROBE, call_within, gone, host_form, write_child
from stbench import sandbox
from stbench.stream import run_case
from stbench.system import CommandSystem, SystemTimeout, timeouts_of
from stream_kit import check_scripted_run, scripted_case

REPO = Path(__file__).resolve().parents[1]  # worked out here, not read from the code under test
PUBLIC = {"public/robot.json", "public/camera.json"}
WRITABLE = {"/sandbox/state/probe-ok", "/tmp/probe-ok"}  # the only two places the probe may write
SESSION, ROBOT, NOW = "0123456789abcdef", {"name": "robot vacuum"}, {"k": 0, "clock": "2026-03-02T08:00:00"}
QUESTION = {"qid": "q", "text": "How many?", "answer_type": "count"}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A dataset folder holding the truth files, empty: what a system's folder is checked against and what no mount
    may reveal."""
    folder = tmp_path_factory.mktemp("dataset")
    for name in ("gold.jsonl", "questions.jsonl", "timeline.json"):
        (folder / name).write_text("", encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def probe_run(dataset, tmp_path_factory):
    """One run of the probe child by hand: a start, one observe of a frame (its image file exists), one ask, a finish
    and a close. Returns the dataset and the answer, whose `fork` is the probe's report at the ask and `live` the one
    report it left in its state at the observe."""
    root = tmp_path_factory.mktemp("probe")
    state, work = root / "state", root / "work"
    state.mkdir()
    (work / "public").mkdir(parents=True)
    (work / "inbox" / "rgb").mkdir(parents=True)
    (work / "public" / "robot.json").write_text('{"name": "robot vacuum"}', encoding="utf-8")
    (work / "public" / "camera.json").write_text("{}", encoding="utf-8")
    (work / "inbox" / "rgb" / "000000.jpg").write_bytes(b"x")
    file = write_child(root / "system", "probe.py", PROBE)
    with sandbox.program(file, "Probe", deny=[dataset]) as prog:
        system = CommandSystem(state, work, program=prog, timeouts=timeouts_of(60))
        try:
            system.start(SESSION, ROBOT, "rgb_pose")
            system.observe([{"k": 0, "rgb": "inbox/rgb/000000.jpg"}])
            answer = system.ask("end", NOW, [QUESTION])[0]["answer"]
            assert system.finish() == {}
        finally:
            system.close()
    assert len(answer["live"]) == 1
    return dataset, answer


def bind_root(report: dict, point: str) -> str:
    return next(root for mount, root, _ in report["binds"] if mount == point)


@pytest.mark.docker
def test_the_child_cannot_import_stbench_or_reach_a_truth_file(probe_run):
    """Contract (VISION rules 3 and 6): a system under test runs with no access to the repository or the dataset: it
    cannot import `stbench`, finds no truth file by walking the disk, sees only the working folder's public files and
    the images of the chunk it was given, and no mount names the repository or the dataset. Regression
    caught: a child that imports `stbench` to find the repository and the gold, one that globs the disk for them, and
    a mount source that reveals the repository or the dataset."""
    dataset, answer = probe_run
    for r in [answer["fork"], *answer["live"]]:
        assert r["spec"] is False and r["import_error"] == "ModuleNotFoundError"
        assert all(p == "/sandbox/code" or p.startswith(("/usr/local/lib/python3.12", "/usr/local/lib/python312"))
                   for p in r["sys_path"]), r["sys_path"]
        assert r["hits"] == []  # no file or folder anywhere named like a truth file or like `stbench*`
        assert r["shim_files"] == ["system_shim.py"] and r["code_files"] == ["probe.py"]
        assert sorted(mount for mount, _, _ in r["binds"]) == ["/sandbox/code", "/sandbox/shim", "/sandbox/state",
                                                                 "/sandbox/work"]
        for secret in (host_form(REPO), host_form(dataset), dataset.name):
            assert secret not in r["mountinfo"].lower()
    for r in answer["live"]:  # while observing, the folder holds the public files and this chunk's images
        assert r["step_files"] and set(r["work_files"]) == PUBLIC | set(r["step_files"])
    assert dataset.name not in json.dumps(answer)


@pytest.mark.docker
def test_the_child_sees_no_host_process_environment_or_network_and_holds_no_privilege(probe_run):
    """Contract (VISION rule 6): the child is alone in its container with the image's own environment and the case's
    (`HOME` in its state folder, the nonce its session), no network, no Docker socket, no capabilities and bounded
    memory and processes. Regression caught: a child that reads its ancestors' command lines (the harness's, as the WMI
    route did), the host's environment reaching the child, a case variable missing in the container, a path out to the
    network, and a dropped hardening flag (user, `--cap-drop`, `no-new-privileges`, the limits)."""
    _, answer = probe_run
    image = json.loads(sandbox.docker("image", "inspect", "--format", "{{json .Config.Env}}", sandbox.IMAGE).stdout)
    image_env = dict(item.partition("=")[::2] for item in image)
    case_env = {"HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
                "XDG_CACHE_HOME", "CLAUDE_CONFIG_DIR", "CODEX_HOME", "STBENCH_CASE_NONCE"}  # the shim's, by hand
    for r in [answer["fork"], *answer["live"]]:
        assert r["pids"] == [1]
        assert r["cmdline"] == ["python", "-I", "-B", "/sandbox/shim/system_shim.py", "/sandbox/code/probe.py", "Probe"]
        assert set(r["env"]) == set(image_env) | {"TMPDIR", "HOSTNAME"} | case_env  # HOME is one of the case's
        assert all(r["env"][k] == v for k, v in image_env.items())
        assert r["env"]["HOME"] == "/sandbox/state/home" and r["env"]["TMPDIR"] == "/tmp"
        assert r["env"]["STBENCH_CASE_NONCE"] == SESSION
        assert r["ifaces"] == ["lo"] and r["dns"] != "resolved" and r["connect"] == "ENETUNREACH"
        assert r["docker_sock"] is False
        assert r["uid"] == r["gid"] == 65534
        assert r["status"]["CapEff"] == r["status"]["CapBnd"] == "0000000000000000"
        assert r["status"]["NoNewPrivs"] == "1"
        assert r["cgroup"] == {"memory.max": str(4 * 2**30), "pids.max": "512"}


@pytest.mark.docker
def test_the_child_can_write_only_its_own_state_and_tmp(probe_run):
    """Contract (VISION rule 6, V0 §9): a system under test changes nothing of the harness's: it writes its state
    directory and its private /tmp and nothing else, and its state directory is the folder the harness gave it.
    Regression caught: a child that writes into the shared `public/` (the mission, say), or anywhere on the image, or a
    state mount that is not the folder it was given."""
    _, answer = probe_run
    for r in [answer["fork"], *answer["live"]]:
        assert {path for path, outcome in r["writes"].items() if outcome == "ok"} == WRITABLE
        assert {outcome for path, outcome in r["writes"].items() if path not in WRITABLE} == {"EROFS"}
        assert bind_root(r, "/sandbox/state").endswith("/state")


@pytest.mark.docker
def test_a_child_that_never_replies_times_out_and_its_container_is_removed(dataset, tmp_path):
    """Contract (HANDOFF 3.2, per-message timeouts): a message not answered within the timeout raises SystemTimeout
    naming the message and the limit, removes the container, and leaves `close` a no-op. Regression caught: a hung
    system that hangs the whole benchmark, and a timed-out container left running."""
    (tmp_path / "work").mkdir()
    file = write_child(tmp_path / "system", "hang.py", HANG)
    with sandbox.program(file, "Hang", deny=[dataset]) as prog:
        cs = CommandSystem(tmp_path / "state", tmp_path / "work", program=prog, timeouts=timeouts_of(8))
        try:
            cs.start(SESSION, ROBOT, "rgb_pose")
            started = time.monotonic()
            with pytest.raises(SystemTimeout, match=r"'observe'.* 8 s"):
                call_within(60, lambda: cs.observe([]))
            assert 8 <= time.monotonic() - started < 30
            assert gone(cs.name) and cs._proc.poll() is not None
            started = time.monotonic()
            call_within(10, cs.close)
            assert time.monotonic() - started < 2
        finally:
            sandbox.remove(cs.name)


@pytest.mark.docker
def test_a_child_that_fails_to_start_is_reported_and_leaves_no_container(dataset, tmp_path, monkeypatch):
    """Contract (V0 §9): a system that dies or fails when `start` reaches it raises with what it said, and no container
    is left waiting for its input; the constructor, which only starts the container, raises nothing. Regression caught:
    a crashed external system that looks like one that abstained, and a failed start that leaves its container
    running."""
    (tmp_path / "work").mkdir()
    boom = write_child(tmp_path / "boom", "boom.py", BOOM)
    missing = write_child(tmp_path / "missing", "echo.py", ECHO)
    for what, file, cls, said in [("it exits while importing", boom, "Boom", "boom"),
                                  ("its class is not there", missing, "Missing", "AttributeError")]:
        name = f"stbench-test-{uuid.uuid4().hex}"  # a name of this test's own, so it can be looked up and removed
        monkeypatch.setattr(sandbox, "container_name", lambda name=name: name)
        try:
            with sandbox.program(file, cls, deny=[dataset]) as prog:
                cs = CommandSystem(tmp_path / f"state-{cls}", tmp_path / "work", program=prog)
                with pytest.raises(RuntimeError, match=said):
                    cs.start(SESSION, ROBOT, "rgb_pose")
            assert gone(name), what
        finally:
            sandbox.remove(name)


@pytest.mark.docker
def test_a_scripted_system_completes_a_case_in_a_container(tmp_path):
    """Contract (plan §4.4, B.1's exit check): the scripted standard-library system of `test_local_system.py`, in the
    Docker sandbox through `run_case`, `CommandSystem` and the shim, completes the same case with the same answers: the
    container gets the container paths, sees each frame's inbox and answers each ask when it is sent. Regression
    caught: an adapter or the shim dropping an op in a container."""
    case, file = scripted_case(tmp_path)
    with sandbox.program(file, "Scripted", deny=[case.dataset]) as program:
        run = run_case(lambda s, w: CommandSystem(s, w, program, timeouts_of(120)), case)
    check_scripted_run(run)


def test_without_docker_the_command_system_refuses(tmp_path, monkeypatch):
    """Contract (VISION rule 6; Docker is the verified mode): without Docker, `CommandSystem` raises
    `SandboxUnavailable` before starting anything, and points at `--local` as the way to run without the sandbox.
    Regression caught: a fallback that runs the external system on the host when Docker is missing."""
    monkeypatch.setattr(sandbox, "DOCKER", "stbench-no-such-docker")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started without Docker"))
    with pytest.raises(sandbox.SandboxUnavailable, match=r"(?s)stbench-no-such-docker.*--local"):
        CommandSystem(tmp_path / "state", tmp_path / "work", program=sandbox.Program(tmp_path, "x.py", "X"))
