"""The sandbox helper the checkpoint window leaned on (DECISIONS "The checkpoint window after review"): `require`
resolves Docker once, so that every docker command costs only itself. The rest of `sandbox` is in test_sandbox.py."""

from __future__ import annotations

import subprocess

import pytest

from stbench import sandbox


def test_docker_is_resolved_once_per_process_and_not_for_every_command(monkeypatch):
    """Contract (DECISIONS "The checkpoint window after review"): once `require` has found docker and the daemon and
    image behind it, every later call with the same `DOCKER`, `IMAGE`, PATH, PATHEXT and working folder returns the path
    at once, with no search of PATH and no probe, so the commands a checkpoint window runs (pause, remove, inspect,
    list, unpause) cost only themselves; a change of any of those looks again, and a failure is never kept. Here ten
    commands cost one search. Regression caught: a search of PATH per command (3 ms each, and every one a place where
    the window's schedule can slip), a cache that outlives a change of PATH or of `DOCKER`, and a refusal that is
    remembered after docker appears."""
    searched = []
    found = {"path": "/usr/bin/docker"}
    monkeypatch.setattr(sandbox, "locate", lambda command: (searched.append(command), found["path"])[1])
    monkeypatch.setattr(sandbox, "problem", lambda docker, image: None)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr(sandbox, "DOCKER", "docker-resolve-a")
    monkeypatch.setattr(sandbox, "IMAGE", "image-resolve-a")
    for _ in range(10):
        sandbox.docker("ps")
    assert searched == ["docker-resolve-a"]
    monkeypatch.setenv("PATH", "/somewhere/else")
    assert sandbox.require() == "/usr/bin/docker" and len(searched) == 2  # another PATH: looked for again
    monkeypatch.setattr(sandbox, "DOCKER", "docker-resolve-b")
    found["path"] = None
    for _ in range(2):  # no docker: refused each time, never remembered
        with pytest.raises(sandbox.SandboxUnavailable):
            sandbox.require()
    assert len(searched) == 4
    found["path"] = "/usr/bin/docker"
    assert sandbox.require() == "/usr/bin/docker" and len(searched) == 5  # and docker appearing is seen at once
