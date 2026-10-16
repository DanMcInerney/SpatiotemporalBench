"""A job's subgoal in the script's own files (`episode.pose_rows`, `leak`): where it is written, that no keyframe lacks
one, that it stays out of the ego the renders are bound to, and that the leak scan reads its text."""

from __future__ import annotations

import json

import pytest

from conftest import script_on_map
from stbench import episode as ep
from stbench import leak
from stbench.leak import find_leaks
from stbench.timeline import load
from subgoal_kit import with_subgoals

MAP = "apartment_a"
POSES = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 0.0), (3.6, 3.4, 0.0, 0.0), (4.0, 3.4, 0.0, 0.0)]


def pose_file(paths) -> list[dict]:
    return [json.loads(line) for line in paths.pose.read_text(encoding="utf-8").splitlines()]


def test_every_pose_row_of_a_job_with_subgoals_carries_one(tmp_path, monkeypatch):
    """Contract (FD-0.5.0-control-impl §2): a job that states a subgoal has one non-empty text on every pose row, as a
    top-level key beside `status` (not inside it), the job's text for that keyframe in keyframe order. Hand values: the
    five keyframes of the script read "cleaning the kitchen, stage 0" to "stage 3" and "stage 0" again (the kit's four
    texts in turn). The text is not part of the ego the renders are bound to: the saved timeline holds no `subgoal` and
    its ego hash is the one the script written for a job without subgoals has. Regression caught: a keyframe with no
    subgoal (a header the agent sees without one), the text put in the status or the ego (which would unbind every
    render), or a row that loses its other keys."""
    tl = script_on_map(MAP, 1, "vacuum", POSES)
    plain = ep.write_script(tmp_path / "plain", tl)
    with_subgoals(monkeypatch)
    paths = ep.write_script(tmp_path / "with", tl)
    rows = pose_file(paths)
    assert [r["subgoal"] for r in rows] == [f"cleaning the kitchen, stage {k}" for k in (0, 1, 2, 3, 0)]
    assert all(set(r) == {"k", "clock", "camera", "pose", "status", "subgoal"} for r in rows)
    assert all("subgoal" not in r["status"] for r in rows)
    assert [{k: v for k, v in r.items() if k != "subgoal"} for r in rows] == pose_file(plain)
    assert "subgoal" not in paths.timeline.read_text(encoding="utf-8")
    assert load(paths.timeline).ego_hash == load(plain.timeline).ego_hash


def test_a_job_without_subgoals_writes_none(tmp_path):
    """Contract: a job whose `subgoals` is None (every job written before the control work, `vacuum/1` among them)
    writes pose rows with no `subgoal` key at all, not an empty one. Regression caught: a default text or a null written
    for a job that states none, which would put an empty `subgoal:` into every header."""
    rows = pose_file(ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", POSES)))
    assert len(rows) == 5 and all("subgoal" not in r for r in rows)


@pytest.mark.parametrize("bad", [lambda tl, m: ["cleaning"] * (tl.n_keyframes - 1),
                                 lambda tl, m: ["cleaning"] * (tl.n_keyframes + 1),
                                 lambda tl, m: ["cleaning", "", *["cleaning"] * (tl.n_keyframes - 2)],
                                 lambda tl, m: ["cleaning", "  ", *["cleaning"] * (tl.n_keyframes - 2)],
                                 lambda tl, m: ["cleaning", None, *["cleaning"] * (tl.n_keyframes - 2)]])
def test_a_job_that_leaves_a_keyframe_without_a_subgoal_is_refused(tmp_path, monkeypatch, bad):
    """Contract: `pose_rows` refuses (ValueError naming the job) a subgoal function that returns too few or too many
    texts, an empty or blank one, or a non-text; nothing is written. Regression caught: a pose file with a gap, whose
    header would have no subgoal for some frames and misalign the rest with the keyframes."""
    with_subgoals(monkeypatch, bad)
    with pytest.raises(ValueError, match="vacuum/1 states a subgoal"):
        ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", POSES))
    assert not list(tmp_path.rglob("pose.jsonl"))


def test_the_leak_scan_allows_the_subgoal_key(tmp_path, monkeypatch):
    """Contract (FD-0.5.0-control-impl §2): `subgoal` is on the pose file's allowlist, so a script whose rows carry one
    scans clean (no `unlisted_key`); the key is the only thing added. Regression caught: the key left off `POSE_KEYS`,
    which would refuse every script of a job with a subgoal."""
    with_subgoals(monkeypatch)
    paths = ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", POSES))
    assert "subgoal" in leak.POSE_KEYS and "subgoal" in paths.pose.read_text(encoding="utf-8")
    assert find_leaks(paths.root, [MAP]) == []


@pytest.mark.parametrize("text, rule", [
    ("going to E0042 to pick it up", "entity_id"),
    ("cleaning region R3", "region_id"),
    ("cleaning the kitchen of apartment_a", "map_name"),
    ("this is the truth about the room", "truth"),
])
def test_the_leak_scan_reads_the_text_of_a_subgoal(tmp_path, monkeypatch, text, rule):
    """Contract: the text rules (an entity id, a region id, the map's name, the word truth) apply to a subgoal's text as
    to the rest of the file, so a job whose subgoal names one is caught at the scan. Hand cases, one per rule.
    Regression caught: an allowed key whose value is never read, through which a job could put the truth into every
    header."""
    with_subgoals(monkeypatch, lambda tl, m: [text] * tl.n_keyframes)
    paths = ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", POSES))
    assert rule in {x.rule for x in find_leaks(paths.root, [MAP])}


def test_ordinary_words_in_a_subgoal_are_not_flagged(tmp_path, monkeypatch):
    """Contract: a subgoal in the words the jobs use (a station, a room, a count) scans clean: the text rules match an
    entity id of four digits or more, a region id and the map's name only. Regression caught: a rule so wide that the
    first real subgoal ("driving to the F12 station to pick up totes (stop 2 of 3)") is refused."""
    with_subgoals(monkeypatch, lambda tl, m: ["driving to the F12 station to pick up totes (stop 2 of 3)"]
                  * tl.n_keyframes)
    paths = ep.write_script(tmp_path, script_on_map(MAP, 1, "vacuum", POSES))
    assert find_leaks(paths.root, [MAP]) == []
