"""The command line: `validate` end to end on a hand-made script, the command table, and the plug-in point other lanes
use to add subcommands."""

from __future__ import annotations

import json
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from conftest import script_on_map
from stbench import cli
from stbench import episode as ep
from stbench import timeline as T
from stbench.geometry import quat_z
from stbench.timeline import Placement
from validate_kit import generic_rules_only  # noqa: F401  (a fixture)

# Hand-made poses on the living room's open floor (x 2.4-3.2, y 3.3-3.6), heading east then north
LIVING_ROOM_FLOOR = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 0.0), (3.2, 3.4, 0.0, 1.5)]


def run(*argv) -> int:
    return cli.main([str(a) for a in argv])


@pytest.fixture
def script(tmp_path):
    return ep.write_script(tmp_path, script_on_map("apartment_a", 1, "vacuum", LIVING_ROOM_FLOOR))


@pytest.mark.usefixtures("generic_rules_only")
def test_a_hand_made_vacuum_script_validates(script, capsys):
    """Contract (M0's exit check): `stbench validate` exits 0, finding no problem and no leak, on a hand-made vacuum
    script on the real apartment (`script_on_map` poses on the living room's open floor, written by `write_script`),
    from the agent folder and from the private one, and says how many keyframes and tasks it holds. Regression caught:
    M0's exit check failing by construction (the fake-map scripts of the synthetic builder fail it)."""
    assert run("validate", script.root) == 0
    out = capsys.readouterr().out
    assert script.id in out and "ok (3 keyframes, 1 tasks)" in out
    assert run("validate", script.private) == 0


def test_validate_fails_on_a_planted_leak(script, capsys):
    """Contract: `stbench validate` also runs the leak scan over the agent folder, so an entity id planted in
    `robot.json` exits 1 and names its rule (`entity_id`). Regression caught: a `validate` that checks the timeline
    alone and never scans the agent-facing files, so a leak reaches a system with exit 0."""
    doc = json.loads(script.robot.read_text(encoding="utf-8"))
    doc["name"] = "E0042 is under the table"
    script.robot.write_text(json.dumps(doc), encoding="utf-8")
    assert run("validate", script.root) == 1
    assert "entity_id" in capsys.readouterr().out


def test_validate_fails_on_the_maps_name_in_an_agent_file(script, capsys):
    """Contract: the scan is given the script's map (read from its timeline) and finds its name in any agent-facing
    file (`map_name`). Regression caught: a scan run with no names (a map's name is no leak unless the caller passes
    it), so a script on any map could carry its map's name to a system with exit 0."""
    doc = json.loads(script.robot.read_text(encoding="utf-8"))
    doc["name"] = "the apartment_a vacuum"
    script.robot.write_text(json.dumps(doc), encoding="utf-8")
    assert run("validate", script.root) == 1
    assert "map_name" in capsys.readouterr().out


def test_validate_fails_on_an_inconsistent_timeline(script, capsys):
    """Contract: a timeline edited into an inconsistent state (a second placement overlapping the first) is reported
    by its code and exits 1. Regression caught: a `validate` that scans for leaks but never runs the timeline's rules,
    so a script that breaks them passes the command line."""
    tl = T.load(script.timeline)
    first = tl.placements[0]
    tl.placements.append(Placement(first.entity, 5.0, None, first.pos, quat_z(0.0), None))
    T.save(tl, script.timeline)
    assert run("validate", script.root) == 1
    assert "placement_overlap" in capsys.readouterr().out


def test_the_commands_over_a_script_take_its_folder(tmp_path):
    """Contract (plan §2.8): `validate`, `perceive` and `render` each take one `SCRIPT_DIR`, the agent folder or the
    private one. Regression caught: a command still asking for the old `episode_dir`."""
    parser = cli.build_parser()
    for command in ("validate", "perceive", "render"):
        assert parser.parse_args([command, str(tmp_path)]).script_dir == tmp_path


def test_the_command_table_names_every_lanes_module():
    """Contract (plan §2.8): `COMMAND_MODULES` is pre-seeded with the module of every lane's command, in the table's
    order, and the stale `render_cmd` is gone, so no lane edits `cli.py` to add its command; `cmd_score` is gone too
    (M2: `run` and `eval` score what they run). Regression caught: a lane that has to touch a shared file to be
    reachable."""
    table = ("visibility", "perception", "cmd_generate", "render_cli", "perceive_cmd", "cmd_compile", "cmd_run",
             "check", "realism_cmd", "asset_cmd", "asset_render", "eval_cmd", "cmd_pilot", "export_cmd", "view_cmd")
    expected = [f"stbench.{name}" for name in table]
    assert list(cli.COMMAND_MODULES) == expected


def test_seed_syntax():
    """Contract: `parse_seeds` reads one seed, an inclusive range and a comma list that mixes both. Regression caught:
    a range that stops one short (`1-3` giving two seeds), or a list that drops its last range."""
    assert cli.parse_seeds("3") == [3]
    assert cli.parse_seeds("1-3") == [1, 2, 3]
    assert cli.parse_seeds("1,4,7-8") == [1, 4, 7, 8]


def test_other_lanes_add_commands_with_a_register_function(tmp_path, monkeypatch, capsys):
    """A module named in COMMAND_MODULES that defines register(subparsers) adds a subcommand; a module that does not
    exist yet is skipped; a module that exists but fails to import raises instead of hiding the error. Regression
    caught: a lane's broken import swallowed (its command silently absent from the command line), or a lane whose module
    is not written yet crashing every command."""
    (tmp_path / "lane_ok.py").write_text(textwrap.dedent("""
        def register(sub):
            p = sub.add_parser("hello")
            p.set_defaults(func=lambda args: print("hello from a lane") or 0)
    """), encoding="utf-8")
    (tmp_path / "lane_noregister.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "lane_broken.py").write_text("import module_that_is_not_installed\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cli, "COMMAND_MODULES", ("lane_missing", "lane_noregister", "lane_ok"))
    assert cli.main(["hello"]) == 0
    assert "hello from a lane" in capsys.readouterr().out
    monkeypatch.setattr(cli, "COMMAND_MODULES", ("lane_broken",))
    with pytest.raises(ModuleNotFoundError, match="module_that_is_not_installed"):
        cli.build_parser()
    sys.modules.pop("lane_broken", None)


def test_version_prints_the_version_of_the_package(capsys):
    """Contract: `stbench --version` prints `stbench <version>` and exits 0, the version being the one the package
    declares in pyproject.toml (read from the file, not from the code under test). Regression caught: a command line
    with no way to say which release it is, so a result cannot name the code that made it."""
    declared = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8"))
    with pytest.raises(SystemExit) as stop:
        run("--version")
    assert stop.value.code == 0
    assert capsys.readouterr().out == f"stbench {declared['project']['version']}\n"
