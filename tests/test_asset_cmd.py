"""`stbench assets ...` end to end through the real command line, against synthetic manifests and a fake network
(patched in for `asset_fetch.urllib_opener`). No download, no network, and the packaged manifest is never written."""

from __future__ import annotations

import argparse
import copy
from pathlib import PurePosixPath

import pytest

from asset_synth import CCBY, MUG_OBJ, FakeNet, asset, raw_manifest, write_manifest
from stbench import asset_cmd, cli
from stbench import asset_fetch as F


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake network, an empty cache, and a way to write manifests; `run` is the command line."""
    net = FakeNet()
    monkeypatch.setattr(F, "urllib_opener", net)
    monkeypatch.setenv("STBENCH_ASSET_CACHE", str(tmp_path / "cache"))

    class World:
        cache = tmp_path / "cache"

        def manifest(self, edit=None):
            raw = copy.deepcopy(raw_manifest())
            if edit:
                edit(raw)
            return write_manifest(tmp_path / "manifest.json", raw)

        def run(self, *argv):
            return cli.main(["assets", argv[0], "--manifest", str(tmp_path / "manifest.json"), *argv[1:]])

    w = World()
    w.net, w.tmp = net, tmp_path
    return w


def test_fetch_without_yes_downloads_nothing_and_prints_sources_licences_and_total(world, capsys):
    """Contract (downloads need `--yes`; sizes are stated first): `fetch` without --yes prints each
    asset with its source, licence, host and size, then the sources and the total, downloads nothing and creates no
    cache. Regression caught: a `fetch` that downloads by default."""
    world.manifest()
    assert world.run("fetch") == 0
    out = capsys.readouterr().out
    assert world.net.calls == [] and not world.cache.exists()
    for needed in ("mug_a", "mat_a", "CC-BY-4.0", "CC0-1.0", CCBY, "cc0.example.org", f"https://{CCBY}/licence",
                   "total: 3 files", "nothing was downloaded"):
        assert needed in out, needed
    assert "mug_b" not in out  # an alternate is not part of the default plan
    assert world.run("fetch", "--include-alternates") == 0 and "mug_b" in capsys.readouterr().out


def test_fetch_yes_fills_the_cache_and_a_second_fetch_has_nothing_to_do(world, capsys):
    """`fetch --yes` fills the cache from the manifest, an entry of which holds only a path and a url; a second `fetch`
    finds nothing to do and downloads nothing. Regression caught: a fetch that downloads what is already cached."""
    world.manifest()
    assert world.run("fetch", "--yes") == 0 and len(world.net.calls) == 3
    assert (world.cache / "mug_a" / "model.obj").read_bytes() == MUG_OBJ
    capsys.readouterr()
    assert world.run("fetch", "--yes") == 0 and len(world.net.calls) == 3
    assert "nothing to fetch" in capsys.readouterr().out


def test_status_lists_every_asset_with_its_state_and_the_total(world, capsys):
    """Contract: `status` is the "sources and sizes before downloading" step: every asset with source, licence, cached
    state and estimated size, the sources with their hosts, and what a download would add up to; it downloads nothing.
    Regression caught: a status that hides an asset, or counts alternates in the default total."""
    world.manifest()
    assert world.run("status") == 0
    out = capsys.readouterr().out
    rows = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.split() and ln.split()[0] in
            ("mug_a", "mug_b", "mat_a", "proc_glass")}
    assert rows["mug_a"][-2] == "0/2" and rows["proc_glass"][-2] == "n/a"
    assert "alternate for mug_a" in out and "mug_b" in rows
    assert "to download: 3 files" in out and "not included: 1 alternates" in out and "hosts cc0.example.org" in out
    assert world.net.calls == []
    world.run("fetch", "--yes")
    capsys.readouterr()
    world.run("status")
    again = capsys.readouterr().out
    assert "to download: 0 files" in again and "2/2" in again and "1/1" in again


@pytest.mark.parametrize("command", ["status", "fetch", "credits"])
def test_a_manifest_with_problems_stops_every_command_before_any_download(world, capsys, command):
    """Contract: a manifest that breaks the licence allowlist or the host list is refused by every command, so nothing
    is downloaded or credited from it (exit 2, the problem on stderr). Regression caught: `fetch --yes` run on a
    manifest `validate` rejects."""
    world.manifest(edit=lambda r: asset(r, "mug_a").update(licence="CC-BY-NC-4.0"))
    assert world.run(command, *(["--yes"] if command == "fetch" else [])) == 2
    assert "allowlist" in capsys.readouterr().err and world.net.calls == []


def test_credits_writes_at_the_dataset_root_and_refuses_scripts_and_private(world, tmp_path, capsys):
    """Contract (AGENTS: agent-facing files live only under scripts/<sid>/): CREDITS.md goes at the dataset root, never
    into scripts/ or private/, where a system could read it or the leak scan would see it. Regression caught: the
    credits written beside the scripts (the guard still naming the retired episodes/ folder)."""
    world.manifest()
    ds = tmp_path / "dataset"
    (ds / "scripts" / "abc").mkdir(parents=True)
    (ds / "private" / "abc").mkdir(parents=True)
    assert world.run("credits", "--out", str(ds / "CREDITS.md")) == 0
    assert "Mugs by Mug Makers" in (ds / "CREDITS.md").read_text(encoding="utf-8")
    for bad in (ds / "scripts" / "CREDITS.md", ds / "scripts" / "abc" / "x.md", ds / "private" / "abc" / "x.md"):
        assert world.run("credits", "--out", str(bad)) == 2
        assert not bad.exists()
    assert "dataset root" in capsys.readouterr().err
    assert world.run("credits") == 0 and "Mugs by Mug Makers" in capsys.readouterr().out


@pytest.mark.parametrize("relative", [
    pytest.param("scripts/CREDITS.md", id="scripts with no dataset around it"),
    pytest.param("scripts/0123456789abcdef/CREDITS.md", id="inside a script's folder"),
    pytest.param("episodes/CREDITS.md", id="the retired episodes folder"),
    pytest.param("a/private/b/c/CREDITS.md", id="private deep in the path"),
    pytest.param("Scripts/CREDITS.md", id="another letter case"),
    pytest.param("ok/../private/CREDITS.md", id="reached through .."),
])
def test_credits_refuses_any_path_through_a_scripts_episodes_or_private_folder(world, tmp_path, capsys, relative):
    """Contract (AGENTS: agent-facing files live only under scripts/<sid>/, the truth only under private/): a resolved
    `--out` with a folder named scripts, episodes or private anywhere in it is refused, whether or not a dataset can be
    recognised there (it may not exist yet). Regression caught: the refusal tied to finding both folders on disk, so a
    CREDITS.md written to a dataset that is not built yet lands in the agent-facing or the private tree; or a guard
    that names only the retired episodes/ folder, as it did after the redesign's layout change."""
    world.manifest()
    bad = tmp_path / "out" / relative
    assert world.run("credits", "--out", str(bad)) == 2
    assert "scripts, episodes or private" in capsys.readouterr().err
    assert not list((tmp_path / "out").rglob("CREDITS.md"))


def test_credits_judges_the_resolved_path_not_how_it_is_spelled(world, tmp_path):
    """`out/scripts/../CREDITS.md` is `out/CREDITS.md`: refusing the spelling would refuse a good path. Regression
    caught: the refusal reading the raw argument instead of the resolved path (so `ok/../private/` slips through)."""
    world.manifest()
    assert world.run("credits", "--out", str(tmp_path / "out" / "scripts" / ".." / "CREDITS.md")) == 0
    assert (tmp_path / "out" / "CREDITS.md").is_file()


@pytest.mark.parametrize(("path", "platform", "refused"), [
    pytest.param("/private/var/folders/t/dataset/CREDITS.md", "darwin", False, id="macOS temp folder"),
    pytest.param("/private/tmp/dataset/CREDITS.md", "darwin", False, id="macOS /tmp"),
    pytest.param("/private/var/folders/t/dataset/private/CREDITS.md", "darwin", True, id="a dataset's private, macOS"),
    pytest.param("/private/tmp/dataset/scripts/CREDITS.md", "darwin", True, id="a dataset's scripts on macOS"),
    pytest.param("/private/dataset/CREDITS.md", "linux", True, id="a root private folder elsewhere"),
])
def test_credits_leaves_macos_system_private_folder_alone(path, platform, refused):
    """Contract: macOS resolves /tmp and /var to /private/tmp and /private/var, the system's folder and not a dataset's,
    so only the folders after it are judged; every other platform judges a root /private like any other. Regression
    caught: `stbench assets credits --out` refusing every path under the macOS temp folders."""
    assert asset_cmd._through_dataset_folder(PurePosixPath(path), platform) is refused


def test_credits_reports_a_path_it_cannot_write_instead_of_crashing(world, tmp_path, capsys):
    """Contract: an `--out` that cannot be written (a parent that is a file, a path that is a folder) is exit 2 with the
    reason on stderr, like every other asset error. Regression caught: an uncaught OSError traceback."""
    world.manifest()
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("a file", encoding="utf-8")
    for bad in (blocker / "CREDITS.md", tmp_path):
        assert world.run("credits", "--out", str(bad)) == 2
        assert "cannot write the credits" in capsys.readouterr().err


def test_other_modules_add_subcommands_to_the_one_assets_group():
    """Contract: `asset_render` adds `library` and `preview` to the same `stbench assets` group, in either import order,
    and the three new command modules are registered with the CLI. Regression caught: a second `assets` parser that
    shadows the first, or a lane's module dropped from COMMAND_MODULES."""
    top = argparse.ArgumentParser().add_subparsers(dest="command")
    first = asset_cmd.group(top)
    assert asset_cmd.group(top) is first
    first.add_parser("library").set_defaults(func=lambda a: 0)
    asset_cmd.register(top)  # registering after another module created the group keeps its subcommands
    parser = top.choices["assets"]
    names = ["library", "status", "fetch", "credits"]
    assert [parser.parse_args([name]).assets_command for name in names] == names
    assert {"stbench.asset_cmd", "stbench.asset_render"} <= set(cli.COMMAND_MODULES)
