"""A script's split follows its map (`episode.write_script`): `heldout` for a map of `map_files.HELD_OUT`, else `dev`,
and a dataset root holds one split. A held-out map here is a renamed copy of `apartment_a` in a temp `STBENCH_MAPS`
folder, named in a patched `HELD_OUT`, so no held-out layout is needed (the variable and the tuple are set with
`monkeypatch` only, so the tests stay parallel-safe)."""

from __future__ import annotations

import pytest

from conftest import script_on_map
from stbench import episode as ep
from stbench import map_files
from stbench.case import read_index
from test_map_files import PRIVATE, write_map

POSES = [(2.4, 3.4, 0.0, 0.0), (2.8, 3.4, 0.0, 1.5), (3.2, 3.4, 0.0, 3.1)]  # the apartment's living room, open floor


@pytest.fixture
def held_out_copy(tmp_path, monkeypatch):
    """`apartment_private_copy` found through `STBENCH_MAPS` and named in `HELD_OUT`; returns a script on it."""
    write_map(tmp_path / "private")
    monkeypatch.setenv(map_files.MAPS_ENV, str(tmp_path / "private"))
    monkeypatch.setattr(map_files, "HELD_OUT", (*map_files.HELD_OUT, PRIVATE))
    return script_on_map(PRIVATE, 1, "vacuum", POSES)


def files(root) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_a_held_out_map_writes_heldout_and_others_dev(held_out_copy, tmp_path):
    """Contract (call 9, D3): the index entry of a script says `heldout` when its map is in `map_files.HELD_OUT` (read
    when the script is written) and `dev` for any other map, with no flag to pass; a split that contradicts the map
    (`dev` or `test` on the held-out copy, `heldout` on `apartment_a`) is a `ValueError` naming the split and the map,
    raised before anything is written, while `test` stays open to a public map. Regression caught: the split read from
    a flag that a build forgets to pass, so a held-out script is indexed `dev` and an export of the root ships it."""
    held = ep.write_script(tmp_path / "held", held_out_copy)
    assert read_index(tmp_path / "held")[held.id]["split"] == "heldout"
    named = ep.write_script(tmp_path / "named", held_out_copy, split="heldout")
    assert read_index(tmp_path / "named")[named.id]["split"] == "heldout"
    public = script_on_map("apartment_a", 1, "vacuum", POSES)
    dev = ep.write_script(tmp_path / "dev", public)
    assert read_index(tmp_path / "dev")[dev.id]["split"] == "dev"
    test = ep.write_script(tmp_path / "test", public, split="test")
    assert read_index(tmp_path / "test")[test.id]["split"] == "test"
    for split in ("dev", "test"):
        with pytest.raises(ValueError, match=rf"{split}.*{PRIVATE}"):
            ep.write_script(tmp_path / "refused", held_out_copy, split=split)
    with pytest.raises(ValueError, match=r"heldout.*apartment_a"):
        ep.write_script(tmp_path / "refused", public, split="heldout")
    assert not (tmp_path / "refused").exists()


def test_a_root_holds_one_split(held_out_copy, tmp_path):
    """Contract (call 9, D3): writing a script whose split differs from that of a script already in the root's index is
    a `ValueError`, in either order, and writes nothing (no folder, no index entry); a script of the root's own split
    still goes in. Regression caught: held-out and public scripts sharing a root, so an export of one ships the
    other."""
    public = script_on_map("apartment_a", 1, "vacuum", POSES)
    other = script_on_map("apartment_a", 2, "vacuum", POSES)
    ep.write_script(tmp_path / "public", public)
    before = files(tmp_path / "public")
    with pytest.raises(ValueError, match="one split"):
        ep.write_script(tmp_path / "public", held_out_copy)
    assert files(tmp_path / "public") == before
    ep.write_script(tmp_path / "held", held_out_copy)
    before = files(tmp_path / "held")
    with pytest.raises(ValueError, match="one split"):
        ep.write_script(tmp_path / "held", public)
    assert files(tmp_path / "held") == before
    ep.write_script(tmp_path / "public", other)  # the root's own split goes in
    assert len(read_index(tmp_path / "public")) == 2
