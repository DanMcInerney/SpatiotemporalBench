"""Fetching assets and the credits file, against a fake network and in-test zips. No download, no network: the opener is
injected, and `urllib_opener`'s guards are exercised before any connection is made."""

from __future__ import annotations

import shutil
import types
import urllib.error
import zipfile
from pathlib import Path

import pytest

from asset_synth import (
    CC0,
    CCBY,
    MAT_ZIP,
    MUG_OBJ,
    MUG_OBJ_URL,
    MUG_TEX,
    MUG_TEX_URL,
    EndlessResponse,
    FakeNet,
    WireResponse,
    asset,
    cut_body,
    make_zip,
    manifest,
)
from stbench import asset_fetch as F
from stbench import asset_manifest as A


def no_parts(cache) -> bool:
    return not list(cache.rglob("*.part"))


def keys(tasks) -> list[tuple[str, str]]:
    return [(t.asset.id, t.file.path) for t in tasks]


# -- planning ----------------------------------------------------------------------------------------------------
def test_plan_skips_alternates_unless_asked_and_names_unknown_assets(tmp_path):
    """Contract: an alternate is never fetched unless asked for (`include_alternates`, or named in `only`); an unknown
    id is an error. Regression caught: fallbacks downloaded by default (an unreviewed 19 MB), or a typo'd --only
    silently selecting nothing."""
    m = manifest()

    def ids(**kw) -> set[str]:
        return {t.asset.id for t in F.plan(m, cache=tmp_path, **kw)}

    assert ids() == {"mug_a", "mat_a"}
    assert ids(include_alternates=True) == {"mug_a", "mug_b", "mat_a"}
    assert ids(only=["mug_b"]) == {"mug_b"}
    assert ids(only=["mat_a"], include_alternates=True) == {"mat_a"}
    with pytest.raises(A.AssetError, match="nope"):
        F.plan(m, only=["nope"], cache=tmp_path)


def test_plan_bytes_counts_each_asset_once_by_its_estimate(tmp_path):
    """Contract: the size shown before a download is each asset's `estimate_bytes`, counted once however many of its
    files are missing, and 0 for an asset the manifest gives no estimate for. Regression caught: an estimate added once
    per file, which overstates the total, or a missing estimate failing the plan."""
    m = manifest(edit=lambda raw: asset(raw, "mat_a").update(estimate_bytes=len(MAT_ZIP)))
    tasks = F.plan(m, cache=tmp_path)  # mug_a: two files, estimate 1234; mat_a: one file
    assert F.plan_bytes([t for t in tasks if t.asset.id == "mug_a"]) == 1234
    assert F.plan_bytes([t for t in tasks if t.asset.id == "mat_a"]) == len(MAT_ZIP)
    assert F.plan_bytes(tasks) == 1234 + len(MAT_ZIP)
    assert F.plan_bytes(F.plan(manifest(), only=["mat_a"], cache=tmp_path)) == 0


# -- fetching --------------------------------------------------------------------------------------------------
def test_fetch_downloads_unpacks_and_never_downloads_a_cached_file_again(tmp_path):
    """Contract: an entry holds only a path and a url, and a file that arrived whole is placed; a zip is kept and
    unpacked to x/; a file already in the cache is never downloaded again, and a missing one is fetched alone.
    Regression caught: a re-run re-downloading everything, or `plan` asking for a file that is there."""
    m, net = manifest(), FakeNet()
    assert F.fetch(F.plan(m, cache=tmp_path), net) == 3
    assert (tmp_path / "mug_a" / "model.obj").read_bytes() == MUG_OBJ
    assert (tmp_path / "mug_a" / "tex" / "texture.png").read_bytes() == MUG_TEX
    assert (tmp_path / "mat_a" / "mat_a_1K.zip").read_bytes() == MAT_ZIP
    assert (tmp_path / "mat_a" / "x" / "maps" / "rough.jpg").read_bytes() == b"jpg-3"
    assert no_parts(tmp_path) and not (tmp_path / "mat_a" / "x.part").exists()
    assert F.plan(m, cache=tmp_path) == []
    assert F.fetch(F.plan(m, cache=tmp_path), net) == 0 and len(net.calls) == 3
    (tmp_path / "mug_a" / "model.obj").unlink()
    redo = F.plan(m, cache=tmp_path)
    assert keys(redo) == [("mug_a", "model.obj")]
    assert F.fetch(redo, net) == 1 and net.calls[-1] == MUG_OBJ_URL and len(net.calls) == 4
    assert (tmp_path / "mug_a" / "model.obj").read_bytes() == MUG_OBJ


def test_fetch_unpacks_a_cached_zip_whose_folder_is_gone_without_downloading_again(tmp_path):
    """A cached zip with no x/ folder is only unpacked. Regression caught: an interrupted unpack leaving a zip that is
    trusted but unusable, or one that forces a re-download."""
    m, net = manifest(), FakeNet()
    F.fetch(F.plan(m, cache=tmp_path), net)
    shutil.rmtree(tmp_path / "mat_a" / "x")
    todo = F.plan(m, cache=tmp_path)
    assert keys(todo) == [("mat_a", "mat_a_1K.zip")]
    assert F.fetch(todo, net) == 0 and len(net.calls) == 3
    assert (tmp_path / "mat_a" / "x" / "color.jpg").read_bytes() == b"jpg-1"


def test_fetch_stops_a_download_that_runs_past_the_size_cap(tmp_path, monkeypatch):
    """Contract: a download is cut off once more bytes arrive than `MAX_FILE_BYTES`, so a hostile or broken server
    cannot fill the disk. Regression caught: reading an endless body to the end."""
    monkeypatch.setattr(F, "MAX_FILE_BYTES", 1000)
    endless = EndlessResponse()
    with pytest.raises(F.FetchError, match="more than"):
        F.fetch(F.plan(manifest(), only=["mug_a"], cache=tmp_path)[:1], lambda url, hosts: endless)
    assert endless.reads == 1 and no_parts(tmp_path)


def test_fetch_downloads_an_entry_with_no_pin_fields_and_refuses_a_body_shorter_than_its_content_length(tmp_path):
    """Contract: integrity without any recorded size or hash. A response that ends before the Content-Length it
    announced is a cut connection, not a short file: it is refused, nothing is placed and no `.part` stays; the same
    file announced correctly is fetched. Regression caught: `http.client` returns a truncated body without an error, so
    a half-downloaded mesh was kept and every later fetch trusted it."""
    m, net = manifest(), FakeNet()
    net.headers[MUG_OBJ_URL] = {"Content-Length": str(len(MUG_OBJ) + 100)}
    with pytest.raises(F.FetchError, match="announced"):
        F.fetch(F.plan(m, only=["mug_a"], cache=tmp_path), net)
    assert not (tmp_path / "mug_a" / "model.obj").exists() and no_parts(tmp_path)
    net.headers[MUG_OBJ_URL] = {"Content-Length": str(len(MUG_OBJ))}
    assert F.fetch(F.plan(m, only=["mug_a"], cache=tmp_path), net) == 2
    assert (tmp_path / "mug_a" / "model.obj").read_bytes() == MUG_OBJ


def test_a_content_coded_response_is_not_held_to_its_content_length(tmp_path):
    """The Content-Length of a content-coded response (a Content-Encoding other than identity) is not the length of the
    bytes read, so it is not compared; the file is kept as it arrived. Regression caught: the length check applied to
    every response, refusing a coded one whose announced length means something else."""
    net = FakeNet()
    net.headers[MUG_OBJ_URL] = {"Content-Encoding": "gzip", "Content-Length": "3"}
    assert F.fetch(F.plan(manifest(), only=["mug_a"], cache=tmp_path)[:1], net) == 1
    assert (tmp_path / "mug_a" / "model.obj").read_bytes() == MUG_OBJ


@pytest.mark.parametrize("chunked, expected", [
    pytest.param(False, "announced", id="a Content-Length body that stops early"),
    pytest.param(True, "IncompleteRead", id="a chunked body that stops early"),
])
def test_a_connection_cut_mid_body_is_a_fetch_error_that_keeps_the_files_before_it(tmp_path, chunked, expected):
    """Contract: with the real `http.client` response (parsed from wire bytes, no socket), a body that ends early is a
    FetchError in both framings, the file is not placed, no `.part` stays and the first file is kept. A cut chunked
    body raises `IncompleteRead`, an HTTPException and not an OSError. Regression caught: `IncompleteRead` escaping
    `fetch` as a raw exception, or a cut Content-Length body accepted."""
    net = FakeNet()

    def opener(url, hosts):
        if url == MUG_TEX_URL:
            return WireResponse(cut_body(MUG_TEX[:5], len(MUG_TEX), chunked), url)
        return net(url, hosts)

    with pytest.raises(F.FetchError, match=expected):
        F.fetch(F.plan(manifest(), only=["mug_a"], cache=tmp_path), opener)
    assert (tmp_path / "mug_a" / "model.obj").read_bytes() == MUG_OBJ
    assert not (tmp_path / "mug_a" / "tex" / "texture.png").exists() and no_parts(tmp_path)


def test_a_network_error_is_a_fetch_error_and_leaves_nothing(tmp_path):
    """Contract: a failed connection (a URLError, a timeout) surfaces as a FetchError the command line reports, never as
    a traceback. Regression caught: only some network errors converted, the rest escaping a 200 MB run."""
    for error in (urllib.error.URLError("no route"), TimeoutError("timed out")):
        def opener(url, hosts, error=error):
            raise error

        with pytest.raises(F.FetchError, match="model.obj"):
            F.fetch(F.plan(manifest(), only=["mug_a"], cache=tmp_path)[:1], opener)
        assert not (tmp_path / "mug_a" / "model.obj").exists() and no_parts(tmp_path)


def test_fetch_refuses_redirects_off_the_source_list_without_reading_the_body(tmp_path):
    """Contract: a redirect off the source's host list (or off https) is refused before a byte of the body is read.
    Regression caught: a redirect to an arbitrary host followed, so a source (or whoever sits between) can serve us
    anything."""
    m = manifest()
    for target in ("https://evil.example.net/m.obj", f"http://{CCBY}/m.obj", "https://" + CC0 + "/m.obj"):
        net = FakeNet(redirects={MUG_OBJ_URL: target})
        with pytest.raises(F.FetchError, match="redirect"):
            F.fetch(F.plan(m, only=["mug_a"], cache=tmp_path)[:1], net)
        assert net.responses[0].bytes_read == 0
        assert not (tmp_path / "mug_a" / "model.obj").exists() and no_parts(tmp_path)


def test_the_real_opener_checks_hosts_before_connecting_and_at_every_redirect(tmp_path):
    """Contract: `urllib_opener` refuses a URL off the host list or off https before it connects, and its redirect
    handler refuses an off-list hop before following it (the test makes no connection). Regression caught: the host
    list checked only on the first URL, so a redirect can lead anywhere."""
    hosts = (CCBY,)
    for url in ("http://" + CCBY + "/m", "https://evil.example.net/m", "file:///etc/passwd"):
        with pytest.raises(F.FetchError, match="hosts"):
            F.urllib_opener(url, hosts)
    guard = F._HostGuard(hosts)
    request = F.urllib.request.Request(f"https://{CCBY}/m")
    for hop in ("https://evil.example.net/m", f"http://{CCBY}/m"):
        with pytest.raises(F.FetchError, match="redirect to"):
            guard.redirect_request(request, None, 302, "Found", {}, hop)
    followed = guard.redirect_request(request, None, 302, "Found", {}, f"https://{CCBY}/other")
    assert followed is not None and followed.full_url == f"https://{CCBY}/other"


# -- unpacking -------------------------------------------------------------------------------------------------
def zip_with(tmp_path, name: str, link: bool = False, raw_name: str | None = None) -> Path:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("fine.txt", b"ok")
        info = zipfile.ZipInfo(name)
        if raw_name is not None:
            info.filename = raw_name
        if link:
            info.external_attr = 0o120777 << 16
        zf.writestr(info, b"payload")
    return path


@pytest.mark.parametrize("name, link, raw_name", [
    pytest.param("../evil.txt", False, None, id="parent traversal"),
    pytest.param("sub/../../evil.txt", False, None, id="nested traversal"),
    pytest.param("/abs/evil.txt", False, None, id="absolute path"),
    pytest.param("C:/evil.txt", False, None, id="drive path"),
    pytest.param("x", False, "a\\..\\evil.txt", id="backslashes"),
    pytest.param("link.txt", True, None, id="symbolic link"),
    pytest.param("nul", False, None, id="Windows device name"),
    pytest.param("CON.txt", False, None, id="device name with an extension"),
    pytest.param("sub/Aux.png", False, None, id="device name in a folder"),
    pytest.param("LPT9.", False, None, id="device name with a trailing dot"),
    pytest.param("prn ", False, None, id="device name with a trailing space"),
    pytest.param("a/.../b.txt", False, None, id="a folder of dots only"),
])
def test_zip_extraction_refuses_traversal_absolute_paths_device_names_and_links(tmp_path, name, link, raw_name):
    """Contract: cache safety. A zip with a path that could leave the target, a Windows device name (which opens the
    device, not a file) or a link is refused before anything is written, even the good member listed before the bad
    one. Regression caught: zip-slip writing outside the cache, or `nul`/`con.txt` members hanging or discarding the
    write on Windows."""
    bad = zip_with(tmp_path, name, link, raw_name)
    with pytest.raises(F.FetchError, match="refusing"):
        F.extract_zip(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "evil.txt").exists()


def test_zip_extraction_keeps_names_that_only_look_like_device_names(tmp_path):
    """`console.txt`, `com10.txt`, `nullable/auxiliary.png` and `a.con` are ordinary names on Windows; the device rule
    refuses a whole component only. Regression caught: a prefix match refusing good textures."""
    members = {"console.txt": b"1", "com10.txt": b"2", "nullable/auxiliary.png": b"3", "a.con": b"4"}
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(make_zip(members))
    F.extract_zip(zip_path, tmp_path / "out")
    assert {n: (tmp_path / "out" / n).read_bytes() for n in members} == members


def test_unpacking_turns_a_disk_error_into_a_fetch_error(tmp_path, monkeypatch):
    """Contract: an OS error while unpacking (a folder that cannot be made, a swap that is refused) is a FetchError, the
    one kind `asset_cmd` reports cleanly, and leaves no `x.part`. Regression caught: a PermissionError or a full disk
    escaping as a traceback after a long download."""
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(MAT_ZIP)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a folder is needed", encoding="utf-8")
    with pytest.raises(F.FetchError, match="cannot unpack"):
        F.extract_zip(zip_path, blocker / "out")
    m, net, cache = manifest(), FakeNet(), tmp_path / "cache"
    F.fetch(F.plan(m, cache=cache), net)
    shutil.rmtree(cache / "mat_a" / "x")

    def refuse(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(F, "os", types.SimpleNamespace(replace=refuse))
    with pytest.raises(F.FetchError, match="cannot unpack"):
        F.fetch(F.plan(m, cache=cache), net)
    assert not (cache / "mat_a" / "x.part").exists()


def test_zip_extraction_unpacks_nested_members_and_refuses_oversized_ones(tmp_path, monkeypatch):
    """The good case works (so the refusals above are not vacuous) and a member over the size cap is refused. Regression
    caught: a zip bomb filling the disk."""
    zip_path = tmp_path / "ok.zip"
    zip_path.write_bytes(MAT_ZIP)
    F.extract_zip(zip_path, tmp_path / "out")
    assert (tmp_path / "out" / "maps" / "rough.jpg").read_bytes() == b"jpg-3"
    monkeypatch.setattr(F, "MAX_MEMBER_BYTES", 4)
    with pytest.raises(F.FetchError, match="refusing"):
        F.extract_zip(zip_path, tmp_path / "out2")
    monkeypatch.setattr(F, "MAX_MEMBER_BYTES", 100)
    monkeypatch.setattr(F, "MAX_ZIP_BYTES", 10)
    with pytest.raises(F.FetchError, match="add up"):
        F.extract_zip(zip_path, tmp_path / "out3")
    zip_path.write_bytes(b"not a zip at all")
    with pytest.raises(F.FetchError, match="not a readable zip"):
        F.extract_zip(zip_path, tmp_path / "out4")


@pytest.mark.parametrize("member, change, expected", [
    pytest.param("color.jpg", b"jpg-X", "CRC32", id="same size, other bytes"),
    pytest.param("normal.jpg", b"jp", "has size 2", id="a shorter file"),
    pytest.param("maps/rough.jpg", None, "is missing", id="a deleted file"),
])
def test_an_unpacked_member_that_differs_from_the_zip_is_found_and_unpacked_again(tmp_path, member, change, expected):
    """Contract: an unpacked folder is trusted only when every member of its zip is there with the zip's size and
    CRC32. `unpack_problem` names the member and what differs; `plan` and `fetch` unpack again from the cached zip,
    without a download. Regression caught: a folder that merely exists passing as unpacked, so a render reads a
    half-written or edited texture."""
    m, net = manifest(), FakeNet()
    F.fetch(F.plan(m, cache=tmp_path), net)
    zip_path, folder = tmp_path / "mat_a" / "mat_a_1K.zip", tmp_path / "mat_a" / "x"
    assert F.unpack_problem(zip_path, folder) is None
    target = folder / Path(member)
    original = target.read_bytes()
    if change is None:
        target.unlink()
    else:
        target.write_bytes(change)
    found = F.unpack_problem(zip_path, folder)
    assert found is not None and found.startswith(f"unpacked {member}") and expected in found
    todo = F.plan(m, cache=tmp_path)
    assert keys(todo) == [("mat_a", "mat_a_1K.zip")]
    assert F.fetch(todo, net) == 0 and len(net.calls) == 3
    assert F.unpack_problem(zip_path, folder) is None and target.read_bytes() == original


# -- credits ---------------------------------------------------------------------------------------------------
def test_credits_list_every_ccby_asset_with_attribution_and_changes():
    """Contract (CC BY 4.0): every CC BY asset in use is credited with its title, creator line, licence URL, source page
    and an indication of changes; CC0 sources follow as a courtesy; an alternate that is not used is not credited.
    Regression caught: a CC BY asset shipped without credit."""
    def second_ccby(raw):
        other = {**asset(raw, "mug_a"), "id": "bag_a", "title": "Bag A", "kind": "bag", "changes": "rescaled, paired",
                 "page": f"https://{CCBY}/pages/bag_a"}
        raw["assets"].append(other)

    text = F.credits(manifest(edit=second_ccby))
    ccby, cc0 = text.index("Attribution required"), text.index("CC0")
    for title, page, change in (("Mug A", f"https://{CCBY}/pages/mug_a", "rescaled, recoloured"),
                                ("Bag A", f"https://{CCBY}/pages/bag_a", "rescaled, paired")):
        line = next(ln for ln in text.splitlines() if f"**{title}**" in ln)
        assert "Mugs by Mug Makers, licensed CC BY 4.0" in line and f"https://{CCBY}/licence" in line
        assert page in line and f"Changes: {change}" in line
    assert "Mug B" not in text and "procedural" not in text
    cc0_part = text[cc0:]
    assert ccby < cc0 and "Cee Zero" in cc0_part and "Material A" in cc0_part and f"https://{CC0}/a/mat_a" in cc0_part
    assert "Mug A" not in cc0_part
