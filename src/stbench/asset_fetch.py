"""Fetch the manifest's assets into the cache and credit them. The only module that opens a network connection (a test
scans for it).

A download streams into `<file>.part` and replaces the final file only when it has arrived whole: a body shorter than
the server's own Content-Length is a cut connection and is refused, a body past the size cap is stopped, and any failure
deletes the `.part`. A redirect off the source's host list is refused before it is followed. A file already in the
cache is never downloaded again. A zip is kept as downloaded and unpacked to `<asset>/x/`; `plan` and `fetch` compare
every unpacked file with the zip (size, CRC32) and unpack again where one differs. A file of a generated source (a
recipe makes it, `asset_manifest`) is never downloaded: `plan` leaves it out. Nothing here runs unasked: `asset_cmd`
needs `--yes` for a download.
"""

from __future__ import annotations

import http.client
import os
import re
import shutil
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .asset_manifest import (
    EXTRACT_DIR,
    GENERATED,
    Asset,
    AssetError,
    AssetFile,
    Manifest,
    Source,
    cache_dir,
    extract_dir,
    file_path,
)

CHUNK = 1 << 20
MAX_FILE_BYTES = 200_000_000  # cut-off for one download
MAX_MEMBER_BYTES = 200_000_000  # largest file allowed inside a zip
MAX_ZIP_BYTES = 1_000_000_000  # largest total allowed inside a zip
TIMEOUT_S = 60
USER_AGENT = "stbench-assets"
# A Windows device name, with or without an extension, is a device wherever it sits in a path (`nul.txt` is NUL).
_DEVICE = re.compile(r"(con|prn|aux|nul|conin\$|conout\$|(com|lpt)[1-9\u00b9\u00b2\u00b3])(\..*)?", re.IGNORECASE)
# (url, allowed hosts) -> a response: a context manager with read(n) and geturl()
Opener = Callable[[str, tuple[str, ...]], object]


class FetchError(AssetError):
    """A download or unpacking was refused or failed."""


@dataclass(frozen=True, eq=False)
class FileTask:
    asset: Asset
    file: AssetFile
    source: Source
    cache: Path

    @property
    def dest(self) -> Path:
        return file_path(self.asset, self.file, self.cache)

    @property
    def unpacked(self) -> Path:
        return extract_dir(self.asset, self.cache)

    @property
    def host(self) -> str:
        return urlsplit(self.file.url).hostname or ""

    @property
    def label(self) -> str:
        return f"{self.asset.id} {self.file.path}"


# -- network -------------------------------------------------------------------------------------------------------
def check_url(url: str, hosts: tuple[str, ...], what: str = "url") -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in hosts:
        raise FetchError(f"{what} {url} is not https on one of the source's hosts {list(hosts)}; if that host really "
                         "is the source's, add it to the source's hosts in the manifest (a reviewed edit)")


class _HostGuard(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to a host on the list, so a source cannot send us to an arbitrary server."""

    def __init__(self, hosts: tuple[str, ...]):
        self.hosts = hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.hosts, "redirect to")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urllib_opener(url: str, hosts: tuple[str, ...]):
    check_url(url, hosts)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.build_opener(_HostGuard(hosts)).open(request, timeout=TIMEOUT_S)


# -- unpacked zips ------------------------------------------------------------------------------------------------
def crc32_of(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
    return crc


def unpack_problem(zip_path: Path, folder: Path) -> str | None:
    """Why `folder` is not the unpacked `zip_path`, or None when every member is there with the zip's size and CRC32. An
    interrupted or edited unpack is caught here, so a render never reads a half-written texture."""
    if not folder.is_dir():
        return "not unpacked"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or _refused(info):
                    continue  # a refused member was never unpacked; `_unpack` reports it
                target = folder.joinpath(*info.filename.split("/"))
                if not target.is_file():
                    return f"unpacked {info.filename} is missing"
                if target.stat().st_size != info.file_size:
                    return f"unpacked {info.filename} has size {target.stat().st_size}, the zip's is {info.file_size}"
                if crc32_of(target) != info.CRC:
                    return f"unpacked {info.filename} differs from the zip (CRC32)"
    except (zipfile.BadZipFile, OSError) as e:
        return f"cannot compare with the zip: {e}"
    return None


# -- planning ------------------------------------------------------------------------------------------------------
def plan(m: Manifest, only: Iterable[str] | None = None, include_alternates: bool = False,
         cache: Path | None = None) -> list[FileTask]:
    """The files that are missing or not (fully) unpacked, of the downloaded assets (a generated asset's recipe makes
    its file). An alternate is included when named in `only` or when `include_alternates` is set."""
    cache, wanted = cache or cache_dir(), None if only is None else set(only)
    if unknown := sorted((wanted or set()) - {a.id for a in m.assets}):
        raise AssetError(f"no such asset(s): {', '.join(unknown)}")
    tasks = []
    for a in m.assets:
        if wanted is not None and a.id not in wanted:
            continue
        if a.alternate_for and not include_alternates and (wanted is None or a.id not in wanted):
            continue
        if m.sources[a.source].kind == GENERATED:
            continue
        for f in a.files:
            t = FileTask(a, f, m.sources[a.source], cache)
            if not t.dest.is_file() or (f.extract and unpack_problem(t.dest, t.unpacked)):
                tasks.append(t)
    return tasks


def plan_bytes(tasks: Iterable[FileTask]) -> int:
    """Bytes to download for the tasks, by each asset's estimate, counted once per asset."""
    return sum(a.download_size() for a in {t.asset.id: t.asset for t in tasks}.values())


# -- downloading ---------------------------------------------------------------------------------------------------
def _announced(resp) -> int | None:
    """The Content-Length the server announced, or None when it announced none or the body is content-coded (the bytes
    read are then not the bytes announced). `http.client` hands back a short body without an error, so this is the only
    check that a cut connection is not taken for a whole file."""
    headers = getattr(resp, "headers", None)
    if headers is None or (headers.get("Content-Encoding") or "identity").strip().lower() != "identity":
        return None
    value = (headers.get("Content-Length") or "").strip()
    return int(value) if value.isdigit() else None


def _download(t: FileTask, opener: Opener) -> None:
    """Stream one file to its place through a `.part`. Raises unless it arrived as long as the server said (a connection
    cut short is not a short file) and within `MAX_FILE_BYTES`."""
    part, n = t.dest.with_name(t.dest.name + ".part"), 0
    t.dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with opener(t.file.url, t.source.hosts) as resp:
            check_url(resp.geturl(), t.source.hosts, "redirect to")
            announced = _announced(resp)
            with open(part, "wb") as out:
                while chunk := resp.read(CHUNK):
                    n += len(chunk)
                    if n > MAX_FILE_BYTES:
                        raise FetchError(f"{t.label}: more than {MAX_FILE_BYTES} bytes arrived; stopped")
                    out.write(chunk)
        if announced is not None and n != announced:
            raise FetchError(f"{t.label}: {n} of the {announced} bytes the server announced arrived; connection cut")
        os.replace(part, t.dest)
    except (OSError, http.client.HTTPException) as e:  # a dropped chunked body is an IncompleteRead, not an OSError
        raise FetchError(f"{t.label}: {e}") from e
    finally:
        part.unlink(missing_ok=True)


def _refused(info: zipfile.ZipInfo) -> str | None:
    """Why a zip member may not be unpacked, or None: no absolute or drive path, no `..`, no backslash, no Windows
    device name (`nul`, `con.txt`), no link and no member over MAX_MEMBER_BYTES."""
    parts = info.filename.rstrip("/").split("/")
    if (info.filename.startswith("/") or "\\" in info.filename or ":" in info.filename
            or any(not p.rstrip(" .") or _DEVICE.fullmatch(p.rstrip(" .")) for p in parts)):
        return f"the member path {info.filename!r}"
    if (info.external_attr >> 16) & 0o170000 == 0o120000:
        return f"the link {info.filename!r}"
    if info.file_size > MAX_MEMBER_BYTES:
        return f"{info.filename!r}, {info.file_size} bytes"
    return None


def extract_zip(path: Path, into: Path) -> None:
    """Unpack a zip into `into`. Every member is checked first, so a refused zip writes nothing (see `_refused`)."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            for info in infos:
                if why := _refused(info):
                    raise FetchError(f"{path.name}: refusing {why}")
            if sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
                raise FetchError(f"{path.name}: the members add up to more than {MAX_ZIP_BYTES} bytes")
            for info in infos:
                target = into.joinpath(*info.filename.rstrip("/").split("/"))
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as e:
        raise FetchError(f"{path.name}: not a readable zip: {e}") from e
    except OSError as e:
        raise FetchError(f"{path.name}: cannot unpack: {e}") from e


def _unpack(t: FileTask) -> None:
    """Unpack into `x.part`, then swap it in, so `x/` is whole or absent."""
    tmp = t.unpacked.with_name(EXTRACT_DIR + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        extract_zip(t.dest, tmp)
        shutil.rmtree(t.unpacked, ignore_errors=True)
        os.replace(tmp, t.unpacked)
    except OSError as e:
        raise FetchError(f"{t.label}: cannot unpack: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(tasks: Iterable[FileTask], opener: Opener | None = None) -> int:
    """Download and unpack what the tasks need; returns the number of files downloaded. A file already in the cache is
    never touched; a zip whose unpacked folder differs from it is unpacked again."""
    tasks, opener, downloaded = list(tasks), opener or urllib_opener, 0
    for t in tasks:
        fresh = not t.dest.is_file()
        if fresh:
            _download(t, opener)
            downloaded += 1
        if t.file.extract and (fresh or unpack_problem(t.dest, t.unpacked)):
            _unpack(t)
    return downloaded


# -- credits -------------------------------------------------------------------------------------------------------
def credits(m: Manifest) -> str:
    """The attribution file for a dataset (`CREDITS.md`): CC BY assets first, each with its creator, licence, page and
    what we changed, as CC BY 4.0 requires; then the CC0 sources, as a courtesy."""
    lines = ["# Credits", "", "Third-party assets used to render this dataset.", "",
             "## Attribution required", ""]
    for a in sorted((a for a in m.assets if a.licence.startswith("CC-BY") and not a.alternate_for), key=lambda a: a.id):
        s = m.sources[a.source]
        lines.append(f"- **{a.title}** ({a.kind}). {s.attribution}. Licence: {s.licence_url} Source: {a.page} "
                     f"Changes: {a.changes}.")
    lines += ["", "## CC0, no attribution required (listed as a courtesy)", ""]
    for key, s in m.sources.items():
        used = sorted((a for a in m.assets if a.source == key and a.licence == "CC0-1.0" and not a.alternate_for),
                      key=lambda a: a.id)
        if used:  # a CC0 source's own credit line, where it asks for one (Poly Haven's API terms), comes first
            credit = f" {s.attribution}." if s.attribution else ""
            lines.append(f"- {s.name} ({s.licence_url}).{credit} " + "; ".join(f"{a.title} ({a.page})" for a in used))
    return "\n".join(lines) + "\n"
