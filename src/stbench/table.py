"""The memory table: what the honest scripted memory keeps of a stream (V0 §10), and the mapper that reads depth.

The table is a step log (`table_log.Log`), tracks built from detections, and the fine cells each step's depth observed.

A track is one thing the system believes it has seen: a description (kind and attributes), every place its
bounding-box centre was seen, in the map frame, and the floor hull of each recognisable sighting (`hulls`, the
detection's `floor_hull`: what `ego_now` and `ego_mem` ask the nearest point of). Detections carry no identity, so
tracks are formed by:
1. matching a detection to a track of the same description whose last position is within MATCH_M, one-to-one within a
   frame, nearest first;
2. otherwise, if the description has exactly one track and that track was not matched in this frame, the detection
   is that track seen somewhere else (a unique description re-identifies a moved item);
3. otherwise, a new track.
With perfect identity (the `oracle` baseline) a track is simply an entity.

The mapper (`Mapper`) applies the free-space rule of `grid` to each depth image a step delivers, with the step's camera
and the robot's band: exactly what the gold reads (`freespace.script_states`), so the oracle reproduces the gold. The
cells a step observed are kept by `k`; `states_of` is `grid.latest` over some steps' cells (the latest frame for a
`now` question), and `latest_states` over every logged step's, up to a step: the space as last seen. An
`Answerer` is what a family module's `ANSWERERS` maps a family to. A detection's `center_map` is used when the channel
gives it, else its `center_cam` is moved to the map with the step's camera (never the base's pose: the camera stands
off the base).
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import grid, wall_seen, zones
from .camera import Camera, cam_to_map
from .descriptions import desc_attrs, matches
from .grid_observer import CellFilter, Observer
from .mapspec import MapSpec
from .render_encode import read_png16
from .robots import Robot
from .rules import MATCH_M
from .sized_cache import SizedCache
from .table_log import Log

Answerer = Callable[["Store", dict, dict], dict]   # (store, a question's slots, the ask's context) -> the answer
Cells = dict[tuple[int, int], str]
ZoneBand = tuple[float, tuple[float, float]]        # `zones.band_cells`' key: (the ground's height, the band above it)


@dataclass
class Track:
    id: int
    kind: str
    attrs: dict
    size: list[float]
    sightings: list[list[float]] = field(default_factory=list)  # [k, x, y, z] of the bounding-box centre
    ident: str | None = None  # entity id, only for a system given perfect identity
    # k -> (the centre's (x, y), the detection's floor hull), for the sightings whose detection had a hull
    hulls: dict[int, tuple[tuple[float, float], list[list[float]]]] = field(default_factory=dict)

    @property
    def last(self) -> list[float]:
        return self.sightings[-1]

    @property
    def key(self) -> tuple:
        return (self.kind, tuple(sorted(self.attrs.items())))


class Seen(dict):
    """What a step's depth said of the floor under the robot's band: that band's fine cells (empty for a flier), and
    `zones`, the cells of each zone band the mapper keeps (`Mapper.zone_cells`'s keys). A dict of the robot band, so a
    mapper that only knows the robot band is still a mapper."""

    zones: dict[ZoneBand, Cells]
    walls: frozenset[wall_seen.Bin] | None   # the (wall id, bin) pairs the depth saw (a flier's mapper); None: not kept

    def __init__(self, cells: Cells, zone_bands: dict[ZoneBand, Cells], walls: frozenset[wall_seen.Bin] | None = None):
        super().__init__(cells)
        self.zones = zone_bands
        self.walls = walls


MAX_SEEN_CELLS = 1_000_000   # the fine cells `SEEN` keeps (a few hundred bytes each): about a case's worth a baseline
SEEN: SizedCache[Seen] = SizedCache(MAX_SEEN_CELLS)   # per process: what the steps any `Mapper` here has read said


class Mapper:
    """`grid.observe` over a step's depth image: its fine cells' states, or None for a step with no depth. Given the
    job's zone rules (`zone_rules`: `JobSpec.zones`' values, `slot_rules`: `JobSpec.slot_zones`) and the map it also
    observes each zone band, keeping the cells of the zones of that band (`zones.band_cells`); a flier has no robot
    band, and given the map it records which wall bins the image saw (`wall_seen`: a flier's `clear_path`, kept by `k`
    in `Store.wall_bins`). The rules stay on the mapper so an answerer finds a question's rule by its words."""

    def __init__(self, robot: Robot, cam: Camera, workdir: Path, m: MapSpec | None = None,
                 zone_rules: Iterable[zones.ZoneRule] = (), slot_rules: Iterable[zones.ZoneRule] = ()):
        self.cam, self.workdir = cam, Path(workdir)
        self.band = None if robot.flies else grid.band(robot)
        self.walls = m.walls if robot.flies and m is not None else ()
        self.zone_rules, self.slot_rules = tuple(zone_rules), tuple(slot_rules)
        self.zone_cells = zones.band_cells(m, [*self.zone_rules, *self.slot_rules]) if m is not None else {}
        self._keep = {key: CellFilter(cells) for key, cells in self.zone_cells.items()}
        self._zoned = CellFilter(frozenset().union(*self.zone_cells.values()))   # every cell some zone reads
        # all a step's cells depend on besides its depth image, camera pose and floor: what the mapper reads them with
        self._config = hashlib.blake2b(repr((self.cam, self.band, self.walls, sorted(
            (key, sorted(cells)) for key, cells in self.zone_cells.items()))).encode(), digest_size=16).digest()

    def observe(self, step: dict) -> Seen | None:
        """What the step's depth image says of the floor. The cells are a function of the image's bytes, the camera's
        pose, the floor and the mapper's configuration alone, so a step any mapper of this process has read (another
        baseline over the same case, or over another case of the script) is not worked out again: `SEEN` holds it."""
        if not step.get("depth"):
            return None
        data = (self.workdir / step["depth"]).read_bytes()
        cam_pose, floor_z = step["camera"], step["pose"]["pos"][2]
        key = (hashlib.blake2b(data, digest_size=16).digest(), tuple(cam_pose["pos"]), tuple(cam_pose["rot"]), floor_z,
               self._config)
        if (found := SEEN.get(key)) is None:
            found = self._read(read_png16(io.BytesIO(data)), cam_pose, floor_z)
            SEEN.put(key, found, 1 + len(found) + sum(map(len, found.zones.values())))
        return found

    def _read(self, depth: np.ndarray, cam_pose: dict, floor_z: float) -> Seen:
        pts = grid.points(depth, self.cam, cam_pose["pos"], cam_pose["rot"])
        seen = Observer(pts)
        robot = {} if self.band is None else seen.observe(floor_z, self.band)
        zoned = seen.within(self._zoned)   # a zone's bands are asked of the points over its cells alone
        return Seen(robot, {key: zoned.observe(*key, keep=keep) for key, keep in self._keep.items()},
                    wall_seen.bins_of(pts, self.walls) if self.walls else None)


class Store(Log):
    def __init__(self, mapper: Mapper | None = None) -> None:
        super().__init__()
        self.mapper = mapper
        self.cells: dict[int, dict[tuple[int, int], str]] = {}  # k -> the fine cells that step's depth observed
        self.zone_cells: dict[ZoneBand, dict[int, Cells]] = {}  # zone band -> k -> its zones' cells that step observed
        self.wall_bins: dict[int, frozenset[wall_seen.Bin]] = {}  # k -> the wall bins that step's depth saw (a flier's)
        self.tracks: list[Track] = []
        self._by_ident: dict[str, Track] = {}
        self._by_key: dict[tuple, list[Track]] = {}

    # -- feeding ---------------------------------------------------------------------------------------------
    def add_step(self, step: dict) -> None:
        """Take one step: log it, map its depth (with a mapper) and track its detections; with an `identity` the
        tracks are entities."""
        self.add(step)
        k = step["k"]
        if self.mapper is not None and (cells := self.mapper.observe(step)) is not None:
            self.cells[k] = cells
            for band, zone_cells in getattr(cells, "zones", {}).items():
                self.zone_cells.setdefault(band, {})[k] = zone_cells
            if (walls := getattr(cells, "walls", None)) is not None:
                self.wall_bins[k] = walls
        cam = step.get("camera")
        dets = [(d, d["center_map"] if "center_map" in d else cam_to_map(d["center_cam"], cam["pos"], cam["rot"]))
                for d in step.get("detections", [])]
        if "identity" in step:
            self._by_identity(k, dets, step["identity"])
        else:
            self._associate(k, dets)

    def _new(self, d: dict, k: int, c, ident: str | None = None) -> Track:
        t = Track(len(self.tracks), d["kind"], desc_attrs(d["attributes"]), list(d["extent_m"]), [], ident)
        self._see(t, d, k, c)
        self.tracks.append(t)
        self._by_key.setdefault(t.key, []).append(t)
        return t

    @staticmethod
    def _see(t: Track, d: dict, k: int, c) -> None:
        """The track `t` is seen at the centre `c` in step `k`, with the detection's floor hull if it carries one."""
        t.sightings.append([k, *map(float, c)])
        if d.get("floor_hull"):
            t.hulls[k] = ((float(c[0]), float(c[1])), d["floor_hull"])

    def _by_identity(self, k: int, dets: list, identity: dict[str, str]) -> None:
        for d, c in dets:
            t = self._by_ident.get(identity[d["d"]])
            if t is None:
                self._by_ident[identity[d["d"]]] = self._new(d, k, c, identity[d["d"]])
            else:
                self._see(t, d, k, c)

    def _associate(self, k: int, dets: list) -> None:
        by_key: dict = {}
        for d, c in dets:
            by_key.setdefault((d["kind"], tuple(sorted(desc_attrs(d["attributes"]).items()))), []).append((d, c))
        for key, group in by_key.items():
            tracks = list(self._by_key.get(key, []))
            pairs = sorted((math.dist(t.last[1:], c), i, j) for i, (_, c) in enumerate(group)
                           for j, t in enumerate(tracks) if math.dist(t.last[1:], c) <= MATCH_M)
            det_to, used = {}, set()
            for _, i, j in pairs:
                if i not in det_to and j not in used:
                    det_to[i] = tracks[j]
                    used.add(j)
            for i, (d, c) in enumerate(group):
                if i not in det_to and len(tracks) == 1 and 0 not in used:
                    det_to[i] = tracks[0]
                    used.add(0)
                if i in det_to:
                    self._see(det_to[i], d, k, c)
                else:
                    self._new(d, k, c)

    # -- reading ---------------------------------------------------------------------------------------------
    def matching(self, kind: str, attributes: dict) -> list[Track]:
        """The tracks a description names: of `kind`, carrying every given attribute (`descriptions.matches`)."""
        return [t for t in self.tracks if t.sightings and matches(t.kind, t.attrs, kind, attributes)]

    def states_of(self, ks) -> dict[tuple[int, int], tuple[str, int]]:
        """`grid.latest` over the free-space cells of the steps `ks` that have any: the cells' latest states and
        the step that observed each."""
        return grid.latest({k: self.cells[k] for k in ks if k in self.cells})

    def latest_states(self, until: int | None = None) -> dict[tuple[int, int], tuple[str, int]]:
        """`grid.latest` over the free-space cells of every logged step up to and including `until` (all when None):
        the space as last seen, and the step that saw each cell."""
        return grid.latest({k: c for k, c in self.cells.items() if until is None or k <= until})

    def zone_states(self, ks: Iterable[int], zone: zones.Zone, until: int | None = None
                    ) -> dict[tuple[int, int], tuple[str, int]]:
        """`grid.latest` over the cells of the steps `ks` that have any, observed on the zone's own ground and band:
        what `zones.zone_state` and `zones.coverage` take. With `until`, only the steps up to and including it."""
        seen = self.zone_cells.get((zone.base_z, zone.band), {})
        return grid.latest({k: seen[k] for k in ks if k in seen and (until is None or k <= until)})
