"""The rows of `stbench realism report` (M1's R.2): the scripts' numbers (`realism_cmd._script`), the
references' (`realism_data.reference_stats`) and the crop scores (`realism_crops.crop_scores`) set against the targets
of `realism_report`. Pure: dicts in, `realism_report.Row`s out, so a test can hand-build every input."""

from __future__ import annotations

from . import realism_noise as N
from . import realism_report as R


def _pool(scripts: list[dict], key: str) -> list:
    return [x for s in scripts for x in s.get(key, [])]


def _spread_row(name: str, values: list[float], bounds) -> R.Row:
    sp = R.spread(values)
    if sp is None:
        return R.Row(name, None, "none" if bounds is None else R.words(bounds), "no data")
    ok = None if bounds is None else (R.within(sp["min"], bounds) and R.within(sp["max"], bounds))
    return R.Row(name, sp, "none" if bounds is None else R.words(bounds), "reported" if ok is None else "ok" if ok else
                 "outside")


def _by_band(tables: dict[str, dict | None], key: str) -> dict:
    """`{who: {band: value}}` of one statistic of the noise tables (`realism_noise.table_by_band`), a table left out
    where it is None, numbers to three figures."""
    return {who: {band: None if row[key] is None else float(f"{row[key]:.3g}") for band, row in t.items()}
            for who, t in tables.items() if t is not None}


def noise_rows(patches: list[dict], refs: dict | None) -> list[R.Row]:
    """Noise, from the flat patches of the frames and of the references (the robot's own frames with their flat fills
    left out): the luma σ (8-bit levels), the colour noise over it and the lag-1 correlation, each by luminance band and
    over all patches."""
    tables = {"frames": {**N.table_by_band(patches), "all": N.overall(patches)},
              "references": None if refs is None else {**N.table_by_band(refs["patches"]),
                                                       "all": N.overall(refs["patches"])}}
    return [R.row(f"noise, flat patches: {what} by luminance band (the references' beside it)",
                  _by_band(tables, key), None)
            for what, key in (("luma σ in 8-bit levels", "sigma"), ("chroma noise over luma noise", "chroma"),
                              ("lag-1 correlation of the residual", "lag1"))]


def reference_rows(refs: dict | None) -> list[R.Row]:
    """What the references say of the robot's own camera: its exposure, and the clipping with and without its fills."""
    if refs is None:
        return [R.row("the references' exposure (the robot's own frames, fills left out)", None, None)]
    return [R.row("the references' exposure: mean sRGB luminance of a picture (the robot's own frames, fills left out)",
                  refs["exposure"], None),
            R.row(f"the references' fills, clipped and crushed ({refs['pictures']} {refs['basis']})", {
                "fills": refs["fills"], "clipped": refs["clipped"], "clipped_with_fills": refs["clipped_with_fills"],
                "crushed": refs["crushed"]}, None)]


def build_rows(scripts: list[dict], refs: dict | None, recog_lum: float, crops: dict | None) -> list[R.Row]:
    """The report's rows from the scripts' numbers (`realism_cmd._script`), the references' (`reference_stats`) and the
    crop scores (`realism_crops.crop_scores`)."""
    rows = [_spread_row("entities a script", [s["counts"]["total"] for s in scripts], R.TARGETS["entities"]),
            _spread_row("floor clutter a script", [s["counts"]["floor_clutter"] for s in scripts], R.FLOOR_SHARE),
            _spread_row("floor clutter a m2 of free floor",
                        [s["counts"]["floor_clutter"] / s["free_m2"] for s in scripts], (0.5, 1.2))]
    standing = [f["objects"] for f in _pool(scripts, "standing_objects")]
    rows.append(R.row("objects in the median standing frame (entities and dressing pieces of 215 px, trim left out)",
                      R.median(standing), R.TARGETS["objects, standing frames"]))
    rows.append(R.row("entities alone in the median standing frame", R.median(f["entities"] for f in _pool(
        scripts, "standing_objects")), None))
    rows.append(R.row("objects in the median vacuum frame", R.median(f["objects"] for f in _pool(
        scripts, "script_objects")), None))
    cards = {key: _pool([{key: s.get("cards", {}).get(key, [])} for s in scripts], key)
             for key in ("lamps", "daylight", "worktop")}
    light = R.light_table(cards)
    lamp_bounds = R.TARGETS["lamps only, a room's median lux"]
    rows += [R.row(f"lamps only, {room}, lux", lux, R.LAMP_ROOM_BOUNDS.get(room, lamp_bounds))
             for room, lux in light["lamps"].items()]
    rows.append(R.row("lamps only, the worktop, lux", light["worktop"], R.TARGETS["lamps only, the worktop's lux"]))
    rows += [R.row(f"daylight only, {room}, lux", lux, None) for room, lux in light["daylight"].items()]
    rows.append(R.row("the lamps' cast (red/green, blue/green)", light["cast"], None))
    cct = R.lamp_cct(_pool(scripts, "lamps"))
    cct_bounds = R.TARGETS["lamps' colour temperature, K"]
    rows.append(R.Row("the lamps' colour temperature, K", cct, R.words(cct_bounds), "no data" if cct is None else "ok"
                      if R.within(cct["min"], cct_bounds) and R.within(cct["max"], cct_bounds) else "outside"))
    rooms = R.by_room(_pool(scripts, "frames"))
    exp = R.TARGETS["exposure, mean sRGB luminance"]
    rows += [R.row(f"exposure, {room}: median mean sRGB luminance (median EV {v['ev']:.2f}, {v['frames']} frames)",
                   v["lum"], exp) for room, v in rooms.items()]
    rows += reference_rows(refs)
    refshare = None if refs is None else {"clipped": refs["clipped"], "crushed": refs["crushed"]}
    rows += [R.row(f"clipped and crushed shares, {room}", {"clipped": v["clipped"], "crushed": v["crushed"],
                                                          "references": refshare}, None) for room, v in rooms.items()]
    rows += noise_rows(_pool(scripts, "patches"), refs)
    stops = _pool(scripts, "stops")
    rows.append(R.row("dynamic range in stops, bright (window or sun over the room's median) and dark (median over 5%)",
                      None if not stops else {"bright": R.spread([s["bright"] for s in stops]),
                                              "dark": R.spread([s["dark"] for s in stops])}, None))
    windows = [w for s in scripts if s.get("daylight") for w in s.get("windows", [])]
    sp = R.spread(windows)
    rows.append(R.row("a window in a daylight frame over the room's median, stops" + (
        "" if sp is None else f" (the median of {len(windows)} frames with a window in view; least {sp['min']:.1f}, "
                              f"most {sp['max']:.1f})"), None if sp is None else sp["median"],
                      R.TARGETS["window over the room's median, stops"]))
    kinds = sorted({k for s in scripts for k in s["relevance"]})
    floor = R.luminance_floor(_pool(scripts, "sightings"), kinds, recog_lum)
    rows.append(R.row("luminance floor: percentiles of lum over sightings of at least 1,024 px of the job's kinds",
                      floor["percentiles"], None))
    rows.append(R.row("candidate RECOG_LUM (the midpoint of the 2nd and 5th percentiles)", floor["candidate"], None))
    share_bounds = R.TARGETS["luminance floor, share of a kind under RECOG_LUM"]
    rows += [R.row(f"luminance floor, {kind}: share under {recog_lum:g} ({v['sightings']} sightings)", v["share_under"],
                   share_bounds) for kind, v in floor["per_kind"].items()]
    for protocol, label in (("forced", "forced choice"), ("free", "free naming")):  # the forced choice is scored first
        scored = None if crops is None else crops.get(protocol)
        if scored is None:
            continue
        rows.append(R.row(f"crop check, {label}: confused crops", len(scored["confused"]), (None, 0)))
        for kind, v in scored["per_kind"].items():
            rows.append(R.row(f"crop check, {label}, {kind}: right, kind and colour ({v['right']} of {v['crops']})",
                              v["rate"], (R.CROP_RIGHT, None) if kind in kinds else None))
    return rows
