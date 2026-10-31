# The dataset's files

Every file of a dataset under the redesign's layout, its fields, units and frames, the two things a command writes
beside a dataset (the folder of a run and the report of a check) and the folder `stbench export` writes from a dataset
("A release" below).
The code that writes them: `stbench.episode` (`write_script`), `stbench.case` (the layout, the index, `write_case`),
`stbench.timeline` and `stbench.timeline_ego`, `stbench.robots`, `stbench.render_cli`, `stbench.visibility`,
`stbench.perception`, `stbench.questions`, `stbench.cmd_run`, `stbench.check` and `stbench.export_cmd`.

The released dataset holds no schema, digest or commit stamp, and no version but the package's own in `release.json`. A
dataset built before it may hold keys this page does not list (`schema`, `version`, `generator`, `compiler`, `render`,
`git_commit`, `git_dirty`, `*_sha256`, `hash`): earlier builds' stamps, which every reader ignores.

Release 0.3's format, `episodes/<id>/` with its mission, floor plan and protocol files, is not described here, and that
release is not part of the public repository.

```text
DATASET/
  scripts/<sid>/                   agent-facing ONLY: the robot's side of one script. <sid> = 16 random hex characters
    robot.json                       the robot's spec (`robots.robot_doc`)
    sensors/  camera.json  pose.jsonl  rgb/%06d.jpg
    perception/  oracle.jsonl
  private/                         never handed to the system under test
    index/<id>.json                  one entry a script or case
    <sid>/  timeline.json  instances/%06d.png  depth/%06d.png  instance_map.json  visibility.jsonl
            perception_ids.jsonl  render_log.json
    <cid>/  case.json  questions.jsonl  gold.jsonl
  check_report.json                what `stbench check` writes
```

Runs are not part of the dataset (0.11.1; before, `DATASET/runs/...`): `stbench run` and `stbench eval` write their
`runs/...` folders (below) under the current directory, so a run started outside the dataset leaves it as it was
downloaded.

A **script** is one robot doing one job on one map for a number of tasks; a **case** is a window of a script's tasks
that a system is run on from its first step (`short`: 1 task, `medium`: 2-3, `long`: 4-6; every release since 0.4.1
ships all three). `<sid>` and `<cid>` are random and say nothing about what they name; an
equal index entry reuses its id, so a rebuild leaves no duplicate. There is no mission, floor plan or protocol file:
**briefs are never files**, the harness composes them at run time.

**What reaches a system.** `scripts/<sid>/` is agent-facing in the sense that it holds no truth, not in the sense that a
system is handed the folder. Of it only `robot.json` and `sensors/camera.json` reach a system whole. The frames, the
pose rows and the detections are in script numbering: they hold the whole script, future frames included, and
each keyframe's place in it. A system gets them only through the stream (`stbench.stream`), rebased to the case (below).

**The agent has a camera, not a depth sensor** (0.6.0). The agents' one channel is `rgb_pose`: the camera image, the
controller's own pose and clock, and the task status. The recorded depth is the benchmark's instrument: it certifies a
sighting and places a thing for a gold, and it is kept in `private/<sid>/depth/`, never in `scripts/`. Only the gate
channel `detections` (a perfect detector, no agent) is sent depth, copied from there. `rgbd_pose` no longer exists, and a
run asking for it is refused. The depth fields in the public `robot.json` (`depth`) and `sensors/camera.json`
(`depth_unit`, `near_m`, `far_m`) describe the scorer's sensor, and no agent channel delivers its images.

`private/` means "never handed to a system", not "secret": the dataset publishes it (the instance images aside), so
anyone can score and check a case and recompute every gold from `timeline.json` and `visibility.jsonl`. Keyframe
numbers `k` count from 0 in a script and index every per-frame file (`rgb/%06d.jpg`).

## Conventions

- **Map frame:** metres, x east, y north, z up. A position `pos` is `[x, y, z]`. An entity's `pos` is the base centre:
  the centre of the bottom face of its bounding box, `size = [x, y, z]` the extents along its local x and y and its
  height, so the footprint is a rectangle centred on `pos.xy` with `size[0]` along the yaw direction. A keyframe has
  two positions, the camera's (`pos`) and the robot's base (`base`), below.
- **Rotations** are unit quaternions `[w, x, y, z]` mapping the body frame (x forward, y left, z up) to the map frame.
  Placements turn about z only. The camera looks along the body x axis; the OpenCV camera frame (x right, y down, z
  forward) relates to the body frame by `x_cv = -y_body, y_cv = -z_body, z_cv = x_body`.
- **Heading and pitch.** A keyframe's camera rotation is `qz(yaw) * qy(-pitch)`: `yaw` is the base's heading,
  counter-clockwise from +x, `pitch` the camera's, positive looking up, roll zero. A heading is never read from the
  camera (a drone's camera at nadir has no heading): the pose row carries it (`pose.yaw_deg`).
- **Time:** `t` is seconds since the script's `episode.start_clock`; a `clock` is the local ISO time, the start clock
  plus `t` rounded half up to a second (a clock a system is sent is rebased by the case and carries tenths of a second,
  below). All intervals are half-open. Two keyframes of one task are at least the robot's
  `min_dt_s` (1.0 s) apart, so every clock of a task is unique at whole seconds. A system never sees `t`: pose and oracle
  rows hold `clock` only, and a case re-bases it (below).
- **Pixels:** `u` is the column and `v` the row, from the top left; `bbox` and `bbox_px` are `[u0, v0, u1, v1]` with
  `u1` and `v1` exclusive. Image coordinates are continuous: pixel column `j` spans `u` in `[j, j + 1)`; depth is planar
  (camera-frame z).
- **Depth:** `private/<sid>/depth/%06d.png` is 16-bit, planar depth in the unit `sensors/camera.json` names, 0 meaning no
  surface or beyond what the unit can hold. The unit is the robot's (`"mm"` for the vacuum, helper and tote robot, up to
  65.5 m; `"cm"` for the drone, up to 655 m): scale by `camera.depth_scale` (metres per unit); a reader that assumes
  millimetres is wrong by ten on a drone's. The robot's range, `near_m` and `far_m` in `camera.json`, cuts the image:
  `render` writes 0 nearer than `near_m` and farther than `far_m` (`render_encode.cut_depth`; a depth exactly on a bound
  stays), and wherever the robot's own body is (`render_encode.self_filter`, which zeroes the instance image there too).
  The RGB is not cut: the renderer's near clip is the robot's (`render_encode.rgb_clip_m`: `near_m`, at most 5 mm), so a
  surface that depth reads is whole in the picture.

## `scripts/<sid>/`

| File | Content |
| --- | --- |
| `robot.json` | `robots.robot_doc`: the robot's `id` and `name`, `flies`, its `body` (shape, size, `climb_m`, `clearance_m`, `under_margin_m`), `speeds_mps` per phase, `turn_dps`, its `camera` (image size, `hfov_deg`, `mount_m` in the base frame, the pitch and its limits, the image processor `isp`), `depth` (unit, `near_m`, `far_m`), `status` (the closed `phases`, the other status `keys` with their types, the `start` set of each task's first step), `frames` (the keyframe policy), `tools` and `sources`. A system parses it with `robots.from_doc`. |
| `sensors/camera.json` | `width height hfov_deg fx fy cx cy depth_unit near_m far_m` (`robots.camera_doc`): 640x480, 80 degrees, pinhole, no distortion; `Camera.load` reads it. |
| `sensors/pose.jsonl` | One row per keyframe, in script numbering (a case re-bases it): below. The robot is localised: no drift. |
| `sensors/rgb/%06d.jpg` | 640x480 RGB JPEG, one per keyframe. Blender Cycles renders a scene-linear image and the robot's camera model (`camera_model.develop`, with the `isp` of `robot.json`) develops it into the JPEG, which holds no EXIF, profile or comment (`render_log.json`, below). |
| `perception/oracle.jsonl` | The perfect-detections channel, one row per keyframe (below). |

A pose row, with sorted keys, one line in the file: the camera, the base, the clock, the status with its task and, for a
job that states one, the subgoal:

```json
{"camera": {"pitch_deg": 5.0, "pos": [5.6, 5.42, 0.08], "rot": [0.706434, -0.030844, -0.030844, -0.706434]},
 "clock": "2026-03-02T08:00:00", "k": 0,
 "pose": {"pos": [5.6, 5.55, 0.0], "rot": [0.707107, 0.0, 0.0, -0.707107], "yaw_deg": -90.0},
 "status": {"battery_pct": 100, "brush": false, "phase": "docked", "task": 0},
 "subgoal": "leaving the dock to clean the kitchen"}
```

`pose` is the base (`pos`, `rot` about z, `yaw_deg` the heading); `camera` is derived from it by `robots.camera_pose`
(the mount turned by the heading, added to the base). `status` is the robot's own: its `phase`, the `task` index in the
script and the keys its spec names (`battery_pct`, `brush`, ...); it carries no room, region or obstacle. Degrees are
rounded to 3 places.

`subgoal` is a top-level key beside `status`, present in every row of a script whose job states subgoals
(`JobSpec.subgoals`; `vacuum/1` states none, and its rows have no such key): one sentence of what the robot is doing at
that keyframe, in the job's own words. It is computed from the timeline when the rows are written and is not stored in
it. A subgoal at keyframe k reads the task plans, the
phases and decisions up to k and the responses of decisions before k, never a coin decided at or after k; the leak scan
reads its text like any file's. The stream hands it to a system with `status`.

**`perception/oracle.jsonl`**: `{"k", "clock", "detections": [...]}`, no `t`. A detection is one entity that covers at
least its kind's seen bound in that frame (215 pixels, 0.07% of the image, unless `seen_table` says otherwise), in a
seeded shuffle:

| Key | Content |
| --- | --- |
| `d` | An 8-hex key, fresh for every detection. It names the detection within the frame only; telling objects apart over time is the memory's job. |
| `kind`, `attributes` | What it is and its visual attributes (`color`, `size`). |
| `bbox_px` | The pixel bounding box of its visible pixels. |
| `center_cam`, `center_map` | The true centre of its bounding box, in the camera frame of that keyframe (metres) and in the map frame. |
| `extent_m` | The entity's size `[x, y, height]`. |
| `floor_hull` | What the frame shows of the entity, on the floor (0.7.0): the convex hull of its own pixels' (x, y), unprojected through the frame's depth, as `[[x, y], ...]` in the map frame, to the millimetre, counter-clockwise from the lowest vertex. `null` when the entity is not recognisable in the frame (`recognise.recognisable`: pixels, light, a box clear of the edge, a depth). It is computed by `nearest_part.floor_hull`, the function the `ego_now` and `ego_mem` gold uses, so a system that keeps the hulls of a thing's sightings reproduces those golds (the point of the hull nearest the camera): the pictures alone give a table the box but not the pixels. **A person's `floor_hull`** (0.8.0) is set exactly when the frame holds person evidence (`person_seen.evidence`: 1,024 pixels, `RECOG_LUM`, a depth at every pixel, the feet tied to the floor at the job's clearance: a pixel of the feet within max(2 cm, a pixel's height) of the floor and within C/2 of the hull's nearest point to the camera, along the floor, read to within C/2 at one pixel; and a box off the bottom edge of the image; the top and the sides may cut the person), not when `recognise.recognisable` holds, whose edge rule would drop the close passes. The gold of `closest_person` and `person_mem` and the oracle's hull call the same function, so a table that keeps a person's hulls reproduces those golds. Every other entity's `floor_hull` is as above. |
| `state` | Doors only: `{"open": true}`. |

The keys and the order of each keyframe are drawn from a generator of its own (the script's map, seed, robot and job, a
fixed seed key, and the keyframe's script `k`), so a keyframe's keys do not depend on the frames before it, but they do
depend on the keyframe's place in its script. The file is therefore never handed to a system whole. **The stream
delivers a frame's detections in a canonical order and under keys of its own** (`stream._rekey`): the detections are
sorted by content (each one's JSON without `d`, keys sorted) and the i-th takes `d` = the first 8 hex characters of the
SHA-256 of the case id, `"d"`, the case's own `k` and i (`seeding.stable_hex`). Neither the order nor the keys then say
anything of the seed or of the keyframe's place in the script or of where the window starts: the same frame in two cases
of one script has two sets of keys, and one case gives its frame the same keys on every run. The privileged `identity`
map the `oracle` baseline is sent is re-keyed to match. Only a run on the `detections` channel is sent them.

## `private/`

### `index/<id>.json`

One small file a script or case (written by temp file and `os.replace`, so parallel writers never rewrite a shared
file):

```json
{"kind": "script", "map": "apartment_a", "seed": 7, "robot": "vacuum",
 "job": "vacuum/2", "tasks": 5, "split": "dev"}
{"kind": "case", "script": "<sid>", "a": 0, "b": 0, "length": "short"}
```

`split` is `dev`, `test` or `heldout`; a held-out script lives in a dataset root of its own, so an export over the public
root never holds it.

A held-out layout's map lives in its held-out root too, at `private/maps/<name>.json`, never in the package or the
repository. Every command finds it by name through `STBENCH_MAPS` (folders separated by `;` on Windows, `:` elsewhere,
searched before the bundled maps; one name, one map), set on each command that reads that root and never in the shell
of a `run --local` (`src/stbench/maps/README.md`).

### `<sid>/timeline.json`

The world timeline: the single source of truth that visibility, questions and gold are
computed from. Canonical JSON: fixed key order, 6-decimal floats, one record a line, every field always written.

| Section | Content |
| --- | --- |
| `episode` | `map seed robot job start_clock`: the script's inputs; `job` is the job's id (`vacuum/2`, `helper/1`, `amr/1`, `drone/1`). |
| `world.regions` | Rooms `{id, name, kind, polygon, level, z0, z1}` (`R1`, ...). `kind` is the room's kind word (the map's `regions[i].kind`, else its name). |
| `world.entities` | `{id, kind, attributes, mobility, size, asset, under_m}`: `id` is `E0001`...; `mobility` is `structure`, `fixed`, `light`, `carried`, `mover` or `clutter` (the mobility table in `timeline.py`: what collides, rests on what, obstructs a ground robot and may be asked about; clutter never is); `asset` is the manifest model of a carried or light item; `under_m` is the height of an open underside (0: solid to the floor). Doors are `fixed` entities with an `open` state and the `hinge` and `swing` of their map opening in `attributes`. |
| `world.placements` | `{entity, t0, t1, pos, rot, on}`: where an entity stands from `t0` to `t1` (null: until the end); `on` is the entity it rests on. An entity exists only while a placement covers `t`. |
| `world.states` | `{entity, key, t0, t1, value}`, e.g. a door's `open`. |
| `ego.keyframes` | The keyframes `{k, t, pos, rot, base, yaw, cam_pitch, task, phase, status}`: `pos` and `rot` the camera's, `base` the robot's base origin, `yaw` its heading and `cam_pitch` the camera's pitch (radians), `task` an index into the tasks, `phase` and `status` the robot's. |
| `ego.tasks` | `{index, k0, k1, plan, targets}`: keyframes `[k0, k1)` (half-open, contiguous from 0), the assignment `plan {kind, params}` (world-independent words the brief may use) and the private `targets` (region or entity ids). |
| `ego.decisions` | `{k, kind}` and, when the job gave one, `response`: the last step before the robot commits to a manoeuvre, a kind of the job's closed vocabulary, and what the robot did after the ask (`go`, `abort`, `skip` or `elsewhere`, following the world's state at the decision). The key is optional: an ego without responses is written as it was before it existed, and a reader takes an absent key as no response. |

### `<sid>/visibility.jsonl`

One row per keyframe, listing every entity with at least one pixel in that frame's instance image.
`{"k", "entities": [{"id", "px", "bbox", "depth_med_m", "lum", "rgb_med", "centroid_cam",
"centroid_map"}]}`: `px` is the pixel count, `lum` the mean relative luminance (0 to 1), `rgb_med` the median of each sRGB
channel (0 to 255, whole numbers) over the entity's pixels in the delivered JPEG, `depth_med_m` the median depth in
metres. The depth and centroid fields are null when the entity has no depth pixel. The compiler reads `px`, `bbox`, `lum`
and `depth_med_m` (the recognisability floor, `recognise.py`) and `rgb_med` (the colour a render shows a thing in,
`colour_seen.py`); the centroids feed the oracle channel or are diagnostics. Computed from the camera block of the pose
rows. `rgb_med` is 0.5.0-control's; an older root's log lacks it, and the compiler then takes a
thing's colour word at its label, a neutral label (black, grey, white) for all three neutrals
([questions.md](questions.md), "Real things and look-alikes"). `stbench perceive` rebuilds the log without
a new render.

### `<sid>/perception_ids.jsonl`

`{"k", "ids": {"<d>": "<entity id>"}}`: the entity behind each detection key of `oracle.jsonl`. It is the oracle
baseline's privileged input (perfect identity).

### `<sid>/instances/%06d.png`, `instance_map.json`

The per-frame instance-index PNGs (16-bit, one `pass_index` per entity) and `{pass_index: entity id}`. Only
`stbench perceive` reads them, to rebuild `visibility.jsonl`; a release leaves them out ("A release", below).

### `<sid>/render_log.json`

Provenance of the render (`render_cli.write_log`): what Blender did, what the frames were made with
and how they were developed. `stbench render` writes it and `stbench develop` rewrites its last part. Blender's
side of it (`render_meta.json`, below):

| Key | Content |
| --- | --- |
| `blender`, `device` | The Blender version and the Cycles device (`type`, `names`); a render on no GPU is refused. |
| `samples`, `fast`, `appearance`, `seed` | RGB samples per pixel, whether `--fast` was given, `placeholder` or `assets`, and the render seed. |
| `resolution`, `frames`, `n_frames` | `[width, height]`; the keyframes rendered, `[first, one past the last]` (the whole script is `[0, n_keyframes]`); and their count. |
| `seconds_per_frame`, `first_frame_s`, `mean_steady_s`, `median_steady_s`, `wall_s` | The time of each frame, the first frame's (it holds Blender's warm-up), the mean and median of the rest, and the wall time of the whole `render`, the encode included. |
| `use_gpu_subdivision` | A Cycles flag. |
| `cycles`, `lighting` | Every Cycles and colour setting the scene ended with; and, for `assets`, what `render/lighting.py` did (null for `placeholder`): its `rig`, `unit_lux`, `balance_k`, `world`, and by the map's catalog the `sun` (`elevation_deg`, `rotation_deg`, `watts_per_m2`, `lux`), `hdri`, `sky` (`lux`, `strength`), `lamps`, `under_cabinet` strips, `portals` and `skylights`. |

What the frames were made with:

| Key | Content |
| --- | --- |
| `unit_lux` | 240. A grey card of albedo 0.18 under this many lux reads 0.18 in the linear pass (`render/cycles_setup.UNIT_LUX`). |
| `linear_clip` | 60,000. The linear pass is clipped here: float16 overflows at 65,504, and anything above 60,000 is white at every exposure the camera model allows. |
| `clip_start_m` | The camera's near clip in metres: the robot's (`render_encode.rgb_clip_m`). |

`stbench build` reuses a render whose last keyframe has its depth and instance images and whose log has the `appearance`
and `fast` asked for, `samples` when `--samples` is given, and an `isp_override` that is false. Nothing else is
compared: a change to the render code or the assets alone reuses the old frames, so delete the script's
`render_log.json` to render it again.

How they were developed:

| Key | Content |
| --- | --- |
| `isp` | The `IspSpec` the frames were developed with, all fifteen fields: `vignette`, `noise_a`, `noise_b`, `meter_sigma`, `target`, `exposure_ev` (`[low, high]`), `gain_ev`, `wb_clamp`, `sharpen`, `jpeg_quality`, `contrast`, `black_level`, `glare`, `denoise_luma_px` and `denoise_chroma_px`. The last five default to their stage off. The stages (`camera_model.py`), in order: the optics (the vignette, then veiling glare, a share of the light scattered over the frame), the meter and the exposure, the sensor's Poisson-Gaussian noise (variance `noise_a` x + `noise_b`) and the denoiser's smoothing of it (luma and colour, each by its own sigma in pixels), the gain, the white balance, the tone curve (contrast about mid-grey, a highlight shoulder, a gamut compression, the black level, the sRGB transfer), the unsharp mask, 8 bits, and a 4:4:4 JPEG. |
| `isp_override` | True when `isp` differs from the robot's own (an `--isp` file, a look-dev candidate). `stbench build` never reuses such a render. |
| `linear_kept` | The folder that holds the kept linear passes (`--keep-linear`, or `develop --linear`), else null. |
| `develop` | One entry a frame, `{k, ev, exposure_ev, gain_ev, metered, wb, clipped, crushed}`: the exposure in EV from the linear unit and its two parts (`ev` = `exposure_ev` + `gain_ev`), the metered luminance, the white-balance gains `[r, g, b]`, and the shares of pixels with a channel at 255 and with an 8-bit luma of 5 or less. |

**`render_meta.json`** is the Blender side of the same record. `render/render_episode.py` writes it into the temporary
`private/<sid>/render_tmp/`, beside the raw passes of each keyframe (`%06d.depth.npy`, planar metres; `%06d.index.npy`,
the object index; `%06d.linear.npy`, the scene-linear Combined pass) and `instance_map.json`; `render` copies what it
needs into `render_log.json` and deletes the folder. Its keys are `blender`, `device`, `samples`, `fast`, `width`,
`height`, `use_gpu_subdivision`, `seed`, `appearance`, `cycles`, `lighting`, `frames` (`{k, render_s, total_s}` a
keyframe), `script_s` and `pid`, and `unit_lux`, `linear_clip` and `clip_start_m`. The lighting record's `exposure_lux`
became `unit_lux`, and it gained `sky`, `under_cabinet` and `sun.lux`.

**The binding.** The render log is what says a script's frames are whole, so `render_cli.render_binding_problems(paths)`
reads it, and `stbench perceive` and `stbench compile` refuse a script for which it names a problem (exit 2, each
problem a sentence that says what to do):
- there is no render log (the script was never rendered);
- `frames` is not `[0, n_keyframes]`: a partial render.

The check is pure: it reads the log and never Blender. Once a script is rendered, `episode.write_script` never writes
its `timeline.json` or `sensors/camera.json` again with other content: it raises, naming the file, and the script's
folder is deleted to regenerate it. The same content proceeds, whatever keys and bytes an earlier build wrote it in, so
a rebuild changes nothing.

### The linear passes and `stbench develop`

`stbench render` develops each frame into its JPEG as it encodes the frame, then deletes the raw passes. With
`--keep-linear DIR` it first moves each keyframe's linear pass into DIR as `%06d.linear.npy` (float16, height x width x
3, in the unit `unit_lux` names and clipped at `linear_clip`: 1.84 MB a frame at 640x480). DIR is outside the dataset;
`stbench build --keep-linear DIR` keeps a script's passes in `DIR/<sid>`.

```sh
stbench develop DATASET/scripts/<sid> --linear DIR [--isp FILE]
```

develops those passes again into the script's `sensors/rgb/` JPEGs, so a change to the camera model, to a robot's
`IspSpec` or to the values of an `--isp` file costs seconds, not a render. The file is a JSON object of `IspSpec`
fields and need hold only the ones it changes; a name that is no field is refused (`render` takes `--isp` too). The
frames develop at the size `sensors/camera.json` names, and the sensor noise is seeded from the timeline's own identity
(map, seed, robot, job and `k`), never from the random script id, so the same values give the same bytes. Depth, the
instance images and the rest of the render stay. `develop` rewrites the log's `isp`, `isp_override`, `linear_kept` and
`develop`.

It refuses (exit 2): a script with no render log, and a DIR with no linear pass for some keyframe.

It deletes the script's `perceive` outputs (`private/<sid>/visibility.jsonl`, `perception_ids.jsonl` and
`scripts/<sid>/perception/oracle.jsonl`, which `stbench render` deletes too): the visibility log's `lum` is read from the
JPEGs. Run `stbench perceive` again, and compile the script's cases again, since a gold compiled before was read from the
old log.

### `<cid>/case.json`

The window of tasks `a` to `b` (script indices, inclusive) a case holds.

```json
{"case": "154c7d545e2d7943", "script": "7d49fa899dfb4d35", "a": 0, "b": 0, "k0": 0, "k1": 2, "t0": 0.0,
 "start_clock": "2026-08-20T07:59:00", "length": "short"}
```

`k0 = tasks[a].k0` and `k1 = tasks[b].k1`: a window never splits a task. `t0` is the script time of `k0`;
`start_clock` is the clock of the case's first step, drawn from the case's own random id (a 2026 date, 07:00-10:00),
never from the script's inputs or the window, so a delivered clock names neither the seed nor the window's place.

**What a case delivers is case-relative**, converted only by the functions of `case.py` (`case_k`, `script_k`,
`case_clock`, `case_task`, `gap_s`, `case_rows`), which the harness and the compiler both import: `k` and the inbox file
names count from 0 at the case's first step; a `clock` is `start_clock` plus the time since `t0`, rounded half up to the tenth
of a second after the shift (`YYYY-MM-DDTHH:MM:SS.s`, one decimal always: a walking person covers a metre in under a
second, and `closest_approach` divides displacements by the clock difference); the task index counts from 0 at task `a`; the first `gap_s` (the unobserved seconds before a task) is 0 and a
later task carries its real gap. Battery, brush, load, tether, phase and the first pose are drawn per task from the
robot's start set and never carried over, so a window reveals nothing of its place in the script. Private records
(evidence keyframes, decision keyframes, `visibility.jsonl`, `perception_ids.jsonl`) stay in script numbering.

### `<cid>/questions.jsonl`, `gold.jsonl`

What a case asks and what answers it, written by `stbench compile` (`questions.write`). How the compiler
chooses the questions and computes each gold is `docs/questions.md`; this is the files. Gold answers come only from code
over the timeline and the visibility log. Each file is one JSON object a line and one row a question, in ask order (the
during questions in decision order, then the end questions), so line i of one is line i of the other. The keys of
`questions.jsonl` are sorted; those of `gold.jsonl` follow `records.GOLD_KEYS`. Every row passes `records.check_question`
or `records.check_gold`, which refuse a missing or an extra key, or the compile stops. The `qid` is 16 lowercase hex
characters, the same in both rows and drawn fresh at every compile (`secrets.token_hex(8)`), so a recompile gives a case
new qids. A reader of gold refuses a case that has no gold.

A question row. A system is sent `qid`, `text` and `answer_type` and nothing else of it (a baseline that sets
`structured` also gets `template` and `slots`):

| Key | Content |
| --- | --- |
| `qid` | The question's id. |
| `when` | `during`, asked after the frame at its decision point, or `end`, asked after the last frame. |
| `decision_k` | The script keyframe a `during` question follows (an integer); null on an `end` row. |
| `text` | The question in words: the family's render of `slots` and nothing else. |
| `template`, `slots` | The family (`object_where`, `same_place`, `out_of_view`, ...: the registry in `docs/questions.md`) and the values its wording takes. The slots of the 0.5.0 families: `object_was` and `person_where` have `object_where`'s (`kind`, `attributes`, `event`, `task`; `task` is null for a window of one task), and `same_place` has `kind`, `attributes`, `events` (the two wordings of the job's events, earlier task first) and `tasks` (the two case task indices). `object_was` and `object_where` render the same words from the same slots, so only `template` tells them apart, and a system is never sent it. |
| `answer_type` | One of `answers.TYPES`: `position`, `offset`, `heading`, `time`, `yesno`, `count`, `number`, `set`, `positions`, `cells`, `route`. An `offset` answer is `{"type": "offset", "ahead_m": a, "left_m": l}`: metres from the camera along the floor, `a` ahead along its heading and `l` to its left (negative is behind or to the right; `ego_frame`). A `heading` answer (`open_heading`) is `{"type": "heading", "deg_left": x}`: `x` a finite number of degrees in [-180, 180] to the left of the way the robot faces, turned in place (to its right is negative); anything else is malformed. |
| `needs` | The sensor capabilities the gold reads, from `rgb`, `depth`, `pose`, `status`, `detections` and `see` (never `now` or `memory`). A run on a channel that lacks one never asks the question. |

A gold row:

| Key | Content |
| --- | --- |
| `qid`, `family` | The question's `qid` and `template`. |
| `gold` | The answer, in the shapes of `answers.parse_gold`; `{"type": "unknown"}` exactly when `slice` is `unknown`. |
| `slice` | `answerable` or `unknown`. A `same_place` row is `answerable` whether its answer is yes or no: the family has no unknown slice, and a coin picks between its two answers. |
| `answer_class` | The declared class a draw balanced, or null where the family declares none: one of `records.ANSWER_CLASSES` (`yes`, `moved`, `vacated`, `earlier`, `later`, `answerable`, `unknown`, `empty`, `items`, `person`, `no_one`). `answerable` and `unknown` are the slice. `records.BALANCE_POOLS` names the two pools of each classed family (`same_place`: `yes` against `moved` and `vacated`), which the balance check reads (docs/balance.md); a class that is in no pool of the row's family is refused. |
| `range_m` | The slant range r, in metres, kept for the distance report's bands, or null; since 0.7.0 it sets no credit (`rules.relative_ladder`, which built a rung from it, is the change families' only). `ego_now`: the slant at the ask. `ego_mem`: the larger of the slant at the nearest recognisable sighting at the gold place and the slant now. `object_where`: the nearest recognisable slant at which the asked place itself was seen (a sighting within `MOVED_NO_M` of it). `object_was`: the nearest recognisable slant at which the asked place itself was seen. `clear_path`: the gold plus the camera-to-front-edge distance. `person_mem`: the slant from the camera now to the visible point nearest it. `closest_person`: null. `same_place`, `task_things` and any family without a ladder: null. The credit of an offset row is the job's clearance C and the row's resolution and says nothing of r, so the distance report by band reads it here. |
| `credit` | How an answer is scored: one of `ladder`, `region`, `relative`, `mra`, `exact`, `f1`, `route`, `graded`, `graded_set`, `regret` (`records.CREDITS`; the table below). Since 0.9.0 every metric family the jobs ask is `graded`, `graded_set` or `regret`; the older credits stay for the families no job asks and for the rows of older compiles. |
| `region` | A polygon of three or more `[x, y]` points when `credit` is `region`, else null. |
| `tolerances` | `{}` or exactly the entries its credit reads (the table below: one for the older credits, a fixed set for `graded`, `graded_set` and `regret`), written in full on every row. A missing or extra key, or a non-positive number, is refused naming the key. |
| `metric` | `xy` or `xyz` on a position gold, the distance a `ladder` judges it by; else null. |
| `scope` | `now` (the latest frame alone gives its gold: the current view answers it) or `memory` (an earlier frame is needed). |
| `reference` | Null; `path_log` for a question the pose log answers by construction (the gates leave `path_log` out of such a family's shortcuts); `kind_range` for `ego_now`, whose declared reference is a typical range per kind (distance is reported above it; it is not a gated shortcut); `map_reader` for `clear_path` and `open_heading`, the two structure families (the public map's raycast from the delivered pose, with the pose-keyed pool of other scripts' golds where it has no map; the family is reported above it, and the gates leave `map_reader` and `other_case` out of its shortcuts and separation on every layout while `path_log`, from the robot's own log alone, stays in); or `other_case` for a fact that is the same in every case of a layout (`adjacent`, asked on held-out layouts only; the gates leave `other_case` out of its shortcuts there). |
| `critical` | Whether the question is task-critical against the job's relevance: its kind is a relevance kind, its place lies in a region the task works in and, for a job whose robot works the floor, the thing stood on the floor; an unknown row is critical by kind (`docs/questions.md`, "How a gold is computed"). |
| `subjects` | What the question is about: `e:<entity id>`, `d:<kind>/<attributes>` (a description absent from the world), `r:<region id>`, `a:<k>` (0.11.0: the sighting of a person at keyframe `k`, the subject of `approach_mem`, who is named by no entity), `w:<k>` or `w:<k0>-<k1>` (a keyframe or an interval of them). |
| `evidence_k` | The script keyframes the gold is read from, sorted. Never empty on an `answerable` gold (methodology, truth rule 2). |
| `entities` | The entity seen at each of `evidence_k`, parallel to it, or empty; each is also a subject, as `e:<id>`. |
| `age_s`, `age_band` | The age of the evidence (0.10.0; `records.GOLD_OPTIONAL`, after `entities`; a gold of an earlier compile lacks both and stays readable). `age_s` is the clock seconds from the gold's oldest evidence keyframe to the ask, read from the timeline's keyframe times (the ask is a `during` question's `decision_k` and an `end` question's last frame). `age_band` is `now` exactly when `scope` is `now`, else `recent` (`age_s` at most 10 s), `task` (the evidence is in the task the ask falls in) or `earlier` (an earlier task; `records.AGE_BANDS`). Both are null on an `unknown` row, which names no evidence. |

Each credit scores certain gold types and reads its own tolerance keys (`records.CREDIT_FITS`, `records.CREDIT_KEYS`);
`check_gold` refuses a row whose credit does not score its gold's type, or whose `tolerances` hold anything but those
keys:

| Credit | Scores | `tolerances` |
| --- | --- | --- |
| `graded` | `offset`, `number`, `position` | `resolution_m` (the gold's one-pixel error `rho`, at least 1 cm), `need_m` (the job's C or R), `harm` (`"over"`, `"radial"` or null) and, for a `number`, `threshold_m` (the body's length plus C); a `position` holds the first three alone (no room: its direction quarter is `e <= R`). The four quarters on the error against the need ([questions.md](questions.md), "Credit"): `harm` `over` halves the need when the answer is longer than the gold, `radial` when an offset's error points away from the camera along the gold's bearing, null never. An offset is judged by its 2-D error, `hypot(d_ahead, d_left)`. The row's `range_m` and the job's numbers do not set the credit. `ego_now`, `ego_mem`, `person_mem` (C, `radial`), `clear_path`, `closest_person` (C, `over`) and `object_where` (R, null) |
| `graded_set` | `positions` | `match_m` (the match radius, widened by the largest item), `resolutions_m` (one number per gold item, in the gold's item order) and `fp_weight` (0.5). The answer's items outside every don't-care circle are matched to the gold's one to one by maximum summed credit, each pair within four times `match_m` earning its four quarters at its item's resolution, then the items inside a circle are matched the same way to the gold items left over (so a circle only adds credit); an unmatched answered item outside every circle is false; `s` is the matched credit over `|gold| + fp_weight |false|`. `task_things` |
| `regret` | `heading` | `profile_m` (the run in metres of each heading, a list of `360 / step_deg` numbers, null where the corridor was not seen), `step_deg`, `resolution_m`, `need_m` (C) and `threshold_m` (the body's length plus C). The answered heading is rounded to the step; its run is `d` (0 for a null) and the regret `r = max(profile) - d` is scored with the four quarters, the direction quarter a run of at least half the best and the harmful side a run under `threshold_m`. Headings wrap at +-180. `open_heading` |
| `ladder` | `position`, `offset`, `number`, `time` | `{"ladder": [t1, t2, ...]}`: strictly increasing positive thresholds, metres for a position, an offset or a number and seconds for a time. An offset is judged by its 2-D error, `hypot(d_ahead, d_left)`. Three rungs are absolute and a rung met is a third of the credit (0.7.0 and 0.8.0): an offset (`ego_now`, `ego_mem`, `person_mem`) and the `number` of `clear_path` and `closest_person` had the job's clearance ladder `rules.clearance_ladder(C)`, C/2, C and 2C; a map place (`object_where`) `clearance_ladder(R)` of the job's re-find radius, each rung grown by `rules.hidden_widening` for a thing whose top the camera never saw. A compile before 0.9.0 still carries them, and 0.9.0's scorer scores such a row exactly as 0.8.0's did. The change families (no job asks them) keep the range-relative `rules.relative_ladder(r, widening)`, three rungs of `max(floor, share * r) + widening` with shares 0.10, 0.25 and 0.50 of the slant range r and floors 5, 10 and 20 cm. `distance_between` (no job asks it) has the job's four rungs, 0.25, 0.5, 1 and 2 m in a home, plus both things' `rules.widening` (it was `relative` in 0.4.x) |
| `region` | `position` | `{"erode_m": x}` with x at least 0, and `region` the polygon |
| `relative` | `number` | `{}`; the gold must be above 0, since the credit divides by it |
| `mra` | `count` | `{}` |
| `exact` | `yesno` | `{}` |
| `f1` | `set`, `positions` | (older compiles' `task_things` and `out_of_view`; 0.9.0 credits `task_things` as `graded_set`) `{"match_m": x}` with x at least 0; `out_of_view`'s x is 0.5 m plus the largest widening of an item on the row, and each `dont_care` place of its gold carries its own `radius` (0.5 m plus its thing's widening). The credit matches answers to gold items first (one to one, within `match_m`, a `set` item also by change type); then an answer that matched nothing and lies within a `dont_care` radius is dropped, neither a hit nor a miss, and every other unmatched answer is a false positive (0.5.0-control; earlier scorers dropped such an answer before matching, which also dropped a correct one) |
| `f1` | `cells` | `{}` |
| `route` | `route` | `{}` |

**`tolerances.catastrophe`** (0.10.0, optional beside any credit's keys; `records.OPTIONAL_TOLERANCES`) is a harm
family's rule for when an answer is a catastrophe, which the reported score charges and no gate reads:
`{"weight": 2 | 3, "rule": "over" | "radial" | "miss", "margin_m": max(C, 2 rho), "threshold_m": x (`over`, optional),
"critical": [[x, y], ...] (`miss`)}`. `weight` is `credit_kernel.CATASTROPHE_THING` (2) or `CATASTROPHE_PERSON` (3). `over`
(a number): the answer exceeds the gold by more than `margin_m` and, when the row has `threshold_m`, the gold is under it
and the answer at or over it. `radial` (an offset): the answer places the point farther from the camera than the gold, along
the gold's bearing, by more than `margin_m`. `miss` (a list): some point of `critical` has no answered item within the
row's `match_m`. `clear_path`'s `over` rule carries the row's `threshold_m` (the body's length plus C), as `closest_person`'s
and `closest_approach`'s do: a corridor overstated is a catastrophe only when the true corridor is under it and the
answer is not. The row has no key where the family asks no harm, where the gold is `unknown`, and, for
`clear_path`, where the gold stands at the question's cap. The rules by family are in [question-design.md](question-design.md), "How answers
are graded".

An `unknown` gold has no answer to score, so its credit may be any and `check_gold` does not tie its tolerances to it;
the compiler writes its family's own tolerances there all the same. `decision_k`, `evidence_k` and the `w:` subjects are
script numbering, like every private record (`case.json`, above); the stream and the question texts are case-relative.

```json
{"qid": "0123456789abcdef", "when": "end", "decision_k": null, "needs": ["see", "pose"],
 "slots": {"first": {"attributes": {"color": "red"}, "kind": "tote"},
           "second": {"attributes": {"color": "brown"}, "kind": "pallet"}},
 "answer_type": "number", "template": "distance_between",
 "text": "How far apart were the red tote and the brown pallet, as you last saw them? Answer in metres."}
{"qid": "0123456789abcdef", "family": "distance_between", "gold": {"type": "number", "value": 6.412},
 "slice": "answerable", "answer_class": null, "range_m": null, "credit": "ladder", "region": null,
 "tolerances": {"ladder": [0.65, 0.9, 1.4, 2.4]},
 "metric": null, "scope": "memory", "reference": null, "critical": true, "subjects": ["e:E0412", "e:E0157"],
 "evidence_k": [61, 118], "entities": ["E0412", "E0157"]}
```

The first row is a line of `questions.jsonl` and the second is the same question's line of `gold.jsonl`; the values are
illustrative. The rungs are the ladder's 0.25, 0.5, 1 and 2 m plus 0.4 m: the tote's widening (0.05 m) and the pallet's
(0.35 m). (`distance_between` is asked by no job; a `graded` row's tolerances are the keys of the table above.)

## `runs/<system>-<cid>/`

`stbench run SYSTEM DATASET/private/<cid>` runs one system over one case through the stream and writes one folder:
`runs/<system>-<case id>/` in the current directory, with every character of SYSTEM outside `[A-Za-z0-9._-]` turned into
`_` (`codex:gpt-6-luna` gives `codex_gpt-6-luna-<case id>`), or the folder `--out` names. A folder under the dataset's
`scripts/` or `private/` is refused. A run scores what it ran, so a case with no `gold.jsonl` is refused (exit 2) before
anything is sent. Exit 0: the case completed. Exit 1: it ended early, the run was invalid or could not be scored, or an
agent's runner failed. Exit 2: the run was refused before anything was sent.

| File | Content |
| --- | --- |
| `answers.jsonl` | `{"qid", "answer"}`, a line for each question the run asked, in the case's question order. A question the harness could not get an answer for holds `{"type": "failed", "reason": ...}` (`stream_protocol.failed`), which scores as malformed. With `--during off` the during questions were not asked and have no line. |
| `rows.jsonl` | One scored row for each gold of the case (`suite.score_rows`, `score.row`), in gold order, with the keys `case`, `script`, `qid`, `family`, `when`, `scope`, `reference`, `critical`, `slice`, `answer_class` (null for a gold of a compile before 0.6.0), `range_m` (the gold's, or null), `age_band` (the gold's evidence age band; null on an `unknown` row and before 0.10.0), `s`, `score`, `score_gate`, `catastrophe`, `outcome`, `layout`, `robot`, `job` and `length`. `s` is the credit in [0, 1]; `score_gate` is the penalised score at t = 0.5 (`score.penalised`: `2s - 1` for an answer on an answerable gold, 0 for `unknown` there, 1 for a correct `unknown`, -1 for an answer on an unknown gold), which every gate reads; `catastrophe` is whether the answer is one (`score.catastrophe`, a boolean); `score` is the reported score (`score.reported`): `score_gate`, except minus the catastrophe's weight for a catastrophe. The reports and the headline read `score`. A row of an earlier run lacks `age_band`, `score_gate` and `catastrophe`, and is read with `score_gate` equal to `score`; `outcome` is `correct`, `partial`, `wrong` (answered with s = 0, a failed answer included) or `abstained` (`unknown` on an answerable gold). A gold the run has no answer for scores as `failed("unanswered")`; with `--during off` the during golds have no row. |
| `summary.json` | How the run went and what it scored, below. |
| `codex.calls.jsonl`, `codex.summary.json` | A `codex:<model>` run only (`cmd_codex`, `docs/codex.md`), below. |

`summary.json`:

| Key | Content |
| --- | --- |
| `case`, `script` | The case's id and its script's. |
| `system`, `channel`, `brief`, `during`, `text_only` | SYSTEM as given; the channel; the brief variant; whether the during questions were asked (a boolean); whether no image was sent (`--text-only`, a boolean). These are options of the run, never part of the case. |
| `status`, `failures` | `complete`; or `ended: <where>: <Type: message>`, with `<where>` one of `start`, `task`, `observe`, `ask`, `finish` and `close` and the message cut at 300 characters; or `ended: case cap`. And the failed questions by reason: `missing`, `answered twice`, `not an object`, `ask error`, `case cap`, or `case ended: <where>`. |
| `wall_s` | Seconds per message, in the order sent (one `observe` for each frame), under each key of `stream_protocol.TIMEOUTS`: `start`, `task`, `observe`, `ask_during`, `ask_end`, `finish`. |
| `state_bytes` | The size of the system's state folder at the end of each task (`"0"`, `"1"`, ...), measured after the task's last frame and the during ask that follows it, and at `"finish"`. |
| `usage` | What `finish` reported: `{"self_reported": true, "models": {model id: {"tokens_in", "tokens_cached_in", "tokens_cache_write_in", "tokens_out"}}, "peak_context", "remote_store_bytes"}`, each null where the system did not say; null when `finish` failed. `tokens_in` counts every input token, cache reads and writes included; `tokens_out` includes reasoning. Unvalidated: `stream_protocol.usage_problems` says what is wrong with it. |
| `remote_memory`, `cost_usd`, `cost_note` | Whether the system reported a remote store (`stream_protocol.remote_memory`: such a run is unverified); the case's cost in USD by the price table (`prices.cost_usd`, never an agent's own figure), or null with the reason in `cost_note`. A reference Codex cost is a lower bound (Codex reports no cache writes: docs/codex.md). |
| `session` | The run's fresh 16-hex token, never the case id. An external system sees it as `STBENCH_CASE_NONCE`, the namespace of a remote memory (the shim sets it at `start`, with the case's own home: docs/agent.md). |
| `sandbox` | Where the system ran: `docker` (the verified mode), `none` (`--local`: not a verified score) or `in-process` (a baseline or an agent). |
| `scores` | The rows aggregated (`score_stats.aggregate`), cut to four keys: `headline`, `{mean, ci}`, the mean of the `now` and `memory` columns, each pooled over its own rows and weighted equally (one column when the rows hold one scope); `columns`, `{scope: {mean, ci, n}}`; `by`, `{dimension: {value: {mean, ci, n}}}`, pooled within each group, over `scope`, `own_path` (`reference` is `path_log`: `"True"` or `"False"`), `family`, `ability`, `kind`, `when`, `length`, `layout`, `robot`, `job`, `critical`, `slice` (`answerable` or `unknown`), `relevance` (`critical`, `situational` or `unknown`), `answer_class` and `age_band` (`now`, `recent`, `task`, `earlier`, `none` for a row with no band: `docs/eval.md`, "The splits"); and `counts`, `{correct, partial, wrong, abstained}`. A `ci` is a bootstrap over scripts, so it is null for one case. `stbench eval`'s `results.json` holds the whole aggregate (`above_floor`, `confidently_wrong_rate`, `abstention`, `job_age_band`, `catastrophe_rate`, ...: `docs/eval.md`). |

The summary is written even when the scoring raises (a gold the scorer cannot read): it then has no `scores`, there is no
`rows.jsonl`, and the run exits 1.

A Codex run adds two files and refuses a folder that already holds a `codex.calls.jsonl`. When the run is invalid (the
leak scan found a hit, or an exec used a tool outside the server's, and `invalid` is true) `stbench run` exits 1 and
writes none of `answers.jsonl`, `rows.jsonl` and `summary.json`; the two Codex files stay to inspect, and an invalid run
always keeps its journal, even when the escape came on the first exec and none finished (the journal's `exec_failed`
record names the tool Codex tried). A run that is not invalid and in which no exec finished (nothing ran or was spent)
removes its journal, so that it does not block a rerun. `docs/codex.md` has the fields.
- `codex.calls.jsonl` is the journal: a `run` event first (the model, the effort, Codex's version, the
  case, channel, brief, `during`, the per-item call bound and resume limit, the timeouts and the tools), then the events
  `exec_started`, `served` (each item the model was shown, its images as hashes; a frame that came with questions also holds their text as `asked`), `call` (each tool call, with when it
  arrived), `answer_outcome` and `exec_failed`, and one record for each `codex exec` (a case is one long exec, and a
  resume for each early end of its turn). It holds the questions' text, so it never goes in the dataset or to a system.
- `codex.summary.json` is written even when the run fails: `model`, `reasoning`, `codex`, `case`, `script`, `channel`,
  `brief`, `during`, `status`, `failures`, `wall_s` (per message, as above), `run_s`, `scan` (the leak scan's hits),
  `invalid`, `text_only`, `compact_at` and, for a run with a memory, `memory`, beside the journal's counts: `execs`
  (`total`, `by_kind`), `frames` served, `frames_asked` (the frames served together with questions), `pulls`,
  `premature_pulls`, `resumes`, `frame_latency` (`n`, `median_s`, `p90_s`: from serving a plain frame to the model's
  next pull), `asked_latency` (the same shape for the frames that came with questions, timed to the pull after the
  answers), `tokens` (`in`, `cached_in`, `out`, `reasoning`), `tokens_lower_bound` (true when a turn was killed or
  failed, so its tokens are missing), `tool_calls`, `tool_errors`, `retries`, `warnings`, `exec_wall_s`, `answers`
  (`submitted`, `repaired`, `failed` by reason, `timed_out`), `exec_failures` and `exec_failure_message` (the last
  failed exec's message).

## `runs/<system>-<channel>[-<brief>][-during-off]/`

What `stbench eval` writes (docs/eval.md has the command, the selection, resume and the spend cap): `results.json`
(written with every case pending before the first case runs, then after each), `results.md`, and
`cases/<cid>/` per case, each exactly a `stbench run` folder (`answers.jsonl`, `rows.jsonl`, `summary.json`, and an
agent's own files). With `--arms LEVER=V1,V2` each value has a run folder of its own under `<lever>-<value>/`. A case is
finished when its `summary.json` says `complete` and its `rows.jsonl` exists; any other case folder is removed and the
case run again on resume. Only finished cases contribute rows, and a results document with any case not finished is
`partial` ("not a benchmark score"). The fields of `results.json` are listed in docs/eval.md ("`results.json`").

## `check_report.json`

```sh
stbench check --data DATASET [--out FILE] [--workers N] [--verdict interval|point|evidence] [--verbose]
              [--select KEY=VALUE ...] [--cases ID,...] [--answers DIR] [--reuse DIR]
```

`stbench check` runs every scripted baseline (`baselines.BASELINES`), and the in-pool twin of each that learns from
other scripts (`baselines.in_pool_twins`: the same system keeping its pool's own best response, which the `shortcuts`
and `memory_needed` gates read beside it, the larger score deciding), over every compiled case of the dataset on the
gate channel (`detections`), gates each (layout, job, family) (`gates.py` owns the six gates and their thresholds:
`ceiling`, `knowable`, `memory_needed`, `shortcuts`, `separation`, `time_needed`) and writes the report to
`DATASET/check_report.json`, or to `--out`. Two of the baselines are references and no shortcut: `kind_range` (the
current view's bearing to the thing at the typical floor range of its kind over other scripts' golds; `unknown` on
every template but `ego_now`) has no in-pool twin, is never read by the `shortcuts` gate, and is a rival of `separation`
in a family whose every oracle row names it as the `reference`, so distance is measured above it; and `map_reader` (the
public map's raycast from the delivered pose, answered as it is, the pose-keyed pool's best response where the held-out
rule allows only on a held-out layout, which has no map; `unknown` on every template but `clear_path`) is the same for a family whose every oracle row names it, so clearance is
measured above it, and in that family `other_case` is left out of the shortcut and separation gates too (`path_log`
stays in). `path_log`'s own-log rules (the detour of `clear_path`, the longest run of `open_heading`) answer from the
robot's own poses alone and stand only where they beat abstaining on other scripts: each prior record carries the rule's
answer as its `own` key (`answer_keys.RECORD_KEYS`; null for a family with no such rule), and the rule stands where the
lower end of the 95% interval of its gain over abstaining, over the scripts of the prior, is above 0
(`baselines_path_own.kept`); the in-pool twin keeps it unasked. A dataset that has a `release.json` is a release, which a check leaves as
released: it writes no report there unless `--out` is given. The runs write no run folders. Only a case that holds at
least one question row is checked: a compiled case whose `questions.jsonl` is empty (no question could be drawn in its
window) is left out of the runs, the gates and the numbers, and is counted instead (`asked_nothing` below, and a printed
line `LAYOUT JOB: N of M compiled cases ask nothing and are not checked`).

- `--workers N`: the processes the runs go over, at least 1 (default 4; 1 runs them in the checking process). The
  runs over one case (every baseline's, and `other_case`'s over the case whose stream it is fed) are one task, the case
  with the most work first, so that what one run works out of a depth image (`table.SEEN`: the free-space cells of the
  image, its camera pose, its floor and the mapper's zones) the next finds in the process. The numbers do not depend on
  `--workers` or on the order the runs finish in. A baseline that opens no picture (`System.reads_files` false) is not
  given the inbox's files.
- `--verdict interval|point|evidence`: the reading the exit code follows. `interval` (the default) wants every gate that
  has an interval to pass on its 95% interval; `point` wants the point estimates alone, the only reading a family with
  few questions can power; `evidence` reads the shortcut-type gates by a gain with an interval above 0 in the group, a
  length or a slice (`gates_evidence`) and adds the balance rule (docs/balance.md): a family whose held-out best
  constant beats `all_unknown` with an interval above 0 fails, with a `BALANCE FAIL` line after the gates' lines.
  `stbench export` requires a report made with `--verdict evidence` whose `passed_evidence` is true and whose `broken`
  is empty: the point and interval readings are not read. The report holds all three; the printout has a `BALANCE` line
  for every family whatever the verdict.
- `--select KEY=VALUE` and `--cases ID,...` (0.11) check only those cases, as `stbench eval` selects them: a scoped
  check. It never writes `DATASET/check_report.json` (only `--out`), says what it was scoped to at the head and the foot
  of its printout, and holds a `scope` in the report. `--answers DIR` stores each scripted system's answers and `--reuse
  DIR` reloads those of the systems that do not read the scorer (`Baseline.reads_scorer`), for a check after a scorer
  change; the report names them in `reused` (docs/eval.md, "Fast loops").
- `--verbose` logs each run to stderr.
- Exit 0: the verdict passes. Exit 1: a gate fails by it, or an instrument is broken; each failed gate is printed as a
  line naming the layout, job, family and gate, the system that did it, the margin and the threshold, and each broken
  instrument as a `BROKEN` line after them (see `broken` below). A `BROKEN` line fails the verdict whatever the gates
  say. Exit 2: refused, with no report: no case has questions and gold (the message counts the compiled cases that ask
  nothing), a (layout, job) group cannot be given its systems
  (no other script has a case of the same layout, job, robot and length to feed `other_case`), or a scripted run failed a question or ended before its case did (the
  message names the first five runs: these systems are the instrument, so their failure is a bug and not a score).

| Key | Content |
| --- | --- |
| `verdict`, `passed` | The `--verdict` reading, and whether it passes: what the exit code follows. |
| `passed_point`, `passed_interval`, `passed_evidence` | The three readings, whatever `--verdict` says, each of them also needing that no instrument is broken (`broken` empty); `passed_evidence` also needs that no family fails the balance rule. |
| `balance` | One entry for each (layout, job, family), `suite_balance.read`'s: `layout`, `job`, `family`, `constant` (the best constant over every script, in words), `mean` and `ci` (the held-out best constant's paired gain over `all_unknown` and its 95% interval over scripts; `ci` null under two scripts), `n`, `scripts`, `thin` (some held-out pool is under the priors' support floor, so the group is reported and cannot fail) and `fails` (the mean above 0 with the interval above 0, in a group that is not thin). |
| `broken` | The broken-instrument lines (`suite_report.broken_instruments`), each starting `BROKEN` and naming the layout, job and family: a prior system none of whose levels above the bare template (`answer_keys.levels`) finds a record of any question of the family (reach rate 0, so its gate is vacuous; a first key that misses while a looser level hits is a prior at work, not a broken one), or a learned shortcut (`blind`, `path_log`, `questionnaire_blind` and the form adversary, each under the held-out rule; not `current_view_prior`, which no shortcut gate lists, and not the in-pool twins) whose mean is more than `0.10` (`suite_report.FLOOR_MARGIN`, the shortcut gate's margin) under `all_unknown`'s on the family (it overfits, and the gate it passes proves nothing). Empty when none is. |
| `asked_nothing` | One entry for each (layout, job) that has a compiled case, in order: `layout`, `job`, `compiled` (its compiled cases) and `asked_nothing` (how many of them ask nothing and are not checked). A group in which every case asks nothing is listed here and has no entry in `groups` or `gates`. |
| `groups` | One entry for each (layout, job): `layout`, `job`, `cases`, `scripts`, `now_share`, `temporal`, `families` and `form_adversary`. |
| `gates` | One row for each (layout, job, family), `gates.evaluate`'s: `layout`, `job`, `family`, `scope` (`now`, `memory` or `mixed`, from the oracle's rows), `scripts` (a count), `n` (questions), `missing` (the systems with no rows), `scores` (`{system: {mean, ci}}` for each baseline, for `form` and for the in-pool twin of each learned shortcut, `<system>_in_pool`), `gates`, `pass`, `point` and `evidence` (the interval, point and evidence readings of the row). `gates` holds `ceiling`, `knowable`, `memory_needed`, `shortcuts`, `separation` and `time_needed`, each `{pass, point, evidence, value, threshold}` and what that gate adds (`ci`, `worst`, `best`, `unknown_share`, `scores`, `lower_ci_threshold`, `cells`, `pooled`, `note`). A null `pass`, `point` or `evidence` is a gate that does not apply or has no rows to read; `time_needed` is null on every family but a temporal one, and `latest_table` and `first_table` (baselines like the rest) are scored in `scores` there. |
| `wall_seconds` | The check's wall time. |
| `scope`, `reused` | Present only when they apply. `scope` (a scoped check): `select` (the selection in words), `groups` (the layout and job of each group checked), `of_groups`, `cases` and `of_cases` (the cases that ask, checked and in all). `reused` (a `--reuse` check): the systems whose stored answers were reloaded instead of run. |

A group's numbers say whether its gates could fail. `scripts` lists the group's script ids. `now_share` is the share of golds whose
scope is `now`. `temporal[length]` (for the lengths the group has, short to long) holds `cases`, `answerable` (the
answerable end questions), `temporal` (those of a temporal family), `share` (their ratio; null with no answerable end
question) and `per_case`: change over time is claimed for a cell only at a share of 0.2 and one a case
(`docs/questions.md`). `families[family]` holds `questions`, `per_case`, `unknown_share`, `empty_set_share` (null unless
the golds are sets; for `out_of_view`, the share the 0.525 bound of `docs/families/out_of_view.md` is read against),
`reference` (the golds' common one, null if they differ), `class_shares` (`{n, classes, pools, draw}`: each declared class's and pool's share of the questions asked, and the draw's `{pool, share, bound}` where the family has one; reported, not gated: docs/balance.md), `hit_rates` (`{system: share}` for `blind`, `path_log` and
`current_view_prior`: the share of the family's questions whose first key some record of the other scripts shares),
`reach_rates` (the same shape: the share of questions for which some level above the bare template is a level of some
record of the other scripts; a system at 0 makes its shortcut gate pass for nothing, except where the family is a fact
of the layout, `reference` `other_case`, on a held-out layout: its priors learn from public layouts only and abstain by
design, and `note` says so) and `current_view_now_mean` (null without `now` rows; under 0.95 the family's `now` tag is
wrong). `form_adversary[family]` holds `feature` (the best single feature of the question's form, null when none beats
the unknown share), `mean`, `unknown_share`, `n` and `means`, each feature's mean.

## A release (`stbench export`)

```sh
stbench export --data DATASET --out DIR [--results RUNDIR_A RUNDIR_B ...]
```

writes the folder that is uploaded as the dataset (`export_cmd`, `export_files`, `export_card`). DIR must
be empty or absent and must not overlap DATASET. The released set is named in `export_files` and nothing else is released:

```text
DIR/
  README.md                       the dataset card: the package's template (stbench/release/dataset-card.md) with its
                                    counts block generated from release.json
  LICENSE                         CC BY 4.0, the package's text
  CREDITS.md                      the attributions of the third-party assets (copied; an export refuses without it)
  check_report.json               the dataset's report (copied)
  release.json                    the document below
  results/<name>.json             one per --results, byte for byte
  scripts/<sid>/**                every file of each script's agent-facing folder (the camera frames; no depth)
  private/index/<id>.json         the index entry of each script and case
  private/<sid>/                  timeline.json  visibility.jsonl  perception_ids.jsonl  render_log.json  depth/%06d.png
  private/<cid>/                  case.json  questions.jsonl  gold.jsonl
```

The depth ships under `private/<sid>/depth/`, never under `scripts/`: `stbench check` on an export reads it for the
`detections` channel.

Left out: `instances/`, `instance_map.json` and `render_tmp/` under `private/<sid>/` (the render's intermediates, read
only by `stbench perceive`, which rebuilds the released visibility log), `runs/`, and anything else the dataset holds.
There is no held-out script and no `instances/` folder in an export.

**`release.json`**, with no timestamp, so two exports of one dataset are byte-identical:

| Key | Content |
| --- | --- |
| `package` | The installed package's version, or null from an uninstalled copy. |
| `groups` | One entry per (layout, job), sorted: `layout`, `job`, `scripts` (a count), `cases` (a count per length: `short`, `medium`, `long`), `asked_nothing` (the cases of the group that ask no question), `questions`, `abilities` (the questions by the ability their family measures, `abilities.ability_of`, sorted by name; the card's second table) and `frames` (keyframes). |
| `cases` | One entry per case, sorted by layout, job and id: `id`, `script`, `layout`, `job`, `length` and `questions`. |
| `results` | The names of the shipped `results/<name>.json`, sorted. |

A **results file** is a `results.json` document (`eval_results.load`) that is not `partial`, names only exported cases,
and whose rows' qids are in their cases' gold. It ships as `results/<name>.json`, `<name>` being the system, the
reasoning effort when there is one, `memory-<name>` for a run with a memory, `compact-<TOKENS>` for one with a
compaction limit, the channel, the brief when it is not `correct` and `during-off` when the during questions were off,
with every character outside `[A-Za-z0-9._-]` turned into `_`; two files of one name are refused. A run of part of the
cases (`--cases`, `--select`) is not partial when it finished all of them.

**Refused** (exit 2, with every check run before the first write, so nothing is written): DIR not empty or overlapping
DATASET; a held-out script (its split, or its map in `map_files.HELD_OUT`); a case that is not compiled; a script whose
render is missing or partial (`render_binding_problems`), or one that lacks a file of the lists above or the rgb or
depth image of a keyframe; any hit of `leak.find_leaks` under `scripts/<sid>`; a released `.json`, `.jsonl` or `.md`
file (the generated ones included) that holds an absolute path of a machine or the name of the user who runs the export;
no `CREDITS.md`; a `check_report.json` that is missing, counts other cases than the export holds (`asked_nothing`:
compiled and silent cases per layout and job), has gate rows that are not the gold's (0.11.0: a (layout, job, family)
of the gold without a row of its question count `n`, or a row the gold lacks, as a report of an earlier compile or a
scoped check would have), was not made with `--verdict evidence`, has `passed_evidence` false or names a `broken`
instrument; and a results file that fails the rule above.

**From the export alone**: `stbench eval all_unknown --data DIR` scores every case that asks a question, and `stbench check
--data DIR --out FILE` reproduces the gate table and the verdict (a release keeps its own report unless `--out` is
given).

## What guards the layout

- **The leak scan** (`stbench.leak.leak_scan(DATASET/scripts/<sid>, [map name])`) fails on any file or key under the
  agent-facing folder that is not on its allowlist (`robot.json`: `robots.DOC_KEYS`; `sensors/camera.json`: the ten
  keys above; `sensors/pose.jsonl`: the row's keys and every robot's status keys; the images; `oracle.jsonl`:
  `leak.PERCEPTION_KEYS`, exactly the keys of the table above), on any file whose path names depth (rule `depth`: the
  agent is fed the camera, so a depth image under `scripts/` is a leak whatever its bytes), on an entity id (`E` + 4 or
  more digits), a region id, the word "truth" and the map's name, in a file's name, its text or an image's
  metadata. The map's name comes from the caller, read from the timeline. The static table `leak.ALLOWED` is the one owner of the allowlist: no writer
  registers anything at import, so a scan in a process that imported no writer sees the same table, and a new
  agent-facing file gets its entry there.
- **The render binding** (`render_cli.render_binding_problems`, under `render_log.json`) makes `stbench perceive` and
  `stbench compile` refuse a script with no render log or with frames for only part of it.
- **The depth move** (`case.depth_move_problem`): `stbench compile`, `check` and `perceive`, and a stream that finds
  its pictures missing, refuse a dataset made before 0.6.0, whose scripts still hold `scripts/<sid>/sensors/depth/`,
  naming `stbench move-depth --data DATASET`. That command moves each such folder to `private/<sid>/depth/` whole
  (idempotent, and it reports what it moved), and refuses a script that holds frames in both places, which needs the
  old folder deleted by hand. A whole `stbench render` deletes the old folder itself. The renderer, `perceive`, the
  free-space rule and the `detections` stream (`stream.delivered`, which copies `private/<sid>/depth/<k>.png` to the
  inbox's `depth/<kc>.png`) read the private folder alone, and an export ships `depth/<k>.png` for each keyframe and no
  other file.
- **The sandbox** refuses a system whose folder holds a name of the private area (`sandbox.TRUTH_FILES`: `gold.jsonl`,
  `questions.jsonl`, `timeline.json`, `case.json`, `visibility.jsonl`, `perception_ids.jsonl`, `instance_map.json`,
  `render_log.json`, `instances`, `index` and `private`; not `depth`, which the `detections` inbox also calls its
  folder of depth images).
- **`stbench validate SCRIPT_DIR`** loads the timeline, checks it against its map (the world's rules, the ego's, the body
  rule and the job's own `checks`) and runs the leak scan.
