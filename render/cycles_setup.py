# SpatiotemporalBench: Blender-side scene code.
# Copyright (C) 2026 the SpatiotemporalBench authors.
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 3 or any later version. It is distributed WITHOUT
# ANY WARRANTY. See <https://www.gnu.org/licenses/gpl-3.0.html>.
"""Cycles settings shared by the episode renderer and the asset preview: the GPU, the denoiser, the colour pipeline.

Runs inside Blender's own Python and is never imported by `stbench`. Exits with status 3 when no GPU is available to
Cycles: it never falls back to the CPU.

The two appearances differ only in what they ask of the light:
- `placeholder` (M1) keeps its 5 bounces and a plain denoiser;
- `assets` (realism Phase 1) renders at 128 samples with adaptive sampling (noise
  threshold 0.01, at least 32 samples), OpenImageDenoise with its albedo and normal guides, 12 bounces in all
  (diffuse 4, glossy 4, transmission 8), no indirect clamp and the light tree. None of it touches the data passes:
  depth and the object index are not sampled or filtered, and `tests/test_render_settings_blender.py` renders both
  and compares them bit for bit. The camera is a pinhole (`camera.json` and `locate` say so): no motion blur, no
  depth of field, no lens distortion.
`settings` reads every one of these back from the scene, after `lighting.setup` has set its exposure and white balance,
for `render_meta.json` and the private `render_log.json`.

The linear unit (`UNIT_LUX`) is what the pictures are in: the Combined pass is scene-linear, and a grey card of
albedo 0.18 under 240 lx reads 0.18, in every catalog. The home's daylight is at its physical intensity, so a sunlit
floor reads about 75 (59,500 lx x albedo 0.3 / 240), and the camera model (`stbench.camera_model`) meters it. The
indirect clamp is off for that reason, measured: the sun is sampled at every bounce, and one bounce off a floor the
sun shines on is worth 200 x albedo in the unit, so any clamp that would cut a firefly cuts the sun's bounce light.
On the kitchen's sun-patch frame (128 samples, adaptive at 0.01, OpenImageDenoise) a clamp of 10 leaves 40% of the
shaded cabinets' light against a 1,024-sample render with no clamp, 30 leaves 68%, 100 leaves 98% there and 73 to 89%
of the unsunlit pixels of five other frames; no clamp leaves 98.8 to 100.8% in all six, with no more fireflies after
denoising (pixels over 8 x the median of their 5 x 5 neighbourhood: at most 0.0062% of a frame, the reference's own
0.0029% on the kitchen's)."""

from __future__ import annotations

import bpy

# the linear unit, one constant with one owner: a grey card of albedo 0.18 under this many lux reads 0.18
UNIT_LUX = 240.0
VIEW_TRANSFORM = "Khronos PBR Neutral"  # keeps a catalog colour's hue and value (DECISIONS, "Realism Phase 1, light")
# What each appearance asks of Cycles. `bounces` are Blender's: `max` is the total, the others cap one ray kind.
SETTINGS = {
    "placeholder": {"bounces": {"max": 5}, "clamp_indirect": 8.0, "adaptive": None,
                    "denoise_passes": None},
    "assets": {"bounces": {"max": 12, "diffuse": 4, "glossy": 4, "transmission": 8, "transparent": 8},
               "clamp_indirect": 0.0, "adaptive": {"threshold": 0.01, "min_samples": 32},
               "denoise_passes": "RGB_ALBEDO_NORMAL"},
}
BOUNCE_ATTRS = {"max": "max_bounces", "diffuse": "diffuse_bounces", "glossy": "glossy_bounces",
                "transmission": "transmission_bounces", "transparent": "transparent_max_bounces",
                "volume": "volume_bounces"}


def setup_device(sc) -> dict:
    """Cycles on a GPU (OptiX, else CUDA). Raises SystemExit(3) rather than rendering on the CPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        prefs.compute_device_type = kind
        prefs.refresh_devices()
        for d in prefs.devices:
            d.use = d.type == kind
        if any(d.use for d in prefs.devices) and prefs.has_active_device():
            break
    else:
        print("STBENCH-ERROR no OptiX or CUDA device is available to Cycles; refusing to render on the CPU", flush=True)
        raise SystemExit(3)
    sc.cycles.device = "GPU"
    return {"type": prefs.compute_device_type,
            "names": [d.name for d in prefs.devices if d.use],
            "all_devices": [{"name": d.name, "type": d.type, "use": bool(d.use)} for d in prefs.devices]}


def configure(sc, width: int, height: int, samples: int, appearance: str = "placeholder") -> dict:
    bpy.context.preferences.system.use_gpu_subdivision = False
    r = sc.render
    for p in r.bl_rna.properties:  # no metadata in any image
        if p.identifier.startswith("use_stamp") and p.type == "BOOLEAN" and not p.is_readonly:
            setattr(r, p.identifier, False)
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = width, height, 100
    r.use_compositing = r.use_sequencer = False
    r.use_persistent_data = True
    r.use_motion_blur = False
    r.film_transparent = False
    device = setup_device(sc)
    cfg = SETTINGS[appearance]
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = cfg["adaptive"] is not None
    if cfg["adaptive"]:
        cy.adaptive_threshold = cfg["adaptive"]["threshold"]
        cy.adaptive_min_samples = min(cfg["adaptive"]["min_samples"], samples)  # a short test render is not adaptive
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    if cfg["denoise_passes"]:
        cy.denoising_input_passes = cfg["denoise_passes"]
    if hasattr(cy, "denoising_use_gpu"):
        cy.denoising_use_gpu = True
    cy.seed = 0
    cy.use_animated_seed = False
    for key, value in cfg["bounces"].items():
        setattr(cy, BOUNCE_ATTRS[key], value)
    cy.sample_clamp_indirect = cfg["clamp_indirect"]
    if appearance == "assets":
        cy.use_light_tree = True
    sc.view_settings.view_transform = VIEW_TRANSFORM
    sc.view_settings.look = "None"
    vl = sc.view_layers[0]
    vl.use_pass_combined = vl.use_pass_z = vl.use_pass_object_index = True
    return device


def settings(sc) -> dict:
    """Every setting that decides what a render looks like, read back from the scene: the sampling, the denoiser, the
    bounces, the clamps, the filter, the seed, the colour management (exposure and white balance included) and the
    camera. `render_meta.json` carries it and `stbench render` copies it to `render_log.json`."""
    cy, r, vs = sc.cycles, sc.render, sc.view_settings
    cam = sc.camera.data if sc.camera else None

    def f(x: float) -> float:  # Blender stores single precision: 0.01 reads back as 0.009999999776
        return round(float(x), 6)

    return {
        "engine": r.engine, "samples": cy.samples, "adaptive_sampling": cy.use_adaptive_sampling,
        "adaptive_threshold": f(cy.adaptive_threshold), "adaptive_min_samples": cy.adaptive_min_samples,
        "denoiser": cy.denoiser, "denoising_input_passes": cy.denoising_input_passes,
        "denoising_prefilter": cy.denoising_prefilter, "denoising_quality": cy.denoising_quality,
        "bounces": {key: getattr(cy, attr) for key, attr in BOUNCE_ATTRS.items()},
        "sample_clamp_direct": f(cy.sample_clamp_direct), "sample_clamp_indirect": f(cy.sample_clamp_indirect),
        "light_tree": cy.use_light_tree, "caustics_reflective": cy.caustics_reflective,
        "caustics_refractive": cy.caustics_refractive, "blur_glossy": f(cy.blur_glossy),
        "pixel_filter": cy.pixel_filter_type, "filter_width": f(cy.filter_width), "seed": cy.seed,
        "animated_seed": cy.use_animated_seed, "motion_blur": r.use_motion_blur,
        "view_transform": vs.view_transform, "look": vs.look, "exposure": f(vs.exposure), "gamma": f(vs.gamma),
        "white_balance": ({"temperature_k": f(vs.white_balance_temperature), "tint": f(vs.white_balance_tint)}
                          if vs.use_white_balance else None),
        "camera": None if cam is None else {"type": cam.type, "depth_of_field": cam.dof.use_dof,
                                            "sensor_fit": cam.sensor_fit, "clip": [f(cam.clip_start), f(cam.clip_end)]},
    }
