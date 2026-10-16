# render/

Blender-side scene and render code. `render/` is GPL-3.0-or-later (see [LICENSE](LICENSE)) because it uses Blender's
Python API; the rest of the repository is MIT, and the dataset is CC BY 4.0. It runs inside Blender's own Python, is not
part of the `spatiotemporalbench` wheel, and `src/stbench` never imports it. You do not need it to use the released
dataset; the build is described in [docs/format.md](../docs/format.md) and [docs/world.md](../docs/world.md).
