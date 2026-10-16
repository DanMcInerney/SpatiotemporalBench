"""Repository rules that are cheap to enforce: module size, the Blender boundary and the fences between
the truth, the baselines and the agent. The fences are prefix rules over module names, so a new module is fenced the day
it arrives, and they read every module under `src/stbench`, subpackages included."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stbench"
RENDER = ROOT / "render"  # Blender's side: modules too
TESTS = Path(__file__).resolve().parent


def modules_under(root: Path) -> dict[str, Path]:
    """The modules below `root` by dotted name (`jobs/vacuum.py` is `jobs.vacuum`; a package is its folder's name)."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(root).with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        if parts:
            found[".".join(parts)] = path
    return found


MODULES = modules_under(SRC)


def too_long(paths, limit: int = 500) -> list[Path]:
    return [p for p in paths if len(p.read_text(encoding="utf-8").splitlines()) >= limit]


def test_modules_and_tests_stay_well_under_500_lines():
    """One concept per module, modules (the package's, subpackages included, and `render/`'s) and test files well under
    500 lines: past the limit a file does too much and must split. Regression caught: a test file (the scorer's reached
    540 lines) or a render module (nothing checked them) growing past what a reader can hold."""
    render = modules_under(RENDER)
    assert len(render) > 10  # the glob finds `render/`'s modules, so the rule below is not vacuous
    files = [*MODULES.values(), *render.values(), *modules_under(TESTS).values()]
    assert [p.relative_to(ROOT).as_posix() for p in too_long(files)] == []


def test_stbench_never_imports_bpy():
    """Contract (AGENTS: Blender code lives in render/, which runs in Blender's own Python): no module of `stbench`
    imports `bpy`, so the package imports and its tests run without Blender. Regression caught: a Blender call added to
    the package (a renderer helper put in `src/stbench`), which breaks every import on a machine without Blender and
    moves GPL-3.0 render code into the MIT package."""
    pattern = re.compile(r"^\s*(import|from)\s+bpy\b", re.M)
    offenders = [n for n, p in MODULES.items() if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_shim_imports_only_the_standard_library():
    """Contract (VISION rule 6, V0 §9): `shim.py` is copied into the container on its own, where there is no `stbench`
    and must not be, so it imports nothing relative and nothing outside the standard library. Regression caught: the
    shim importing the package (to reuse `System`, say), which would fail at start in every container and, if the
    package were ever mounted, give the system under test a path to the repository."""
    tree = ast.parse((SRC / "shim.py").read_text(encoding="utf-8"))
    froms = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    relative = [n.lineno for n in froms if n.level > 0]
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in froms if n.module and n.level == 0}
    assert {"json", "os", "sys"} <= imported  # the scan sees the shim's imports, so the rule below is not vacuous
    assert relative == [] and imported - sys.stdlib_module_names == set()


# -- the fences: which module may import which -------------------------------------------------------------------
# A module is on a side when any dotted component of its name fullmatches the side's regex.
TRUTH = re.compile(r"^(timeline\w*|visibility|sightings|freespace\w*|questions\w*|questionnaire|job_\w+|generate|route"
                   r"|mover_\w+|coverage|keyframes|world_\w+|people\w*|changes|validate\w*|perception|clutter|case"
                   r"|brief|suite\w*|stream|check|eval_\w+|export\w*|build|cmd_\w+|episode)$")
BASELINE_SIDE = re.compile(r"^(baseline\w*|blind|places|table\w*|form_check|answer_\w+)$")
AGENT_SIDE = re.compile(r"^(llm_\w+|codex_\w+|agent_\w+)$")
NEUTRAL = ("stream_protocol", "records", "grid", "recognise", "events", "rules", "camera", "geometry", "robots",
           "answers", "mapspec", "map_files", "prices", "grid_cover", "change_events", "zones", "zone_shapes",
           "rack_stock", "adjacency", "abilities", "ego_frame", "task_predicate", "nearest_part", "pixels")
LIST_OWNERS = re.compile(r"^(timeline\w*|validate\w*|generate|route|job_\w+)$")


def on_side(side: re.Pattern[str], name: str) -> bool:
    return any(side.fullmatch(part) for part in name.split("."))


def package_imports(source: str, name: str, *, is_package: bool = False,
                    known: frozenset[str] = frozenset()) -> set[str]:
    """Dotted names of the stbench modules a source file imports (`from .x import y`, `from . import x`, `from stbench.x
    import y`, `import stbench.x`), relative imports resolved at any level against the importing module's own package:
    `name` is its dotted name, `is_package` says it is a package's `__init__`. `from .x import y` names `x.y` only when
    it is in `known`, the modules that exist (`y` may be a function)."""
    here = name.split(".") if is_package else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            parts = node.module.split(".") if node.module else []
            if node.level:
                base = here[:max(0, len(here) - node.level + 1)]
            elif parts[:1] == ["stbench"]:
                base, parts = [], parts[1:]
            else:
                continue
            target = base + parts
            if target:
                found.add(".".join(target))
            for alias in node.names:
                sub = ".".join([*target, alias.name])
                if not parts or sub in known:  # `from . import x`: x is a module
                    found.add(sub)
        elif isinstance(node, ast.Import):
            found.update(a.name.removeprefix("stbench.") for a in node.names if a.name.startswith("stbench."))
    return found


def import_closure(roots: tuple[str, ...], modules: dict[str, Path] = MODULES) -> set[str]:
    """The modules `roots` import, and those they import, and so on; importing `a.b` runs the package `a` first."""
    known = frozenset(modules)
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        name = todo.pop()
        if name in seen or name not in modules:
            continue
        seen.add(name)
        path = modules[name]
        todo += package_imports(path.read_text(encoding="utf-8"), name, is_package=path.name == "__init__.py",
                                known=known)
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
    return seen


def test_import_scanner_sees_relative_and_absolute_imports():
    """Contract: the scanner behind the fences finds `from .x import`, `from . import x`, `from stbench.x import` and
    `import stbench.x` in a top-level module, and in a subpackage module every relative form at any level (`from ..x
    import`, `from .. import x`, `from .sibling import`), resolved against the module's own package, a package's
    `__init__` included. Regression caught: a scanner that misses a form or resolves a level wrongly, so a fence passes
    vacuously (map X-H10)."""
    top = ("from .timeline import q\nfrom . import visibility\nfrom stbench.questions import x\n"
           "import stbench.sightings\n")
    assert package_imports(top, "mod") == {"timeline", "visibility", "questions", "sightings"}
    nested = ("from ..timeline import q\nfrom .sibling import z\nfrom . import peer\nfrom .. import top\n"
              "from ..deep.mod import f\nfrom stbench.jobs.vacuum import g\nfrom stbench import events\n"
              "import stbench.jobs.vacuum\nimport os\nfrom os import path\n")
    assert package_imports(nested, "jobs.vacuum") == {"timeline", "jobs.sibling", "jobs", "jobs.peer", "top",
                                                      "deep.mod", "jobs.vacuum", "events"}
    assert package_imports("from .vacuum import x\nfrom . import peer\n", "jobs", is_package=True) == {
        "jobs.vacuum", "jobs", "jobs.peer"}
    submodule = "from .jobs import vacuum, helper\n"
    assert package_imports(submodule, "mod") == {"jobs"}
    assert package_imports(submodule, "mod", known=frozenset({"jobs.vacuum"})) == {"jobs", "jobs.vacuum"}


def test_the_fences_see_subpackages(tmp_path):
    """Contract (plan §2.8): a module in a subfolder is found by its dotted name, its relative imports are read against
    its package (`from ..timeline import q` in `jobs/vacuum.py` is `timeline`), it is on a side when any component of
    its name is, and the size rule flags it. Regression caught: a `jobs/` package escaping every rule (map X-H10)."""
    pkg = tmp_path / "stbench"
    (pkg / "jobs").mkdir(parents=True)
    (pkg / "jobs" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "jobs" / "vacuum.py").write_text("from ..timeline import q\n" + "x = 1\n" * 599, encoding="utf-8")
    (pkg / "timeline.py").write_text("", encoding="utf-8")
    found = modules_under(pkg)
    assert set(found) == {"jobs", "jobs.vacuum", "timeline"}
    assert package_imports(found["jobs.vacuum"].read_text(encoding="utf-8"), "jobs.vacuum") == {"timeline"}
    assert import_closure(("jobs.vacuum",), found) == {"jobs", "jobs.vacuum", "timeline"}
    assert on_side(TRUTH, "jobs.job_vacuum") and not on_side(TRUTH, "jobs.vacuum")
    assert too_long(found.values()) == [found["jobs.vacuum"]]


def test_a_new_truth_module_is_fenced_on_arrival():
    """Contract (plan §2.8): a module named like `questions_new`, `job_new`, `timeline_new`, `validate_ego` or `cmd_new`
    is a truth module with no list edited, and no neutral module is on any side. Regression caught: a new truth module
    unprotected until someone remembers to list it, or a neutral module (`robots`, `events`) swept into a side."""
    for name in ("questions_new", "job_new", "timeline_new", "validate_ego", "cmd_new"):
        assert on_side(TRUTH, name), name
    sides = (TRUTH, BASELINE_SIDE, AGENT_SIDE)
    assert [n for n in NEUTRAL if any(on_side(side, n) for side in sides)] == []


def test_baselines_and_the_agent_reach_no_truth():
    """Contract (V0 §1, VISION rule 6): the import closure of every baseline-side module (the scripted systems, the
    free-space-free helpers, the form adversary) and of every agent-side module (the reference agent's tools and its
    Codex transport), and of each neutral module that exists, reaches no truth module: a system and the model's tools
    read only what a system may read. `robots` imports only `geometry`, so both sides may read a robot through it.
    Regression caught: a baseline or the model's tools reaching the timeline, the visibility log, the compiler or the
    dataset layout through a shared import, which would make the oracle and knowable gates meaningless."""
    baseline = tuple(n for n in MODULES if on_side(BASELINE_SIDE, n))
    agent = tuple(n for n in MODULES if on_side(AGENT_SIDE, n))
    neutral = tuple(n for n in NEUTRAL if n in MODULES)
    assert {"baselines", "blind"} <= set(baseline) and {"agent_tools", "llm_render", "codex_exec"} <= set(agent)
    assert {"robots", "geometry", "mapspec"} <= set(neutral)  # the scan finds the sides, so the rules are not vacuous
    reached = {side: sorted(n for n in import_closure(roots) if on_side(TRUTH, n))
               for side, roots in (("baseline", baseline), ("agent", agent))}
    reached |= {n: sorted(t for t in import_closure((n,)) if on_side(TRUTH, t)) for n in neutral}
    assert {side: hits for side, hits in reached.items() if hits} == {}
    assert import_closure(("robots",)) == {"robots", "geometry"}


# -- the reference agent names no model --------------------------------------------------------------------------
PROVIDER_PREFIX = re.compile(r"(?<![a-z0-9])(openai|anthropic|google|qwen|deepseek|mistralai|meta-llama|x-ai|z-ai|"
                             r"moonshotai)/")
FAMILY_NAMES = ("gpt-", "claude", "gemini")
FUNCTIONS = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def string_constants(source: str) -> list[str]:
    """The string constants of a source file (the literal parts of an f-string included), docstrings left out."""
    tree = ast.parse(source)
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, FUNCTIONS) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def test_string_constant_scanner_skips_docstrings_and_sees_f_strings():
    """Contract: the scanner behind the rule below returns every string constant but docstrings: a plain one, a dict
    key, the literal part of an f-string. Regression caught: a scanner that misses a form (or reads a docstring), so
    the no-model-names rule passes vacuously or fails on prose."""
    src = ('"""gpt-4 in a module docstring"""\n'
           'def f(x):\n'
           '    """claude in a function docstring"""\n'
           '    return {"k": "openai/x", **{f"gemini {x}": 1}}\n')
    assert sorted(string_constants(src)) == ["gemini ", "k", "openai/x"]


def test_the_agent_modules_name_no_provider_or_model_family():
    """Contract (AGENTS "Never"): nothing special-cases one model, one provider or one family, so no string constant in
    an agent-side module (docstrings aside) holds a provider prefix (`openai/`, `anthropic/`, ...) or a family name
    (`gpt-`, `claude`, `gemini`); the model id arrives only on the command line. Regression caught: a per-model branch
    or default, which would make a row of the benchmark depend on who the model is."""
    agent = {n: p for n, p in MODULES.items() if on_side(AGENT_SIDE, n)}
    assert len(agent) > 3  # the scan finds the agent's modules, so the rule below is not vacuous
    offenders = {name: sorted(c for c in string_constants(path.read_text(encoding="utf-8"))
                              if PROVIDER_PREFIX.search(c.lower()) or any(f in c.lower() for f in FAMILY_NAMES))
                 for name, path in agent.items()}
    assert {n: hits for n, hits in offenders.items() if hits} == {}


# -- the timeline is read through its query helpers ---------------------------------------------------------------
RAW_LISTS = {"placements", "states", "keyframes"}


def raw_list_reads(source: str) -> list[str]:
    """Attribute reads of the timeline's record lists (`x.placements`, `x.states`, `x.keyframes`) in a source file."""
    hits = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute) and n.attr in RAW_LISTS]
    return [f"{n.attr} line {n.lineno}" for n in sorted(hits, key=lambda n: n.lineno)]


def test_raw_list_scanner_finds_attribute_reads():
    """Contract: the scanner behind the rule below flags `tl.placements`, `tl.ego.keyframes` and `x.states`, and not
    the keyword in a constructor call. Regression caught: a scanner that sees nothing, so the rule passes vacuously."""
    src = "a = tl.placements\nb = len(tl.ego.keyframes)\nc = x.states\nd = Timeline(placements=[], states=[])\n"
    assert raw_list_reads(src) == ["placements line 1", "keyframes line 2", "states line 3"]


def test_the_timeline_lists_are_read_only_inside_the_timeline_modules():
    """Contract (V0 §3): everything reads the timeline through `placement_of`, `placements_of`, `state_of`,
    `keyframe`, `n_keyframes` and the other query helpers, so continuous tracks can be added without breaking callers.
    Only the timeline modules and the modules that build or validate one (`LIST_OWNERS`: generate, route, validate and
    their splits, the jobs) touch the lists. Regression caught: a caller indexing `tl.placements` directly, which a
    different storage would break."""
    offenders = {name: hits for name, path in MODULES.items() if not on_side(LIST_OWNERS, name)
                 if (hits := raw_list_reads(path.read_text(encoding="utf-8")))}
    assert offenders == {}
