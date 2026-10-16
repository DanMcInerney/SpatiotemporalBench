"""The compiler's output in a second process (`determinism_kit.py`: nothing is recorded, the two are compared).

`compile_digest` below is the SHA-256 of the compiler's constants (the corridor's sub-cell `grid.SUB_M` among them) and
of what it compiles for nine synthetic reference cases (`reference_cases`). The compiler of Release 0.3 was retired by
the redesign, so nothing of it is tested."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

from determinism_kit import digest
from space_kit import box, inputs, record, things_run
from stbench import (
    ego_frame,
    grid,
    job_registry,
    questionnaire,
    questions,
    questions_looks,
    questions_where,
    questions_words,
    recognise,
    records,
    rules,
)

ROOT = Path(__file__).resolve().parents[1]


def reference_cases():
    """Nine synthetic cases, each a script and a case record with a fixed id (so the draws that hash it repeat): the
    run with things on three seeds, with a box in the corridor and with an open floor, a window over the second of two
    tasks, a medium window over both, two more ids over the first run, and a medium window over three tasks (the task
    slot and the per-task events). Only the decision's own keyframe (21, and 49 in the second task) has a render: the
    rule certifies nothing where there is no depth."""
    wall = box(2.23, 2.43, -3.0, 3.0, 0.0, 2.0)
    ahead = box(1.43, 1.73, -0.15, 0.15, 0.0, 0.3)

    def run(cid, a=0, b=None, length="short", **fields):
        script = things_run(depth_ks={21, 49}, **fields)
        return script, record(script, a=a, b=b, cid=cid, length=length)

    return [run("ref-a"), run("ref-b", seed=2, boxes=[wall, ahead]), run("ref-c", seed=3, boxes=[]),
            run("ref-d", a=1, n_tasks=2, seed=5), run("ref-e", a=0, b=1, length="medium", n_tasks=2, seed=5),
            run("ref-f", seed=6), run("ref-g", seed=7), run("ref-h", seed=8),
            run("ref-i", a=0, b=2, length="medium", n_tasks=3, seed=9)]


def predicate(job) -> dict | None:
    """The job's `task_things` predicate as plain data, its sets sorted (a set's repr depends on the process's hash
    seed)."""
    if job.task_things is None:
        return None
    return {k: sorted(v) if isinstance(v, frozenset) else v for k, v in dataclasses.asdict(job.task_things).items()}


def compile_digest() -> str:
    """The digest of the compiler's output: the constants its rules and draws read (the free-space rule's and the
    corridor's sub-cell, the recognisability floor, the questionnaire's sizes and its share per length, the ladder, the
    ordinals and the cap, the widening, the never-seen pool's seeds), the questions and gold of the reference cases,
    qids dropped, and the families each job asks (`decisions`, `ends`)."""
    constants = {
        "min_floor_px": grid.MIN_FLOOR_PX, "free_share": grid.FREE_SHARE,
        "fine_m": grid.FINE_M, "sub_m": grid.SUB_M, "cell_m": grid.CELL_M, "eps": grid.EPS,
        "recog_px": recognise.RECOG_PX, "recog_lum": recognise.RECOG_LUM, "gap_steps": questionnaire.GAP_STEPS,
        "max_during": questionnaire.MAX_DURING, "max_end": questionnaire.MAX_END,
        "unknown_percent": questionnaire.UNKNOWN_PERCENT, "max_cap_m": questions_words.MAX_CAP_M,
        "ladder": questions_words.DISTANCE_LADDER, "ordinals": questions_words.ORDINALS,
        "default_ends": questionnaire.DEFAULT_ENDS, "look_per_frames": questions_looks.LOOK_PER_FRAMES,
        "look_min_steps": questions_looks.LOOK_MIN_STEPS, "look_gap_steps": questions_looks.LOOK_GAP_STEPS,
        "other_seeds": questions_where.OTHER_SEEDS, "widen_free_m": rules.WIDEN_FREE_M,
        "gold_seen_scale": rules.GOLD_SEEN_SCALE, "rel_shares": rules.REL_SHARES, "rel_floors_m": rules.REL_FLOORS_M,
        "answer_classes": records.ANSWER_CLASSES, "balance_pools": {f: {p: sorted(c) for p, c in pools.items()}
                                                                      for f, pools in records.BALANCE_POOLS.items()},
        "band_units": [ego_frame.NEAR_UNITS, ego_frame.FAR_UNITS],
    }
    compiled = []
    for script, rec in reference_cases():
        qs, gs = questions.compile_inputs(inputs(script), rec)
        compiled.append([[{k: v for k, v in row.items() if k != "qid"} for row in rows] for rows in (qs, gs)])
    jobs = {job.id: {"decisions": job.decisions, "ends": job.ends, "looks": job.looks,
                     "look_decisions": job.look_decisions, "task_things": predicate(job)}
            for job in job_registry.all_jobs()}
    return digest({"constants": constants, "cases": compiled, "jobs": jobs})


def test_the_compiler_digest_is_deterministic():
    """Contract: the compiler's output (its constants, the corridor's sub-cell included, and the compile of nine
    synthetic cases, qids dropped) is the same in a second process (another `PYTHONHASHSEED`), so no draw of it depends
    on process state (hash seeds, set order, the clock). Regression caught: a draw that is deterministic only inside
    one process."""
    here = compile_digest()
    other = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'tests'); "
                            "import test_determinism_compile as t; print(t.compile_digest())"],
                           cwd=ROOT, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": "12345"})
    assert other.stdout.strip() == here
