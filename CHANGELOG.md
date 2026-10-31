# Changelog

All notable changes to SpatiotemporalBench are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The goals and the rules are in
[docs/vision.md](docs/vision.md). Entries before 0.7.0 cite the development records (DECISIONS, FINDINGS) by the title
of an entry; those records are not part of the public repository. The public history starts at 0.11.0: the tags and
branches of earlier versions that entries below mention are not published.

## [0.11.1] - 2026-10-30

A fix release with the memory features that were waiting under [Unreleased]. The renders, the 144 cases and every
question but two are 0.11.0's. The drone's corridor question is no longer asked where its evidence frame cannot place
the wall's foot, which removes two questions from the data; the scorer, the place a run is written, the Blender lookup
and the tests are fixed; and `--memory`, `--compact-at` and the carried record, added since 0.11.0, are listed under
"Added" with the memory-feed fix. No model ran again: the reference rows are 0.11.0's runs with the answers to the two
removed questions dropped, scored again.

### Changed

- **The drone's corridor is asked only where its evidence frame reads the wall's foot to within C/2 at one pixel**
  (`clear_path`, `drone/1`; [docs/families/clear_path.md](docs/families/clear_path.md), "Tied to its foot"). A ground
  robot's corridor already had to pass this tie (`corridor_screens.error_tied`); the flier's did not, and its evidence
  frame is the latest keyframe that saw the wall, which may be one from the nest: a camera 0.17 m up, 45 m from a wall
  whose ground is 0.15 m, reads one pixel there as 270 m, and the credit's band then spans every answer from 22.5 m to
  89.7 m. **The data.** Compiling the drone's 36 cases again removes exactly two questions and changes nothing else
  (content and order): two `memory` corridors with a gold of 44.9 m, whose gold's resolution was 271.86 m and 31.93 m
  against the drone's C/2 of 1.2 m. The questions of this release are 0.11.0's compiled data with those two rows
  removed, so every other question keeps its id and the stored answers still score. The drone has 262 questions
  (`clear_path` 194, `ego_mem` 68; 0.11.0: 264, `clear_path` 196), the release 1,759 (1,761), its clearance questions
  1,072 (1,074), and the drone's clearance `memory` cell 98 (100). **The rows.** The answers the three models and the
  floor gave to the two questions are dropped and the 42 cases scored again, with no model call. Only `downtown_a` moves
  (headline, `now`, `memory`; 0.11.0, then 0.11.1):

  | System | Headline | `now` | `memory` |
  | --- | --- | --- | --- |
  | `claude:opus` (medium) | +0.614, +0.609 | +0.596, +0.596 | +0.632 (46), +0.622 (44) |
  | `claude:sonnet` (medium) | +0.366, +0.369 | +0.380, +0.380 | +0.351 (46), +0.358 (44) |
  | `claude:haiku` (medium) | +0.185, +0.191 | +0.109, +0.109 | +0.261 (46), +0.272 (44) |
  | `all_unknown` | +0.132, +0.137 | +0.000, +0.000 | +0.263 (46), +0.274 (44) |

  Opus scored +1.00 on both questions, and sonnet, haiku and the floor answered `unknown` (0), so their removal lowers
  opus's `memory` column and raises the others'. The `apartment_a` and `warehouse_a` rows are unchanged to the last
  digit. Opus's full-credit share of the drone's answerable corridors falls from 11% to 7% (5 of 44, then 3 of 42), and
  its remembered clearance cell from +0.701 to +0.681 (32 to 30 questions). The free check of the new data passes in
  every reading, as 0.11.0's did, per job and in full. Two gate values moved: the drone's `clear_path` `separation`
  reads +0.907 where it read +0.908, on 194 questions where it had 196, and the helper's `task_things` shortcut lead
  reads +0.000 where it read +0.007, from the scorer fix below (the in-pool shortcuts choose their constants by the
  score they would earn).
- **Behaviour change: `stbench run` and `stbench eval` write their run folder to `runs/` in the current directory, not
  to `DATASET/runs/`** ([docs/format.md](docs/format.md), [docs/eval.md](docs/eval.md), "RUNDIR"). A dataset is
  downloaded and kept as it was published, so a run started outside it no longer changes it (started inside it, the run
  still writes `runs/` there). `--out` names another folder, as before, and a folder under the dataset's `scripts/` or
  `private/` is still refused. `stbench view` finds the dataset by itself only for a run inside it, so a run at the
  default `runs/` needs `--data`. A script or a habit that reads `DATASET/runs/<system>-<case id>` must read `runs/...`
  where it was run from.

### Fixed

- **Blender is found on any system.** `STBENCH_BLENDER` first, then the `blender` on the PATH, then where the installer
  puts it on Windows or macOS (`blender_run.blender_path`). It was a fixed Windows path.
- **The scorer: an item inside a don't-care circle never lowers a list's credit.** `score_set.graded_set` pairs the
  answered items outside every circle first and lets an item inside a circle take only the gold items left over, so
  adding one can no longer lower the credit ([docs/questions.md](docs/questions.md), "A list scores per item"). The
  scripted shortcuts of `stbench check` that choose their best constant by the score read lists differently, which moved
  the in-pool shortcuts' `task_things` values (the helper's lead above).
- **A huge offset answer keeps its bearing.** The bearing and the radial error are computed on scaled vectors, so an
  answer near the float limit no longer overflows to NaN (read as 0 degrees), a place answer that large no longer raises
  `OverflowError`, and the distance report reads the scorer's bearing. With the list fix above, no published score
  changes: the 1,552 answers of the four reference rows, and the map reader's 388, score as before.
- **The tests no longer break on pytest 9.2**, which rejects `pytest.approx` on nested data: floats and counts are
  compared apart.
- **The README's far-field example states its pixel unrounded** (13.65 cm), so its scores reproduce, and the door table
  of [docs/question-design.md](docs/question-design.md) gives the pixels it uses.
- **The memory's sensor feed no longer runs one frame ahead** (`--memory SPEC`, Codex and Claude alike). The harness fed
  a frame to the memory when it read it in, but serves a frame only when the next message shows whether questions join
  it, so the memory held frame k+1 while the model was shown frame k. A frame is now fed as it is served
  (`codex_system._feed_held`), and a task is fed after the frame before it: the memory holds exactly the frames the
  model has been served. The model's own messages are unchanged and a run without a memory is unaffected. **Memory runs
  made before this change had the one-frame lookahead** (the earlier paid memory runs among them): a `view` of the store
  could show the frame after the one the agent was looking at, so their memory rows are not directly comparable with new
  ones.

### Added

- **CI also tests on macOS** (`.github/workflows/ci.yml`: Ubuntu, Windows and macOS).
- **`--memory SPEC` and `--compact-at TOKENS` for the Claude Code agent** (`claude:<model>`; [docs/claude.md](docs/claude.md),
  "A memory beside the agent" and "Compaction"). The Codex agent's memory arm now runs on Claude Code, from the one SPEC:
  the memory's MCP server joins the case's one `claude -p` process (a second entry of `--mcp-config`, every tool allowed
  by the server's name under `dontAsk`), its guide follows the system prompt, its sensor feed is written each task and
  frame before the model is served it, a carried record rides on `next_frame`, its calls are journaled (`memory_call`)
  and read by the leak scan, and the store is kept beside the case as `<name>-store`. Every message the model is sent
  stays the reference agent's, byte for byte, and a run without a memory differs from before only by `compact_at: null` in
  its journal and summary. `--compact-at` sets Claude Code's two auto-compaction variables (a window of 100,000 and a
  percentage; limits from 1 to 66,400 tokens, rounded up to a multiple of 800), and a `compact_boundary` event is a
  warning of its turn. The tests run against an extended fake Claude (`tests/claude_cli_fake.py`); nothing was run
  against the real CLI.

- **`--memory SPEC` for the Codex agent** (`stbench run` and `stbench eval`; [docs/codex.md](docs/codex.md), "A memory
  beside the reference agent"). A memory store is measured under the reference agent's own transport, prompt and images:
  its MCP server joins the case's one exec (and every resume), its guide follows the base instructions, and its sensor
  feed (one process for the case, a JSON line in and out for each message) is given each task and each frame, never a
  question. Every message the model is sent stays the reference agent's, byte for byte. `results.json` names the memory,
  `eval compare` takes it as part of the agent lever, the memory's tool calls are journaled and counted by name, and the
  store is kept beside the case as `<name>-store`. Refused on `detections`, with `--text-only` and for other agents.
- **A memory may carry its record on the pull** (SPEC key `"carry": true`, optional; [docs/codex.md](docs/codex.md), "A
  memory that carries its record"). With it `next_frame` takes an optional `memory` object, the agent's record about the
  frame it was just shown, which the harness writes to the memory's sensor feed (`{"record": ...}`) before it counts the
  pull; the feed's reply is the last text block of the item that pull returns. The journal's `call` event holds the reply
  and the `served` event gains it; the leak scan reads it. Without `carry` the tools and every message are unchanged.
- **`--compact-at TOKENS` for the Codex agent** (`stbench run` and `stbench eval`; [docs/codex.md](docs/codex.md),
  "Compaction is Codex's own"). Every exec of the case, the resume included, is launched with
  `-c model_auto_compact_token_limit=TOKENS`, so Codex compacts the thread earlier than its own limit, to measure a memory
  beyond the context. Without it the argv is unchanged. The journal's `run` event and `codex.summary.json` carry
  `compact_at`, `results.json` names it when it is set, the reports name the system
  `codex:<model>, compacts at 50k tokens`, `eval compare` takes it as part of the agent lever, and a resume into a
  folder made with another limit is refused. Refused for other agents.

## [0.11.0] - 2026-10-16

Two changes, asked for on 2026-10-15 and 2026-10-16: scoring "on a graded curve", and questions that are somewhat better
or more complex but still task-relevant and not arbitrarily hard. Every distance is scored on one need rule, and the
heading question counts; and the helper is asked the way to a thing it saw, the vacuum a list of what its brush could
catch, and the prompt says a named thing may never have been seen. There is no new render, as in 0.10.0: the same 64
shifts, cut into the same 144 cases, compiled again. A second pass of 2026-10-16 ("hard") scores a list item as a place
and grades the empty list, asks the AMR and the helper how close the person they saw last, now out of view, would come
(`approach_mem`), gives the drone no new family, and takes the map reader out of the published results.

| Job | Place | Scripts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 711 | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 541 | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 245 | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 264 | 2,534 |

**The questions** (the 0.11 compile of the dev root, 1,729 over 144 cases, after the bound on `way_to` golds noted
under "Added", which dropped one of 30 and gave its slot to `ego_mem`): `clear_path` 607, `ego_mem` 327,
`open_heading` 188, `ego_now` 154, `tangle_ahead` 133, `task_things` 114, `object_where` 97, `closest_approach` 46,
`closest_person` 34 and `way_to` 29. The vacuum's list asks about 0.6 a case (22 over its 36), and the glimpse screen
drops 6 of the 95 `unknown` rows. Every (job, ability, column) cell holds at least 20 questions (the least hold 22: the
AMR's clearance `now` and the vacuum's relevance cell, its `task_things`), so every cell weighs 1 and the new weight
moves no headline; those two sit at the edge, which is what a continuous weight is for. The free check of the 0.11 dev
root passes in every reading on every job: `ALL GATES PASS` and `EVIDENCE VERDICT PASSES`, one job at a time. The largest
lead a shortcut has over `all_unknown` is +0.04 on `open_heading` for the helper and +0.00 for the vacuum, +0.03 on
`way_to`, and +0.00 on the vacuum's `task_things`, against a gate line of +0.10; the family pages record the scoped
checks that decided `way_to` and the vacuum's list. With `approach_mem` and the place rule the compile holds
1,761 questions (the table above is this compile's): `clear_path` 607, `ego_mem` 331, `open_heading` 188,
`tangle_ahead` 133, `task_things` 116, `ego_now` 114, `object_where` 97, `approach_mem` 66, `closest_approach` 46,
`closest_person` 34 and `way_to` 29.

**The stored answers, scored again.** The reference rows of 0.10.0 are model answers to 0.10.0's questions, so they
cannot be run again without a model. They can be scored again: each stored answer of the 42 cases, through 0.11.0's
scorer, paired by question id (the scorer reads the rows already compiled, and the new need rule and the heading credit
need no new data). The headlines under the 0.11 scorer, against 0.10.0's as published (`apartment_a` / `warehouse_a` /
`downtown_a`):

| System | `apartment_a` | `warehouse_a` | `downtown_a` |
| --- | --- | --- | --- |
| astra | +0.615 | +0.602 | +0.790 |
| sol | +0.449 | +0.558 | +0.615 |
| luna | -0.045 | +0.321 | +0.050 |
| `all_unknown` | +0.031 | +0.083 | +0.132 |

0.10.0 as published: astra +0.638 / +0.598 / +0.768, sol +0.461 / +0.554 / +0.593, luna -0.091 / +0.308 / +0.008,
`all_unknown` +0.032 / +0.083 / +0.132. The order astra, sol, luna holds in every
place. The apartment moves by two changes in opposite directions: the tenth lifts sol and astra by about 0.03 and
counting `open_heading` lowers them by 0.04 to 0.05. On the 38 `open_heading` rows among the 42 cases the credit scores
pooled luna -0.334, sol +0.028 and astra +0.162. These are 0.10.0's questions and prompt, answered by Codex models; the
Claude rows below read 0.11.0's.

**Results.** The reference models changed from Codex to Claude. 0.10.0's rows were three Codex models (astra, sol and
luna). The harness runs the same reference agent through either CLI ([docs/claude.md](docs/claude.md)), so 0.11.0's
rows are three Claude models, `claude:haiku`, `claude:sonnet` and `claude:opus`, each once at medium effort, with the
floor `all_unknown`. They ran the same 42 of the 124 short and medium cases that ask a question as 0.10.0's rows, on
0.11.0's questions, prompt and scorer.
On those cases, `claude:haiku` (medium) scores +0.152 [+0.076, +0.250] on `apartment_a`, +0.185 [+0.019, +0.325] on `downtown_a`, +0.273 [+0.112, +0.440] on `warehouse_a`; `claude:sonnet` (medium) scores +0.286 [+0.159, +0.404] on `apartment_a`, +0.366 [+0.213, +0.505] on `downtown_a`, +0.307 [+0.077, +0.499] on `warehouse_a`; `claude:opus` (medium) scores +0.567 [+0.518, +0.642] on `apartment_a`, +0.614 [+0.517, +0.717] on `downtown_a`, +0.419 [+0.188, +0.638] on `warehouse_a`; the floor
scores `apartment_a`: `all_unknown` (rgb_pose) +0.034 [+0.019, +0.048]; `downtown_a`: `all_unknown` (rgb_pose) +0.132 [+0.078, +0.219]; `warehouse_a`: `all_unknown` (rgb_pose) +0.056 [+0.000, +0.125].
Coverage: `all_unknown`: 42 of 42 cases; `claude:haiku` (medium): 42 of 42 cases; `claude:sonnet` (medium): 42 of 42 cases; `claude:opus` (medium): 42 of 42 cases.
The Claude rows and 0.10.0's Codex rows differ in the models, the questions, the prompt and the scorer, so they do not
compare. 0.10.0's rows are in [docs/history/report-0.10.0.md](docs/history/report-0.10.0.md), and their answers scored
again by 0.11.0's scorer are above. All are dev rows; the cells, comparisons and costs are in
[docs/report.md](docs/report.md).

### Changed

- **One need rule for every distance** (0.11.0; [docs/methodology.md](docs/methodology.md) rule 7,
  [docs/question-design.md](docs/question-design.md) section 6). An answer is scored on the need
  `N' = max(N, 0.1 g, 2 rho)`, where `N` is the job's own need (the clearance C, the re-find radius R or a list's match
  radius M), `g` the distance the answer is judged over (a number's gold value, an offset's ground range from the
  camera, a heading's best run) and `rho` the gold's one-pixel resolution. The first two terms are halved on the harmful
  side and `2 rho` is not. A place and a list item keep R and M. Within ten needs of the robot (1.0 m on the vacuum, 1.5
  m on the helper, 4.5 m on the AMR, 24 m on the drone) nothing changes. Beyond that the margin is a tenth of the
  distance, the same for every robot, where it was whatever two pixels of the robot's own camera resolved. A door 5 m
  away, answered 4 m, scores 0.00 (as good as `unknown`) on the vacuum, the helper and the AMR; it scored -0.03, -0.42
  and -0.06 in 0.10.0. 4.5 m scores +0.50 and 6.0 m scores -0.42 on the helper, and the drone keeps its 2.4 m clearance.
  An over-estimate costs more than an under-estimate of the same size, because the distance term is halved on the
  harmful side. The catastrophe term is unchanged. The constant is `credit_kernel.KAPPA` = 0.1, the largest slack that
  keeps every constant answer at the floor and the models apart (a fifth lifts the AMR's constants).
- **`open_heading` counts, and its credit is room to move** (0.11.0;
  [docs/families/open_heading.md](docs/families/open_heading.md)). Its ability is clearance
  (`FAMILY_ABILITY["open_heading"]`), and `abilities.DIAGNOSTIC` is the retired `prediction` alone. The credit is the
  same kernel on the regret (the best run less the answered heading's) with the need rule above and `g` the best run.
  The direction quarter is room to move: a run of the room the body needs, T (its length plus C), or longer, or the best
  run less a pixel where no heading has that much; the harmful side is a run under T. Through 0.10.0 the direction was a
  run of half the best, and the credit was a cliff (a heading 2 to 10 degrees off a doorway scored -0.4 to -1). The
  chosen credit was measured against the others on the 0.10.0 rows: the robot's own log alone scores +0.04 on the helper
  and -0.14 on the vacuum under it, and +0.38 on the helper under a credit that paid every heading with room linearly in
  that room, which fails the shortcut line and is not used. The map reader falls to the floor (-0.05 and -0.02).
- **A list item is a place, and an empty list is scored on the same curve** (0.11.0;
  [docs/questions.md](docs/questions.md), "Credit"; [docs/methodology.md](docs/methodology.md) rule 7;
  `score_set.graded_set`). Where there was nothing to list, the empty list is the one thing to find, and false places
  dilute it as they dilute a list of one thing: `s = 1 / (1 + FP_WEIGHT |false|)`. One false place scores +0.33 and
  three -0.20; through 0.10.0 any false place scored -1, as ten did. This affects 80 of the dev root's 247 list
  questions: 28 of `task_things` and 52 of `tangle_ahead`. The Claude Code fast tier found it. Both of Opus's -1s on
  the apartment's `task_things` were one thing placed 0.7 m and 0.9 m from a thing the question counts neither way. The
  prompt's list bullet says it: "When there were none, the empty list is the one thing: the credit is 1 divided by 1
  plus half the number of false places."
  - **The place rule.** A list item is a place, so a pair of an answered place and a gold item earns its direction
    quarter only within the match radius M of the item, as a place does within R (rule 7, "A place's direction is its
    radius"). Through 0.10.0 every list pair earned that quarter at any distance up to four radii, the one place
    answer outside the rule: a pair 1 m off with M 0.5 m earned 0.5 and now earns 0.25. The pairing reach, the other
    three quarters and the don't-care circles are unchanged. The prompt says it: "its right direction is being within a
    matching radius of the thing, and the matching radius is its margin".
  - **Why both.** The graded empty rule alone failed the helper's `task_things` gates, because ending the cliff
    unmasked the free direction quarter: on a replay of the check's own pools (the same answers, the same scorer
    patched in process), `blind_in_pool` read +0.143 against the line 0.10, `current_view_prior` +0.075 against 0.05
    and `form_in_pool` +0.141. Every helper list item lies in one 2.5 m by 1.9 m patch of the living room that the
    helper's way always crosses, so a blind list of its two densest places earned the quarter for 47 of the 78 item
    rows. With the place rule they read -0.124, 0.000 and 0.000, and the other list families stay under their lines
    (the vacuum's `task_things` -0.504, the AMR's -0.159, the vacuum's `tangle_ahead` 0.000 on `blind_in_pool`). A
    per-kind spot test marks no helper kind (none of 22 items recurs at 0.1 m) and one over any kind marks 19 of 22 and
    makes a gold depend on the other scripts, so neither was used.
  - **The models.** Scored again with no model call, astra, sol and luna move by -0.02 to +0.04 on the 42 reference
    cases (astra's `apartment_a` / `warehouse_a` / `downtown_a` +0.615 / +0.602 / +0.790 become +0.621 / +0.581 /
    +0.790; sol's +0.449 / +0.558 / +0.615 become +0.450 / +0.537 / +0.615; luna's -0.045 / +0.321 / +0.050 become
    -0.010 / +0.332 / +0.050), and the order holds. Opus's apartment headline on the Claude Code fast tier rises from
    +0.418 to +0.511, because the cliff's -1s go. The scoped checks on the 0.11.0 compile give the list families' shortcuts as
    `task_things` +0.01 on the helper and +0.00 on the vacuum and the AMR, `tangle_ahead` +0.01, against the line of 0.10, with `memory_needed` +0.00, ceiling 1.00 and separation 1.00 on each.
- **The map reader leaves the published results** (0.11.0, 2026-10-16: an agent that reads the map has extra knowledge,
  and what is tested is the agent's spatiotemporal perception and memory). The agent is never given a map, and how a
  system that holds one scores is not what is tested. Its rows are out of the README's results, the technical report and
  the dataset card, and the README's quickstart no longer runs it; `docs/vision.md` carries a dated amendment. The
  baseline stays as an internal design diagnostic: `stbench check` computes it and reports it beside `clear_path` and
  `open_heading` (a new family must put a system that holds the layout near the floor), `stbench eval map_reader` still
  runs, and no gate changes. The claim that "the map answers the drone's corridor" is gone too: downtown's corridors are
  bounded only by static building walls, which the camera sees, so they measure seeing big static structure
  ([docs/job-drone.md](docs/job-drone.md)).
- **A cell weighs `min(1, n / 20)`** (0.11.0; [docs/methodology.md](docs/methodology.md) rule 4). It replaces the rule
  that left a (job, ability, column) cell of fewer than 20 questions out of the headline, which flipped a cell in and
  out between compiles (the helper's clearance `now` cell held 26, 14 and 51 across 0.10.0's). A cell's mean pools its
  rows by weight, a cell weighs in its column the largest weight of its rows, and a column in the headline the largest
  weight of its cells, so a column of thin cells alone does not count as half the headline. One cell set serves a
  release's dev, fast-tier and held-out scores: `--cells-from DEVROOT`. A dataset whose cells all hold 20 or more is
  unweighted.
- **The prompt's words** ([docs/questions.md](docs/questions.md), "Credit"; `llm_credit`). The margin bullet reads "For
  an offset, a distance in metres or a heading, the margin is {C} or a tenth of the right distance (from your camera to
  the right point, the right number of metres or the best heading's run), whichever is larger." The heading bullets say
  its right direction is room to move and that a shorter run is its harmful side, where the margin is half as large; the
  worked zero of the harmful side names the margin. The room the body needs reads "the robot body's length plus 15 cm"
  (the job's C, 15 cm on the helper), never "plus the margin": the far field's tenth does not widen it. A job that asks
  `way_to` is told how that heading is credited (below): "Its margin is 20 cm: the largest of 15 cm, a tenth of 1 m and
  twice the 10 cm cell of the floor that the way is worked out on." "Quarter 4 is whole within 10 cm of the best
  heading's progress, falling smoothly to nothing at the margin." "Any heading with the room the body needs (the robot
  body's length plus 15 cm) has the right direction, even one that leads away from the thing." The numbers are the
  scorer's own (`llm_credit.route_margin`), and a test ties them to it.
  The last bullet is new: "A question may name a thing your camera never showed clearly enough to tell what it
  is, or one that is not in this place at all; then `unknown` is right." It is true of every named thing by construction
  (an answerable thing has a recognisable sighting, an `unknown` one has none in the window or does not exist) and it
  names no question. The vacuum's prompt already carried the list bullets, through `tangle_ahead`, so its list adds no
  text.
- **A narrow screen on `unknown` questions: S4** (0.11.0; `questions_where.glimpse_hides`;
  [docs/families/ego_mem.md](docs/families/ego_mem.md)). An `unknown` candidate of `ego_mem` or `object_where` is
  dropped when, between the case's first frame and the ask, a thing of a look-alike kind was glimpsed (sighted and never
  recognisable) in a frame whose colour may be taken for the asked one: a person asked for the blue skip could not tell
  the glimpse from it. It costs 6 of the 95 `unknown` rows (6%). The wider screens (any colour of a look-alike glimpse,
  48%; any glimpse in the colour, 57%) would leave the AMR's `ego_mem` and the helper's `object_where` no `unknown` rows
  and are not applied.
- **The helper's second slot is `ego_mem` 65 and `way_to` 35** (the 35 it drew nothing on), so `ego_mem` keeps its
  supply; the helper declares `look_phases["way_to"]` (`navigating`, `carrying`).
- **The leak check reads two things** ([docs/eval.md](docs/eval.md), "D2's verdict"): see "Added".
- **The export checks the gate rows against the gold** (0.11.0, final review B1; `export_files.gate_row_problem`;
  [docs/format.md](docs/format.md)). `stbench export` refuses a `check_report.json` whose gate rows differ from the
  dataset's gold by (layout, job, family) and question count `n`: a report of an earlier compile, or a scoped check, can
  agree with a dataset on the cases that ask nothing and still gate other questions. The message names a group the gold
  holds that the report does not gate, one it gates that the gold lacks, or one with another count.
- **The documents** (0.11.0). The README, the dataset card and the question documents say what the code does now. The
  technical report ([docs/report.md](docs/report.md)) states 0.11.0's rules, and its results, like the README's and the
  card's, are filled by `tools/fill_report.py` from the Claude reference rows. The report of 0.10.0, whose rows were
  Codex models, is kept in [docs/history/report-0.10.0.md](docs/history/report-0.10.0.md).

### Added

- **`approach_mem`** (AMR and helper; harm kind, clearance ability, the `memory` cell;
  [docs/families/approach_mem.md](docs/families/approach_mem.md)): "If you kept moving as you are now, how close would
  you come in the next 5 seconds to the person you saw last, where you last saw them? Answer the smallest distance in
  metres along the floor from the point below your camera to that person; answer 10 if you would not come within 10 m."
  It is `closest_approach` for the person now out of view, such as the AMR's blind aisle end, and it needs memory of a
  dynamic thing that is in no layout, composed with the robot's own motion. The gold is code: the last frame of the task
  before the ask that holds an evidence sighting of anyone, the person there only if exactly one stood
  (`approach_rule.last_seen`, which the table systems share), and the least distance of `closest_approach`'s arithmetic
  (`approach_rule.closest_m`) against that place with the robot's own velocity. A point is a dead band (the gold's own
  screens in `questions_approach`, which the table systems do not apply) where someone is in view (then it is
  `closest_approach`'s), where no one was seen yet, where two people stood at the last sighting, where someone was shown
  and not placed, where the person was walking when seen (the `WALKING` screen: a person who stood still at the sighting
  is the only one asked), or where the person has moved since by truth (a reader cannot know it, but its best belief,
  that people stay put, is right on every asked row). The credit and the catastrophe are
  `closest_approach`'s (C, harmful side `over`, weight 3). It is asked in any phase: the AMR's slot 3 is now `ego_now` 50
  and `approach_mem` 50, the helper's `ego_now` 20, `open_heading` 15 and `approach_mem` 30 (of the 65% it drew nothing
  on). Supply is large on the AMR, where 41% of keyframes have a person within 10 m who was seen recently and is out of
  view, against about 10% where `closest_approach` finds someone in view. The design's offline proxies put the shortcuts
  at the floor: the best constant scores -0.31 on the AMR and -0.33 on the helper, a pose-keyed pool -0.37, and a
  static-layout system answers `unknown`. On the 0.11.0 compile it holds 47 rows on the AMR and 19 on the helper,
  every one a person (the share of `no_one` golds is 0 on both), and the AMR's `ego_now` falls from 105 to 65; its
  gates read ceiling 1.00, `knowable` +0.00, `memory_needed` +0.00, shortcuts +0.00 and separation 1.00 on both jobs, the best constant `unknown`.
- **`way_to`** (helper; help kind, memory ability): "If you had to go now to the router you saw, which way would you set
  off to get there by the shortest way your body fits through, of the ways you have seen?", a heading
  ([docs/families/way_to.md](docs/families/way_to.md)). The helper must recall where it saw a thing and which doorway to
  take. The gold is code over the seen floor (`route_field`): the 0.1 m cells last seen free or driven over, eroded by
  half the body's width, and the Dijkstra distance in metres from the cells within R of the thing's place at its last
  recognisable sighting; each of 360 headings has a null outcome (less room than the body needs, or ending on floor
  never seen) or how much nearer one R of driving along it brings the robot. The credit (`route`) is the kernel on the
  regret against the best heading's outcome, with N = C and the field's 0.1 m cell for `rho`, so the margin is the
  largest of C, a tenth of R and twice the cell (20 cm on the helper) and quarter 4 is whole within one cell (10 cm); a
  null heading scores -1, so a wrong doorway costs its whole regret. Its subjects and its `unknown` pool are
  `ego_mem`'s, a seen thing with no seen route is a dead band and never an `unknown` gold, and the robot's own log
  ("turn back the way you came") is its `path_log`. It passes every gate on the helper (36 cases). It asked 30
  questions (25 on the short and medium cases) where the design expected about 45, because the end questions reserve
  their things first; all 19 answerable golds on the short and medium cases ask for a turn of at least 30 degrees.
  The gold drops a best outcome above the geometric bound (final review S5; `route_field.max_outcome`): a drive of one
  R cannot bring the robot nearer than the field's own metric lets it, which is 1.44 per R on the helper, and a best
  outcome above it means the eroded floor and the body's corridor disagree there. It dropped 1 of those 30 rows on the
  dev compile (an outcome of 2.99 against that bound; the other golds lie between 0.94 and 1.21).
- **The vacuum's `task_things`** ([docs/families/task_things.md](docs/families/task_things.md)): "Where were the things
  on the floor of the rooms you cleaned that your brush could catch (a cable bundle, an extension cord, a power strip or
  a towel; a bath mat, a clothes pile or a pedestal mat counts neither way) in your first task, which you would have to
  steer around next time?" `tangle_ahead` names the same kinds in the same words (`hazard_rule.catch_words`, built from
  the two kind sets below), so a reader is never charged for a thing the words leave open. A cord left on the floor
  stops the next clean. `TaskThings.item_kinds` names the kinds a brush
  catches as the only items (`hazard_rule.CATCH_KINDS`), and the kinds that recur at one spot from script to script
  (`SPOT_KINDS`: bath mat, pedestal mat, clothes pile) are don't-care, the split `tangle_ahead` makes; it was dropped in
  0.9.0 for a layout prior that this removes (`path_log_in_pool` +0.258 then, -0.298 now). A thing of any other kind is
  no item and no don't-care circle, so a list holding it is charged a false place. About 0.6 questions a case, and
  `empty_keep_percent` 20 by the declared rule from the first compile's empty share. It passes every gate on the vacuum
  (36 cases).
- **The abstention report** (`score_abstain`; an "Abstention" section of `results.md`): per model, job and family,
  recall (the share of `unknown`-gold questions answered `unknown`), precision (the share of `unknown` answers on an
  `unknown` gold) and the rate of `unknown` on answerable questions, with Wilson intervals. On 0.10.0's answers: sol
  recall 0.84 and precision 0.31, astra 0.78 and 0.48, luna 0.47 and 0.28, and `unknown` on 17%, 8% and 11% of the
  answerable questions.
- **Fast loops** ([docs/eval.md](docs/eval.md), "Fast loops"). `stbench check` takes `--select KEY=VALUE` and
  `--cases ID,...` (a scoped check, which never writes `check_report.json`, and says what it was scoped to),
  `--answers DIR` (store each scripted system's answers) and `--reuse DIR` (reload the answers of the systems that do
  not read the scorer, `Baseline.reads_scorer` false, and run the rest); `stbench eval` takes `--cells-from DEVROOT`.
  `case_select` is the one owner of the case selection that `eval` and `check` share. The fast tier is 24 dev cases
  chosen by a fixed rule (3 scripts a job that host a short and a medium case, ordered by
  `stable_hash("fast-tier-0.11", script id)`), for sol and luna after a change to the questions, the prompt or the
  frames.
- **The leak check's two readings** (`pilot_leak`; `stbench pilot report --leak ON OFF`). The verdict (D2's rule, margin
  0.05, unchanged) reads the pairs whose gold is answerable; the pairs whose gold is `unknown` are an abstention shift,
  reported apart with their own d and interval and no verdict; each (job, family) has its rows. On 0.9.0's run the
  answerable pairs gave d +0.001 [-0.063, +0.063] over 171 pairs and the `unknown` pairs +0.364 [+0.087, +0.737] over
  22: the vacuum's failing +0.178 was two `unknown`-gold pairs.
- **`heading_screen` and `FamilySpec.own_log_screen`** (off): the fallback declared before the check for a heading
  family on whose job `path_log` or its in-pool twin reads over the shortcut line. A heading point is a dead band when
  the robot's own log already points within about twice the need of the best way, as `World.told` never asks what the
  log tells. It is not on in any spec.
- **`route_field`, `table_way`, `baselines_path_way`, `questions_way`, `cell_weights`, `check_answers`, `case_select`**;
  `credit_kernel.KAPPA`, `score_graded.route_credit`, `records.CREDIT_KEYS["route"]` (the heading keys `profile_m`,
  `step_deg`, `step_m`, `resolution_m`, `need_m`) and `BALANCE_POOLS["way_to"]`.

### Not in 0.11.0

- **No new render.** The 0.7.0 renders are compiled again, so the questions are new and the pictures are not.
- **Codex rows.** No Codex model has read the 0.11.0 questions, the new prompt words or the new credit: the
  reference rows are Claude models. Codex's 0.10.0 rows are in
  [docs/history/report-0.10.0.md](docs/history/report-0.10.0.md), and their answers scored again by 0.11.0's scorer are
  above.
- **A drone family.** The drone gets no new family on this data. Every candidate fails a validity test: people near the
  drop points stand at layout spots (639 of 643 downtown dwell places lie within 2 m of another script's); a remembered
  person at the drop scores +0.42 with a constant and a drop clearance +0.34; a people list at the drop is the helper
  list's layout prior; flight routes are fixed by the map, so a pose-keyed pool learns any route or heading answer;
  vehicles stand at parking slots; and 12 of 27 carried things move between flights, but in its 16 medium cases the
  drone sees both sides of 11 changes and can name 3. Downtown is reported as it is. What
  would change it is a data round with renders: people placed by a coin around each drop point, vehicles drawn per
  flight, and more walkers on its streets ([docs/question-design.md](docs/question-design.md), section 3).
- **A held-out set.** It is designed (fresh seeds, the code frozen before the set is compiled, reported apart from dev
  with a t-interval over scripts) and deferred, since no render was wanted near the release. Every row is a dev row.
- **Change over time.** Still untested: no question asks what moved, appeared or went since an earlier pass. The change
  families stay in the code and are asked by no job. The fix is new data (scripts that revisit places after the world
  changes), which needs new renders.
- **The leak check on 0.11.0, read after the release (2026-10-17).** `claude:sonnet` at medium effort, during on against
  off, on every short and medium case of the helper, the vacuum and the AMR: pooled d -0.007, interval [-0.092, +0.079],
  over 180 answerable pairs of 44 scripts (the AMR -0.010, the helper -0.005, the vacuum -0.003). Inconclusive by the
  rule, since the upper bound is above the margin of 0.05, so the during questions ship and a leak above +0.079 is
  excluded. A subset reading before it (33 pairs) had put the AMR's `closest_person` at +0.29; at full size it is +0.03
  over 30 pairs.
- **A stopping-distance catastrophe.** The trigger stays the body's length plus C: a reaction time and a braking
  deceleration for each robot have no primary source.
- **`way_to` for the vacuum, the AMR and the drone,** and the other questions the design considered and left out (a
  route between named places, the offset between two remembered things, a tighter tolerance): the reasons are in
  [docs/question-design.md](docs/question-design.md), "What is deliberately not asked".

## [0.10.0] - 2026-10-15

Two changes: one frame at a time, pulled by the agent, in one session; and questions that are the decisions each
robot's brain faces, asked far more often. The renders are 0.7.0's, as in 0.9.0, with no new render: the same 64
shifts, cut into the same 144 cases (16 short, 16 medium and 4 long a job), compiled again.

| Job | Place | Scripts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 689 | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 513 | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 235 | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 264 | 2,534 |

**One frame at a time.** Every system gets exactly one frame in each `observe` and its reply is the request for the next,
so the agent sets the pace; the reference Codex agent holds one session for the whole case and pulls frames inside it. The
batch of up to 8 frames leaves the protocol and becomes the question compiler's look window, and `now` becomes the latest
frame. This change alone left every question and gold value as 0.9.0's and moved 44 `scope` tags.

**Task-decision questions** ([docs/question-design.md](docs/question-design.md) is the reasoning). A look point can be any
frame, the robot's decision frames come first in the draw, a point has three slots filled harm first, and a case now asks
one question in 12.7, 8.7 and 13.3 frames on the drone, helper and vacuum and one in 24.7 on the AMR, against one in 29.8,
14.1, 69.2 and 35.8 in 0.9.0 (short and medium cases, after the final review's repairs; see "Measured density" below).
Two families are new for the jobs that ask them (`closest_approach` and `tangle_ahead`) and one is built and asked by no
job (`drop_zone_people`), the floor robots are asked only about things on the floor, every gold row records the age of
its evidence, and an answer whose action would end in harm is charged a flat penalty in the reported score, never in a
validity gate. The question set and the gold values therefore differ from 0.9.0's.

**The final review's repairs.** An independent review of the branch found one blocker and several faults, all repaired
without moving a threshold to a score: the helper's `before_place` frames no longer come first in the look draw, because
a decision frame put its corridor question at the same plan pose in every script and made the evidence verdict fail;
`drop_zone_people` leaves the drone (2 questions); every step carries the robot's own velocity, so a `closest_approach`
row whose person stands is `now`; a corridor's catastrophe needs the true corridor under the body's length plus C; a
`now` row's age is 0; and the prompt states what omitting a cord from a list costs. On the dev root `stbench check` now
prints `ALL GATES PASS` and `EVIDENCE VERDICT PASSES`. They are listed under "Changed" and "Removed" below.

The release ships reference rows on 42 of the 124 short and medium cases that ask a question, run on the redesigned
questions and the pull: the cases all three models had finished when the Codex plan's weekly quota ran out (under
"Added"). The leak check was not rerun, and the questions do not test change over time: see "Not in 0.10.0".

### Changed

- **Every system gets one frame per `observe`** (0.10.0; [docs/agent.md](docs/agent.md)). `steps` is a list of exactly one
  step, so an agent written for batches runs unchanged (`minimal_agent` and `template_agent` do). The reply to an
  `observe` is the request for the next frame: the harness sends nothing more until it has come. The inbox holds only
  the current frame's files, during its `observe` and during the ask after it. The timeouts (600 s a message, 3,600 s the
  end ask, 12 h a case), the failure rules, `task`, `ask` and `finish` are unchanged, and the step gains one key, `velocity`
  (below). A side benefit:
  nothing in the stream marks where a decision point falls, so the two arms of the leak check stream identical messages
  by construction.
- **The batch is the compiler's look window** (0.10.0; `look_windows.py`). The rule moved word for word: a window is at
  most 8 steps, cut after every decision point and every task's last step. `World` renames `batches`, `batch_ks`,
  `batch_of` and `end_ks` to `windows`, `window_ks`, `window_of` and `end_window`. The window still decides where look
  points fall and the difficulty rules of `ego_mem` (a thing seen before the window and not in it) and `person_mem` (a
  person last seen before it); a system never sees one. So the questions asked and their gold values are unchanged.
  `stbench view` marks look windows where it marked batches.
- **`now` is the latest frame** (0.10.0; `docs/questions.md`, "How a gold is computed"). A question is `now` when the
  frame at its point alone gives its gold: `clear_path` (ground and flying), `open_heading` and `rear_clear` are `now`
  only when frame k's own cells, or the wall frame k saw, give the gold within the row's resolution; `ego_now` already
  read frame k alone; `current_view`, the memoryless floor, answers from the one frame it was last sent. The scope used
  to be read over the look window's cells, so a corridor that needed an earlier frame of the window was still `now`, and
  under one-frame delivery `now` has to mean what a system holds when it is asked. On 0.9.0's data it moves 44 of 668
  rows from `now` to `memory` and none the other way: `clear_path` 78 to 43 `now` (drone 9 of 32, helper 12 of 26,
  vacuum 14 of 20) and `open_heading` 15 to 6 (helper 7 of 9, vacuum 2 of 6); `ego_now`'s 65 stay. A compile draws
  fresh question ids, so the recompiled data pair with 0.9.0's runs by case and question order, not by id.
  - **Cells move across the 20-question line.** On the recompile, the helper's clearance `now` cell holds 14 questions
    (it held 26) and the vacuum's 6 (it held 20), so both are thin; the headline's `now` column is then distance plus
    the drone's clearance cell (23 questions, from 32). The drone's clearance `memory` cell holds 22 (from 13) and is
    counted. The rule is not loosened for this: the question redesign (below) raises question density, which is the
    remedy. On its compile the only thin cell left was the helper's heading `now` (2 questions, a diagnostic); after the
    repair of `closest_approach` the AMR's clearance `now` cell is thin as well (6 questions; see "The gates" below).
- **The reference Codex agent pulls its frames** (0.10.0; [docs/codex.md](docs/codex.md), "How a run goes"). A case is
  one long `codex exec` on one thread. The tools are `next_frame` (the robot's sensor feed: each call returns the next
  task message, frame, ask or word that the job is over, and the frame's image goes as an MCP image block) and
  `submit_answer`. The model's next `next_frame` call means it is done with the item; a call made while questions are
  unanswered gets one error naming them, and the second fails them `no_submit`. The time from serving an item to the
  next pull is bounded by that message's timeout, and past it the case ends as `timeout`. If the model ends its turn
  early the thread is resumed with one fixed line, three times at most before the case ends as `stalled`. Why: a Codex
  turn waits about 12 to 13 s for its first token whatever it carries, while a request after a tool result inside a
  turn took about 3 to 5 s, so an exec for each frame would pay the first wait for every frame. The "How this works"
  bullets of the system prompt describe the pull; nothing is said of memory.
- **Look points may fall on any frame, decision frames first, three slots a point, harm first** (0.10.0;
  [docs/questions.md](docs/questions.md), "Look points"). The candidates are the frames at least 4 steps into their task
  but not its last; the frames of each job's priority decisions (`JobSpec.look_decisions`: before a crossing, a station, a
  grasp, a doorway, a lowering, a room done; the helper's `before_place` was one until the final review) come first in the `stable_hash(case id, k, "ask")` draw; a frame is
  kept at least 4 frames (`LOOK_GAP_STEPS`, was 8) from every frame kept, up to `ceil(frames / LOOK_PER_FRAMES)` points with
  `LOOK_PER_FRAMES` 6 (was a cap of 6, 10 or 14 by length). The draw reads the case id and the plan, never the world, so a
  question can show a frame is a decision moment, which the robot's status and subgoal already show. A point asks one
  question a slot of the job's table (`JobSpec.look_shares`): slot 1 the job's harm questions, slot 2 what the robot saw
  before, slot 3 the rest. The tables are in [docs/question-design.md](docs/question-design.md), section 5; no score enters a
  share.
- **Measured density** on the dev root after the final review's repair pass (short and medium cases, 32 a job; one
  question in N frames, look questions then all questions, the harm share of all): AMR 37.5 and 24.7, 25%; drone 12.7
  and 12.7, 74%; helper 11.5 and 8.7, 23%; vacuum 14.0 and 13.3, 65%. (Before the repair pass they were 37.5 and 24.7;
  12.6 and 12.6; 12.5 and 9.3 before `person_mem` left the helper; 14.0 and 13.3; 0.9.0: 35.8, 29.8, 14.1 and 69.2.)
  Every job meets the target of one in 15 except the AMR, which is supply-limited (every slot is whole already, a person
  is in range at few draws, and the floor holds few things to name twice). Harm holding half of the questions is out of
  reach by supply on the AMR and the helper.
- **The gates after the repairs** (0.10.0; [docs/question-design.md](docs/question-design.md), section 8). On the dev root
  after the final review's repair pass, `stbench check` finds every gate passing in the point
  verdict and in the evidence verdict, and the balance rule everywhere:
  - **The helper's `clear_path` `memory_needed` no longer fails.** It failed the evidence verdict after the first
    compile (+0.136 [0.022, 0.283] on the answerable `memory` rows, +0.177 on the long cases) and was first recorded as the
    fixed structure of the helper's recurring routes. That diagnosis was wrong. The review split the rows: 44 of the 106 sat
    at decision frames, 8 of the carrier gold's 10 at `before_place`, and a decision frame puts a structure question at
    the same plan pose in every script. The decision-frame rows alone read +0.211, the others +0.000, and the rows without
    `before_place` passed. `before_place` left `look_decisions` and the cells read +0.000 (96 questions: 29 `now`, 55
    `memory` answerable, 12 `unknown`). No gate, threshold or share changed.
  - **`closest_approach` passes every gate on both jobs** (AMR 24 questions, helper 22), and so do `tangle_ahead` and every
    other family. The first wiring failed the AMR's `knowable` (+0.119), which the two fixes below removed. The helper's
    `closest_approach` has no `memory` row now, so its `memory_needed` gate has nothing to read.
  - **`person_mem` is no longer asked** (below).
  - **The AMR's clearance `now` cell counts again** (22 questions, was 6): see `closest_approach` below.
- **The recent view slides with the point** (`look_windows.recent`): `ego_mem` asks of a thing, and `person_mem` (which
  no job asks now) of a person, not seen in the last 8 frames of the current task up to the point, where it was the look
  window that ended at the point. The case's last 8 frames are the view the end questions are asked from
  (`World.end_window`).
- **The floor robots are asked only about things on the floor** (`World.on_floor`): the AMR's `ego_now` and `ego_mem` and the
  vacuum's `ego_mem`, and the `unknown` pool of `ego_mem` too, so a thing's kind does not predict the slice. This cut the
  AMR's `ego_now` by 26%, the AMR's `ego_mem` by 44% and the vacuum's by 29%. The helper and the drone ask of every thing.
- **A reported score and a gate score** (0.10.0; [docs/methodology.md](docs/methodology.md), rule 7). The penalised score
  `2s - 1` is read by every gate, `knowable` and the evidence verdict (a result row's `score_gate`); the reported score, which
  the headline, the reports and the reference rows read (the row's `score`), is the same except for a catastrophe. A
  catastrophe is an answer whose action would end in harm, by a harmful error larger than `max(C, 2 rho)`; it scores minus a
  weight whatever credit it earned: 2 (`credit_kernel.CATASTROPHE_THING`) for contact with a thing the robot saw (`clear_path`
  answered longer than it is, where the true corridor is shorter than the body's length plus C and the answer is not: the
  rule has the row's `threshold_m`, as the people's has, since an overstated corridor with room to spare is no collision), 3 (`CATASTROPHE_PERSON`) for contact with a person or a hazard of the robot's own kind
  (`closest_person`, `closest_approach`, `person_mem`, `drop_zone_people`, `tangle_ahead`). `unknown` is never one. The rule
  is on the gold row (`tolerances.catastrophe`). Replayed on 0.9.0's answers, the term lowers the apartment's headline for sol
  from +0.443 to +0.364, luna's from -0.133 to -0.340 and astra's from +0.663 to +0.615, leaves the warehouse's unchanged,
  and fires on no answer with a credit of 0.75 or more; with the term off the replay reproduces the published headlines
  exactly, and `stbench check`'s verdicts are identical with it in.
- **Evidence age on every gold row** (`age_s`, `age_band`: `now`, `recent` up to 10 s, `task`, `earlier`; null on an
  `unknown` row). The reports give the reported score by band, overall and for each job, and the catastrophe rate per job.
- **A frame and its questions reach the reference agent together** ([docs/codex.md](docs/codex.md), "How a run goes"). The
  harness holds a frame until its next message shows whether questions come with it; an ask at the held frame's step joins
  it, so one `next_frame` reply carries the header line, the image and the questions. The frame's deadline starts when it is
  served. Several asks at one point stay one question each, so a later question's wording cannot inform an earlier answer.
  Other systems still get `observe`, then `ask`.
- **Delivered clocks carry tenths of a second** (`YYYY-MM-DDTHH:MM:SS.s`, one decimal always; `case.format_clock`): the second-
  rounded clock moved `closest_approach`'s gold by more than C on 6% of person rows. The date and the time of day are drawn per
  case; the differences between clocks are true.
- **Every step carries the robot's own velocity** (0.10.0; final review S3; [docs/agent.md](docs/agent.md)). `velocity` is
  `[vx, vy]`, metres a second in the map frame: the displacement of the delivered base position from the previous delivered
  frame of the same task, over the true time between them (not the tenth-rounded clocks), to the millimetre a second, and
  `[0.0, 0.0]` at a task's first frame (`case.base_velocity`, the one owner). It is the controller's odometry, which the
  robot knows of itself. The reference agent's header shows it (`velocity (vx, vy) m/s`, between the heading and the
  camera pitch; the header still starts `[k<k> `), the table and memoryless baselines read it, and `closest_approach`'s gold
  and table use it, so no frame before the ask is needed to read the robot's motion. A `closest_approach` row whose person
  stands (or has no earlier sighting to move against) is therefore `now`, with the ask the only evidence and an age of 0,
  and a row whose person moves is `memory` with their earlier sighting as evidence. On the dev root that is 32 `now` person
  rows, 12 "no one" rows and 2 `memory` rows (the AMR's two walkers, 1.06 and 1.14 m/s): the motion half is built, tested
  and exercised by two rows. The final review had counted 0 walking among 33 rows, by the evidence set, which a walker
  sighted at the frame before the ask does not lengthen. `current_view` reproduces every `now` row (1.000), and the oracle
  and `memory_table` still reproduce every row. The AMR's clearance `now` cell holds 22 questions and counts (it held 6);
  the `memory` cell holds 36.
- **The helper's `before_place` frames no longer come first in the look draw** (final review B1; `look_decisions` is
  `before_grasp` and `before_doorway`). A decision frame puts a structure question at the same plan pose, by the same
  furniture, in every script, which let a memoryless prior learn the helper's corridor lengths and failed the evidence
  verdict. The helper now draws 96 corridor questions (106), and the memory cells read +0.000. The second form of the
  repair, no decision frame first for a structure family, was not needed.
- **`clear_path`'s catastrophe has a threshold** (final review S5): the body's length plus C, as `closest_person`'s, so an
  overstated corridor is a catastrophe only where the true one is shorter than that and the answer is not, and by more than
  the margin. It was a flat -2 for any overstatement past `max(C, 2 rho)`, including a 30 m answer to a 24 m drone corridor.
- **A `now` row's `age_s` is 0** (final review S6): the latest frame answers it, so the seconds from an evidence frame it
  names (35 `open_heading` and 9 `clear_path` rows of 0.9.0's data) are no age.
- **The prompt states the list catastrophe** (final review S7): for a job that asks `tangle_ahead` during the job, a list
  with no place within a matching radius of a thing the brush could catch within 30 cm of the front edge scores -3, with the
  weight and the distance read from `credit_kernel` and `hazard_rule`; the helper, the AMR and the drone are told nothing of
  it (`llm_credit.LIST_CRITICAL`, `brief.list_hazards`).
- **`closest_approach` asks about "the person you can see nearest you"**, where it said "the person nearest you": the gold
  is the evidence person in frame k, and a person the frame shows without placing makes the point a dead band.
- **A list question says what it counts.** The `positions` answer shape in the prompt gives the form only; `task_things` ends
  with the sentence that says what a thing is and says why the things matter ("which you would have to steer around next
  time"), and the two new list questions say what they count in their own words.
- **[docs/vision.md](docs/vision.md), "Speed", is amended** (2026-10-14): the sentence that harder questions "never add
  questions or time to a case" is replaced, since the frames are a case's fixed cost. **[docs/methodology.md](docs/methodology.md)**
  now separates the gate score from the reported score (rule 7, which read "the penalised score is unchanged") and records
  the evidence age (rule 4).
- **The Claude Code example** (`examples/claude_code_agent`) runs one `claude -p` for each message, so one for each
  frame now, and pays its start-up for each. It is an example and not a reference row.
- **The documents** (0.10.0). The README, the dataset card and the technical report ([docs/report.md](docs/report.md))
  are written for 0.10.0, and their tables of results and gates are pasted from the runs and the check by
  `tools/fill_report.py`. They say plainly what the release does not test (change over time, below) and that the map
  reader is a reference, not a competitor: every case starts a fresh agent that has never seen the place, and the map
  reader answers corridors from the public floor plan only to show which questions ask about fixed structure. The report
  of 0.9.0 is kept in [docs/history/report-0.9.0.md](docs/history/report-0.9.0.md).

### Added

- **`closest_approach`** (AMR, helper; harm, clearance): "If you kept moving as you are now, how close would the person
  you can see nearest you come to you in the next 5 seconds?", the least floor distance over 5 s if the robot and the person each keep
  the velocity they have, credited on the job's clearance C ([docs/families/closest_approach.md](docs/families/closest_approach.md)).
  Two validity fixes came out of its first check: a row with a person was `memory` (the robot's velocity then needed the
  frame before the ask, and `now` is the latest frame; the velocity is delivered with every step now and a row whose person
  stands is `now` again, above), and a person's motion counts only when the sensors pair it, which is
  when exactly one evidence person is in the ask's frame and in the earlier frame used (a detection's key names nothing, so
  a frame with two people is a dead band). The AMR's `knowable` gate went from +0.119 to +0.000.
- **`tangle_ahead`** (vacuum; harm, clearance): "List anything lying on the floor within 1 m of you, ahead or to your sides,
  that your brush could catch", a list of places; bath mats, pedestal mats and clothes piles recur at the same spots from
  script to script and are don't-care ([docs/families/tangle_ahead.md](docs/families/tangle_ahead.md)).
- **`drop_zone_people`** (harm, clearance; asked by no job): "Before you lower the package: where are the people within 4 m
  of the point below you?", a list of places, asked in `inspect`, `descend` and `lower` when the 4 m disc is wholly in the
  frame. The drone asked it in the 0.10.0 draft and drew 2 questions on the dev root, under the 20 a counted cell needs,
  with no per-family diagnostic, so the final review took it off the drone (below); the code and the page stay
  ([docs/families/drop_zone_people.md](docs/families/drop_zone_people.md)).
- **`JobSpec.look_decisions` and `look_phases`**, the decision kinds drawn first and the status phases a hazard family is
  asked in; `questions_age`, `score_catastrophe`, `questions_harm`, `approach_rule` and `hazard_rule`; `score.reported` and
  `score.catastrophe`; the reports' `by.age_band`, `job_age_band` and `catastrophe_rate` and the `results.md` sections "Per age
  of the evidence" and "Catastrophes"; the journal's `asked` text on a joined frame and the summary's `frames_asked` and
  `asked_latency`.
- **[docs/question-design.md](docs/question-design.md)**: what is asked, when, how it is graded and why, with what is
  deliberately not asked.
- **`look_windows`** (`LOOK_WINDOW`, `windows`) and the compiler's `World.windows`.
- **`codex_feed`, `codex_live` and `codex_asks`.** The harness end of the pull (a listener on the loopback interface
  with a random key, over `multiprocessing.connection`; no module imports `socket`), the long exec read as it goes with
  every event checked as it arrives (a tool that escapes kills the exec at once), and the table of an ask's answers.
  `codex_mcp`, the tool server, is a thin relay.
- **A fuller journal and summary.** The journal gains `served` (each item the model was shown, its images as hashes),
  `call` and `exec_started` events; the summary counts frames served, pulls, premature pulls, resumes, the per-frame
  latency (median and p90) and whether the token count is a lower bound (a turn the harness kills prints no usage).
- **`tools/codex_canary.py` runs the pull path**, and the tests' fake Codex pulls from the tool server as the model would.
- **Reference rows** on 42 of the 124 short and medium cases that ask a question (379 of their 1,239 questions; 20
  apartment cases, 11 warehouse, 11 downtown), with the camera channel, the correct brief and the during questions on:
  `codex:gpt-6.1-sol` at medium effort, `codex:gpt-6-luna` and `codex:gpt-6-astra` at xhigh, the floor `all_unknown` and
  the `map_reader`. Every system ran the same 42, the cases all three models had finished when the Codex plan's weekly
  quota ran out; which finished first was set by the order the runs took them. A first start on 14 October ended on a
  Codex-side failure; its cases were run again from the start, and every case that started in the runs that make the
  rows ended complete.
  They are dev rows: scored on the shifts the questions were refined on. The headlines, with their 95% interval over
  shifts:
  `codex:gpt-6-luna` (xhigh) scores -0.091 [-0.253, +0.046] on `apartment_a`, +0.008 [-0.132, +0.125] on `downtown_a`, +0.308 [+0.137, +0.459] on `warehouse_a`; `codex:gpt-6.1-sol` (medium) scores +0.461 [+0.325, +0.595] on `apartment_a`, +0.593 [+0.433, +0.742] on `downtown_a`, +0.554 [+0.431, +0.646] on `warehouse_a`; `codex:gpt-6-astra` (xhigh) scores +0.638 [+0.524, +0.717] on `apartment_a`, +0.768 [+0.630, +0.849] on `downtown_a`, +0.598 [+0.448, +0.777] on `warehouse_a`; the floor and the map reader score
  `apartment_a`: `all_unknown` (rgb_pose) +0.032 [+0.016, +0.050], `map_reader` (rgb_pose) +0.220 [+0.134, +0.309]; `downtown_a`: `all_unknown` (rgb_pose) +0.132 [+0.078, +0.219], `map_reader` (rgb_pose) +0.647 [+0.543, +0.763]; `warehouse_a`: `all_unknown` (rgb_pose) +0.083 [+0.000, +0.167], `map_reader` (rgb_pose) +0.083 [+0.000, +0.167].

### Removed

- **`drop_zone_people` is no longer asked** (the drone's; final review S2). It drew 2 questions over the 36 dev drone cases
  (413 of its 593 draws fell outside its three phases and 174 more found nobody within 4 m), under the 20 a counted cell
  needs, and its rows sat inside the drone's clearance `now` cell with no per-family diagnostic. It leaves the drone's
  `looks`, `look_shares` (the drone has two slots now: `clear_path` 60, `ego_mem` 70) and `look_phases`; nothing else is
  rebalanced. The family, its gold, its table answers and its page stay for a dataset with more supply. The drone's density
  is 1 question in 12.7 frames (12.6).
- **`person_mem` is no longer asked** (the helper's; 0.10.0; [docs/families/person_mem.md](docs/families/person_mem.md)). It
  failed `separation` on the first 0.10.0 compile: 19 questions, 7 answerable and 12 with an `unknown` gold, so a perfect
  reader led a blind `unknown` by exactly the answerable share, +0.368, against the gate's 0.4. The evidence rule drops it
  rather than re-tuning it. Its 40% of the helper's second slot went to `ego_mem`, which has 65% of points; the helper's
  slots are now `clear_path` 50 and `closest_approach` 50, `ego_mem` 65, and `ego_now` 20 and `open_heading` 15. The family,
  its gold, its table answers and its catastrophe rule stay in the code. Where an out-of-view person is goes unasked, an
  open gap.
- **`stream_protocol.BATCH` and `batches`**, and the batch in the messages, the viewer and the documents.
- **One `codex exec` for each message,** `Codex.run`, and the per-exec limits on tool calls (40 in a task's exec, 100 in
  a batch's, 60 in an ask's); an item gets at most 100 calls, and `next_frame` is exempt.
- **The look-window-end rule for look points, the per-length cap of 6, 10 or 14 points, the 8-step spacing and the two-slot
  tables of 0.9.0.** A look point is any frame; see above.
- **The counting sentence of the `positions` answer shape in the prompt** (what a "thing" is): it served `task_things`
  alone and now lives in that question's text.
- **The batch's wording in the Codex messages:** the title line over a batch, the line naming each attached image, and
  the prompt's "reply with one short line after each task message and each batch".

### Not in 0.10.0

- **Change over time.** No question asks what moved, appeared or went since an earlier pass, or where a moving person is
  going. The world changes only between tasks, so the short cases hold no change; in the medium cases the robot sees both
  sides of only 11 to 28 of the 90 to 165 changes a job's 16 medium cases hold, whether a thing moved is largely given by
  its kind, and balancing that leaves a fair change question 4 questions in the whole release. The fix is new data
  (scripts that revisit places after the world changes, changes drawn per thing by a coin rather than by kind, more
  people walking), which needs new renders. The change families stay in the code and are asked by no job.
- **Walking people in the data.** The motion half of `closest_approach` rests on two AMR rows; the helper's resident never
  walks at a frame tried. A dataset with walking people would exercise it.
- **Reference rows on every case.** The rows cover 42 of the 124 short and medium cases. The pull re-reads the whole
  session at every frame, so a case reads about 3 million input tokens (short) to 11 million (medium), nearly all of it
  cached: about five times 0.9.0's, against a projection of three to four times. In 73 minutes, 18 execs at once (6 a
  model) ran 157 cases with no rate-limit error, and took the ChatGPT Pro plan's weekly quota from 53% to 96%, most of
  it astra's at xhigh. The rest of the rows wait for a later revision.
- **The leak check.** It was not rerun. Its last reading is 0.9.0's, on 0.9.0's questions (`gpt-6.1-sol`, inconclusive:
  pooled d +0.043, interval [-0.030, +0.118]; the vacuum's own reading, +0.178, was above the margin). 0.10.0 asks one and a half to five times as many questions during the
  job, so that reading does not carry over, and a leak is not excluded.
- **The help kind in the headline.** `open_heading` stays a diagnostic until its credit is redesigned, so the help kind is
  counted in no mean.
- **Questions not built:** at the start of a task, blind corners, what stays and what moves, the way to the goal, and answers
  with uncertainty; the reasons are in [docs/question-design.md](docs/question-design.md), "What is deliberately not asked".
- **Reworked unasked families.** The table answerers of `out_of_view`, `clearance_then`, `slots_empty` and `zone_clear`
  see a window of one frame and no longer reproduce their golds, and `zone_clear` and `path_ahead` keep a fixed `now`
  scope and a gold over the look window. No job asks any of them, so no score is affected; a job that asks one again
  first reworks it.

## [0.9.0] - not released

0.9.0 was finished on 2026-10-14 and never tagged or uploaded: 0.10.0 replaces its questions and ships its data.

Questions as decisions, scored by consequence. Every question now asks a decision the robot's brain faces in its job,
about an action and what it would do: what an action now would hit (harm), which action now moves the task forward
(help), what is around the robot (situation) and what is worth remembering to do the job here again (reuse). Every
metric answer is credited by the consequence of its error in four quarters, so the right direction earns some credit,
the right thing or action more, the right amount more, and only precision at the camera's own resolution earns the
whole. The renders are 0.7.0's, with no new render: 64 shifts, 16 a job, cut into 144 cases (16 short, 16 medium and 4
long a job). Only the questions and the scoring are new, so the data are compiled again: 668 questions,
550 of them on the 128 short and medium cases (121 of which ask anything), against 0.7.0's 785 and 643. 0.8.0 was an
internal checkpoint and was never released; what survives of it is folded into this entry, marked with the version it
came from.

| Job | Place | Shifts | Cases (short / medium / long) | Questions (short and medium) | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 100 (81) | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 311 (253) | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 156 (133) | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 101 (83) | 2,534 |

The vacuum asks nothing in 6 of its 36 cases and the drone in 1, all short or medium.

### Added

- **Questions as decisions** (0.9.0; [docs/vision.md](docs/vision.md), "How questions are made and scored"). The
  families fall into the four kinds, and the reports split by them (`by.kind`):
  - **harm:** `clear_path` (how far the body could move before it touched anything), `closest_person` and `person_mem`;
  - **help:** `open_heading` (which way gives the body the most room);
  - **situation:** `ego_now` and `ego_mem` (where things are from the camera, in view and out of view);
  - **reuse:** `object_where` and `task_things` (where things are kept, and what stood near the route).

  Answers are open: a distance, a heading, a place or a set, never a multiple choice. The strongest questions ask
  about space the camera cannot see now, which only memory answers.
- **Credit by consequence** (0.9.0; `docs/questions.md`, "Credit"). An answer earns four quarters, each of
  the vision's layers in its order: the right direction (the side, the order, blocked or clear), the right thing within
  twice the job's need, the right amount within the need, and precision, which is whole only within one pixel of the
  camera at the geometry the gold rests on (`resolution_m`, at least 1 cm) and falls to nothing at the need. Credit
  falls smoothly between those points, so no answer coarser than a pixel earns the whole. The need is the job's: the
  clearance C for an offset, a distance in metres or a heading, the re-find radius R for a place, a list's matching
  radius for a list. The penalised score (`2s - 1`) and `unknown` are unchanged. This replaces 0.7.0's three flat
  rungs and 0.8.0's clearance rungs.
  - **Harm costs more than caution.** On the harmful side the need is halved: a clearance answered longer than it is, a
    closest approach answered wider, a thing or a person placed farther away than it is, a heading toward too little
    room. A place to re-find a thing has no harmful side.
  - **A set scores per item.** Items are paired one to one with the gold's by the largest summed credit; a missed item
    costs one and a false one half.
  - **A decision scores by its regret,** the room it gives up against the best heading.
  - The kernel is one standard-library module, `credit_kernel`, which the scorer, the reports and the compiler import.
- **`clear_path` in every direction a robot can move** (ahead and back in 0.8.0; left and right in 0.9.0). The helper is
  asked all four, the vacuum ahead, back and left, the drone ahead and back; the direction is drawn from the case id and
  the point alone (`JobSpec.clear_directions`). The ground gold reads what the robot last saw of every cell of the
  window up to the point, not the batch alone, so a corridor behind or beside the robot needs memory, or the knowledge
  that it never looked there, and space never seen is `unknown`. What the corridor rule fixed in 0.8.0:
  - the stop is read at the 2 cm sub-cells, and a surface at most 3 mm over the climb is floor, for every robot;
  - a body sweeps its own shape, a disc a disc and a box its rectangle (the gold had been up to 12 cm short of a round
    body's travel);
  - a question is asked only where the stopper is tied to a floor contact a camera can read to within C/2 and is
    visible, only where no obstacle within 1 cm of the body's side and no height within 5 mm of the band's edge decides
    the gold, and only in the phases in which the robot drives; a turned corridor is not asked where the body's turn
    would not clear a seen obstacle (0.9.0);
  - the drone flies at least 8.35 m up, above the street lamps' 7.95 m heads, and the first wall its corridor meets
    decides: `unknown` if no frame showed that wall.
- **`open_heading`** (0.9.0), the help kind: which heading, turned in place, would let the body go farthest before it
  touched anything it has seen, scored by its regret. It is asked, gated and reported, but it is a diagnostic for now:
  its ability, `heading`, is counted in no headline mean. The calibration's log review found every gold right and the
  convention followed, but the credit is a cliff. A run changes by metres within a few degrees at a doorway, while the
  regret is read against 10 to 15 cm, so a heading 2 to 10 degrees off the right opening scores -0.4 to -1, as does one
  that leaves 1 to 2 m of room. That breaks the vision's "roughly right ones still score". The next version redesigns
  what a heading's regret is measured against, before any of its scores are read.
- **`closest_person` and `person_mem`** (0.8.0). `closest_person` is the AMR's end question: how close a person came to
  the robot during a task, as the robot saw them, one question a case, in `memory`. `person_mem` is the helper's: where
  the person it last saw is from it now, an offset, `unknown` only when it never saw anyone. Both rest on one person
  evidence rule (`person_seen.evidence`): 1,024 pixels, light, a depth at every pixel, the feet tied to the floor at the
  job's clearance and a box off the bottom edge of the frame, so a person cut by the top of the image still counts. The
  oracle channel's person `floor_hull` follows the same function.
- **The job tables** (0.8.0, extended in 0.9.0). Each job declares the families it asks at look points
  (`JobSpec.looks`), the share each takes of a point's two slots (`look_shares`), the share of `unknown` golds it keeps
  (`look_unknown_keep`), its end families (`ends`), the directions of its corridor (`clear_directions`), the phases in
  which it asks one (`clear_phases`) and the drone's `fly_floor_m`. The draw reads the case id and the table, never the
  world. A table that passes 100 percent, or omits a family, is refused when the job registers.
- **The budget holds** (0.8.0, kept in 0.9.0). No job's or length's mean questions or asks a case rise above 0.7.0's:
  harder questions replace easy ones, they never add questions or time to a case. `ego_now` left the drone and the
  vacuum, and `object_where` left the AMR and the drone (0.8.0). In 0.9.0 the helper's `ego_now` gives slot share to the
  turned corridors, and the helper's `person_mem` and `ego_mem` and the vacuum's `ego_mem` give share to
  `open_heading`.
- **Clearance is an ability** (0.8.0). It has a cell in each column of the headline, `clear_path` in the column its
  evidence puts it in (`now` when the current batch alone answers it), `closest_person` and `person_mem` in `memory`.
  A (job, ability, column) cell with fewer than 20 questions in the dataset is a diagnostic and counts in no mean.
- **The reports** (0.9.0). The aggregate adds `by.kind`, a split by harm, help, situation and reuse. The credit bands
  follow the four quarters (full, within the need, the right thing, the right direction, miss, `unknown`). `stbench eval
  distance` reads each row's need and resolution and reports the share of errors on the harmful side.
- **The map reader as a published reference** (0.8.0, answering its ray since 0.9.0). `clear_path` and `open_heading`
  read the place's fixed structure, and the maps are public. The agent is never given a map, so the structure is
  perceived and remembered, and the `map_reader` (the public map's raycast from the delivered pose; on a held-out place,
  the pose-keyed pool of other scripts' golds) is the declared reference of both, printed beside them. It is never a
  gate; every other shortcut stays one. On a public place a structure family's score is a claim only about a system that
  holds no copy of the map (methodology rule 5 names these two families, and another needs the rule amended).
- **`path_log`'s held-out bar** (0.9.0). The robot's own-log rules (`clear_path`'s detour, `open_heading`'s longest
  swept run) are no proof of an obstacle and had scored below `all_unknown`, so each answers only where the lower end of
  the 95% interval of its gain over `unknown`, over other scripts' records, is above 0. Otherwise it abstains.
- **Speed** (0.8.0 and 0.9.0; [docs/vision.md](docs/vision.md), "Speed"). The free validity check of the whole release
  is about ten times faster: the table baselines read each step's depth image once a process, the runs over one case are
  one pool task, every group goes through one pool, and route search and the raycast are quicker with the same answers.
  It takes about 8 minutes, where 0.8.0's took 45 minutes at 18 workers and 2 to 3 hours at 6. Cases run in parallel
  (`--workers`), and the calibration ran three models over 47 cases at once.
- **The reference agent's prompt** (0.9.0). It states the credit by consequence (the four quarters, the
  halved margin on the harmful side, a list per item and a heading by its regret), the robot's body, the corridor in
  every direction, the heading answer and that a place names the middle of the thing. The brief adds "which way the
  robot has the most room" to the during sentence of a job that asks `open_heading`; the drone is not told of questions
  after the job, since it has none (0.8.0).
- **Reference rows** on the 121 short and medium cases that ask a question, with the camera channel, the correct brief
  and the during questions on: `codex:gpt-6.1-sol` at medium effort, `codex:gpt-6-luna` and `codex:gpt-6-astra` at
  xhigh, the floor `all_unknown` and the `map_reader`. They are dev rows (below). The headlines, with their 95% intervals
  over shifts:
  `codex:gpt-6-luna` (xhigh) scores -0.133 [-0.246, -0.010] on `apartment_a`, -0.215 [-0.357, -0.077] on `downtown_a`, +0.004 [-0.115, +0.116] on `warehouse_a`; `codex:gpt-6.1-sol` (medium) scores +0.443 [+0.367, +0.523] on `apartment_a`, +0.559 [+0.457, +0.683] on `downtown_a`, +0.494 [+0.391, +0.589] on `warehouse_a`; `codex:gpt-6-astra` (xhigh) scores +0.663 [+0.604, +0.716] on `apartment_a`, +0.725 [+0.618, +0.830] on `downtown_a`, +0.695 [+0.636, +0.754] on `warehouse_a`; the floor and the map reader score
  `apartment_a`: `all_unknown` (rgb_pose) +0.092 [+0.061, +0.122], `map_reader` (rgb_pose) +0.231 [+0.175, +0.284]; `downtown_a`: `all_unknown` (rgb_pose) +0.109 [+0.042, +0.196], `map_reader` (rgb_pose) +0.609 [+0.542, +0.696]; `warehouse_a`: `all_unknown` (rgb_pose) +0.030 [+0.006, +0.062], `map_reader` (rgb_pose) +0.030 [+0.006, +0.062].

- **The leak check** on the reference agent (`gpt-6.1-sol`, `--arms during=on,off`, the helper's, the vacuum's and the
  AMR's short and medium cases): pooled d = +0.043, interval [-0.030, +0.118], over
  193 pairs of 42 scripts, verdict
  inconclusive, so the during questions ship. The vacuum's own reading is above the margin
  on 21 pairs and is disclosed ([the 0.9.0 report](docs/history/report-0.9.0.md), "Validity").

### Changed

- **A place's direction quarter is "within R"** (0.9.0). The room a thing is in follows from the public map and the
  thing's kind, without the camera, so a quarter for "the right room" paid a system that holds the layout. Under it a
  sensorless layout prior led the helper's `object_where` on long cases by +0.126; with the quarter at R it falls to
  -0.062, and the models' answers move by at most 0.025. R itself is unchanged.
- **The headline's cells** (0.8.0, 0.9.0). Each place's `now` and `memory` columns are the equal mean of their cells:
  distance, memory, relevance and clearance, with `heading` a diagnostic. The apartment's `now` column is distance and
  clearance, its `memory` column memory, relevance and clearance; the warehouse's `now` is distance alone and downtown's
  clearance alone.
- **A person's floor hull** (0.8.0): a person's `floor_hull` is set exactly when the frame holds person evidence. The
  stream and the scorer's depth are unchanged.
- **The documents** (0.9.0). The README, the dataset card and the technical report ([docs/history/report-0.9.0.md](docs/history/report-0.9.0.md)) are
  written for 0.9.0, and their tables of results and gates are pasted from the runs and the check by
  `tools/fill_report.py`. The report of 0.7.0 is kept in [docs/history/report-0.7.0.md](docs/history/report-0.7.0.md).

### Removed

- **Definition version stamps, recorded digests and pins** (0.9.0): results, reports, the release and new data files
  carry no version, schema, hash or commit. Resume, `eval compare` and the export pair runs with their gold by question
  id. `MANIFEST.sha256`, `stbench assets pin` and `verify`, the asset hashes, the sandbox image digest and `uv.lock` are
  gone; the release number is the version.
- **The vacuum's `task_things`** (0.9.0), by rule. The evidence verdict found `path_log_in_pool`, the robot's own log
  with the layout prior of other scripts' golds, scoring +0.258 over `all_unknown` on its 14 long rows, +0.135 on the 24
  medium and -0.152 on the 11 short, against a gate of +0.10. 90% of the vacuum's distinct gold items recur within 0.5 m
  in another script, and the predicate ("on the floor of the rooms you cleaned") reaches every recurring spot, so a copy
  of the layout earns without the camera (methodology rule 5). A family that fails a gate is dropped, never forced;
  widening the false-item weight would be tuning the credit to pass a gate. The vacuum's end is `object_where` alone and
  the apartment's relevance cell is the helper's. The way back is a held-out place.
- **`ego_now` for the drone and the vacuum, `object_where` for the AMR and the drone** (0.8.0), to make room in the
  budget for the decision questions. Nothing the drone names can be hit from 15 m, and the vacuum's 16 rows were under
  the 20-question line.
- **Not asked, measured and left out:** the AMR's corridors (a held-out constant scores +0.33 ahead, and most golds are
  the 5 m cap), the vacuum's right turn (a constant scores +0.28: it follows edges with the wall on its right), sideways
  flight, `open_heading` for the AMR and the drone, `person_now`, and any people question for the drone and the vacuum.
  Three reuse decisions are answered by the pose log the agent is given and are not asked: where the dock is from here,
  a route between two places the robot has been, and what stays put and what moves.

### Not in 0.9.0

- **A held-out confirmation.** The rows are dev rows: the questions, the screens and the credit were settled on these 64
  shifts, and no fresh shifts were rendered. The gold is public and each place is one layout.
- **A graded heading.** `open_heading` counts in no headline mean until its credit is redesigned.
- **`closest_person` at the form adversary's bar.** The AMR's `closest_person` fails the point reading of its shortcut
  gate by a hair: the form adversary's in-sample best constant scores +0.103 over `all_unknown` against a bar of +0.10,
  and +0.013 held out. The binding evidence line passes. The AMR's C of 0.45 m is wide against closest approaches of
  1.35 to 3.8 m, and the credit was not tuned to pass the gate.
- **`unknown` on a glimpse.** Of 24 wrong answers to `unknown` golds in the calibration, 7 landed on a same-coloured thing
  seen only in glimpses too small to show what it is. A sentence in the prompt (a question may name a thing never seen
  or not there, and then `unknown` is right) is for the next version.
- **Thin cells.** The drone's and the vacuum's clearance `memory` cells (13 and 4 questions) are diagnostics, and the
  vacuum's clearance `now` cell sits at exactly 20.
- **Everything 0.7.0 did not do:** memory at shift length, a hard drone, more places, a clean apartment floor, change
  over time and prediction, and a closed-loop check that the score predicts control.

## [0.7.0] - not released

0.7.0 was an internal checkpoint and was never released: 0.9.0 replaces its questions and ships its data.

Camera only, and credit by the job. The robot has a camera and no depth sensor: the agent is sent each frame's picture
with the robot's pose, clock and status, estimates every distance itself, and is asked where a thing in view is from
it now, where a thing it saw earlier is from it now, where on the map it last saw a thing, and where the things near
its task's route were. Each metric answer is credited by what the job needs to act on it: the job's clearance C for an
offset from the camera, its re-find radius R for a place on the map. 0.5.0 and 0.6.0 were internal checkpoints and were
never released; what survives of them is folded into this entry, marked with the version it came from. The data are a
new render: 64 shifts, 16 a job (12 of 2 or 3 tasks, 4 of 4 to 6), cut into 144 cases (16 short, 16 medium and 4 long
a job), with 785 questions and 16,830 frames, per job as below.

| Job | Place | Shifts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 16 | 16 / 16 / 4 | 157 | 6,460 |
| `helper/1` | `apartment_a` | 16 | 16 / 16 / 4 | 341 | 3,157 |
| `amr/1` | `warehouse_a` | 16 | 16 / 16 / 4 | 174 | 4,679 |
| `drone/1` | `downtown_a` | 16 | 16 / 16 / 4 | 113 | 2,534 |

### Added

- **Camera only** (0.6.0). The agents' channel, `rgb_pose`, sends each frame's 640 × 480 camera JPEG, the base's pose,
  the camera's pose and pitch, the robot's status and its subgoal, and no depth. Depth is recorded for the
  scorer alone and ships under `private/<sid>/depth/`. The reference agent's prompt states the camera's size, field of
  view and mount, and that any distance it gives is its own estimate.
- **Four question families, one for each ability** (0.6.0, worded as in 0.7.0):
  - `ego_now`, distance, the `now` column: where the nearest visible point of a named thing in view is from the
    camera, as an offset along the floor;
  - `ego_mem`, memory: the same for a thing seen earlier and not in view, measured from the camera now;
  - `object_where`, memory: where on the map the robot last saw a thing;
  - `task_things`, relevance: the places of the things near the task's route, by the job's own predicate, which the
    question states (the drone asks none).

  The during questions come at look points: batch ends drawn from the case id and the stream's cuts, never from the
  world, up to 6, 10 and 14 in a short, medium and long case. About a fifth of the `ego_mem` and `object_where`
  questions have `unknown` for answer, in the same words as the rest (`docs/questions.md`, `docs/families/`).
- **Credit by the job** (0.7.0). An offset is credited against the job's clearance C (vacuum 0.10 m, helper 0.15 m,
  AMR 0.45 m, drone 2.4 m), the margin its robot's own operating spec keeps beside its path; a place on the map against
  its re-find radius R (1 m in the home and the warehouse, 2 m for the drone). Each value traces to a source of its
  kind (`docs/questions.md`, "Credit"). The rungs are C/2, C and 2C, or R/2, R and 2R, a third of the credit each, and
  do not widen with range. The prompt states them in metres and centimetres. This replaces 0.6.0's credit
  relative to a thing's range.
- **The nearest visible point** (0.7.0). An offset's target is what the frames show: the point of the thing's floor
  hull (the convex hull of its visible pixels, dropped onto the floor) nearest the floor point under the camera. For
  `ego_mem` the hull is the union over every recognisable sighting at the thing's current place. A pixel-perfect reader
  of the frames reproduces the gold. `object_where` keeps the thing's middle.
- **The screens** (0.7.0). Each removes questions and adds none, and each is the same for every job:
  - the resolvability screen and its surface tie: a thing is asked only where a frame ties its gold point to a surface
    whose height a camera-only reader can read, either a contact with the floor, a shelf or a table top read to within
    C/2 at one pixel's error, or the near edge of a support at or above the camera;
  - the set-back screen: a thing resting on a surface at or above the camera is not asked when it stands more than C/2
    behind the surface's near edge, which no frame from below shows;
  - new look-alike groups: the home's soft goods ({folded clothes, towel, book, book stack} and {jacket, hanging coat,
    clothes pile, folded clothes}) and {cable bundle, extension cord};
  - thin things such as a remote or a cable are don't-care places in `task_things`, counted neither for nor against a
    list.
- **The oracle channel's floor hull** (0.7.0). Each recognisable detection on the `detections` channel carries
  its `floor_hull`, so a memory table built from a channel a system can be sent reproduces every gold.
- **The reports** (0.7.0). Beside the penalised headline, each `results.md` gives the unpenalised mean credit, the
  credit bands (full, within C, within 2C, miss, `unknown`) and the abstention bound. `stbench eval distance` breaks
  the offset answers down against C and by range band, and re-scores the run with C and R scaled by 0.5, 1 and 2
  (`docs/eval.md`).
- **The balance rule and the evidence verdict** (0.6.0). No constant answer may score above `all_unknown` on a family:
  the compile draws the answer classes so that none can, and `stbench check --verdict evidence` confirms it. A learned
  shortcut answers only where it beats abstaining on the shifts it left out. `stbench export` accepts
  only a check report that passes this verdict.
- **The text-only option** (`--text-only`, the Codex agent only): every image is withheld, a zero-shot check of what
  the questions give away without the pictures.
- **Reference rows** on the 120 short and medium cases that ask a question, with the camera channel, the correct brief
  and the during questions on. They are dev rows (below). The headline, with its 95% interval over shifts:

  | Place | `codex:gpt-6.1-sol` (medium) | `codex:gpt-6-luna` (xhigh) | `all_unknown` |
  | --- | --- | --- | --- |
  | `apartment_a` | +0.55 [+0.48, +0.61] | −0.18 [−0.28, −0.09] | +0.05 [+0.04, +0.07] |
  | `warehouse_a` | +0.42 [+0.31, +0.53] | −0.16 [−0.29, −0.03] | +0.06 [+0.03, +0.08] |
  | `downtown_a` | +0.77 [+0.62, +0.90] | −0.12 [−0.27, +0.04] | +0.13 [+0.06, +0.19] |

  The README gives each place's distance, memory and relevance.
- **The checks behind the rows.** On the final compile, every family passes every gate on every job: the oracle and
  the memory table score 1.00, no shortcut beats `all_unknown` by more than +0.07, and no constant beats it at all.
  A review of sol's lost credit there, against the frames and masks, put 1% of it on the instrument on the AMR, 0% on
  the drone and 2% on the helper, and about a fifth on the vacuum, whose causes were repaired by the screens above
  except the two named below.

### Changed

- **The reference agent is unassisted** (0.5.0 to 0.7.0). Its one tool is `submit_answer`. It
  has no memory tool, no `locate` and no instruction on what to keep, and it remembers with its own context, which
  Codex compacts itself. Rows made with 0.4.x's scaffold are not comparable with its rows.
- **The brief gives no help** (0.7.0): no line on what to keep and no list of the kinds and colours a question may
  use. Its during sentence names the kinds of question asked during the job. Each frame's header carries the robot's
  subgoal, which the `neutral` brief withholds.
- **The headline** (0.5.0-control and 0.6.0) is the mean of the `now` and `memory` columns, which count equally
  whatever their numbers of questions, with the abilities equal inside a column. The ability mean and the pooled mean
  are published beside it.

### Removed

- **Retired families.** No job asks `zone_clear`, `out_of_view`, `adjacent`, `where_was_i` or `distance_between`. The
  change families `object_was` and `same_place` left every job: at sourced change rates they ask too few questions to
  read. Their modules and tests stay.
- **The `rgbd_pose` channel and the agent-facing depth** (0.6.0), and the `swapped` brief: the subgoal states the true
  task in every frame, so a swapped brief would contradict it.
- **The reference agent's memory tools and `locate`.**

### Not in 0.7.0

- **A held-out confirmation.** The rows are dev rows: the questions, the screens and the instrument repairs were
  settled on these 64 shifts, after the dev pilot's answers there were read, and no fresh shifts were rendered to
  confirm them. The gold is public, and each place is one layout.
- **A hard drone.** The drone sits near the top of what its sourced 2.4 m clearance allows. This is reported, not
  tightened; streets that would ask more were not built.
- **Memory at shift length.** The rows cover 1 to 3 tasks. The 16 long cases (4 to 6 tasks) have no 0.7.0 row, and
  whole shifts at sourced lengths were not built.
- **More places:** one apartment, one warehouse, one downtown.
- **A clean apartment floor.** A cable-like mark in the apartment's floor texture, which sol twice took for a cable
  bundle in the dev pilot; removing it needs a re-render. One question sits at exactly one pixel's error, under the
  rule declared before data.
- **Change over time and prediction** (too few questions at sourced rates), and a closed-loop check that the score
  predicts control.

## [0.4.1] - 2026-10-10

The long tier. 0.4.1 adds the long windows (4 to 6 tasks) to 0.4.0 and changes no definition: `compile/13`,
`score/4` and `baselines/6` stay as they were, so every 0.4.0 case is unchanged and its scores stay valid on 0.4.1's
short and medium cases (DECISIONS, "The long tier follows as 0.4.1, decided before its data is read"). The release is 46
scripts and 134 cases in all three lengths: 12 scripts (seeds 2001 to 2012, 36 cases, 12 of each length) of each of
`helper/1`, `amr/1` and `drone/1`, and 10 (26 cases: 10 short, 10 medium and 6 long) of `vacuum/2`; it holds
1,076 questions and 17,904 frames, per job as below.

| Job | Place | Scripts | Cases (short / medium / long) | Questions | Frames |
| --- | --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 10 | 10 / 10 / 6 | 168 | 5,978 |
| `helper/1` | `apartment_a` | 12 | 12 / 12 / 12 | 200 | 3,076 |
| `amr/1` | `warehouse_a` | 12 | 12 / 12 / 12 | 463 | 5,899 |
| `drone/1` | `downtown_a` | 12 | 12 / 12 / 12 | 245 | 2,951 |

### Added

- **The long tier.** The long windows (4 to 6 tasks) of the scripts 0.4.0 shipped: 12 each of `helper/1`, `amr/1` and
  `drone/1` (seeds 2001 to 2012) and 2 of `vacuum/2` (seeds 2001 and 2002), which 0.4.0 cut but did not ship because a
  long reference-agent run takes about an hour (DECISIONS, "The long tier is pinned"). Each script now ships as a
  `short`, a `medium` and, where its day has one, a `long` case; the vacuum's days of seeds 2003 to 2006 are days of 2
  or 3 tasks and have no long window. 0.4.1 is the first release whose memory curve reaches long windows. Whether a
  case outgrows the agent's context follows the frames it streams, not its length (see Changed).
- **Four fresh 4-to-6-task `vacuum/2` days (seeds 2013 to 2016),** rendered for the long tier: 12 cases, 4 of each
  length. The vacuum then has 10 short, 10 medium and 6 long cases, so its long cell has 6 scripts (a cell of 2 is
  below the minimum of 6 for an interval). Seeds 2013 to 2016 and not 4-to-6-task versions of seeds 2003 to 2006, which
  would share their worlds with the shipped 2-or-3-task days of the same seeds and put near-duplicate worlds in one
  job; and not seeds 2007 to 2012, which on `apartment_a` are the helper's worlds.
- **The AMR's 12 and the vacuum's 2 earlier long cases were recompiled** (`compile/13`, as before) before their rows
  ran, because they still held the dropped `when_seen`. A row ships only if its question ids are those of the case's
  gold on disk.
- **Reference rows over all three lengths:** `codex:gpt-6-luna` (xhigh) scores +0.407 [+0.312, +0.495] on
  `apartment_a`, +0.270 [+0.179, +0.359] on `downtown_a`, +0.361 [+0.303, +0.413] on `warehouse_a`;
  `codex:gpt-6.1-sol` (medium, the neutral brief) scores +0.804 [+0.735, +0.865] on `downtown_a`, +0.669 [+0.579,
  +0.774] on `warehouse_a`; `codex:gpt-6.1-sol` (medium, the swapped brief) scores +0.687 [+0.613, +0.761] on
  `downtown_a`, +0.654 [+0.605, +0.730] on `warehouse_a`; `codex:gpt-6.1-sol` (medium, the during questions off) scores
  +0.673 [+0.569, +0.793] on `apartment_a`, +0.601 [+0.530, +0.667] on `warehouse_a`; `codex:gpt-6.1-sol` (medium, the
  default arm: the correct brief, the during questions on) scores +0.713 [+0.633, +0.788] on `apartment_a`, +0.700
  [+0.612, +0.787] on `downtown_a`, +0.675 [+0.622, +0.734] on `warehouse_a`. The `results/5`
  documents are in the dataset export's `results/` folder and summarised in `docs/report.md`, section 4. The memory
  curve over case length has a long point, read per cell with its number of cases; a paired reading uses the scripts
  that have all three lengths.
- **The gates and the leak check on what ships.** The binding gate check, one whole-root `stbench check` over the 134
  cases, passed every family with no BROKEN line, so no family was dropped and every job ships all its cases (the rule
  decided before its data was read would have parked a job's long cases again). The leak check is read once more on this
  release's pool, which now holds the long pairs, by the rule declared before: d = +0.031, interval
  [-0.011, +0.070], over 483 pairs in 32 scripts, verdict
  inconclusive. Per job: `amr/1` d +0.048 [+0.014, +0.087] (inconclusive); `drone/1` n/a; `helper/1` n/a; `vacuum/2` d +0.007 [-0.079, +0.085] (inconclusive). The during questions ship; publish d, the interval and n, and state that the check cannot exclude a leak of the upper bound (+0.070). Both readings are
  published (0.4.0's, before the long pairs, is in its entry and in `docs/report.md`, section 3).

### Changed

- **README, the dataset card and `docs/report.md` state the new size and windows.** The "no long cases" limit is gone.
  The memory limit now reads that the context window is the agent's memory and that the frames a case streams, not its
  length, decide whether it outgrows the context: `gpt-6.1-sol` compacted in 5 of 10 `vacuum/2` medium, 4 of 6 vacuum
  long and 4 of 12 `amr/1` long case-runs and never on `helper/1` or `drone/1`, and `gpt-6-luna` in 12 of 12 AMR long,
  3 of 12 AMR medium, 5 of 10 vacuum medium, 4 of 6 vacuum long, 1 of 10 vacuum short and 1 of 12 helper long; the score barely moves with
  length. The limit of no question about change over time stands (the question redesign is 0.5.0).
- **The vacuum has 10 scripts, not 6** (seeds 2001 to 2006 and 2013 to 2016), and the card's links point at the tag
  `v0.4.1`.

### Unchanged

- Every definition and every 0.4.0 case: its questions, gold and frames are byte for byte those of 0.4.0, and its
  scores stay valid on 0.4.1's short and medium cases.

## [0.4.0] - 2026-10-09

The first draft of the redesign. SpatiotemporalBench becomes a benchmark of task-conditioned spatiotemporal ability,
perception and memory both: an agent with its own memory is the brain of a robot doing a realistic job, and is asked the
questions a controller would ask itself, as the job goes and after it (VISION.md; DECISIONS, "The redesign" and "The
first draft's scope"). It is four jobs on three places, one public split with the gold public, and no held-out set.
Release 0.3 (the memory-quiz benchmark) stays as tagged (`v0.3.0`) and on `main`, and nothing here is on `main` yet. The
release is 42 scripts and 84 cases (short and medium, no long): 12 scripts (seeds 2001 to 2012, 24 cases) of each of
`helper/1`, `amr/1` and `drone/1` and 6 (seeds 2001 to 2006, 12 cases) of `vacuum/2`; it holds 541 questions and 15,689
frames, per job as below.

| Job | Place | Scripts | Questions | Frames |
| --- | --- | --- | --- | --- |
| `vacuum/2` | `apartment_a` | 6 | 60 | 3,763 |
| `helper/1` | `apartment_a` | 12 | 113 | 3,076 |
| `amr/1` | `warehouse_a` | 12 | 236 | 5,899 |
| `drone/1` | `downtown_a` | 12 | 132 | 2,951 |

### Added

- **`stbench view RUN_DIR [--data DATASET] [--out FILE.html]`.** One self-contained HTML page of an `eval` run (or of
  its arms, or of one `run` folder) to troubleshoot a system with: a sortable, filterable table of the cases with their
  status and outcome counts, each case opening in place into its questions (the solver's answer, the gold with its credit
  and tolerance, the outcome and score, the evidence frames) and the transport log of an agent. It reads the folders as
  they stand, so it shows a run that is still going or has no `results.json`; it needs no server and no network; every
  string of the run is shown as text only; and it is refused under the dataset's `scripts/` and `private/`, since it holds
  the gold (docs/eval.md, "Looking at a run").
- **Four jobs on three places.** `vacuum/2` and `helper/1` on `apartment_a`, `amr/1` on `warehouse_a` and `drone/1` on
  `downtown_a` (`docs/jobs.md`, `docs/job-helper.md`, `docs/job-amr.md`, `docs/job-drone.md`). A script is up to six
  tasks of one robot's job with unobserved gaps between them, in which the place changes: things move, doors open and
  shut, and a resident, pickers or pedestrians come and go (`docs/world.md`, `docs/people.md`). A robot is a profile
  (`robot/1`): its body, its camera and mount, its speeds, its status and its keyframe policy, and nothing special-cases
  one robot.
- **Cases are windows of a script.** The release cuts each script (a day of 4 to 6 tasks; the vacuum's last four, seeds
  2003 to 2006, are days of 2 or 3) into a `short` (1 task) and a `medium` (2 or 3) case, each delivered from step 0 on
  a fresh clock, so nothing tells the agent where its window sits in the script. A `long` window (4 to 6) is defined and
  `stbench build` still cuts it, but the release ships none: a long case takes a reference-agent run about an hour
  (DECISIONS, "The long tier is pinned"). The long cases already run are kept, not shipped, and the long tier returns in
  a later release.
- **The in-stream protocol, `stream/2`.** The robot's camera, depth, pose and status arrive in batches of up to eight
  steps in time order. Questions are asked at the job's decision points (about the robot's own situation) and in one
  questionnaire after the job. The brief is composed by the harness at run time and never a file (`brief/1`: the job and
  the place, never what will be asked), in a `correct`, a `swapped` and a `neutral` variant. `finish` reports token usage
  per model, and each case has its own home and session nonce.
- **The question families** (`docs/questions.md`, `compile/13`). The first draft asks `where_was_i` (`vacuum/2`,
  `amr/1`) and `out_of_view` (`amr/1`) during the job, and `object_where` (all four jobs) and `distance_between`
  (`helper/1`, `amr/1`, `drone/1`) at its end. `helper/1` and `drone/1` ask nothing during the job. Each job maps the
  families it asks; a family that fails its gate check is dropped from its job, never re-tuned, and the drops are
  recorded. `vacuum/2` lost `changed` (M2's gate check) and `out_of_view` and `uncovered` (M2's fast gate check), and
  `order` and `distance_between` left its `ends` (under one question a case). At the release gate check (the first six
  scripts of each job, seeds 2001 to 2006), `helper/1` lost `where_was_i` (`other_case` +0.24), `out_of_view` (its
  learned shortcuts BROKEN after one repair round), `changed` (shortcuts: `blind` +0.33) and `order` (2 questions in 18
  cases); `amr/1` lost `zone_clear` (`blind` +0.67: its memory rows are clear bays), `rear_clear` (`memory_needed`),
  `changed` (2 questions in 18 cases) and `order` (4 questions); `drone/1` lost `zone_clear` and `clearance_then` (first
  `other_case` +0.27 and +0.21 on its first six scripts, on golds a fair coin draws; then both failed the one re-read
  declared before the data, over all 12 of its scripts, and are dropped for good) and `order` (0.28 a case). `when_seen`
  was dropped from `helper/1` and `drone/1` after the repair of its time shortcut (`baselines/6`), and from `amr/1` and
  `vacuum/2` by the binding gate check over the 84 cases that ship (the AMR's `questionnaire_blind` BROKEN; the vacuum's
  failed `memory_needed` and `shortcuts`, with `blind` and `path_log` BROKEN), so **no job asks `when_seen`**. The
  helper's, the AMR's and the drone's families are confirmed on the release data they ship on, with no separate
  development render, and every family that ships passed the binding gate check with no BROKEN line. **The first draft
  asks no `now` question**: every zone family failed where it was asked and no job asks `path_ahead`, so perception is
  in every answer but is not scored apart from memory. Every gold is code over the timeline and the visibility log and
  names its evidence. `path_ahead`, `rear_clear`, `zone_clear`, `uncovered`, `changed`, `order`, `clearance_then`,
  `slots_empty`, `when_seen` and `adjacent` are in the registry and asked by no first-draft job.
- **The validity gates over scripted systems** (`stbench check`): nine scripted systems and a form adversary, and five
  gates (`ceiling`, `knowable`, `memory_needed`, `shortcuts`, `separation`) read per layout, job and family, on a point
  and an interval verdict. A shortcut that learns from other scripts (`blind`, `path_log`, `current_view_prior`,
  `questionnaire_blind` and the form adversary) answers a pool only when its best response beats abstaining on a
  script it left out (`baselines/5`); a pool whose winner only wins in the pool is answered `unknown`. Time answers are
  learned as lags behind the clock of the question's ask, so they are credited on any day, and `other_case`'s are put back
  on the asking case's clock; each learned shortcut also runs under its pool's own best response, and the `shortcuts` and
  `memory_needed` gates read the larger of the two scores (`baselines/6`).
- **`stbench eval`**: a system over a dataset's cases, resumable and in parallel, with one result per layout
  (`results/5`), a spend cap priced by `prices/1`, arms for the leak check (`--arms during=on,off`) and the brief swap
  (`--arms brief=correct,swapped,neutral`), and `stbench eval compare`, which refuses results that differ in more than one
  lever (`docs/eval.md`).
- **`stbench pilot report`**: the pilot's measures (the variance, the design effect and the correlation between windows
  of a script, the leak check's verdict, the brief swap) and the rule that sets the release's script count
  (`docs/eval.md`). The leak check's final reading, on the pool declared before the arms were read (the vacuum's
  development and release cases and the AMR's release cases), is d = +0.033, interval [-0.026, +0.087], over 284 pairs
  in 28 scripts, verdict inconclusive. The AMR on its own is above the 0.05 margin (`docs/report.md`, section 3).
- **`stbench export`** (`release/4`): writes the folder to upload as the dataset (scripts, the private files that score
  and check need, the gate report, licence, credits, the dataset card, `release.json` and `MANIFEST.sha256`) and refuses
  anything that is not releasable: a leak-scan hit, a machine path, a stale compile, a failed gate or a partial result
  (`docs/format.md`, "A release").
- **The reference agents.** The Codex CLI agent (`codex:<model>`, `codex/2`) on `gpt-6-luna` (xhigh) and `gpt-6.1-sol`
  (medium), with no API key and no dollars; the free scripted baselines; and `examples/claude_code_agent`, an agent that
  drives the Claude Code CLI (built and tested against a fake, with no paid run yet), beside `examples/minimal_agent` and
  `examples/template_agent`. The Codex agent reads images only and refuses the `detections` channel, so there is no agent
  row on perfect perception: that reference is the scripted systems on `detections`. The reference rows are
  `gpt-6.1-sol` (medium) in its default arm (the correct brief, the during questions on) and in three more (the during
  questions off, the swapped brief and the neutral brief), `gpt-6-luna` (xhigh), and the five free scripted rows
  (`all_unknown` on `rgbd_pose`; `oracle`, `memory_table`, `current_view` and `path_log` on `detections`): ten
  `results/5` documents, in the dataset export's `results/` folder and summarised in `docs/report.md`, section 4.
- **A camera model and photoreal places.** Blender Cycles scene-linear passes developed through each robot's camera model
  (`render/1`; the vacuum's is fitted to real Roomba j7 frames) with free-licensed photoreal assets. The apartment's
  look for the vacuum was approved.

### Changed

- **The scorer, `score/4`.** A right answer scores +1, a near miss partial credit, `unknown` 0 where an answer exists
  and 1 where `unknown` is right, and a wrong answer costs more than `unknown`. The headline is reported beside its
  `now`, `memory` and own-path columns, with a bootstrap interval over scripts.
- **Pictures and harness run without Blender or a GPU.** The released frames are in the dataset; Blender is needed only
  to build new scripts. Code is MIT, `render/` (it runs inside Blender) stays GPL-3.0-or-later, and the dataset is
  CC BY 4.0.
- **One public split, gold public.** There is no held-out set and no lite set, and each place is one layout. The dataset
  card and `docs/format.md` say what a result on it can and cannot mean.

### Removed

- The Recall and Perceive question sets, the two labelled tracks (text and pictures; two channels replace them,
  `rgbd_pose` for agents and `detections` for the gates), the tiers (T1 to T3), the episode layout (`episodes/<id>/` with
  its mission, floor plan and protocol files), the city's movers and Release 0.3's `eval` and `export` (about 30,000
  lines; they are on the tag).

### Not in the first draft

- Long cases (4 to 6 tasks): the format and `stbench build` still define and cut them, but a long case takes a
  reference-agent run about an hour, so the release ships none and its memory curve stops at medium windows. The long
  tier returns in a later release.
- Held-out layouts, and with them `adjacent` (its code stays; it asks only on a held-out layout, so it asks nothing
  here), and the fulfilment hall.
- The `house_floor` and `suburb` maps, which are built and tested but not rendered.
- A time-of-last-sighting question: `when_seen` failed the gates in all four jobs (the helper's and the drone's by its
  time shortcut, the AMR's and the vacuum's at the binding gate check), so no job asks it. The family stays in the
  registry.
- A reading of task conditioning: every question is critical to its task by construction and each brief swap pairs
  tasks with the same relevance, so the swap measures how much the assignment's wording moves the answers (on the drone
  the neutral brief scored above the correct one: correct minus neutral -0.117 [-0.219, -0.025], 132 pairs of 12
  scripts), not whether the agent keeps what its task needs.
- A `now` question: every zone family failed its gate check where it was asked, so the `now` column is empty and
  perception is not scored apart from memory. A later release needs a `now` family whose answers the place does not
  predict.
- An agent row on perfect perception: the Codex agent refuses the `detections` channel.
- A closed-loop subset, where the agent chooses the robot's actions, and a real-capture calibration set. The score is
  not claimed to predict control.

The two entries below, dated 2026-10-07 and 2026-10-05, are the notes this branch kept while it was built, as written.

### Changed (2026-10-07): the redesign's first slice (M1), on its own branch

The benchmark is being rebuilt as a task-conditioned benchmark: an agent with its own memory is the brain of a robot
doing a real job, and is asked the questions a controller would ask itself (the goals are now in
[docs/vision.md](docs/vision.md)). This is on its own branch, not `main`; Release 0.3 stays as tagged (`v0.3.0`).
Nothing here is released yet.

- **Retired** (the sweep): the Recall and Perceive question sets, the episode-tier orchestration, the city's movers and
  the old eval and export commands, about 30,000 lines (they are on the tag).
- **The contracts** (M0): `robot/1` (four robots' bodies and cameras), `timeline/1` with a job registry and neutral
  events, scripts and cases (`case/1`, a case's steps and clock start fresh), `stream/1` (start, task, observe, ask,
  finish), typed answers and their records, the free-space grid, and the apartment map's edit.
- **The slice** (M1): the vacuum cleaning one room of `apartment_a` (`vacuum/1`, generator 0.8.0), the brief
  (`brief/1`), `stbench run` and the runner, the end questionnaire's `object_where` (`compile/10`, `oracle/2`), the
  penalised scorer (`score/3`), nine scripted baselines and the five gates (`stbench check`), the Codex reference agent
  on `stream/1` (`codex/2`), a camera model fitted to real Roomba j7 frames and the room light (`render/1`), and 42
  clutter models placed as world content. The apartment's look for the vacuum was approved on 2026-10-07.
- **Measured:** 20 dev scripts rendered (3,170 frames); every gate passes on `object_where`; the Codex smoke run
  completes on both test models (FINDINGS, "M1 closed").
- **Dropped from M1:** `path_ahead`, the during question at the doorway, because the robot's own path log answered it
  (it waits for M2's sites).

### Changed (2026-10-05): `perceive/2`, after the first model runs on the Perceive track

The first model runs found that the warehouse's rack stock (beige cartons on bare wooden pallets, no entity) matched
"all the pallets", "all the boxes", the warehouse `findings` phrase and kind-only `where_went` referents on the
pictures. An independent review then found smaller defects (FINDINGS, "The first model runs on the Perceive track";
DECISIONS, "`perceive/2`"). Nothing released changes: the Perceive set is not in a release, and Recall's questions,
gold, prompts and `results.md` are byte-identical.

- **Fixed scenery** (`scenery.py`). The Perceive mission adds one sentence per catalog with non-entity look-alikes: the
  warehouse's rack stock is part of the building and no question is about it. The look-alike colours are declared per
  kind (box, pallet); a test keeps them off every colour a tracked thing of that kind can have and ties them to the
  render's carton colour.
- **A recognisability floor.** A thing counts in a visit only when one of its seen keyframes there has at least 1,024
  px and a luminance of at least 0.01. Below that it is a don't-care place, or its question is not asked.
  `where_went`'s rule 5 counts only look-alikes that were recognisable since the reset; the strongest context guess
  then scores 0.000 on the kept episodes.
- **The `findings` wording.** The apartment's names small household items and leaves out furniture and robots; the
  warehouse's leaves out the rack stock.
- **Capture attribution.** A miss is "captured, then lost" when its twin scored above it, and each cell reports
  `capture_loss` and `retention_loss`, and names the rule that split its misses (`split`, `twin>base`). Perceive results
  print the mean credit beside the penalised mean. This is `results/4`; `results/3` still loads, its retention loss
  shown as n/a and its `split` `twin=1`.
- **Prompts** `notes-perceive/2` and `notes-rgb-perceive/2` say that a place a question gives as (x, y, z) is the
  thing's middle.
- **`codex:` takes `--reasoning xhigh`.** An effort the model's Codex catalog entry does not list is refused before any
  exec; `llm:` refuses `xhigh`.
- **An episode with no Perceive set.** An episode whose compile would leave a checkpoint empty gets
  `perceive/none.json` (`{compiler, reason}`) instead of the set; `compile` prints the reason and exits 0, `run` and
  `score` refuse it, `check` and `eval` skip it with a note, and `export` records it. On the kept episodes one apartment
  T1 episode has it, so 149 carry the set.
- **Measured at `perceive/2`.** On the 150 kept episodes of Release 0.2 (in a scratch copy: 149 of them carry the
  set), 3,469 questions, 1,837 scored and 1,632 capture twins, and every gate passes on both maps. On the city's 60
  episodes `perceive/1` passed every gate (1,632 questions); `perceive/2`'s yields and gates there are measured at the
  merge that brought Release 0.3 into this branch (FINDINGS).
- **The downtown's `findings` wording.** Its phrase and mission sentence say "things you keep track of" and leave street
  furniture (lamp posts, signal poles, hydrants) out; the predicate is unchanged. The downtown has no scenery sentence:
  its scenery is street furniture in the kerb strips and a bare carriageway past the map's edge, no look-alike of a
  tracked kind.

The Perceive track, built after Release 0.2 and merged into `main` on 2026-10-04, is carried by Release 0.3 as code
only (below); its data is in no release yet and ships in the release after 0.3, compiled on all three maps. No kept
data, question or result of Release 0.2 or 0.3 changes: the set is compiled beside Recall's, and only scratch model
runs exist on it.

## [0.3.0] - 2026-10-04

The city. Release 0.3 adds a third map, `downtown_a`, to Release 0.2's apartment and warehouse, and a second robot
profile to see it with: a drone flying 12 m above the streets. It is 210 episodes, 34,228 frames and 6,058 questions,
all counted, scored and gated per map, never pooled. The apartment is 90 episodes (T1-T3, 30 seeds each), 14,352 frames
and 2,557 questions, and the warehouse 60 episodes (T2 and T3), 10,098 frames and 1,681 questions, both with questions
and gold identical to Release 0.2's once the question ids are dropped; the downtown is 60 episodes (T2 and T3, seeds
1-30), 9,778 frames and 1,820 questions (T2 933, T3 887). Every gate passes on all three maps: the apartment's 17 rows,
the warehouse's 13 and the downtown's 13 (7 in T2 and 6 in T3), the downtown's after one fix to the `memory_table`
baseline (Fixed, below).

Release 0.3 also carries the Perceive track, built after Release 0.2 and merged into this release's branch on
2026-10-04: the code of a second question set per episode (`perceive/1`) and the schemas it moved, but none of the
set's questions, which are compiled into the release after 0.3 (every episode, since `export` refuses a partial set).
No Recall question, gold or score moves, and the downtown's 13 gate rows equal the full build's check on every number.
No model has been run on the new set.

### Added

- **A third map, `downtown_a`.** One fixed 100 x 96 m downtown district in the apartment's map-file style: a 27 m high
  street running east-west, two 20 m side streets (`west street` and `east street`) running 45 m north from it between
  blocks, and a 30 x 24 m `square` south of it behind the middle block, with buildings 18.9-44.1 m high on six facade
  styles. The `room` slot of a question names the high street, a side street or the square. It holds 22 parking bays, 2
  loading zones, 8 sidewalk strips and 6 paving pads (38 surfaces for items), and its street ends open onto static
  scenery. Tiers T2 and T3, seeds 1-30 (60 episodes), with the same eight question families and no new one.
- **A second robot profile, `drone`.** A body-free camera 12 m up, pitched 35 degrees down, at least 3 m from a facade,
  flying at 3 m/s (a keyframe every 4.5 m). It flies over everything lower than 9 m (street furniture, parked vehicles,
  the movers) and only buildings obstruct it, so the route grammar (three visits, 140-180 keyframes) holds unchanged.
  `stbench generate` and `stbench build` take `--profile drone`. A profile now carries its speed and whether it flies
  (`Profile.key_spacing`, `Profile.flies`), because they belong to a robot; the ground profiles' paths are unchanged.
- **The `downtown` catalog.** Eight carried kinds, 43 (kind, colour) pairs, 26-34 items a world: `car`, `van`, `pickup`,
  `box truck`, `skip`, `dumpster`, `portable toilet` and `food cart`, parked along the kerb, on the sidewalks and on the
  square's pads. Identical sets of portable toilets and dumpsters; a van is the delivery, and cars, vans and pickups
  leave. The T3 movers are a taxi (4-8 m/s) and a 9 m midibus (3-7 m/s), which keep to the roadway. The mission is a
  city drone's. There are no light pieces of furniture, so nothing shifts in a gap.
- **Procedural vehicles and a street scene, no new download.** Ten procedural models (the eight kinds and the two
  movers), sidewalks with kerbs, painted bays and loading zones, the square's paving, six facade styles, scenery
  beyond the open street ends, the map's fixed sun and a sky (an outdoor scene, with no ceiling), rendered once like the
  other maps. The renderer is GPL-3.0 like the rest of `render/`.
- **Depth in the map's unit.** A map says whether its depth PNGs are in millimetres (the default, as before) or
  centimetres (`depth_unit`; `camera.json` always carried it and every reader already scaled by it). The downtown's are in
  centimetres: from 12 m up at 35 degrees, 1.4% of a frame lies 65.5 m or further, beyond what 16-bit millimetres can
  hold, and a depth of 0 means "nothing nearer". `check_map` refuses a map whose size does not fit its unit.
- **The downtown's free rows** (the scripted baselines and `examples/minimal_agent`) are in the README, the dataset card and
  the report (0.3.0's technical report, section 6.3): the oracle and `memory_table` score
  +1.000 in both tiers, `no_memory` and `all_unknown` the floor (+0.100 in T2, +0.158 in T3), and the minimal agent
  +0.243 in T2 (it logs its poses, so `ego_where` is +1.000) and the floor in T3. Its model rows (`codex:gpt-6-astra`,
  `gpt-6-sol` and `gpt-6-luna`) are pending, because the Codex plan's weekly quota resets on 2026-10-11; they run then
  with the commands of the report's section 9 and `--maps downtown_a`.
- **The Perceive set's code (`perceive/1`), with no Perceive data in this release.** A second question set per episode,
  for a system that does its own perceiving (VISION §3): four families, `inventory` ("Where were all the boxes you saw
  during your visit starting at ...?"), `findings` (the same for what the mission names), `where_went` (where did the
  thing you saw at (x, y, z) go?) and `distance_between` (the horizontal distance between two things), with a new
  `positions` answer (a list of places, scored by F1). Compiled from the same timeline and visibility log into
  `private/<id>/perceive/` and `episodes/<id>/perceive/mission.md`; no new render. Measured in scratch copies: on the
  150 kept episodes of Release 0.2, 3,789 questions (2,061 scored, 1,728 capture twins), and on the city's 60, 1,632
  (974 scored, 658 twins); every gate passes on all three maps (18 + 8 rows).
- **Capture twins and an attribution report.** Every `inventory`, `findings` and `distance_between` question is asked
  again, as the same question, at the end of the visit it is about (a checkpoint at the end of visit 1). A twin is never
  scored; `stbench score`, `results.json` and `results.md` report each miss as never captured or captured, then lost.
- **`--set recall|perceive`** on `compile`, `run`, `score`, `check` and `eval` (default `recall`). A run answers one
  set; with `--set perceive` and no `--channel` the channel is `rgb`, the Perceive track (`oracle` is the diagnostic
  `perceive-text` row). The set tags the default answers file and run folder, so two sets never overwrite each other.
- **Findings predicates per catalog** (`src/stbench/findings.py`): the apartment's floor, rugs and mat, the warehouse's
  floor level in the aisles, and the city's side-street sidewalks and square (a carried thing with its base at 0.10 m or
  higher there). The mission file of the set adds one sentence that says what they are.
- **`stbench check --set perceive`**: Recall's six gates plus `no_memory_now` in `memory_needed` and a seventh gate,
  `pictures` (a perfect perceiver with a perfect memory scores at least 0.95); the report is
  `check_report.perceive.json`. The baselines answer the new families (`table_perceive.py`, `blind` and `pose_only` over
  dense places) and the pictures ceiling is `pictures_ceiling.py`.
- **State size.** The harness records the bytes of a system's state directory at each checkpoint and reports it beside
  the scores (`results/3`, `answers/<id>.state.json`); there is no budget.
- **The documentation:** `docs/plan-perceive.md` (the design and its measurements), `docs/questions.md` ("The Perceive
  set"), and the Perceive files in `docs/format.md`, `docs/agent.md` and `docs/episode.md`.

### Changed

- **`compile/9`.** The in-view rule's far bound is 150 m (it was 30 m). From 12 m up the drone sees a car out to about
  69 m and a box truck to about 116 m, and a removal it saw from 40 m was neither an event nor a don't-care region at 30
  m, so a correct report of it was charged. The bound cannot decide on the other two maps, whose largest in-map distance
  is 27.3 m (the warehouse) and 11.2 m (the apartment), so their questions and gold are unchanged: all 150 episodes of
  Release 0.2 recompiled in memory at 150 m are identical to their stored gold without the question ids (4,238
  questions), and so are all 150 on the build's recompile of all 210 episodes at `compile/9` (150 of 150 identical
  without the question ids and the compiler stamp). Question ids are random per compile, so the apartment's and the
  warehouse's are new again, and a result made on Release 0.1's or 0.2's episodes cannot be paired with one made on
  Release 0.3's (`eval compare` refuses different gold): compare within a release. The apartment's Release 0.1
  reference rows and the warehouse's Release 0.2 rows stand for 0.3 (identical questions and gold), as the apartment's
  did for 0.2.
- **Generator 0.7.0.** The downtown catalog, the `drone` profile, routes planned per profile (a flying profile plans with
  its own clearance and spacing over everything under 9 m), the flying rules in the movers' walks and the validator, and
  a world with no light furniture draws its shift count and shifts nothing. `PLANNING_CLEARANCE` is the largest of the
  ground profiles' only, so no ground path moves. The apartment's and the warehouse's worlds are byte for byte
  unchanged: their kept-world pins hold, and their episodes keep their stamps (0.4.0, 0.5.0 and 0.6.0).
- **The map file has three optional keys,** `depth_unit`, `outdoor` (the sun and sky of an outdoor map) and a wall's
  `facade`. `apartment_a.json` and `warehouse_a.json` do not use them, so their bytes and hashes are unchanged.
- **Not changed:** `runner/4` (the quotas, so the cap of 21 questions a checkpoint), protocol 1, `oracle/1`,
  `notes-rgb/2`, `rgb-image/1` and `locate/1`. The scorer and the two schemas moved with the Perceive track (`score/2`,
  `results/3`, `release/3`, below), and none of them changes a Recall number. The reference agent's prompt and tools are
  the ones of Release 0.2, so every earlier reference row is the same system. `stbench check`, `eval`, `eval compare`
  and `export` already work per map and need no change for a third.
- **The README, the dataset card, the report and the docs say three maps**: the results are one table a map, and the
  downtown's pictures-track ceiling is stated beside the warehouse's.
- **`score/2`** adds the `positions` answer type (one F1 core for `set` and `positions`); it scores every `score/1`
  input exactly as `score/1` did (a test pins it), so Recall's results do not move.
- **`results/3`** adds `set`, the tracks `perceive` and `perceive-text`, per-episode `state_bytes` and a `capture`
  report; `results/2` and `results/1` still load, read as Recall's. `eval compare` refuses results of two sets.
- **`release/3`** names each question set a release holds: per episode `sets: {set: {questions, gold_sha256}}`,
  `compilers` and `check_reports` per set, and `counts.questions` per set, map and tier. A release carries the Perceive
  set for every episode or for none. `release/2` is still what Release 0.2 is; Release 0.3's `release.json` is
  `release/3`, with the one set `recall`.
- **A Docker system has three checkpoint windows on the Perceive set**, 21 minutes an episode at the default slot, not 14.
- **The reference agent reads its own prompts on the Perceive set** (`notes-perceive/1`, `notes-rgb-perceive/1`); the
  Recall prompts (`notes/1`, `notes-rgb/2`) keep their bytes.

### Fixed

After the full build:

- **`memory_table` no longer reports its own association error as a change.** The first `stbench check` of the 210
  episodes failed the downtown's `changed` on `knowable` in both tiers (+0.09 and +0.08 against 0.02); every other row
  of all three maps passed. The gold was right (the oracle scores 1.000 on all 269 downtown `changed` questions): the
  baseline's tracker joined the two yellow loading zones, one in each side street and never in one frame, into one
  track, and reported the zone in the street it ended a visit in as added or removed. `table_answers.changed` now makes
  no item of a description seen at two places at least the moved threshold apart within one visit, as the compiler does.
  With it downtown `changed` has `knowable` 0.00 and `solvable` 1.00 in both tiers; no other baseline's mean moves on
  any map and Release 0.1's and 0.2's baseline rows stand. The questions, the gold and every version are unchanged
  (DECISIONS, "The city's `changed` failed `knowable` through a baseline bug").

The independent reviews of the downtown's code and render, before the full build:

- **One boundary for "flown over".** A thing is flown over exactly when its top is below the height the drone clears,
  decided in one place (`profiles.flies_over`) that the route, the validator, the light-furniture guard and the movers
  all call. Furniture at exactly that height used to obstruct while a mover at it did not; no kind or piece is at 9 m,
  so no episode changes. The route's key spacing comes from the profile for every profile.
- **A map's depth unit is `mm` or `cm`, never `m`.** A metre step rounds a depth by up to 0.5 m, beyond the 0.1 m
  in-view tolerance, so a visible point would read as hidden behind its own surface. `write_episode` takes the map's
  depth unit when a caller gives none.
- **A mover's body may reach the scenery road past an open street end.** The movers' free grid keeps a mover's centre
  inside a region, so a taxi or the bus body reaches past an open end (up to 4.35 m for the bus and 1.9 m for the taxi
  over the 30 T3 episodes of the build, with no overlap of scenery); the renderer keeps that carriageway free of scenery
  objects. The documents said that nothing leaves the map; they now say this.

### Known limitations

- **The downtown has no people, no T1 tier, no trees in the map and no light furniture.** People need an asset download
  and posed meshes, and nothing shows that they would be recognisable from 12 m up; canopies over the sidewalks would
  hide items at random; a light piece cannot stand on a blocking sidewalk. The sun is fixed (no weather, no time of
  day), so shading never changes between visits without an entity to explain it. The vehicles and street kinds are
  procedural shapes with a vehicle's drawing, not scanned models, and were checked for recognisability on the render
  (rule 10), not on real footage.
- **The pictures track has a much lower ceiling on the downtown's position questions.** A position gold is the centre of
  the thing's box, and `locate` returns a point on what the drone sees, a vehicle's roof and near side. Measured on the
  full build (the 493 answerable position golds of the 60 episodes; 0.412 over all 2,738 answerable candidates), a
  perfect perceiver with a perfect memory that reports the centroid of a thing's visible pixels has a mean position
  credit of 0.447 on the downtown (the warehouse 0.81, the apartment 1.00): `where_at` 0.30, `where_last` 0.49 and
  `where_then` 0.50, and 0.19 for the bus; one that boxes the visible points reaches 0.78, and on the dev pass the
  reference agent's single `locate` pixel at the middle of a thing gave 0.32. A mean credit of 0.447 is a penalised
  score of about -0.11 on an answerable position question, below the 0 of answering `unknown`, for an agent that always
  answers with that centroid. The prompt (`notes-rgb/2`) and the ladder are unchanged so that every published pictures
  result stays comparable; compare pictures rows within a map. The text track is unaffected.
- **A difference between the downtown and another map is not a profile effect.** Only the drone patrols the downtown and
  only ground profiles patrol the other maps, so the map and the robot change together (VISION rule 13). The downtown's
  answerable `where_at` golds lie a median 24.8 m from the camera (1.9 to 55.9 m, over the 129 asked), the warehouse's
  5.4 m and the apartment's 2.6 m. A drone indoors and a human-height camera in the street are deferred.
- **The map's bays are public.** The packaged map file lists the parking bays and loading zones, and in the plan's proxy
  62% of the `where_last` golds lie within 1 m of one. `blind` and `pose_only` are gated against it like every shortcut
  (at most the unknown share + 0.10). On the built set `blind` scores +0.075 on `where_last` (T2 and T3) and +0.083
  (T2) and +0.100 (T3) on `when_last` against the floor's +0.25, `pose_only` +0.075 on `where_last` and +0.046 and
  +0.071 on `when_last`, and no shortcut of any downtown row is above the unknown share (the `shortcuts` column reads
  +0.00 on all 13).
- **The warehouse's light-equipment residual is not fixed in 0.3.** The fix would change `changed` and `moved_between`
  questions in 11 apartment and 49 warehouse episodes, so all 14 earlier reference rows would be rerun for a +0.02 to
  +0.03 residual that is under the gates' margin and published. It goes with the first release that reruns those rows
  anyway; this supersedes 0.2's "fixed at the next compile".

## [0.2.0] - 2026-10-04

The warehouse. Release 0.2 adds a second map, `warehouse_a`, to Release 0.1's apartment: 150 episodes, 24,450 frames and
4,238 questions, all counted, scored and gated per map, never pooled. The apartment is 90 episodes (T1-T3, 30 seeds
each), 14,352 frames and 2,557 questions, with questions and gold identical to Release 0.1's; the warehouse is 60
episodes (T2 and T3, seeds 1-30), 10,098 frames and 1,681 questions. Both maps pass every gate (the apartment's 17 rows,
the warehouse's 13).

### Added

- **A second map, `warehouse_a`.** One fixed 17.4 x 19.8 m warehouse in the apartment's map-file style: a dock with four
  dock doors, four staging lanes, two packing tables and two shelving units, and three look-alike dead-end aisles of
  single-deep pallet racks (72 bay-levels, half of them free surfaces for items and half static stock that is part of
  the rack's drawing). Tiers T2 and T3, seeds 1-30 (60 episodes), with the same eight question families and the regions
  `dock`, `west aisle`, `middle aisle` and `east aisle` in the `room` slot. The warehouse robot's mission says that it
  is asked where it has been and when.
- **Warehouse stock and equipment.** Six carried kinds, each item with one colour (pallet, tote, drum, box, sack, bin;
  22-28 items a world; the box is not brown there, because the racks' stock is brown cartons), three pieces of light
  equipment (two pallet jacks, a roll cage) and two T3 movers (a forklift and a tote robot). Nine new procedural kinds,
  a procedural concrete floor, racks with stock and wire decks and high-bay lights, rendered with no new download: the
  only downloaded assets are CC0 ones already in the library (the cardboard box, two ambientCG textures).
- **`--maps` on `stbench eval`, `stbench check` and `stbench export`**: run, check or export some of the maps a dataset
  holds (default all), so the warehouse can be evaluated or re-checked alone, or exported as a folder of its own.
- **`docs/figures/frames.jpg`**, four camera frames of each map, shown at the top of the README and the dataset card
  (CC BY 4.0, like the dataset; it is not in the sdist).
- **The warehouse's reference rows** (`codex:gpt-6-astra`, `gpt-6-sol` and `gpt-6-luna` on both tracks, 60 of 60
  episodes each): on text astra scores +0.959 and +0.955 in T2 and T3, sol +0.778 and +0.856, luna +0.294 and +0.436; on
  pictures only astra is above the floor, in T2 (+0.403 against +0.100; +0.171 against +0.158 in T3), and sol and luna
  are below it (0.3.0's technical report, section 6.2).

### Changed

- **`compile/8`.** The in-view rule's far bound is 30 m (it was 8 m): a removal the robot could see from further than 8
  m was neither an event nor a don't-care region, so a correct report of it was charged. The never-seen pool is drawn
  from the map's own catalog. The apartment's 90 episodes were recompiled at `compile/8` and keep exactly the questions
  and gold of `compile/7` (checked over all 90 episodes and 2,557 questions), but their question ids are new, because
  ids are random per compile. So the apartment's Release 0.1 reference results stand for Release 0.2, and a result made
  on Release 0.1's apartment cannot be paired with one made on Release 0.2's (`eval compare` refuses different gold):
  compare within a release. "Never recompile a release" still holds.
- **Generator 0.6.0.** A `Catalog` per map (kinds, deliveries, movers, mission, square yaws, a mover's own camera
  clearance). The apartment's worlds are byte for byte unchanged.
- **`stbench check`, `eval`, `eval compare` and `export` work per map.** `check` gates every (map, tier, family) with
  baselines that learn from the same map's episodes only, and `check_report.json` is now `{versions, passed, maps,
  episodes, wall_seconds}` (each map's block is the old top level). `eval` writes `results.json` as `results/2`: every
  episode and row names its map and the aggregate is one per map (`maps: {map: aggregate}`), and `results.md` has a
  block per map. `eval compare` has a column per (map, tier) and still reads Release 0.1's `results/1` files (as the
  apartment's).
- **`stbench export` writes `release/2`.** `release.json` names each episode's map and counts the episodes per map and
  tier (`counts.by_map`), not per tier over both maps; the dataset card lists a row per map from those counts.
  `release/1` (Release 0.1, the apartment alone) had `counts.by_tier` and no map per episode.
- **The README, the dataset card and the docs say both maps**: the results are one table a map, and the apartment's are
  Release 0.1's runs.

### Fixed

The independent review of the warehouse, before the full render:

- **No brown box in the warehouse.** The warehouse's `box` kind keeps white, blue, red and green (so 31 (kind, colour)
  pairs, not 32): every rack's stock is brown cartons and no entity, so "the brown box" matched hundreds of things that
  are not items.
- **A wire deck on the free upper bay-levels** (levels 1 and 2): 846 of the dev set's 1,140 elevated placements rested
  on nothing between two beams.
- **The aisles are named `west aisle`, `middle aisle` and `east aisle`** (the map's hash changed): the question
  templates said "in the aisle 3".
- **The export is per map** (`release/2`; the card has a row for each map and no longer says "one apartment").
- **The pictures track's `locate` ceiling is documented per map**: a mean position credit of 0.809 for a perfect
  perceiver on the warehouse, 1.000 on the apartment (measured on every episode of both maps).
- **Documentation drift:** a mover's speed is no longer said to be under 1 m/s (the forklift does 0.8-1.5), and the
  README, the dataset card, `V0.md`, the maps README and `docs/format.md` say `compile/8` or both maps where they said
  `compile/7` or "one apartment".

### Known limitations

- The warehouse has no people, no blocked-aisle questions (the dock sees down every aisle, so too few are askable) and
  no T1 tier; the forklift's and the tote robot's speeds have no cited source.
- The pictures track has a lower ceiling on the warehouse's position questions: a position gold is the centre of the
  thing's box and `locate` returns its visible surface, so a perfect perceiver with a perfect memory scores a mean
  position credit of 0.809 there (forklift 0.553, pallet 0.682) against 1.000 on the apartment. The prompt
  (`notes-rgb/2`) is unchanged so the apartment's released pictures results stay comparable; compare within a map.
- The warehouse's pallet jacks and roll cage never move before visit 0, so their visit-0 place is public (the packaged
  map's `center`): 10 `moved_between` questions are answerable without memory, +0.022 (T2) and +0.023 (T3) on the
  family's mean, under the gates' shortcut margin. It is fixed at the next compile, not by recompiling this release.

## [0.1.0] - 2026-10-04

The first release: a benchmark of an LLM's or agent's spatio-temporal memory of a world it perceives through a robot's
sensors, with the dataset, the harness that runs and scores it, and a reference agent.

### Added

- **The dataset: 90 episodes.** One apartment, patrolled three times in a day by a simulated home robot, in three tiers
  of 30 seeds each: T1 (a static world), T2 (things are moved, added and removed between patrols) and T3 (T2 plus things
  that move by themselves). 14,352 rendered frames (RGB and 16-bit depth) and 2,557 questions in 8 question families
  (`where_last`, `when_last`, `ego_where`, `ego_last_in`, `moved_between`, `changed`, `where_then`, `where_at`), with
  typed answers (a map position, a clock time, yes/no, a set of changes, or `unknown`). Every gold answer is computed by
  code from the simulator's full record of the world, never written by hand or by a model. The dataset is CC BY 4.0
  (Hugging Face: `DanMcInerney/spatiotemporalbench`).
- **Two tracks, always named.** *Pictures: memory through perception* (`--channel rgb`: RGB and depth images and the
  robot's pose) and *text: memory only* (`--channel oracle`: depth, pose and a perfect list of what is in view). A
  result says its track. The per-family gap between the two, for an agent, is the cost of the pictures track:
  perception, and what the agent writes down about what it sees.
- **The `stbench` commands** `run`, `score`, `eval` and `export`, beside the build commands (`generate`, `validate`,
  `compile`, `render`, `perceive`, `build`, `assets`, `probe`) and `check`. `eval` runs a whole dataset (`--tiers`,
  `--seeds`, `--jobs`), resumes where it stops and writes `results.json` and `results.md`; `eval compare` sets results
  side by side with paired differences and refuses results scored against different gold; `export` writes the folder
  that is uploaded as the dataset (episodes, the private files that score and check need, licence, card, manifest).
- **Reference agents.** `llm:<model id>` makes paid model calls through OpenRouter, from your machine, under a dollar
  cap; `codex:<model id>` runs the same agent (the same prompt, chunks, memory tools and `submit_answer`) through the
  Codex CLI on a ChatGPT plan, with no key and no dollars. On the pictures track both have the `locate` tool, which
  turns a pixel of a frame into a map point from that frame's depth image and pose. A `codex:` and an `llm:` result for
  one model are two systems. `codex:` refuses (exit 2) a Codex home folder that holds a non-empty `AGENTS.md` or
  `AGENTS.override.md`, because Codex would add it to every exec as a message the reference agent is not sent.
- **Your own agent.** A Python class with `observe(steps)` and `answer(question)`, run in a Docker sandbox (the verified
  mode) or with `--local` as a plain subprocess (stamped `sandbox: none`, its score marked unverified: `run` and `eval`
  write the stamp beside the answers, `score` prints it and puts it in `score --json`, and `eval compare` marks the row
  `(unsandboxed)`). Two examples (`examples/minimal_agent`, `examples/template_agent`) and `docs/agent.md`.
- **Reference results and a technical report.** `codex:gpt-6-astra`, `codex:gpt-6-sol` and `codex:gpt-6-luna`
  (reasoning low) on both tracks over all 90 episodes, the scripted baselines and the minimal example agent (an
  unsandboxed `--local` run), in the README, the dataset card and `docs/report.md`, generated by `stbench eval compare`.
  On the text track astra is near the ceiling (+0.971, +0.931 and +0.953 in T1, T2 and T3), so that track separates
  weaker memories; the pictures track is the open one (astra +0.444, +0.580 and +0.658, far above sol, with luna below
  the floor). The report covers the design, the protocol, scoring, validity, the reference agent, the results on both
  tracks, limitations and how to reproduce them.
- **Validity gates.** `stbench check` runs the scripted baselines over the dataset and holds each (tier, family) row to
  six gates: ceiling, solvable, knowable, shortcuts, memory needed and separation. The baselines are an oracle and a
  memory table (the solvability references), `all_unknown` (the floor), and the shortcut adversaries `blind`,
  `pose_only`, `other_seed`, `no_memory_prior` and `form`, none of which may beat abstaining by more than 0.10. All 90
  episodes pass every gate on all 17 rows. A leak scan guards the agent-facing files.
- **The licences.** Code MIT; `render/` (it runs inside Blender's Python) GPL-3.0-or-later, as is any other script that
  imports `bpy`; the dataset CC BY 4.0, with the attributions for the third-party 3D assets in its `CREDITS.md`. The
  wheel carries the MIT code only (numpy and Pillow are its only dependencies); the sdist holds the code, docs and
  examples and none of the GPL scripts, the tests or the data (CI runs the tests from the repository).
- **Continuous integration** (`.github/workflows/ci.yml`): lint and the model-free tests on Ubuntu and Windows, and a
  build job that installs the built wheel in a fresh environment and runs `stbench --help`.

### Known limitations

- **One layout per environment.** The release is one apartment. Each episode starts a fresh agent, so a single layout
  acts as a new apartment every time, but nothing here says how a system fares in another building or on real footage.
- **The gold is public.** The release carries the questions, the gold and the truth they are computed from, so there is
  no secret test split: this is a development split, and results on it can be overfit. Report the track, the tier, the
  number of episodes and the versions in `release.json`. A `--local` agent runs with your permissions and can read the
  gold, and the Docker sandbox refuses only truth-file names in the agent's folder, so results are only as honest as the
  system: compare results only from trusted runs.
- **The pictures track mixes perception with memory.** Whether an object counts as "seen" is a fixed pixel bound, not a
  measured recognisability, and a position answer needs a picture and a pose turned into map metres. For about 38% of
  the answerable questions about an item or a mover the largest view of it in any evidence frame is under 1,024
  pixels.
- **Templated questions.** Each question is filled from a template of one of the 8 families, so the wording is
  narrower than a person's. Question ids are random per compile, so never recompile the release.
- **Developed on Windows.** The code was written and the data built on Windows. The model-free tests also pass on
  Linux (run in a Python 3.12 container) and CI runs them on Ubuntu and Windows. The Docker sandbox tests need Docker
  and the pinned image and are skipped without them, so CI does not run them; a verified (Docker) agent run takes about
  14 minutes an episode.

[Unreleased]: https://github.com/DanMcInerney/SpatiotemporalBench/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/DanMcInerney/SpatiotemporalBench/releases/tag/v0.11.0
